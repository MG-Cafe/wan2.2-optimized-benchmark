#!/usr/bin/env python3
"""
Parse SGLang benchmark logs into structured JSON results.

Usage:
    python3 scripts/parse_results.py results/run_YYYYMMDD_HHMMSS/

Reads all *_output.log files from benchmark result directories and extracts
timing data into results/benchmark_results.json
"""
import json
import os
import re
import sys
from pathlib import Path


def parse_sglang_log(log_path: str) -> dict:
    """Parse a single SGLang output log file for timing information."""
    result = {
        "log_file": str(log_path),
        "status": "unknown",
        "denoising_sec_per_step": None,
        "denoising_total_sec": None,
        "decoding_sec": None,
        "total_sec": None,
        "pipeline_stages": {},
        "oom_stage": None,
        "errors": [],
    }

    if not os.path.exists(log_path):
        result["status"] = "missing"
        return result

    with open(log_path, "r") as f:
        content = f.read()

    # Check for OOM
    if "CUDA out of memory" in content or "OutOfMemoryError" in content:
        result["status"] = "OOM"
        oom_match = re.search(r"(\w+Stage).*CUDA out of memory", content)
        if oom_match:
            result["oom_stage"] = oom_match.group(1)
        return result

    # Check for errors
    if "Error" in content or "FAILED" in content:
        error_lines = [l.strip() for l in content.split("\n") if "Error" in l or "FAILED" in l]
        result["errors"] = error_lines[:5]

    # Parse stage timing (SGLang stage logging format)
    # Pattern: "Stage: StageName took X.XXXs"  or similar
    stage_patterns = [
        r"(\w+(?:Stage)?)\s*(?:took|:)\s*([\d.]+)\s*s",
        r"Stage\s+(\w+)\s*:\s*([\d.]+)\s*s",
        r"\[(\w+)\]\s*([\d.]+)\s*s",
    ]

    for pattern in stage_patterns:
        for match in re.finditer(pattern, content):
            stage_name = match.group(1)
            stage_time = float(match.group(2))
            result["pipeline_stages"][stage_name] = stage_time

    # Parse "average time per step" (SGLang denoising output)
    step_match = re.search(r"average time per step[:\s]*([\d.]+)\s*s", content, re.IGNORECASE)
    if step_match:
        result["denoising_sec_per_step"] = float(step_match.group(1))

    # Parse total denoising time
    denoise_match = re.search(r"[Dd]enoising[:\s]*([\d.]+)\s*s", content)
    if denoise_match:
        result["denoising_total_sec"] = float(denoise_match.group(1))

    # Parse decoding time
    decode_match = re.search(r"[Dd]ecod(?:ing|er)[:\s]*([\d.]+)\s*s", content)
    if decode_match:
        result["decoding_sec"] = float(decode_match.group(1))

    # Parse total generation time
    total_patterns = [
        r"[Tt]otal[:\s]*([\d.]+)\s*s",
        r"Generated.*in\s*([\d.]+)\s*s",
        r"Pixel data generated.*?([\d.]+)\s*s",
    ]
    for pattern in total_patterns:
        total_match = re.search(pattern, content)
        if total_match:
            result["total_sec"] = float(total_match.group(1))
            break

    # Calculate denoising from per-step if not found directly
    if result["denoising_sec_per_step"] and not result["denoising_total_sec"]:
        result["denoising_total_sec"] = result["denoising_sec_per_step"] * 40  # 40 steps

    # Determine status
    if result["total_sec"] or result["denoising_total_sec"]:
        result["status"] = "success"
    elif result["errors"]:
        result["status"] = "error"

    return result


def parse_wall_time(results_dir: str, model_type: str) -> float | None:
    """Read wall time from the saved timing file."""
    path = os.path.join(results_dir, f"{model_type}_wall_time_ms.txt")
    if os.path.exists(path):
        with open(path) as f:
            return float(f.read().strip()) / 1000.0  # Convert ms to seconds
    return None


def parse_all_results(base_dir: str) -> dict:
    """Parse all benchmark results from a run directory."""
    results = {
        "run_dir": base_dir,
        "benchmarks": [],
    }

    # Look for result directories in vm1_results/ and vm2_results/
    search_dirs = [
        os.path.join(base_dir, "vm1_results"),
        os.path.join(base_dir, "vm2_results"),
        base_dir,
    ]

    seen_ids = set()

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        for bench_dir in sorted(Path(search_dir).rglob("*")):
            if not bench_dir.is_dir():
                continue

            bench_id = bench_dir.name

            # Skip non-benchmark directories
            if bench_id in ("vm1_results", "vm2_results", "results"):
                continue

            if bench_id in seen_ids:
                continue

            # Check for output logs
            t2v_log = os.path.join(str(bench_dir), "t2v_output.log")
            i2v_log = os.path.join(str(bench_dir), "i2v_output.log")

            if os.path.exists(t2v_log) or os.path.exists(i2v_log):
                seen_ids.add(bench_id)
                bench_result = {
                    "benchmark_id": bench_id,
                    "t2v": parse_sglang_log(t2v_log) if os.path.exists(t2v_log) else None,
                    "i2v": parse_sglang_log(i2v_log) if os.path.exists(i2v_log) else None,
                    "t2v_wall_time_sec": parse_wall_time(str(bench_dir), "t2v"),
                    "i2v_wall_time_sec": parse_wall_time(str(bench_dir), "i2v"),
                }
                results["benchmarks"].append(bench_result)

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/parse_results.py <results_dir>")
        print("Example: python3 scripts/parse_results.py results/run_20260420_160000/")
        sys.exit(1)

    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)

    results = parse_all_results(base_dir)

    output_path = os.path.join(base_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Parsed {len(results['benchmarks'])} benchmark results")
    print(f"Saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Benchmark ID':<30} {'T2V Total(s)':<15} {'I2V Total(s)':<15} {'Status'}")
    print("=" * 80)

    for bench in results["benchmarks"]:
        bid = bench["benchmark_id"]
        t2v_time = bench.get("t2v", {})
        i2v_time = bench.get("i2v", {})
        t2v_total = t2v_time.get("total_sec", "N/A") if t2v_time else "N/A"
        i2v_total = i2v_time.get("total_sec", "N/A") if i2v_time else "N/A"
        t2v_status = t2v_time.get("status", "N/A") if t2v_time else "N/A"
        i2v_status = i2v_time.get("status", "N/A") if i2v_time else "N/A"

        t2v_str = f"{t2v_total:.1f}" if isinstance(t2v_total, (int, float)) else str(t2v_total)
        i2v_str = f"{i2v_total:.1f}" if isinstance(i2v_total, (int, float)) else str(i2v_total)

        print(f"{bid:<30} {t2v_str:<15} {i2v_str:<15} T2V:{t2v_status} I2V:{i2v_status}")

    print("=" * 80)


if __name__ == "__main__":
    main()
