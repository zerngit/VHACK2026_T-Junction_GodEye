#!/usr/bin/env python3
"""
Train/Test split evaluation runner for LangGraph trace metrics.

Produces quantitative evidence artifacts:
- per-run records
- split-level aggregates (mean/std/min/max)
- success-rate summary
- CSV for easy inspection

Default split:
- Train: A, B, C
- Test: D, E

Run examples:
    python evaluate_split_metrics.py
    python evaluate_split_metrics.py --runs-per-scenario 3 --max-ticks 120
    python evaluate_split_metrics.py --simulate-ai --runs-per-scenario 2
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
import random
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

from core.mesa_drone_rescue_mcp import SCENARIOS
from controllers.mesa_drone_rescue_langgraph_trace import LangGraphTraceDroneRescueModel


TRAIN_SCENARIOS = [
    "A: Center quake (clustered)",
    "B: Two hotspots",
    "C: Perimeter scattered",
]
TEST_SCENARIOS = [
    "D: City with high buildings",
    "E: Palu city",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _kpi_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = [_safe_float(r.get(key, 0.0)) for r in rows]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 4),
        "std": round(pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _pick_target(rng: random.Random, model: LangGraphTraceDroneRescueModel) -> Tuple[int, int]:
    width = int(model.width)
    height = int(model.height)

    candidates: List[Tuple[int, int]] = []
    for _ in range(24):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        if not model.is_high_building((x, y)):
            candidates.append((x, y))
    if candidates:
        return candidates[0]
    return (rng.randint(0, width - 1), rng.randint(0, height - 1))


def _drive_headless_policy(model: LangGraphTraceDroneRescueModel, rng: random.Random) -> None:
    """Simple deterministic policy to produce tool calls when AI is disabled."""
    fleet = model._tools.discover_drones().get("drones", [])
    for drone_row in fleet:
        if not isinstance(drone_row, dict):
            continue
        drone_id = str(drone_row.get("id", ""))
        if not drone_id:
            continue
        if bool(drone_row.get("disabled", False)):
            continue

        battery = _safe_float(drone_row.get("battery", 0.0), 0.0)
        if battery <= 12.0:
            model._tools.recall_to_base(drone_id)
            continue

        tx, ty = _pick_target(rng, model)
        model._tools.move_and_scan(drone_id, tx, ty, reason="split-eval coverage sweep")


def _run_single(
    scenario: str,
    split: str,
    run_index: int,
    max_ticks: int,
    simulate_ai: bool,
    seed_offset: int,
) -> Dict[str, Any]:
    base_seed = int(SCENARIOS.get(scenario, {}).get("seed", 0))
    run_seed = base_seed + int(seed_offset) + int(run_index)
    rng = random.Random(run_seed)

    model = LangGraphTraceDroneRescueModel(
        width=24,
        height=16,
        num_drones=4,
        num_survivors=12,
        scenario=scenario,
        simulate_ai=bool(simulate_ai),
        ai_delay_s=0.0,
    )

    for _ in range(int(max_ticks)):
        if not simulate_ai:
            _drive_headless_policy(model, rng)
        model.step()

        snap = model.get_metrics_snapshot()
        if int(snap.get("survivors_found", 0)) >= int(snap.get("survivors_total", 0)) and int(snap.get("survivors_total", 0)) > 0:
            break

    final = model.get_metrics_snapshot()
    success = bool(
        _safe_float(final.get("survivor_detected_rate", 0.0)) >= 90.0
        and _safe_float(final.get("coverage_pct", 0.0)) >= 65.0
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "split": split,
        "scenario": scenario,
        "scenario_seed": base_seed,
        "run_index": run_index,
        "run_seed": run_seed,
        "simulate_ai": bool(simulate_ai),
        "max_ticks": int(max_ticks),
        "final_tick": int(final.get("tick", 0)),
        "success": success,
        "fhc_ticks": _safe_float(final.get("fhc_ticks", -1.0)),
        "fhc_seconds": _safe_float(final.get("fhc_seconds", -1.0)),
        "coverage_pct": _safe_float(final.get("coverage_pct", 0.0)),
        "area_coverage_rate": _safe_float(final.get("area_coverage_rate", 0.0)),
        "survivors_found": int(final.get("survivors_found", 0)),
        "survivors_total": int(final.get("survivors_total", 0)),
        "survivor_detected_rate": _safe_float(final.get("survivor_detected_rate", 0.0)),
        "exploration_score": _safe_float(final.get("exploration_score", 0.0)),
        "redundancy_penalty": _safe_float(final.get("redundancy_penalty", 0.0)),
        "dispersion_penalty": _safe_float(final.get("dispersion_penalty", 0.0)),
        "avg_battery": _safe_float(final.get("avg_battery", 0.0)),
        "tool_total": int(final.get("tool_total", 0)),
        "tool_valid": int(final.get("tool_valid", 0)),
        "tool_call_accuracy": _safe_float(final.get("tool_call_accuracy", 0.0)),
        "discovery_events": int(final.get("discovery_events", 0)),
        "discovery_latency_ticks": _safe_float(final.get("discovery_latency_ticks", 0.0)),
        "discovery_latency_seconds": _safe_float(final.get("discovery_latency_seconds", 0.0)),
    }


def _split_aggregates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "runs": 0,
            "success_rate": 0.0,
            "kpis": {},
        }

    success_rate = sum(1 for r in rows if bool(r.get("success"))) / max(1, len(rows))
    keys = [
        "fhc_ticks",
        "coverage_pct",
        "area_coverage_rate",
        "survivor_detected_rate",
        "exploration_score",
        "avg_battery",
        "tool_call_accuracy",
        "discovery_latency_ticks",
    ]

    return {
        "runs": len(rows),
        "success_rate": round(success_rate * 100.0, 2),
        "kpis": {key: _kpi_stats(rows, key) for key in keys},
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("\n")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run train/test split evaluation for drone metrics.")
    parser.add_argument("--runs-per-scenario", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=120)
    parser.add_argument("--simulate-ai", action="store_true", help="Use LangGraph AI; default uses deterministic headless policy.")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="evaluation_reports")
    args = parser.parse_args()

    train_set = [s for s in TRAIN_SCENARIOS if s in SCENARIOS]
    test_set = [s for s in TEST_SCENARIOS if s in SCENARIOS]

    overlap = set(train_set).intersection(test_set)
    if overlap:
        raise RuntimeError(f"Invalid split: overlap detected -> {sorted(overlap)}")

    print("=" * 88)
    print("SPLIT EVALUATION RUNNER")
    print("=" * 88)
    print(f"Train scenarios: {train_set}")
    print(f"Test scenarios : {test_set}")
    print(f"Runs/scenario   : {args.runs_per_scenario}")
    print(f"Max ticks       : {args.max_ticks}")
    print(f"Simulate AI     : {bool(args.simulate_ai)}")

    all_rows: List[Dict[str, Any]] = []

    for split_name, scenarios in (("train", train_set), ("test", test_set)):
        for scenario in scenarios:
            for run_idx in range(int(args.runs_per_scenario)):
                print(f"[RUN] split={split_name} scenario='{scenario}' run={run_idx}")
                row = _run_single(
                    scenario=scenario,
                    split=split_name,
                    run_index=run_idx,
                    max_ticks=int(args.max_ticks),
                    simulate_ai=bool(args.simulate_ai),
                    seed_offset=int(args.seed_offset),
                )
                all_rows.append(row)

    train_rows = [r for r in all_rows if r.get("split") == "train"]
    test_rows = [r for r in all_rows if r.get("split") == "test"]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "runs_per_scenario": int(args.runs_per_scenario),
            "max_ticks": int(args.max_ticks),
            "simulate_ai": bool(args.simulate_ai),
            "seed_offset": int(args.seed_offset),
        },
        "split_manifest": {
            "train": train_set,
            "test": test_set,
        },
        "counts": {
            "total_runs": len(all_rows),
            "train_runs": len(train_rows),
            "test_runs": len(test_rows),
        },
        "train": _split_aggregates(train_rows),
        "test": _split_aggregates(test_rows),
        "per_run": all_rows,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"split_eval_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "summary.json")
    csv_path = os.path.join(out_dir, "per_run.csv")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    _write_csv(csv_path, all_rows)

    print("\nArtifacts:")
    print(f"  - JSON summary: {json_path}")
    print(f"  - CSV per-run : {csv_path}")
    print("\nSplit-level success rates:")
    print(f"  - Train: {summary['train']['success_rate']}%")
    print(f"  - Test : {summary['test']['success_rate']}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
