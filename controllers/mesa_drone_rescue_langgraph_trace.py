"""
Dedicated Mesa launcher for LangGraph Trace Bridge.

Run this in a separate terminal:
    python mesa_drone_rescue_langgraph_trace.py

This starts a Mesa UI that uses LangGraphMesaTraceController directly and writes
full per-tick I/O trace records to `langgraph_tick_trace_log.jsonl`.
"""

from __future__ import annotations

import os
import sys
# Ensure parent directory (V-Hack) is in sys.path so 'core' and others can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from datetime import datetime
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import core.mesa_drone_rescue_mcp as base

try:
    import tornado.web  # type: ignore
except Exception:
    tornado = None  # type: ignore

try:
    from controllers.langgraph_mesa_trace_bridge import LangGraphMesaTraceController
    TRACE_IMPORT_ERROR: Optional[str] = None
except Exception as exc:
    LangGraphMesaTraceController = None  # type: ignore
    TRACE_IMPORT_ERROR = str(exc)


_SPLIT_EVAL_STATE: Dict[str, Any] = {
    "process": None,
    "started_at": None,
    "log_file": None,
    "command": None,
}


def _latest_split_eval_summary_path(workspace_dir: str) -> Optional[str]:
    reports_root = os.path.join(workspace_dir, "evaluation_reports")
    if not os.path.isdir(reports_root):
        return None
    candidates = [
        os.path.join(reports_root, name)
        for name in os.listdir(reports_root)
        if name.startswith("split_eval_") and os.path.isdir(os.path.join(reports_root, name))
    ]
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    summary = os.path.join(latest, "summary.json")
    if os.path.exists(summary):
        return summary
    return None


def _tail_text_file(path: str, line_count: int = 40) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-line_count:])
    except Exception:
        return ""


class LangGraphTraceDroneRescueModel(base.DroneRescueModel):
    """Mesa model wired only to LangGraph trace bridge controller."""

    def __init__(
        self,
        width: int = 24,
        height: int = 16,
        num_drones: int = 4,
        num_survivors: int = 12,
        scenario: str = "A: Palu city",
        simulate_ai: bool = True,
        ai_delay_s: float = 0.15,
        langgraph_model: str = "qwen/qwen3-14b",
        langgraph_operator_model: str = "qwen/qwen-2.5-7b-instruct",
    ):
        super().__init__(
            width=width,
            height=height,
            num_drones=num_drones,
            num_survivors=num_survivors,
            scenario=scenario,
            simulate_ai=False,
            ai_delay_s=ai_delay_s,
            use_gemini_ai=False,
            use_ollama_ai=False,
            use_crew_ai=False,
        )

        self.simulate_ai = bool(simulate_ai)
        self.langgraph_model = str(langgraph_model or "qwen/qwen3-14b")
        self.langgraph_operator_model = str(
            langgraph_operator_model or "qwen/qwen-2.5-7b-instruct"
        )
        self._langgraph_trace_ai: Optional[Any] = None

        if self.simulate_ai:
            if LangGraphMesaTraceController is None:
                reason = TRACE_IMPORT_ERROR or "Unknown import error"
                print(f"[WARN] LangGraph trace bridge unavailable: {reason}")
            else:
                self._langgraph_trace_ai = LangGraphMesaTraceController(
                    self._tools,
                    model_name=self.langgraph_model,
                    operator_model_name=self.langgraph_operator_model,
                    action_delay_s=self.ai_delay_s,
                )
                print(
                    "[AI MODE] LangGraph TRACE bridge enabled "
                    f"(commander={self.langgraph_model}, operator={self.langgraph_operator_model})."
                )

        # Match OpenRouter launcher behavior: start recording immediately
        # so users can export without manually toggling first.
        self._frame_capture.capturing = True

        self._trace_file = "langgraph_tick_trace_log.jsonl"
        self._voting_flow = {
            "active": False,
            "phase": "IDLE",
            "idle_drone_id": None,
            "current_tick": 0,
            "total_ticks": 5,
            "reasoning": {},
            "vote_tally": {},
            "winning_sector": None,
            "winning_action": None,
            "last_message": "",
        }
        self._idle_warning_drone_id: Optional[str] = None
        self._idle_warning_visible: bool = False
        self._voting_flow_log: List[dict] = []

        self._metrics_started_at = time.time()
        self._metrics_series: List[Dict[str, Any]] = []
        self._metrics_write_interval_ticks: int = 5
        self._last_metrics_persisted_tick: Optional[int] = None
        self._last_summary_written_tick: Optional[int] = None

        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        self._metrics_output_dir = os.path.join(workspace_dir, "metrics_stream")
        os.makedirs(self._metrics_output_dir, exist_ok=True)

        scenario_slug = "".join(
            ch.lower() if ch.isalnum() else "_" for ch in str(self.scenario)
        ).strip("_")
        if not scenario_slug:
            scenario_slug = "scenario"

        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._metrics_run_id = f"{run_stamp}_{scenario_slug}"
        self._metrics_tick_jsonl_path = os.path.join(
            self._metrics_output_dir, f"metrics_ticks_{self._metrics_run_id}.jsonl"
        )
        self._metrics_run_summary_path = os.path.join(
            self._metrics_output_dir, f"summary_{self._metrics_run_id}.json"
        )
        self._metrics_latest_summary_path = os.path.join(
            self._metrics_output_dir, "summary_latest.json"
        )

        self._hard_stop_requested: bool = False
        self._hard_stopped: bool = False
        self._metrics: Dict[str, Any] = {
            "first_contact_tick": None,
            "first_contact_seconds": None,
            "total_scans": 0,
            "duplicate_scans": 0,
            "scan_counts": {},
            "tool_total": 0,
            "tool_valid": 0,
            "tool_invalid_schema": 0,
            "tool_invalid_logic": 0,
            "tool_execution_failed": 0,
            "tool_history": [],
            "known_drones": set(),
            "pending_discovery": {},
            "discovery_latencies_ticks": [],
            "discovery_latencies_seconds": [],
            "snapshot": {},
        }
        self._install_metric_tool_wrappers()
        self._configure_metrics_datacollector()
        self._update_metrics_state()
        self._write_offline_summary(force=True)

    def hard_stop(self) -> Dict[str, Any]:
        self._hard_stop_requested = True
        self._hard_stopped = True
        self.running = False
        self.simulate_ai = False
        self._langgraph_trace_ai = None
        self._voting_flow["active"] = False
        self._voting_flow["phase"] = "HARD_STOP"
        self._voting_flow["last_message"] = "Hard stop triggered"
        return {
            "ok": True,
            "message": "⛔ Hard stop applied. Simulation loop halted.",
            "tick": int(self.schedule.steps),
            "running": bool(self.running),
        }

    def resume_from_hard_stop(self) -> Dict[str, Any]:
        self._hard_stop_requested = False
        self._hard_stopped = False
        self.running = True
        return {
            "ok": True,
            "message": "▶ Simulation resumed. You can press Run again.",
            "tick": int(self.schedule.steps),
            "running": bool(self.running),
        }

    def _split_label(self) -> str:
        if str(self.scenario).startswith(("A:", "B:", "C:")):
            return "train"
        if str(self.scenario).startswith(("D:", "E:")):
            return "test"
        return "unspecified"

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _extract_drone_id(
        self,
        tool_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Optional[str]:
        if tool_name in {"discover_drones", "get_mission_state"}:
            return None
        if "drone_id" in kwargs:
            return str(kwargs.get("drone_id"))
        if len(args) >= 1:
            return str(args[0])
        return None

    def _validate_tool_args(
        self,
        tool_name: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        if tool_name == "discover_drones":
            return

        drone_id = self._extract_drone_id(tool_name, args, kwargs)
        if not drone_id:
            raise ValueError("missing drone_id")

        if tool_name in {"move_to", "move_and_scan"}:
            x_val = kwargs.get("x", args[1] if len(args) >= 2 else None)
            y_val = kwargs.get("y", args[2] if len(args) >= 3 else None)
            if x_val is None or y_val is None:
                raise ValueError("missing x/y")
            int(x_val)
            int(y_val)

    def _is_logically_sound(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        if result.get("error"):
            return False

        reason = str(result.get("reason", "")).lower()
        bad_reason_markers = (
            "drone disabled",
            "battery depleted",
            "insufficient battery",
            "already moved this tick",
            "not at a station",
        )
        if any(marker in reason for marker in bad_reason_markers):
            return False

        if "moved" in result and result.get("moved") is False:
            return False
        if "scanned" in result and result.get("scanned") is False:
            return False
        if "charged" in result and result.get("charged") is False:
            return False
        if "ok" in result and result.get("ok") is False:
            return False
        return True

    def _record_scan_coordinate(self, result: Dict[str, Any]) -> None:
        coord = result.get("coordinate") or result.get("new_pos")
        if not (isinstance(coord, list) and len(coord) >= 2):
            return
        x_val, y_val = int(coord[0]), int(coord[1])
        key = f"{x_val},{y_val}"
        scan_counts: Dict[str, int] = self._metrics["scan_counts"]
        if scan_counts.get(key, 0) >= 1:
            self._metrics["duplicate_scans"] += 1
        scan_counts[key] = scan_counts.get(key, 0) + 1
        self._metrics["total_scans"] += 1

    def _record_discovery(self, result: Dict[str, Any], tick: int, now_s: float) -> None:
        drones = result.get("drones", [])
        if not isinstance(drones, list):
            return
        known: set = self._metrics["known_drones"]
        pending: Dict[str, Dict[str, Any]] = self._metrics["pending_discovery"]
        for row in drones:
            if not isinstance(row, dict):
                continue
            drone_id = str(row.get("id", ""))
            if not drone_id:
                continue
            if drone_id not in known:
                known.add(drone_id)
                pending[drone_id] = {"tick": tick, "time": now_s}

    def _record_tasking_latency(self, drone_id: Optional[str], tick: int, now_s: float) -> None:
        if not drone_id:
            return
        pending: Dict[str, Dict[str, Any]] = self._metrics["pending_discovery"]
        if drone_id not in pending:
            return
        row = pending.pop(drone_id)
        d_tick = int(row.get("tick", tick))
        d_time = self._safe_float(row.get("time", now_s), now_s)
        self._metrics["discovery_latencies_ticks"].append(max(0, tick - d_tick))
        self._metrics["discovery_latencies_seconds"].append(max(0.0, now_s - d_time))

    def _install_metric_tool_wrappers(self) -> None:
        wrapped_tools = [
            "discover_drones",
            "move_to",
            "move_and_scan",
            "thermal_scan",
            "recall_to_base",
            "charge_drone",
        ]

        for tool_name in wrapped_tools:
            original = getattr(self._tools, tool_name, None)
            if not callable(original):
                continue

            def make_wrapper(name: str, func: Any):
                def wrapped(*args: Any, **kwargs: Any):
                    tick = int(self.schedule.steps)
                    now_s = time.time()
                    self._metrics["tool_total"] += 1

                    try:
                        self._validate_tool_args(name, args, kwargs)
                    except Exception as exc:
                        self._metrics["tool_invalid_schema"] += 1
                        self._metrics["tool_history"].append(
                            {
                                "tick": tick,
                                "tool": name,
                                "syntax_ok": False,
                                "logical_ok": False,
                                "error": str(exc),
                            }
                        )
                        return {"ok": False, "error": f"Invalid arguments: {exc}"}

                    drone_id = self._extract_drone_id(name, args, kwargs)
                    try:
                        result = func(*args, **kwargs)
                    except Exception as exc:
                        self._metrics["tool_execution_failed"] += 1
                        self._metrics["tool_invalid_logic"] += 1
                        self._metrics["tool_history"].append(
                            {
                                "tick": tick,
                                "tool": name,
                                "drone_id": drone_id,
                                "syntax_ok": True,
                                "logical_ok": False,
                                "error": str(exc),
                            }
                        )
                        raise

                    logical_ok = self._is_logically_sound(result)
                    if logical_ok:
                        self._metrics["tool_valid"] += 1
                    else:
                        self._metrics["tool_invalid_logic"] += 1

                    if isinstance(result, dict):
                        if name in {"move_and_scan", "thermal_scan"}:
                            self._record_scan_coordinate(result)
                        if name == "discover_drones":
                            self._record_discovery(result, tick, now_s)
                        if name in {"move_to", "move_and_scan", "thermal_scan", "recall_to_base", "charge_drone"}:
                            self._record_tasking_latency(drone_id, tick, now_s)

                    self._metrics["tool_history"].append(
                        {
                            "tick": tick,
                            "tool": name,
                            "drone_id": drone_id,
                            "syntax_ok": True,
                            "logical_ok": logical_ok,
                        }
                    )
                    if len(self._metrics["tool_history"]) > 800:
                        self._metrics["tool_history"] = self._metrics["tool_history"][-800:]
                    return result

                return wrapped

            setattr(self._tools, tool_name, make_wrapper(tool_name, original))

    def _average_pairwise_distance(self) -> float:
        drones = [a for a in self.schedule.agents if isinstance(a, base.DroneAgent)]
        if len(drones) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for idx in range(len(drones)):
            for jdx in range(idx + 1, len(drones)):
                p1 = drones[idx].pos
                p2 = drones[jdx].pos
                total += abs(int(p1[0]) - int(p2[0])) + abs(int(p1[1]) - int(p2[1]))
                pairs += 1
        return total / max(1, pairs)

    def _configure_metrics_datacollector(self) -> None:
        self.datacollector = base.DataCollector(
            model_reporters={
                "SurvivorsFound": lambda m: int(getattr(m, "_metrics", {}).get("snapshot", {}).get("survivors_found", 0)),
                "SectorsDone": lambda m: int(getattr(m, "_metrics", {}).get("snapshot", {}).get("sectors_done", 0)),
                "AvgBattery": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("avg_battery", 0.0)),
                "FHC_Ticks": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("fhc_ticks", -1)),
                "AreaCoverageRate": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("area_coverage_rate", 0.0)),
                "SurvivorDetectedRate": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("survivor_detected_rate", 0.0)),
                "ExplorationScore": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("exploration_score", 0.0)),
                "ToolCallAccuracy": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("tool_call_accuracy", 0.0)),
                "DiscoveryLatencyTicks": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("discovery_latency_ticks", 0.0)),
                "CoveragePercent": lambda m: float(getattr(m, "_metrics", {}).get("snapshot", {}).get("coverage_pct", 0.0)),
                "DroneWaitCount": lambda m: int(getattr(m, "_metrics", {}).get("snapshot", {}).get("drone_wait_count", 0)),
            }
        )

    def _update_metrics_state(self) -> None:
        tick = int(self.schedule.steps)
        elapsed_s = max(0.0, time.time() - self._metrics_started_at)

        survivors = [a for a in self.schedule.agents if isinstance(a, base.SurvivorAgent)]
        found_count = sum(1 for s in survivors if bool(getattr(s, "detected", False)))
        total_survivors = len(survivors)

        scanned_cells = sum(1 for t in self.tile_map.values() if bool(getattr(t, "scanned", False)))
        total_cells = max(1, int(self.width * self.height))
        coverage_pct = (scanned_cells / total_cells) * 100.0
        area_coverage_rate = coverage_pct / max(1.0, float(tick + 1))

        survivor_detected_rate = (found_count / max(1, total_survivors)) * 100.0

        if found_count > 0 and self._metrics["first_contact_tick"] is None:
            self._metrics["first_contact_tick"] = max(1, tick)
            self._metrics["first_contact_seconds"] = max(0.001, elapsed_s)

        sector_done_count = 0
        for sid, sdef in base.SECTOR_DEFS.items():
            origin = sdef["origin"]
            size = sdef["size"]
            ox, oy = int(origin[0]), int(origin[1])
            w, h = int(size[0]), int(size[1])
            total = max(1, w * h)
            scanned = sum(
                1
                for xi in range(ox, ox + w)
                for yi in range(oy, oy + h)
                if self.tile_map.get((xi, yi)) and self.tile_map[(xi, yi)].scanned
            )
            if (scanned / total) >= 0.999:
                sector_done_count += 1

        drones = [a for a in self.schedule.agents if isinstance(a, base.DroneAgent)]
        avg_battery = (
            sum(self._safe_float(getattr(d, "battery", 0.0)) for d in drones) / max(1, len(drones))
        )

        moved_this_tick = {
            str(entry[1])
            for entry in self.movement_history
            if isinstance(entry, tuple) and len(entry) >= 2 and int(entry[0]) == tick
        }
        drone_wait_count = max(0, len(drones) - len(moved_this_tick))

        total_scans = max(1, int(self._metrics["total_scans"]))
        duplicate_scans = int(self._metrics["duplicate_scans"])
        redundancy_penalty = duplicate_scans / total_scans

        avg_pairwise = self._average_pairwise_distance()
        max_pairwise = max(1.0, float((self.width - 1) + (self.height - 1)))
        dispersion_norm = max(0.0, min(1.0, avg_pairwise / max_pairwise))
        dispersion_penalty = 1.0 - dispersion_norm

        coverage_norm = max(0.0, min(1.0, coverage_pct / 100.0))
        exploration_score = 100.0 * (
            0.60 * coverage_norm
            - 0.25 * redundancy_penalty
            - 0.15 * dispersion_penalty
        )
        exploration_score = max(0.0, min(100.0, exploration_score))

        tool_total = int(self._metrics["tool_total"])
        tool_valid = int(self._metrics["tool_valid"])
        tool_call_accuracy = (tool_valid / max(1, tool_total)) * 100.0

        lat_ticks = self._metrics["discovery_latencies_ticks"]
        lat_secs = self._metrics["discovery_latencies_seconds"]
        avg_lat_ticks = (
            sum(int(v) for v in lat_ticks) / len(lat_ticks)
            if lat_ticks
            else 0.0
        )
        avg_lat_secs = (
            sum(self._safe_float(v) for v in lat_secs) / len(lat_secs)
            if lat_secs
            else 0.0
        )

        snapshot = {
            "tick": tick,
            "elapsed_seconds": round(elapsed_s, 3),
            "split": self._split_label(),
            "scenario": str(self.scenario),
            "survivors_found": found_count,
            "survivors_total": total_survivors,
            "fhc_ticks": (
                int(self._metrics["first_contact_tick"])
                if self._metrics["first_contact_tick"] is not None
                else -1
            ),
            "fhc_seconds": (
                round(self._safe_float(self._metrics["first_contact_seconds"]), 3)
                if self._metrics["first_contact_seconds"] is not None
                else -1.0
            ),
            "coverage_pct": round(coverage_pct, 2),
            "area_coverage_rate": round(area_coverage_rate, 4),
            "survivor_detected_rate": round(survivor_detected_rate, 2),
            "sectors_done": int(sector_done_count),
            "avg_battery": round(avg_battery, 2),
            "drone_wait_count": int(drone_wait_count),
            "total_scans": int(self._metrics["total_scans"]),
            "duplicate_scans": duplicate_scans,
            "redundancy_penalty": round(redundancy_penalty, 4),
            "dispersion_norm": round(dispersion_norm, 4),
            "dispersion_penalty": round(dispersion_penalty, 4),
            "exploration_score": round(exploration_score, 2),
            "tool_total": tool_total,
            "tool_valid": tool_valid,
            "tool_invalid_schema": int(self._metrics["tool_invalid_schema"]),
            "tool_invalid_logic": int(self._metrics["tool_invalid_logic"]),
            "tool_execution_failed": int(self._metrics["tool_execution_failed"]),
            "tool_call_accuracy": round(tool_call_accuracy, 2),
            "discovery_events": len(lat_ticks),
            "discovery_latency_ticks": round(avg_lat_ticks, 3),
            "discovery_latency_seconds": round(avg_lat_secs, 3),
        }
        self._metrics["snapshot"] = snapshot
        if self._metrics_series and int(self._metrics_series[-1].get("tick", -1)) == int(snapshot["tick"]):
            self._metrics_series[-1] = dict(snapshot)
        else:
            self._metrics_series.append(dict(snapshot))
        self._append_metrics_tick_record(snapshot)
        if len(self._metrics_series) > 5000:
            self._metrics_series = self._metrics_series[-5000:]

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        return dict(self._metrics.get("snapshot", {}))

    def get_metrics_series_tail(self, limit: int = 120) -> List[Dict[str, Any]]:
        n = max(1, int(limit))
        return list(self._metrics_series[-n:])

    def _build_metrics_payload(self, series_limit: Optional[int] = None) -> Dict[str, Any]:
        snapshot = self.get_metrics_snapshot()
        if series_limit is None:
            time_series = list(self._metrics_series)
        else:
            n = max(1, int(series_limit))
            time_series = list(self._metrics_series[-n:])

        return {
            "generated_at": datetime.now().isoformat(),
            "run_id": self._metrics_run_id,
            "source": "mesa_drone_rescue_langgraph_trace",
            "split": self._split_label(),
            "scenario": str(self.scenario),
            "scenario_seed": base.SCENARIOS.get(str(self.scenario), {}).get("seed"),
            "models": {
                "commander": self.langgraph_model,
                "operator": self.langgraph_operator_model,
            },
            "formula": {
                "exploration_score": "100 * (0.60*coverage_norm - 0.25*redundancy_penalty - 0.15*dispersion_penalty)",
                "tool_call_accuracy": "100 * tool_valid / max(1, tool_total)",
                "area_coverage_rate": "coverage_pct / max(1, tick+1)",
                "survivor_detected_rate": "100 * survivors_found / max(1, survivors_total)",
            },
            "latest_snapshot": snapshot,
            "time_series": time_series,
            "tool_history": list(self._metrics.get("tool_history", [])),
            "train_test_policy": {
                "train": ["A: Center quake (clustered)", "B: Edge quake (scattered)", "C: North-East quake (line)"],
                "test": ["D: City with high buildings", "E: City with buildings (all low)"],
            },
            "files": {
                "tick_jsonl": self._metrics_tick_jsonl_path,
                "summary_latest": self._metrics_latest_summary_path,
                "summary_run": self._metrics_run_summary_path,
            },
        }

    def _append_metrics_tick_record(self, snapshot: Dict[str, Any]) -> None:
        tick = int(snapshot.get("tick", self.schedule.steps))
        if self._last_metrics_persisted_tick is not None and tick == self._last_metrics_persisted_tick:
            return

        record = {
            "generated_at": datetime.now().isoformat(),
            "run_id": self._metrics_run_id,
            "source": "mesa_drone_rescue_langgraph_trace",
            "split": self._split_label(),
            "scenario": str(self.scenario),
            "tick": tick,
            "snapshot": dict(snapshot),
        }

        try:
            with open(self._metrics_tick_jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._last_metrics_persisted_tick = tick
        except Exception as exc:
            print(f"[WARN] Failed to append metrics tick record: {exc}")

    def _write_json_atomic(self, path: str, payload: Dict[str, Any]) -> None:
        temp_path = f"{path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(temp_path, path)

    def _write_offline_summary(self, force: bool = False) -> None:
        snapshot = self.get_metrics_snapshot()
        tick = int(snapshot.get("tick", self.schedule.steps))

        if not force:
            if tick <= 0:
                return
            if (tick % max(1, int(self._metrics_write_interval_ticks))) != 0:
                return
            if self._last_summary_written_tick is not None and tick == self._last_summary_written_tick:
                return

        payload = self._build_metrics_payload(series_limit=2000)
        try:
            self._write_json_atomic(self._metrics_run_summary_path, payload)
            self._write_json_atomic(self._metrics_latest_summary_path, payload)
            self._last_summary_written_tick = tick
        except Exception as exc:
            print(f"[WARN] Failed to write offline summary: {exc}")

    def export_metrics_to_file(self) -> Dict[str, Any]:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_eval_{now}.json"
        output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')), filename)

        payload = self._build_metrics_payload(series_limit=None)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        return {
            "message": f"✅ Exported metrics report to {output_path}",
            "file": filename,
            "path": output_path,
            "rows": len(self._metrics_series),
            "split": self._split_label(),
        }

    def _append_voting_trace(self, payload: dict) -> None:
        rec = {
            "kind": "voting_flow",
            "timestamp": datetime.now().isoformat(),
            "sim_tick": int(self.schedule.steps),
            **payload,
        }
        try:
            with open(self._trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def _voting_log(self, message: str) -> None:
        stamped = f"[VOTING T{self.schedule.steps:03d}] {message}"
        print(stamped)
        base.log_to_file(stamped)
        self._voting_flow_log.append(
            {
                "tick": int(self.schedule.steps),
                "message": stamped,
            }
        )

    def start_idle_voting_flow(self, drone_id: str) -> dict:
        if bool(self._voting_flow.get("active")):
            return {
                "ok": False,
                "message": "A 5-tick demo is already running. Let it finish first.",
                "phase": self._voting_flow.get("phase", "UNKNOWN"),
                "current_tick": self._voting_flow.get("current_tick", 0),
                "total_ticks": self._voting_flow.get("total_ticks", 5),
            }

        drone = self.get_drone(drone_id)
        self._idle_warning_drone_id = None
        self._idle_warning_visible = False
        self._voting_flow = {
            "active": True,
            "phase": "QUEUED",
            "idle_drone_id": drone.unique_id,
            "current_tick": 0,
            "total_ticks": 5,
            "reasoning": {},
            "vote_tally": {},
            "winning_sector": None,
            "winning_action": None,
            "last_message": "5-tick voting demo queued",
        }
        self._voting_log(f" ⚠ 🔴 {drone.unique_id} found idle / disabled.")
        self._append_voting_trace(
            {
                "phase": "QUEUED",
                "idle_drone": drone.unique_id,
                "message": "5-tick voting demo queued",
            }
        )
        return {
            "ok": True,
            "message": f"5-tick voting demo queued for {drone.unique_id}",
            "idle_drone": drone.unique_id,
            "phase": "QUEUED",
            "current_tick": 0,
            "total_ticks": 5,
        }

    def _run_voting_flow_tick(self) -> None:
        demo = self._voting_flow
        if not demo.get("active"):
            return

        current_tick = int(demo.get("current_tick", 0)) + 1
        demo["current_tick"] = current_tick
        idle_drone_id = str(demo.get("idle_drone_id") or "")
        if not idle_drone_id:
            demo["active"] = False
            demo["phase"] = "FAILED"
            demo["last_message"] = "Missing idle drone id"
            self._idle_warning_drone_id = None
            self._idle_warning_visible = False
            return

        if current_tick == 1:
            idle_drone = self.get_drone(idle_drone_id)
            idle_drone.disabled = True
            demo["phase"] = "IDLE_DETECTED"
            demo["last_message"] = f"{idle_drone_id} forced to IDLE state"
            self._voting_log(f"Tick 1/5: {idle_drone_id} is currently idle/disabled.")
            if idle_drone_id == "d_0":
                self._idle_warning_drone_id = idle_drone_id
                self._idle_warning_visible = True
                time.sleep(2.0)

        elif current_tick == 2:
            voters = [
                a.unique_id
                for a in self.schedule.agents
                if isinstance(a, base.DroneAgent) and a.unique_id != idle_drone_id
            ]
            reasoning = {
                voter_id: "Sector 1 is selected as fallback rescue route for idle recovery."
                for voter_id in voters
            }
            demo["reasoning"] = reasoning
            demo["phase"] = "REASONING"
            demo["last_message"] = "All active drones posted reasoning"
            self._voting_log("Tick 2/5: reasoning collected from active drones.")
            for voter_id, reason in reasoning.items():
                self._voting_log(f"  {voter_id} reasoning: {reason}")

        elif current_tick == 3:
            reasoning = demo.get("reasoning", {}) or {}
            vote_count = len(reasoning.keys())
            vote_tally = {1: vote_count}
            demo["vote_tally"] = vote_tally
            demo["winning_sector"] = 1
            demo["phase"] = "VOTE_COMPLETE"
            demo["last_message"] = f"Votes finalized: {vote_tally}"
            self._voting_log(f"Tick 3/5: vote tally completed -> {vote_tally}")

        elif current_tick == 4:
            winning_action = {
                "drone_id": idle_drone_id,
                "type": "move_and_scan",
                "target": [2, 10],
                "reason": "Vote result: recover by scanning Sector 1",
            }
            demo["winning_action"] = winning_action
            demo["phase"] = "ACTION_DETERMINED"
            demo["last_message"] = "Winning action locked for idle drone"
            self._voting_log(
                "Tick 4/5: action determined -> "
                f"move_and_scan({idle_drone_id}, 2, 10)"
            )

        elif current_tick == 5:
            winning_action = demo.get("winning_action") or {}
            idle_drone = self.get_drone(idle_drone_id)
            idle_drone.disabled = False
            tx, ty = (2, 10)
            if isinstance(winning_action.get("target"), list) and len(winning_action["target"]) >= 2:
                tx, ty = int(winning_action["target"][0]), int(winning_action["target"][1])
            action_result = self._tools.move_and_scan(
                idle_drone_id,
                tx,
                ty,
                reason="IDLE recovery action",
            )
            demo["phase"] = "EXECUTED"
            demo["last_message"] = f"Idle drone executed action at ({tx}, {ty})"
            demo["active"] = False
            if self._idle_warning_drone_id == idle_drone_id:
                self._idle_warning_drone_id = None
                self._idle_warning_visible = False
            self._voting_log(
                "Tick 5/5: idle drone executed recovery action -> "
                f"{action_result}"
            )

        else:
            demo["active"] = False
            demo["phase"] = "COMPLETE"
            demo["last_message"] = "Voting demo completed"
            self._idle_warning_drone_id = None
            self._idle_warning_visible = False

        self._append_voting_trace(
            {
                "phase": demo.get("phase"),
                "idle_drone": demo.get("idle_drone_id"),
                "current_tick": demo.get("current_tick"),
                "total_ticks": demo.get("total_ticks"),
                "reasoning": demo.get("reasoning"),
                "vote_tally": demo.get("vote_tally"),
                "winning_sector": demo.get("winning_sector"),
                "winning_action": demo.get("winning_action"),
                "message": demo.get("last_message"),
            }
        )

    def step(self):
        if bool(self._hard_stop_requested):
            self.running = False
            return

        self._tools.reset_tick_state()
        if bool(self._voting_flow.get("active")):
            self._run_voting_flow_tick()
        else:
            trace_ai = self._langgraph_trace_ai
            if trace_ai is not None:
                trace_ai.think_and_act()
        self._update_metrics_state()
        self._write_offline_summary(force=False)
        self.datacollector.collect(self)
        self.schedule.step()
        
        # Check for mission complete condition (copied from base model or we could just check it)
        ms = self._tools.get_mission_state()
        sectors = ms.get("sectors", [])
        all_sectors_scanned = len(sectors) > 0 and all(float(s.get("coverage_pct", 0)) >= 99.0 for s in sectors)
        
        if all_sectors_scanned and self.running:
            self.running = False
            surv_found = ms.get("survivors_found", 0)
            surv_total = ms.get("actual_survivors_total_hidden", 0)
            accuracy = (surv_found / max(1, surv_total)) * 100
            print("\n" + "═" * 72)
            print("🚀 MISSION ACCOMPLISHED: All sectors have been 100% scanned!")
            print(f"📊 Accuracy: Found {surv_found}/{surv_total} survivors ({accuracy:.1f}%)")
            print("═" * 72 + "\n")
            
            try:
                import tkinter as tk
                
                root = tk.Tk()
                root.withdraw()
                
                popup = tk.Toplevel()
                popup.title("Mission Accomplished")
                popup.geometry("380x240")
                popup.configure(bg="#ffffff")
                popup.attributes('-topmost', True)
                
                tk.Label(popup, text="🎉 Mission Accomplished!", font=("Segoe UI", 16, "bold"), fg="#0284c7", bg="#ffffff").pack(pady=(20, 5))
                tk.Label(popup, text="All search grids have been completely scanned.", font=("Segoe UI", 10), fg="#64748b", bg="#ffffff").pack(pady=(0, 10))
                
                stats_frame = tk.Frame(popup, bg="#f0fdfa", bd=1, relief="solid")
                stats_frame.pack(fill="x", padx=30, pady=5)
                
                tk.Label(stats_frame, text=f"Survivors Found: {surv_found} / {surv_total}", font=("Segoe UI", 12, "bold"), fg="#0d9488", bg="#f0fdfa").pack(pady=(12, 2))
                tk.Label(stats_frame, text=f"Overall Accuracy: {accuracy:.1f}%", font=("Segoe UI", 10, "bold"), fg="#334155", bg="#f0fdfa").pack(pady=(0, 12))
                
                btn = tk.Button(popup, text="Close", command=root.destroy, bg="#38bdf8", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", activebackground="#0ea5e9", activeforeground="white", cursor="hand2")
                btn.pack(ipadx=20, ipady=4, pady=15)
                
                popup.update_idletasks()
                x = (popup.winfo_screenwidth() // 2) - (380 // 2)
                y = (popup.winfo_screenheight() // 2) - (240 // 2)
                popup.geometry(f"+{x}+{y}")
                
                root.mainloop()
            except Exception as e:
                print(f"[UI Popup Error] {e}")
                pass
                
        self._frame_capture.capture_frame()


class SidebarStyleElement(base.TextElement):
    def render(self, model):
        return """
        <style>
        /* Modern Light Theme inspired by the user's reference image */
        body {
            background-color: #f0fdfa !important; /* Extremely soft teal/white */
            color: #334155 !important;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;   
        }

        /* Sidebar Container */
        .sidebar, #sidebar-nav, .col-md-3, .col-lg-3 {
            background-color: #ffffff !important;
            border-right: none !important;
            border-radius: 0 16px 16px 0 !important;
            padding: 24px !important;
            box-shadow: 4px 0 24px rgba(20, 184, 166, 0.08) !important;
        }

        /* Override the giant blue Bootstrap labels */
        .label, .badge, .control-label, label {
            background-color: #ccfbf1 !important; /* light cyan/teal background */
            color: #0f766e !important; /* dark teal text */
            font-weight: 700 !important;
            font-size: 11px !important;
            padding: 4px 10px !important;
            border-radius: 12px !important; /* Small rounded pill */
            display: inline-flex !important;
            align-items: center;
            width: fit-content !important; /* Fixes huge full-width bars */
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: none !important;
            margin-top: 22px !important;
            margin-bottom: 6px !important;
            line-height: 1.4 !important;
        }

        /* Checkbox formatting (Simulate AI drives drones) */
            div:has(input[type="checkbox"]),
            p:has(input[type="checkbox"]),
            .checkbox {
                margin-top: 24px !important;
                margin-bottom: -10px !important; /* pull it close to LangGraph commander model label */
                margin-left: 0 !important;
                display: flex !important;
            }

            label:has(input[type="checkbox"]),
            .checkbox label,
            input[type="checkbox"] + label {
                background-color: #ccfbf1 !important; /* light cyan/teal background */
                color: #0f766e !important; /* dark teal text */
                font-weight: 700 !important;
                font-size: 11px !important;
                padding: 4px 10px !important;
                border-radius: 12px !important; /* Small rounded pill */
                display: inline-flex !important;
                align-items: center;
                width: fit-content !important; 
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border: none !important;
                margin: 0 !important;
                pointer-events: none !important; /* Ignore clicks */
            }

        /* Selects and inputs */
        select.form-control, select.form-select, input.form-control, select, input[type="text"] {
            background-color: #f8fafc !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 20px !important; /* pill shape */
            color: #0f766e !important;
            padding: 4px 12px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            box-shadow: none !important;
            transition: all 0.2s ease !important;
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 85% !important;
            background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2314b8a6%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.4-12.8z%22%2F%3E%3C%2Fsvg%3E") !important;
            background-repeat: no-repeat !important;
            background-position: right 12px top 50% !important;
            background-size: 12px auto !important;
        }

        select.form-control:focus, input.form-control:focus, select:focus {
            border-color: #14b8a6 !important;
            outline: none !important;
            box-shadow: 0 0 0 4px rgba(20, 184, 166, 0.15) !important;
            background-color: #ffffff !important;
        }

        /* Core Buttons */
        .btn {
            border-radius: 20px !important; /* Soft pill shape */
            font-weight: 700 !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
            margin-bottom: 10px !important;
            border: none !important;
        }

        /* Play/Start button */
        .btn-success, #play-pause {
            background-color: #2ed573 !important; 
            color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(46, 213, 115, 0.3) !important;
        }
        .btn-success:hover, #play-pause:hover {
            background-color: #26b360 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 14px rgba(46, 213, 115, 0.4) !important;
        }

        /* Step button */
        .btn-primary, #step {
            background-color: #14b8a6 !important;
            color: #ffffff !important;
            box-shadow: 0 4px 10px rgba(20, 184, 166, 0.3) !important;
        }
        .btn-primary:hover, #step:hover {
            background-color: #0d9488 !important;
            transform: translateY(-2px) !important;
        }

        /* Reset button */
        .btn-default, .btn-secondary, #reset, button.btn.btn-default {
            background-color: transparent !important;
            color: #14b8a6 !important;
            border: 2px solid #14b8a6 !important;
        }
        .btn-default:hover, .btn-secondary:hover, #reset:hover,
        .btn-default:focus, .btn-secondary:focus, #reset:focus,
        .btn-default:active, .btn-secondary:active, #reset:active {
            background-color: #0ea5e9 !important; /* Nice dark/saturated sky blue */
            color: #ffffff !important;            /* White text will now pop nicely */
            border-color: #0ea5e9 !important;
            transform: translateY(-2px) !important;
            box-shadow: none !important;
            outline: none !important;
        }

                /* Sliders / Ranges - Soft Light Blue */
        input[type="range"] {
            -webkit-appearance: none !important;
            width: 85% !important;
            height: 10px !important;
            background: #e0f2fe !important; /* Very soft light blue background */
            border-radius: 6px !important;
            outline: none !important;
            margin: 8px 0 16px 0 !important;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 24px !important;
            height: 24px !important;
            border-radius: 50% !important;
            background: #38bdf8 !important; /* Brighter, softer cyan-blue for the slider button */
            cursor: pointer !important;
            box-shadow: 0 2px 6px rgba(56, 189, 248, 0.4) !important;
            border: 3px solid #ffffff !important;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.1) !important;
        }
        
        /* Bootstrap Slider overrides for Mesa standard */
        .slider-track {
            background: #e0f2fe !important;
            border-radius: 6px !important;
            height: 10px !important;
        }
        .slider-selection {
            background: #38bdf8 !important;
            border-radius: 6px !important;
        }
        .slider-handle {
            background: #38bdf8 !important;
            width: 24px !important;
            height: 24px !important;
            border: 3px solid #ffffff !important;
            box-shadow: 0 2px 6px rgba(56, 189, 248, 0.4) !important;
        }
        .slider-tick-label-container {
            font-size: 12px !important;
            color: #38bdf8 !important;
            font-weight: 600 !important;
        }
        .slider {
            margin-bottom: 12px !important;
        }input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none !important;
            appearance: none !important;
            width: 24px !important;
            height: 24px !important;
            border-radius: 50% !important;
            background: #14b8a6 !important;
            cursor: pointer !important;
            box-shadow: 0 2px 6px rgba(20, 184, 166, 0.4) !important;
            border: 3px solid #ffffff !important;
            transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2) !important;
        }

        /* Checkboxes -> Toggles (like the image) */
        input[type="checkbox"] {
            appearance: none !important;
            -webkit-appearance: none !important;
            display: none !important;
              /* width: 46px !important;
              height: 24px !important; */
            background: #cbd5e1 !important; 
            border-radius: 24px !important;
            position: relative !important;
            cursor: pointer !important;
            outline: none !important;
            transition: background 0.3s ease !important;
            vertical-align: middle !important;
            margin: 0 !important;
        }

        input[type="checkbox"]::after {
            content: '' !important;
            position: absolute !important;
            top: 2px !important; /* Adjusted so thumb centers nicely */
            left: 2px !important;
            display: none !important;
            border-radius: 50% !important;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.15) !important;
        }

        input[type="checkbox"]:checked {
            background: #2ed573 !important; 
        }

        input[type="checkbox"]:checked::after {
            transform: translateX(22px) !important;
        }

        /* Force elements container to fit grid width so chart isn't squished */
        #elements {
            min-width: 980px !important;
        }

        /* Map fixes to preserve grid styling */
        .sidebar {
             font-size: 14px;
        }
        </style>
        """

class VideoExportElement(base.TextElement):
    """Mesa TextElement that shows recording status and Export/Clear buttons."""

    def render(self, model):
        fc = model._frame_capture
        frame_count = len(fc.frames)
        is_capturing = fc.capturing

        status_color = "#14b8a6" if is_capturing else "#94a3b8"
        status_text = f"🔴 Recording: {frame_count} frames" if is_capturing else f"⏸ Stopped: {frame_count} frames"

        return f"""
        <div id="video-export-comp-container" style="font-family:'Segoe UI', system-ui, sans-serif;padding:16px;margin-bottom:10px;margin-top:20px;
                    background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;box-shadow:0 4px 12px rgba(20, 184, 166, 0.05);transform:translateX(-18px);width:90%;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;justify-content:center;">
            <span style="color:{status_color};font-weight:700;font-size:14px;background:#f0fdfa;padding:6px 12px;border-radius:12px;">
              {status_text}
            </span>
          </div>
          <div style="display:flex;flex-direction:column;gap:8px;">
            <button onclick="fetch('/export_video', {{method:'POST'}})
                            .then(async r=>{{
                                const txt = await r.text();
                                try {{
                                    const d = JSON.parse(txt);
                                    alert(d.message || d.error || txt || 'No response body');
                                }} catch (e) {{
                                    alert(txt || ('HTTP '+r.status));
                                }}
                            }})
                            .catch(e=>alert('Export failed: '+e));"
                    onmouseover="this.style.backgroundColor='#e0f2fe';this.style.borderColor='#38bdf8';" onmouseout="this.style.backgroundColor='#f8fafc';this.style.borderColor='#e2e8f0';"
                    style="padding:10px 14px;background:#f8fafc;color:#0284c7;border:2px solid #e2e8f0;border-radius:20px;cursor:pointer;font-weight:700;font-size:13px;width:100%;text-align:center;transition:all 0.2s ease;">
              🎬 Export MP4
            </button>
            <button onclick="fetch('/clear_frames', {{method:'POST'}})
                            .then(r=>r.json())
                            .then(d=>alert(d.message||'Cleared'))
                            .catch(e=>alert('Error: '+e));"
                    onmouseover="this.style.backgroundColor='#ffedd5';this.style.borderColor='#fb923c';" onmouseout="this.style.backgroundColor='#f8fafc';this.style.borderColor='#e2e8f0';"
                    style="padding:10px 14px;background:#f8fafc;color:#ea580c;border:2px solid #e2e8f0;border-radius:20px;cursor:pointer;font-weight:700;font-size:13px;width:100%;text-align:center;transition:all 0.2s ease;">
              🗑️ Clear Frames
            </button>
            <button onclick="fetch('/toggle_capture', {{method:'POST'}})
                            .then(r=>r.json())
                            .then(d=>alert(d.message))
                            .catch(e=>alert('Error: '+e));"
                    onmouseover="this.style.backgroundColor='#f3e8ff';this.style.borderColor='#c084fc';" onmouseout="this.style.backgroundColor='#f8fafc';this.style.borderColor='#e2e8f0';"
                    style="padding:10px 14px;background:#f8fafc;color:#9333ea;border:2px solid #e2e8f0;border-radius:20px;cursor:pointer;font-weight:700;font-size:13px;width:100%;text-align:center;transition:all 0.2s ease;">
              ⏯ Toggle Capture
            </button>
            <button onclick="fetch('/hard_stop', {{method:'POST'}})
                            .then(r=>r.json())
                            .then(d=>alert(d.message||d.error||'Hard stop sent'))
                            .catch(e=>alert('Hard stop error: '+e));"
                    onmouseover="this.style.backgroundColor='#fee2e2';this.style.borderColor='#f87171';" onmouseout="this.style.backgroundColor='#f8fafc';this.style.borderColor='#e2e8f0';"
                    style="padding:10px 14px;background:#f8fafc;color:#dc2626;border:2px solid #e2e8f0;border-radius:20px;cursor:pointer;font-weight:700;font-size:13px;width:100%;text-align:center;transition:all 0.2s ease;">
              ⛔ HARD STOP
            </button>
            <button onclick="fetch('/resume_simulation', {{method:'POST'}})
                            .then(r=>r.json())
                            .then(d=>alert(d.message||d.error||'Resume sent'))
                            .catch(e=>alert('Resume error: '+e));"
                    onmouseover="this.style.backgroundColor='#dcfce7';this.style.borderColor='#34d399';" onmouseout="this.style.backgroundColor='#f8fafc';this.style.borderColor='#e2e8f0';"
                    style="padding:10px 14px;background:#f8fafc;color:#15803d;border:2px solid #e2e8f0;border-radius:20px;cursor:pointer;font-weight:700;font-size:13px;width:100%;text-align:center;transition:all 0.2s ease;">
              ▶ Resume
            </button>
          </div>
        </div>
        <img src="x" onerror="
    let el = document.getElementById('video-export-comp-container');
    if(el && el.parentElement && el.parentElement.parentElement && el.parentElement.parentElement.id === 'elements') {{
        let sidebar = document.getElementById('sidebar');
        if(sidebar) {{
            sidebar.appendChild(el.parentElement);
        }}
    }}
" style="display:none;">
        """


class VotingSimulatorElement(base.TextElement):
    """Mesa TextElement showing Voting controls."""

    def render(self, model):
        voting_flow = model._voting_flow
        is_voting = bool(voting_flow.get("active"))
        state_color = "#e11d48" if is_voting else "#0f766e"
        state_icon = "🗳️" if is_voting else "✅"

        drones = [a for a in model.schedule.agents if isinstance(a, base.DroneAgent)]
        warning_drone_id = str(getattr(model, "_idle_warning_drone_id", "") or "")
        warning_visible = bool(getattr(model, "_idle_warning_visible", False))
        options_html = ""
        for d in drones:
            warning_suffix = " ⚠" if warning_visible and d.unique_id == warning_drone_id else ""
            options_html += f'<option value="{d.unique_id}">Drone {d.unique_id}{warning_suffix}</option>'
        if not options_html:
            options_html = '<option value="d_0">Drone d_0</option>'

        phase = voting_flow.get("phase", "IDLE")
        current_tick = voting_flow.get("current_tick", 0)
        total_ticks = voting_flow.get("total_ticks", 5)
        message = voting_flow.get("last_message", "Ready")

        return f"""
        <div id="voting-simulator-comp-container" style="font-family:'Segoe UI', system-ui, sans-serif;padding:16px;margin-bottom:10px;margin-top:0px;
                    background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;box-shadow:0 4px 12px rgba(20, 184, 166, 0.05);transform:translateX(-18px);width:90%;">
          
          <div style="display:flex;align-items:center;margin-bottom:8px;justify-content:center;">
            <span style="color:{state_color};font-weight:700;font-size:13px;background:{'#ffe4e6' if is_voting else '#ccfbf1'};padding:6px 12px;border-radius:12px;text-align:center;line-height:1.3;">
              {state_icon} Swarm Voting Simulation<br/>{phase} ({current_tick}/{total_ticks})
            </span>
          </div>

          <div style="color:#64748b;font-size:11px;margin-bottom:12px;text-align:center;line-height:1.4;background:#f8fafc;padding:6px;border-radius:8px;">
              <i>idle → reasoning → vote → action → execute</i><br/>
              <b>Last:</b> {message}
          </div>

          <div style="display:flex;flex-direction:column;gap:8px;">
            <select id="voting_drone_select" style="padding:8px 12px;border-radius:20px;border:2px solid #e2e8f0;background:#f8fafc;color:#0f766e;font-weight:600;font-size:13px;width:100%;outline:none;cursor:pointer;">
                {options_html}
            </select>
            <button onclick="const drone = document.getElementById('voting_drone_select').value;
                            fetch('/trigger_voting', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{drone_id: drone}})}})
                            .then(r=>r.json())
                            .then(d=>alert(d.message||d.error||'Voting demo queued'))
                            .catch(e=>alert('Error: '+e));"
                    onmouseover="this.style.backgroundColor='#ffe4e6';this.style.borderColor='#fb7185';" onmouseout="this.style.backgroundColor='#f8fafc';this.style.borderColor='#e2e8f0';"
                    style="padding:10px 14px;background:#f8fafc;color:#e11d48;border:2px solid #e2e8f0;border-radius:20px;cursor:pointer;font-weight:700;font-size:13px;width:100%;text-align:center;transition:all 0.2s ease;">
                🗳️ Start 5-Tick Idle Demo
            </button>
          </div>
        </div>
        <img src="x" onerror="
            let el = document.getElementById('voting-simulator-comp-container');
            if(el && el.parentElement && el.parentElement.parentElement && el.parentElement.parentElement.id === 'elements') {{
                let sidebar = document.getElementById('sidebar');
                if(sidebar) {{
                    sidebar.appendChild(el.parentElement);
                }}
            }}
        " style="display:none;">
        """


class VotingMovementDashboard(base.TextElement):
    def render(self, model):
        movement_logs = getattr(model, "movement_history", [])
        voting_logs = getattr(model, "_voting_flow_log", [])
        warning_drone_id = str(getattr(model, "_idle_warning_drone_id", "") or "")
        warning_visible = bool(getattr(model, "_idle_warning_visible", False))

        combined_rows = []

        for entry in movement_logs:
            if len(entry) == 5:
                tick, d_id, p_from, p_to, reason = entry
            else:
                tick, d_id, p_from, p_to = entry
                reason = "Moving"
            combined_rows.append(
                {
                    "kind": "move",
                    "tick": int(tick),
                    "drone": str(d_id),
                    "p_from": p_from,
                    "p_to": p_to,
                    "reason": str(reason),
                }
            )

        for idx, entry in enumerate(voting_logs):
            combined_rows.append(
                {
                    "kind": "voting",
                    "tick": int(entry.get("tick", 0)),
                    "message": str(entry.get("message", "")),
                    "idx": idx,
                }
            )

        combined_rows.sort(
            key=lambda row: (row.get("tick", 0), row.get("idx", 0) if row.get("kind") == "voting" else 0)
        )
        recent_rows = combined_rows[-120:]

        log_rows = ""
        for row in reversed(recent_rows):
            if row["kind"] == "move":
                reason = row["reason"]
                short_reason = reason if len(reason) < 120 else reason[:117] + "..."
                drone_label = row["drone"]
                if warning_visible and row["drone"] == warning_drone_id:
                    drone_label = f"{drone_label} ⚠"
                log_rows += (
                    f"<div style='border-bottom: 1px solid #e0e0e0; padding: 6px 0; margin: 0; font-size: 13px; display: flex; justify-content: space-between;'>"
                    f"<div>"
                    f"<span style='color:#a0a0a0; font-family: monospace; font-size: 11px; margin-right: 8px;'>[T:{row['tick']:03d}]</span> "
                    f"<strong style='color:#333; margin-right: 4px;'>{drone_label}</strong> <span style='color:#666;'>moved</span> <span style='color:#0078d4;'>{row['p_from']}</span> &rarr; <span style='color:#28a745;'>{row['p_to']}</span>"
                    f"</div>"
                    f"<div style='color: #6c757d; font-style: italic; font-size: 12px; max-width: 600px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='{reason}'>{short_reason}</div>"
                    f"</div>"
                )
            else:
                msg = row["message"]
                log_rows += (
                    "<div style='border-bottom: 1px dashed #d8c5ff; padding: 6px 0; margin: 0; font-size: 12px; display: flex; align-items: flex-start;'>"
                    f"<span style='color:#7c3aed; font-weight:600; margin-right:6px;'>🗳️</span>"
                    f"<span style='color:#4b5563; font-family: monospace;'>{msg}</span>"
                    "</div>"
                )

        if not log_rows:
            log_rows = "<div style='color: #888; font-style: italic; font-size: 13px; padding: 8px; text-align: center;'>Awaiting drone deployment...</div>"

        return f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 10px auto; width: 980px; box-sizing: border-box; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; height: 350px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background: white;">
            <div style="background: #2c3e50; color: white; padding: 10px 15px; font-weight: 600; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
                <span>🛰️ Drone Movement Log</span>
                <span style="font-size: 11px; font-weight: normal; opacity: 0.8; background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 10px;">Chain of Thought</span>
            </div>
            <div style="padding: 0 15px; overflow-y: auto; flex: 1; background: #fafafa;">
                {log_rows}
            </div>
        </div>
        <img src="x" onerror="
    let el = document.getElementById('video-export-comp-container');
    if(el && el.parentElement && el.parentElement.parentElement && el.parentElement.parentElement.id === 'elements') {{
        let sidebar = document.getElementById('sidebar');
        if(sidebar) {{
            sidebar.appendChild(el.parentElement);
        }}
    }}
" style="display:none;">
        """


class MissionMetricsDashboard(base.TextElement):
        """Mesa TextElement showing mission/swarm/agentic evaluation metrics."""

        def render(self, model):
                m = model.get_metrics_snapshot()

                fhc_ticks = m.get("fhc_ticks", -1)
                fhc_secs = m.get("fhc_seconds", -1.0)
                fhc_text = "pending" if int(fhc_ticks) < 0 else f"{int(fhc_ticks)} ticks / {float(fhc_secs):.2f}s"

                return f"""
                <div style="font-family:Arial;padding:10px;margin-bottom:10px;background:#111827;border-radius:8px;border:1px solid #374151;color:#e5e7eb;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span style="font-weight:700;font-size:14px;">📊 Evaluation Dashboard ({m.get('split', 'unspecified').upper()})</span>
                        <span style="font-size:11px;color:#9ca3af;">Scenario: {m.get('scenario', 'N/A')} | T:{int(m.get('tick', 0)):03d}</span>
                    </div>

                    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;">
                        <div style="background:#1f2937;padding:8px;border-radius:6px;">
                            <div style="font-size:12px;font-weight:600;color:#93c5fd;">Mission Efficacy</div>
                            <div style="font-size:12px;margin-top:4px;">FHC time: <b>{fhc_text}</b></div>
                            <div style="font-size:12px;">Area coverage rate: <b>{float(m.get('area_coverage_rate', 0.0)):.3f}</b></div>
                            <div style="font-size:12px;">Survivor detected rate: <b>{float(m.get('survivor_detected_rate', 0.0)):.1f}%</b></div>
                        </div>

                        <div style="background:#1f2937;padding:8px;border-radius:6px;">
                            <div style="font-size:12px;font-weight:600;color:#86efac;">Swarm Intelligence</div>
                            <div style="font-size:12px;margin-top:4px;">Exploration score: <b>{float(m.get('exploration_score', 0.0)):.1f}</b></div>
                            <div style="font-size:12px;">Redundancy penalty: <b>{float(m.get('redundancy_penalty', 0.0)):.3f}</b></div>
                            <div style="font-size:12px;">Dispersion penalty: <b>{float(m.get('dispersion_penalty', 0.0)):.3f}</b></div>
                            <div style="font-size:12px;">Average battery: <b>{float(m.get('avg_battery', 0.0)):.1f}</b></div>
                            <div style="font-size:12px;">Drone wait count / tick: <b>{int(m.get('drone_wait_count', 0))}</b></div>
                        </div>

                        <div style="background:#1f2937;padding:8px;border-radius:6px;">
                            <div style="font-size:12px;font-weight:600;color:#fca5a5;">Agentic + MCP</div>
                            <div style="font-size:12px;margin-top:4px;">Tool call accuracy: <b>{float(m.get('tool_call_accuracy', 0.0)):.1f}%</b></div>
                            <div style="font-size:12px;">Discovery latency: <b>{float(m.get('discovery_latency_ticks', 0.0)):.2f} ticks / {float(m.get('discovery_latency_seconds', 0.0)):.2f}s</b></div>
                            <div style="font-size:12px;">Calls: valid {int(m.get('tool_valid', 0))}/{int(m.get('tool_total', 0))}</div>
                        </div>
                    </div>

                    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
                        <button onclick="fetch('/export_metrics', {{method:'POST'}})
                                                        .then(r=>r.json())
                                                        .then(d=>alert(d.message||d.error||'Export complete'))
                                                        .catch(e=>alert('Error: '+e));"
                                        style="padding:6px 14px;background:#0ea5e9;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;">
                            📤 Export Metrics Report
                        </button>
                        <button onclick="fetch('/run_split_eval', {{method:'POST'}})
                                                        .then(r=>r.json())
                                                        .then(d=>alert(d.message||d.error||'Split evaluation started'))
                                                        .catch(e=>alert('Error: '+e));"
                                        style="padding:6px 14px;background:#2563eb;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;">
                            🧪 Run Train/Test Evaluation
                        </button>
                        <button onclick="fetch('/split_eval_status')
                                                        .then(r=>r.json())
                                                        .then(d=>alert((d.message||'Status') + (d.latest_summary ? ('\nSummary: '+d.latest_summary) : '')))
                                                        .catch(e=>alert('Error: '+e));"
                                        style="padding:6px 14px;background:#334155;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px;min-width:140px;text-align:center;">
                            📡 Check Eval Status
                        </button>
                    </div>
                </div>
                """


class ExportVideoHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /export_video — exports captured frames to timestamped MP4."""

    def initialize(self, server):
        self.server = server

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            fc = model._frame_capture
            if not fc.frames:
                self.write(json.dumps({"error": "No frames captured yet. Run the simulation first."}))
                return
            output_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_recording_{ts}.mp4"
            output_path = os.path.join(output_dir, filename)
            fc.export_to_mp4(output_path, fps=4)
            self.write(json.dumps({
                "message": f"✅ Exported {len(fc.frames)} frames to {output_path}",
                "path": output_path,
                "file": filename,
                "frames": len(fc.frames),
            }))
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"error": str(exc)}))


class ClearFramesHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /clear_frames — clears all captured frames."""

    def initialize(self, server):
        self.server = server

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            count = len(model._frame_capture.frames)
            model._frame_capture.clear_frames()
            self.write(json.dumps({"message": f"🗑️ Cleared {count} frames."}))
        except Exception as exc:
            self.write(json.dumps({"error": str(exc)}))


class ToggleCaptureHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /toggle_capture — toggles frame capture on/off."""

    def initialize(self, server):
        self.server = server

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            fc = model._frame_capture
            fc.capturing = not fc.capturing
            state = "Recording" if fc.capturing else "Stopped"
            self.write(json.dumps({"message": f"⏯ Capture is now: {state}"}))
        except Exception as exc:
            self.write(json.dumps({"error": str(exc)}))


class TriggerVotingHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /trigger_voting — starts 5-tick idle voting demo."""

    def initialize(self, server):
        self.server = server

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            body = json.loads(self.request.body.decode('utf-8'))
            drone_id = body.get('drone_id', 'd_0')

            result = model.start_idle_voting_flow(drone_id)

            self.write(json.dumps(result))
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"error": str(exc)}))


class ExportMetricsHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /export_metrics — exports dashboard metrics to JSON file."""

    def initialize(self, server):
        self.server = server

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            result = model.export_metrics_to_file()
            self.write(json.dumps(result))
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"error": str(exc)}))


class MetricsSnapshotHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """GET /metrics_snapshot — lightweight live metrics payload for external dashboards."""

    def initialize(self, server):
        self.server = server

    def get(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            self.write(
                json.dumps(
                    {
                        "ok": True,
                        "timestamp": datetime.now().isoformat(),
                        "snapshot": model.get_metrics_snapshot(),
                    }
                )
            )
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"ok": False, "error": str(exc)}))


class MetricsSeriesHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """GET /metrics_series?limit=120 — recent metrics timeseries for external dashboards."""

    def initialize(self, server):
        self.server = server

    def get(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            limit_q = self.get_query_argument("limit", "120")
            limit = max(1, min(2000, int(limit_q)))
            self.write(
                json.dumps(
                    {
                        "ok": True,
                        "timestamp": datetime.now().isoformat(),
                        "series": model.get_metrics_series_tail(limit=limit),
                    }
                )
            )
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"ok": False, "error": str(exc)}))


class HardStopHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /hard_stop — immediate emergency stop for simulation loop."""

    def initialize(self, server):
        self.server = server

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            result = model.hard_stop()
            self.write(json.dumps(result))
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"error": str(exc)}))


class ResumeSimulationHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /resume_simulation — clear hard-stop and allow running again."""

    def initialize(self, server):
        self.server = server

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            model = self.server.model
            result = model.resume_from_hard_stop()
            self.write(json.dumps(result))
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"error": str(exc)}))


class RunSplitEvalHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /run_split_eval — run split evaluator in background."""

    def post(self):
        self.set_header("Content-Type", "application/json")
        try:
            existing = _SPLIT_EVAL_STATE.get("process")
            if existing is not None and getattr(existing, "poll", lambda: None)() is None:
                self.write(
                    json.dumps(
                        {
                            "error": "Split evaluation already running",
                            "pid": int(getattr(existing, "pid", -1)),
                            "log_file": _SPLIT_EVAL_STATE.get("log_file"),
                        }
                    )
                )
                return

            body: Dict[str, Any] = {}
            if self.request.body:
                try:
                    body = json.loads(self.request.body.decode("utf-8"))
                except Exception:
                    body = {}

            runs = int(body.get("runs_per_scenario", 1))
            max_ticks = int(body.get("max_ticks", 120))
            simulate_ai = bool(body.get("simulate_ai", False))

            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(workspace_dir, f"split_eval_runner_{ts}.log")

            command = [
                sys.executable,
                "evaluate_split_metrics.py",
                "--runs-per-scenario",
                str(max(1, runs)),
                "--max-ticks",
                str(max(10, max_ticks)),
            ]
            if simulate_ai:
                command.append("--simulate-ai")

            log_handle = open(log_file, "w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                cwd=workspace_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )

            _SPLIT_EVAL_STATE["process"] = proc
            _SPLIT_EVAL_STATE["started_at"] = datetime.now().isoformat()
            _SPLIT_EVAL_STATE["log_file"] = log_file
            _SPLIT_EVAL_STATE["command"] = command

            self.write(
                json.dumps(
                    {
                        "message": "🧪 Split evaluation started in background",
                        "pid": int(proc.pid),
                        "log_file": log_file,
                        "command": command,
                    }
                )
            )
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"error": str(exc)}))


class SplitEvalStatusHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """GET /split_eval_status — status of current/last split evaluator run."""

    def get(self):
        self.set_header("Content-Type", "application/json")
        try:
            proc = _SPLIT_EVAL_STATE.get("process")
            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
            latest_summary = _latest_split_eval_summary_path(workspace_dir)
            log_file = _SPLIT_EVAL_STATE.get("log_file")
            tail = _tail_text_file(str(log_file or ""), line_count=12)

            if proc is None:
                self.write(
                    json.dumps(
                        {
                            "running": False,
                            "message": "No split evaluation has been started yet.",
                            "latest_summary": latest_summary,
                            "log_tail": tail,
                        }
                    )
                )
                return

            exit_code = proc.poll()
            if exit_code is None:
                self.write(
                    json.dumps(
                        {
                            "running": True,
                            "pid": int(proc.pid),
                            "message": "Split evaluation is running.",
                            "started_at": _SPLIT_EVAL_STATE.get("started_at"),
                            "log_file": log_file,
                            "latest_summary": latest_summary,
                            "log_tail": tail,
                        }
                    )
                )
                return

            status_text = "completed successfully" if int(exit_code) == 0 else f"failed (exit {exit_code})"
            self.write(
                json.dumps(
                    {
                        "running": False,
                        "pid": int(proc.pid),
                        "exit_code": int(exit_code),
                        "message": f"Split evaluation {status_text}.",
                        "started_at": _SPLIT_EVAL_STATE.get("started_at"),
                        "log_file": log_file,
                        "latest_summary": latest_summary,
                        "log_tail": tail,
                    }
                )
            )
        except Exception as exc:
            traceback.print_exc()
            self.write(json.dumps({"error": str(exc)}))


def trace_portrayal(agent):
    base_portrayal = base.portrayal(agent)
    if base_portrayal is None:
        return None

    try:
        if isinstance(agent, base.DroneAgent):
            model = getattr(agent, "model", None)
            warning_drone_id = str(getattr(model, "_idle_warning_drone_id", "") or "")
            warning_visible = bool(getattr(model, "_idle_warning_visible", False))
            voting_flow = getattr(model, "_voting_flow", {}) or {}
            voting_active = bool(voting_flow.get("active", False))

            if (
                warning_visible
                and voting_active
                and warning_drone_id == "d_0"
                and str(getattr(agent, "unique_id", "")) == "d_0"
            ):
                patched = dict(base_portrayal)
                current_text = str(patched.get("text", "") or "")
                patched["text"] = f"{current_text}⚠" if current_text else "⚠"
                patched["text_color"] = "#ffffff"
                return patched
    except Exception:
        return base_portrayal

    return base_portrayal



grid = base.CanvasGrid(trace_portrayal, 24, 16, 980, 650)
chart = base.ChartModule(
    [
        {"Label": "SurvivorsFound", "Color": "Green"},
        {"Label": "SectorsDone", "Color": "Blue"},
        {"Label": "AvgBattery", "Color": "Orange"},
        {"Label": "CoveragePercent", "Color": "Purple"},
        {"Label": "ExplorationScore", "Color": "Red"},
        {"Label": "ToolCallAccuracy", "Color": "Teal"},
        {"Label": "DiscoveryLatencyTicks", "Color": "Brown"},
        {"Label": "DroneWaitCount", "Color": "Black"},
    ],
    canvas_height=300,
    canvas_width=980,
    data_collector_name="datacollector",
)
legend = base.Legend()
movement_board = VotingMovementDashboard()
style_element = SidebarStyleElement()
video_export = VideoExportElement()
voting_simulator = VotingSimulatorElement()

server = base.ModularServer(
    LangGraphTraceDroneRescueModel,
    [style_element, video_export, legend, grid, chart, voting_simulator, movement_board],
    "Drone Fleet Search & Rescue — LangGraph TRACE Bridge",
    {
        "width": 24,
        "height": 16,
        "scenario": base.Choice(
            "Scenario",
            value="A: Palu city",
            choices=list(base.SCENARIOS.keys()),
            description="Pick scenario. Trace output writes to langgraph_tick_trace_log.jsonl",
        ),
        "num_drones": base.Slider("Drones", value=4, min_value=3, max_value=4, step=1),
        "num_survivors": base.Slider("Survivors", value=12, min_value=5, max_value=20, step=1),
        "simulate_ai": True,
        "langgraph_model": base.Choice(
            "LangGraph commander model",
            value="qwen/qwen3-14b",
            choices=["qwen/qwen3-14b", "qwen/qwen-2.5-7b-instruct"],
            description="OpenRouter model for Commander",
        ),
        "langgraph_operator_model": base.Choice(
            "LangGraph operator model",
            value="qwen/qwen-2.5-7b-instruct",
            choices=["qwen/qwen-2.5-7b-instruct", "qwen/qwen3-14b"],
            description="OpenRouter model for per-drone Operator",
        ),
        "ai_delay_s": base.Slider("AI delay (sec)", value=0.15, min_value=0.0, max_value=1.5, step=0.05),
    },
)


def _pick_free_port(preferred: List[int]) -> int:
    for port in preferred:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("", port))
            return port
        except OSError:
            continue

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    server.port = _pick_free_port([8544, 8545, 8546, 8547])
    extra_handlers = [
        (r"/export_video", ExportVideoHandler, {"server": server}),
        (r"/clear_frames", ClearFramesHandler, {"server": server}),
        (r"/toggle_capture", ToggleCaptureHandler, {"server": server}),
        (r"/metrics_snapshot", MetricsSnapshotHandler, {"server": server}),
        (r"/metrics_series", MetricsSeriesHandler, {"server": server}),
        (r"/hard_stop", HardStopHandler, {"server": server}),
        (r"/resume_simulation", ResumeSimulationHandler, {"server": server}),
        (r"/trigger_voting", TriggerVotingHandler, {"server": server}),
        (r"/export_metrics", ExportMetricsHandler, {"server": server}),
        (r"/run_split_eval", RunSplitEvalHandler),
        (r"/split_eval_status", SplitEvalStatusHandler),
    ]
    server.add_handlers(r".*", extra_handlers)
    print(f"Launching LangGraph TRACE bridge server at http://127.0.0.1:{server.port}")
    print("Trace output file: langgraph_tick_trace_log.jsonl")
    print("🎬 Video export endpoints: /export_video, /clear_frames, /toggle_capture")
    print("📡 Metrics API endpoints: /metrics_snapshot, /metrics_series")
    print("⛔ Stop endpoints: /hard_stop, /resume_simulation")
    print("🗳️ Voting endpoint: /trigger_voting")
    print("📊 Metrics endpoint: /export_metrics")
    print("🧪 Split eval endpoints: /run_split_eval, /split_eval_status")
    server.launch()
