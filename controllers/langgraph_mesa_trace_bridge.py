"""
LangGraph ↔ Mesa UI trace bridge.

This controller is intentionally verbose and records full per-tick data flow:
- what is passed into Commander
- Commander raw/parsed output
- what is passed into each Operator
- Operator raw/parsed output
- staged commands
- final flush execution results

Trace output is appended to `langgraph_tick_trace_log.jsonl`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
import json
import os
import time

import core.mesa_drone_rescue_mcp as base

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except Exception:
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    ChatOpenAI = None  # type: ignore


class TickState(TypedDict):
    tick: int
    global_mission_state: Dict[str, Any]
    active_drone_telemetry: Dict[str, Any]
    available_drones: List[str]


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    import re

    clean = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"```\s*", "", clean)
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(clean[start : end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


class LangGraphMesaTraceController(base.GeminiAiController):
    """Verbose LangGraph-like controller with full tick tracing for Mesa UI."""

    def __init__(
        self,
        tools: base.InUiToolServer,
        model_name: str = "qwen/qwen3-14b",
        operator_model_name: str = "qwen/qwen-2.5-7b-instruct",
        action_delay_s: float = 0.0,
        trace_file: str = "langgraph_tick_trace_log.jsonl",
    ):
        super().__init__(tools, model_name, action_delay_s, 100)
        self.tools = tools
        self.model_name = model_name
        self.operator_model_name = operator_model_name
        self._tick = 0
        self.trace_file = trace_file

        self._commander_client = self._maybe_client(self.model_name)
        self._operator_client = self._maybe_client(self.operator_model_name)

        # Custom JSON mission log initialization
        self._mission_log_file = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mission_log.json'))
        self._mission_log_data: Dict[str, Any] = {"ticks": []}
        self._mission_successful_logged = False

    @staticmethod
    def _json_pretty(payload: Any, limit: int = 12000) -> str:
        try:
            txt = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            txt = str(payload)
        if len(txt) > limit:
            return txt[:limit] + "\n...<truncated>"
        return txt

    @staticmethod
    def _rule(title: str) -> None:
        pass

    def _print_block(self, title: str, payload: Any) -> None:
        pass

    def _print_text(self, title: str, text: str) -> None:
        pass

    def _print_tick_decision_summary(
        self,
        tick: int,
        commander: Dict[str, Any],
        operators: List[Dict[str, Any]],
        flush_results: List[Dict[str, Any]],
    ) -> None:
        print("\n" + "-" * 72)
        print(f"[TICK {tick}] COT + ACTION SUMMARY (TERMINAL)")
        print("-" * 72)

        commander_cot = str(commander.get("cot_summary") or "No commander summary available")
        assignments = commander.get("assignments", {})
        print("[COMMANDER COT]")
        print(f"- {commander_cot}")
        print("[COMMANDER ACTION]")
        print(f"- assignments: {assignments}")

        tick_data: Dict[str, Any] = {
            "tick": tick,
            "commander": {
                "cot": commander_cot,
                "action": assignments
            },
            "operators": []
        }

        executed_by_drone: Dict[str, Dict[str, Any]] = {}
        executed_result_by_drone: Dict[str, Dict[str, Any]] = {}
        tick_detected_survivors: List[Dict[str, Any]] = []
        for fr in flush_results:
            cmd = fr.get("command", {}) if isinstance(fr, dict) else {}
            res = fr.get("result", {}) if isinstance(fr, dict) else {}
            did = str(cmd.get("drone_id", ""))
            if did:
                executed_by_drone[did] = cmd
                if isinstance(res, dict):
                    executed_result_by_drone[did] = res
                    found_now = res.get("survivors_found", [])
                    if isinstance(found_now, list):
                        for s in found_now:
                            if isinstance(s, dict):
                                tick_detected_survivors.append(s)

        if tick_detected_survivors:
            print("[TICK DETECTION SUMMARY]")
            print(
                f"- Newly detected this tick: {len(tick_detected_survivors)} "
                f"{tick_detected_survivors}"
            )
        else:
            print("[TICK DETECTION SUMMARY]")
            print("- Newly detected this tick: 0")

        for op in operators:
            drone_id = str(op.get("drone_id", "?"))
            decision = op.get("decision", {}) if isinstance(op, dict) else {}
            action = str(decision.get("action", "wait"))
            x = decision.get("x")
            y = decision.get("y")
            reason = str(decision.get("reason", ""))
            executed = executed_by_drone.get(drone_id, {})
            executed_result = executed_result_by_drone.get(drone_id, {})
            ex_action = str(executed.get("action", action))
            ex_x = executed.get("x", x)
            ex_y = executed.get("y", y)

            filter_result = executed_result.get("filter_result") if isinstance(executed_result, dict) else None
            filter_note = ""
            if isinstance(filter_result, dict):
                if bool(filter_result.get("human_detected", False)):
                    filter_note = (
                        " After data filtering (this scan): survivor detected "
                        f"(intensity={filter_result.get('intensity')} >= threshold={filter_result.get('threshold')})."
                    )
                else:
                    filter_note = (
                        " After data filtering (this scan): no survivor suspected "
                        f"(intensity={filter_result.get('intensity')} < threshold={filter_result.get('threshold')})."
                    )

            print(f"[OPERATOR {drone_id} COT]")
            print(f"- {(reason or 'No reasoning provided')}")
            if ex_x is not None and ex_y is not None:
                op_action = f"{ex_action}({drone_id}, x={ex_x}, y={ex_y})"
                print(f"- {op_action}")
            else:
                op_action = f"{ex_action}({drone_id})"
                print(f"- {op_action}")
            
            tick_data["operators"].append({
                "drone_id": drone_id,
                "cot": reason or "No reasoning provided",
                "action": op_action
            })

        # --- Capture logic for mission_log.json ---
        self._mission_log_data["ticks"].append(tick_data)
        self._dump_mission_log()
        
    def _dump_mission_log(self) -> None:
        try:
            with open(self._mission_log_file, "w", encoding="utf-8") as f:
                json.dump(self._mission_log_data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[WARN] Failed to write mission_log.json: {exc}")

    def _maybe_client(self, model_string: Optional[str] = None):
        if not model_string:
            return None
        if ChatOpenAI is None:
            return None
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=model_string,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "MesaDroneRescue-Trace"),
            },
        )

    def _record_voting_trace(self, tick: int) -> Optional[Dict[str, Any]]:
        """Record a voting round trace if one is in progress."""
        voting_sim = self.tools.voting_simulator
        if voting_sim.state == "IDLE" or not voting_sim.idle_drone_id:
            return None
        
        voting_state = voting_sim.get_voting_state()
        
        voting_trace: Dict[str, Any] = {
            "type": "drone_voting",
            "tick": tick,
            "timestamp": time.time(),
            "state": voting_sim.state,
            "idle_drone_id": voting_sim.idle_drone_id,
        }
        
        if voting_sim.state == "VOTING" or voting_sim.state == "VOTING_COMPLETE":
            voting_trace["votes"] = voting_state["votes"]
            voting_trace["vote_tally"] = voting_state["vote_tally"]
            voting_trace["winning_sector"] = voting_state["winning_sector"]
        
        if voting_sim.state == "VOTING_COMPLETE":
            voting_trace["winning_action"] = voting_state["winning_action"]
        
        if voting_sim.state == "EXECUTING":
            voting_trace["execution_result"] = voting_sim.execution_result
        
        return voting_trace

    def _print_voting_trace(self, voting_trace: Dict[str, Any]) -> None:
        """Pretty-print a voting trace."""
        self._rule("[VOTING ROUND] Mock drone voting simulation")
        print(f"Idle Drone: {voting_trace.get('idle_drone_id')}")
        print(f"State: {voting_trace.get('state')}")
        
        votes = voting_trace.get("votes", [])
        if votes:
            print(f"\n[VOTES] {len(votes)} drone(s) voted:")
            for vote in votes:
                print(f"  {vote['voter_id']} → Sector {vote['target_sector']}: {vote['reasoning']}")
        
        tally = voting_trace.get("vote_tally", {})
        if tally:
            print(f"\n[TALLY]")
            for sector, count in sorted(tally.items()):
                is_winner = sector == voting_trace.get("winning_sector")
                marker = " ← WINNER" if is_winner else ""
                print(f"  Sector {sector}: {count} vote(s){marker}")
        
        action = voting_trace.get("winning_action", {})
        if action:
            print(f"\n[ACTION]")
            print(f"  Drone: {action.get('drone_id')}")
            print(f"  Type: {action.get('type')}")
            print(f"  Target: ({action.get('x')}, {action.get('y')})")
            print(f"  Reason: {action.get('reason')}")


    def _append_trace(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[TRACE] Failed writing trace file: {exc}")

    @staticmethod
    def _fallback_assignments(ms: Dict[str, Any], drones: List[str]) -> Dict[str, int]:
        sectors = ms.get("sectors", [])
        incomplete = [
            sector
            for sector in sectors
            if float(sector.get("coverage_pct", 0)) < 100.0 and sector.get("id") is not None
        ]
        ordered = sorted(
            incomplete,
            key=lambda sector: (float(sector.get("coverage_pct", 0)), int(sector.get("id", 0))),
        )
        if not ordered:
            ordered = sectors # All complete, just assign them anywhere
            
        out: Dict[str, int] = {}
        for i, drone_id in enumerate(drones):
            if ordered:
                sector = ordered[i % len(ordered)]
                out[drone_id] = int(sector["id"])
            else:
                out[drone_id] = 1 # Absolute fallback
        return out

    def _run_commander(
        self,
        ms: Dict[str, Any],
        drones: List[str],
        telemetry: Dict[str, Any],
        trace: Dict[str, Any],
    ) -> Dict[str, int]:
        if not self._commander_client or SystemMessage is None or HumanMessage is None:
            trace["commander"] = {
                "status": "unavailable",
                "reason": "Commander client unavailable",
            }
            return self._fallback_assignments(ms, drones)

        system_prompt = (
            "You are the Mission Commander. Assign all available drones to sectors.\n"
            "Focus on sectors with coverage_pct < 99.\n"
            "To increase overall efficiency, analyze the Unscanned grids and map each drone to a specific sector that contains those remaining grids.\n"
            "No overlap across drones.\n"
            "Output ALL text, reasoning, and JSON keys/values STRICTLY AND ONLY in English.\n"
            "Return ONLY JSON: {\"assignments\": {\"d_0\": 1, \"d_1\": 2}}"
        )
        unscanned_grids = ms.get('unscanned_grids', [])
        unscanned_text = f"\nUnscanned:[{json.dumps(unscanned_grids, separators=(',', ':'))[:1200]}]" if unscanned_grids else ""

        # Compact the list of dictionaries to save tokens
        sectors_compact = []
        for s in ms.get('sectors', []):
            sectors_compact.append({"id": s["id"], "cov": s["coverage_pct"]})

        user_prompt = (
            f"Drones:{drones}\n"
            f"Telem:{json.dumps(telemetry, separators=(',', ':'))}\n"
            f"Sectors:{json.dumps(sectors_compact, separators=(',', ':'))}"
            f"{unscanned_text}"
        )

        if "qwen3" in self.model_name.lower():
            system_prompt = "/no_think\n" + system_prompt

        assignments: Dict[str, int] = {}
        raw = ""
        parsed: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        used_fallback = False
        cot_summary = ""

        self._print_block(
            "[COMMANDER IN] payload passed to Commander agent",
            {
                "model": self.model_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        )

        if self._tick == 1:
            print("\n" + "=" * 72)
            print("[TICK 1] PROMPT SENT TO COMMANDER")
            print("=" * 72)
            print("[SYSTEM]")
            print(system_prompt)
            print("\n[USER]")
            print(user_prompt)
            print("=" * 72)

        try:
            response = self._commander_client.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = str(response.content)
            parsed = _extract_json_object(raw)
            candidate = (parsed or {}).get("assignments", {})
            if isinstance(candidate, dict) and candidate:
                assignments = {str(k): int(v) for k, v in candidate.items()}
                # Fill missing drones
                missing = [d for d in drones if d not in assignments]
                if missing:
                    fallback_partial = self._fallback_assignments(ms, missing)
                    assignments.update(fallback_partial)
                
                cot_summary = (
                    "Prioritized incomplete sectors and distributed drones without overlap "
                    "based on current sector coverage and live drone availability."
                )
            else:
                used_fallback = True
                assignments = self._fallback_assignments(ms, drones)
                cot_summary = "Commander output was invalid/empty, so fallback assigned drones to lowest-coverage sectors."
        except Exception as exc:
            error = str(exc)
            used_fallback = True
            assignments = self._fallback_assignments(ms, drones)
            cot_summary = "Commander call failed, so fallback assigned drones to lowest-coverage sectors."

        self._print_text("[COMMANDER OUT] raw return from Commander agent", raw)
        self._print_block(
            "[COMMANDER OUT] parsed + final assignment result",
            {
                "parsed_json": parsed,
                "assignments": assignments,
                "used_fallback": used_fallback,
                "error": error,
            },
        )

        trace["commander"] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw,
            "parsed_json": parsed,
            "cot_summary": cot_summary,
            "assignments": assignments,
            "used_fallback": used_fallback,
            "error": error,
        }
        return assignments

    def _run_operator(
        self,
        drone_id: str,
        assigned_sector_id: Optional[int],
        drone_telemetry: Dict[str, Any],
        ms: Dict[str, Any],
    ) -> Dict[str, Any]:
        battery = float(drone_telemetry.get("battery", 0))
        at_base = bool(drone_telemetry.get("at_base", False))

        sector_info = "Unknown (free roam)"
        if assigned_sector_id is not None:
            for s in ms.get("sectors", []):
                if s.get("id") == assigned_sector_id:
                    origin = s.get("origin", [0, 0])
                    size = s.get("size", [1, 1])
                    sector_info = (
                        f"Sector {assigned_sector_id}: "
                        f"x-range [{origin[0]} to {origin[0] + size[0] - 1}], "
                        f"y-range [{origin[1]} to {origin[1] + size[1] - 1}]"
                    )
                    break

        operator_trace: Dict[str, Any] = {
            "drone_id": drone_id,
            "assigned_sector_id": assigned_sector_id,
            "drone_telemetry": drone_telemetry,
            "stage_result": None,
        }

        self._print_block(
            f"[OPERATOR IN] payload passed to Operator agent for {drone_id}",
            {
                "model": self.operator_model_name,
                "drone_id": drone_id,
                "assigned_sector_id": assigned_sector_id,
                "drone_telemetry": drone_telemetry,
            },
        )

        if assigned_sector_id is None:
            staged = self.tools.stage_drone_command(
                drone_id, "wait", reason="No commander sector assignment available"
            )
            operator_trace["decision"] = {"action": "wait", "reason": "missing_assignment"}
            operator_trace["stage_result"] = staged
            self._print_block(
                f"[MCP-STYLE TRIGGER] stage_drone_command called for {drone_id}",
                {
                    "tool": "stage_drone_command",
                    "arguments": {
                        "drone_id": drone_id,
                        "action": "wait",
                        "reason": "No commander sector assignment available",
                    },
                    "result": staged,
                },
            )
            return operator_trace

        if battery < 30 and not at_base:
            staged = self.tools.stage_drone_command(
                drone_id, "recall_to_base", reason="Low battery fallback"
            )
            operator_trace["decision"] = {"action": "recall_to_base", "reason": f"low_battery : Battery at {battery}% is below 30% threshold, and drone is not at base"}
            operator_trace["stage_result"] = staged
            self._print_block(
                f"[MCP-STYLE TRIGGER] stage_drone_command called for {drone_id}",
                {
                    "tool": "stage_drone_command",
                    "arguments": {
                        "drone_id": drone_id,
                        "action": "recall_to_base",
                        "reason": "Low battery fallback",
                    },
                    "result": staged,
                },
            )
            return operator_trace

        if at_base and battery < 80:
            staged = self.tools.stage_drone_command(
                drone_id, "charge_drone", reason="Replenishing battery at base"
            )
            operator_trace["decision"] = {"action": "charge_drone", "reason": f"charge_at_base : Battery at {battery}% is below 20% threshold, but drone is at base and can recharge"}
            operator_trace["stage_result"] = staged
            self._print_block(
                f"[MCP-STYLE TRIGGER] stage_drone_command called for {drone_id}",
                {
                    "tool": "stage_drone_command",
                    "arguments": {
                        "drone_id": drone_id,
                        "action": "charge_drone",
                        "reason": "Replenishing battery at base",
                    },
                    "result": staged,
                },
            )
            return operator_trace

        if not self._operator_client or SystemMessage is None or HumanMessage is None:
            staged = self.tools.stage_drone_command(
                drone_id, "wait", reason="Operator client unavailable"
            )
            operator_trace["decision"] = {"action": "wait", "reason": "operator_unavailable"}
            operator_trace["stage_result"] = staged
            self._print_block(
                f"[MCP-STYLE TRIGGER] stage_drone_command called for {drone_id}",
                {
                    "tool": "stage_drone_command",
                    "arguments": {
                        "drone_id": drone_id,
                        "action": "wait",
                        "reason": "Operator client unavailable",
                    },
                    "result": staged,
                },
            )
            return operator_trace

        grid_bounds = ms.get("grid", [24, 16])
        obstacle_positions = json.dumps(ms.get("obstacle_positions", []), separators=(',', ':'))[:1000]

        system_prompt = (
            "You are an automated Drone Operator AI.\n"
            "Valid actions: move_and_scan, recall_to_base, charge_drone, wait.\n"
            "For move_and_scan you MUST provide integer x and y.\n"
            "To increase efficiency, prioritize moving your drone to scan the specific Unscanned grids within your Assigned Sector.\n"
            "Data preprocessing is handled internally; do not output verification actions.\n"
            "Output ALL text, reasoning, and JSON keys/values STRICTLY AND ONLY in English.\n"
            "Return exactly one JSON object and nothing else."
        )
        unscanned_grids = ms.get('unscanned_grids', [])
        unscanned_text = f"Unscanned:[{json.dumps(unscanned_grids, separators=(',', ':'))[:1200]}]\n" if unscanned_grids else ""

        user_prompt = (
            f"Drone ID: {drone_id} | Bat: {battery}%\n"
            f"Pos: {drone_telemetry.get('pos')}\n"
            f"Assigned Sector: {sector_info}\n"
            f"Grid Bounds: x=[0,{grid_bounds[0]-1}], y=[0,{grid_bounds[1]-1}]\n"
            f"Obstacles: {obstacle_positions}\n"
            f"{unscanned_text}"
            "CRITICAL: Output ONLY a single JSON object with the keys 'action', 'x', 'y', and 'reason'. Do not regurgitate the state.\n"
            "Example: {\"action\": \"move_and_scan\", \"x\": 10, \"y\": 15, \"reason\": \"Scanning assigned sector\"}"
        )

        if "qwen3" in self.operator_model_name.lower():
            system_prompt = "/no_think\n" + system_prompt

        raw = ""
        parsed: Optional[Dict[str, Any]] = None
        error: Optional[str] = None

        try:
            self._print_block(
                f"[OPERATOR IN] full prompts for {drone_id}",
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
            )
            response = self._operator_client.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            raw = str(response.content)
            parsed = _extract_json_object(raw)
            action = str((parsed or {}).get("action", "wait"))
            x = (parsed or {}).get("x")
            y = (parsed or {}).get("y")
            reason = str((parsed or {}).get("reason", "Operator LLM"))
            staged = self.tools.stage_drone_command(drone_id, action, x, y, reason)
            self._print_text(
                f"[OPERATOR OUT] raw return from Operator agent for {drone_id}",
                raw,
            )
            self._print_block(
                f"[OPERATOR OUT] parsed decision for {drone_id}",
                {
                    "parsed_json": parsed,
                    "decision": {"action": action, "x": x, "y": y, "reason": reason},
                },
            )
            self._print_block(
                f"[MCP-STYLE TRIGGER] stage_drone_command called for {drone_id}",
                {
                    "tool": "stage_drone_command",
                    "arguments": {
                        "drone_id": drone_id,
                        "action": action,
                        "x": x,
                        "y": y,
                        "reason": reason,
                    },
                    "result": staged,
                },
            )
            operator_trace.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": raw,
                    "parsed_json": parsed,
                    "decision": {"action": action, "x": x, "y": y, "reason": reason},
                    "stage_result": staged,
                    "error": None,
                }
            )
            if not staged.get("staged", False):
                fallback = self.tools.stage_drone_command(
                    drone_id,
                    "wait",
                    reason=f"Validation failed: {staged.get('reason', 'unknown')}",
                )
                operator_trace["fallback_stage_result"] = fallback
                self._print_block(
                    f"[MCP-STYLE TRIGGER] fallback stage_drone_command for {drone_id}",
                    {
                        "tool": "stage_drone_command",
                        "arguments": {
                            "drone_id": drone_id,
                            "action": "wait",
                            "reason": f"Validation failed: {staged.get('reason', 'unknown')}",
                        },
                        "result": fallback,
                    },
                )
        except Exception as exc:
            error = str(exc)
            staged = self.tools.stage_drone_command(drone_id, "wait", reason="Operator parsing error")
            self._print_text(
                f"[OPERATOR OUT] raw return from Operator agent for {drone_id}",
                raw,
            )
            self._print_block(
                f"[OPERATOR ERROR] exception and fallback for {drone_id}",
                {
                    "error": error,
                    "fallback_stage_result": staged,
                },
            )
            operator_trace.update(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": raw,
                    "parsed_json": parsed,
                    "decision": {"action": "wait", "reason": "operator_exception"},
                    "stage_result": staged,
                    "error": error,
                }
            )

        return operator_trace

    def think_and_act(self) -> None:
        self._tick += 1

        # ── Record voting round if in progress ──
        voting_trace = self._record_voting_trace(self._tick)
        if voting_trace:
            self._print_voting_trace(voting_trace)
            # Append voting trace immediately
            try:
                with open(self.trace_file, "a") as f:
                    f.write(json.dumps(voting_trace, default=str) + "\n")
            except Exception as exc:
                print(f"[TRACE] Warning: Could not write voting trace: {exc}")

        ms = self.tools.get_mission_state()
        drones_info = self.tools.discover_drones().get("drones", [])
        telemetry = {d["id"]: self.tools.get_drone_status(d["id"]) for d in drones_info}
        available = [d["id"] for d in drones_info if not d.get("disabled")]

        tick_trace: Dict[str, Any] = {
            "type": "langgraph_mesa_trace",
            "tick": self._tick,
            "timestamp": time.time(),
            "models": {
                "commander": self.model_name,
                "operator": self.operator_model_name,
            },
            "input": {
                "mission_state": ms,
                "telemetry": telemetry,
                "available_drones": available,
            },
        }

        self._print_block(
            "[TICK IN] state entering LangGraph trace bridge",
            {
                "tick": self._tick,
                "models": tick_trace["models"],
                "available_drones": available,
                "telemetry": telemetry,
                "sectors": ms.get("sectors", []),
                "obstacle_positions": ms.get("obstacle_positions", []),
            },
        )

        if not available:
            tick_trace["result"] = {"status": "no_available_drones"}
            self._append_trace(tick_trace)
            print("[TRACE] No available drones for this tick.")
            return

        start = time.time()
        assignments = self._run_commander(ms, available, telemetry, tick_trace)

        operators: List[Dict[str, Any]] = []
        for drone_id in available:
            operators.append(
                self._run_operator(
                    drone_id=drone_id,
                    assigned_sector_id=assignments.get(drone_id),
                    drone_telemetry=telemetry.get(drone_id, {}),
                    ms=ms,
                )
            )

        # self._rule("[MCP-STYLE TRIGGER] flush_staged_commands execution")
        # print("This bridge uses in-process MCP-style tools on InUiToolServer:")
        # print("1) stage_drone_command(...) called during operator decisions")
        # print("2) flush_staged_commands() executes staged commands on live Mesa model")
        flush_results = self.tools.flush_staged_commands()
        self._print_block("[MCP-STYLE OUT] flush_staged_commands returned", flush_results)
        elapsed = time.time() - start

        tick_trace["operators"] = operators
        tick_trace["flush_results"] = flush_results
        tick_trace["result"] = {
            "status": "ok",
            "elapsed_s": round(elapsed, 3),
            "assignments": assignments,
        }

        self._append_trace(tick_trace)

        self._print_block("[TICK OUT] complete trace object for this tick", tick_trace)
        # print(f"[TRACE] Commander assignments: {assignments}")
        # print(f"[TRACE] Operators executed: {len(operators)}")
        # print(f"[TRACE] Flush results: {len(flush_results)}")
        # print(f"[TRACE] Trace append -> {self.trace_file}")

        self._print_tick_decision_summary(
            tick=self._tick,
            commander=tick_trace.get("commander", {}),
            operators=operators,
            flush_results=flush_results,
        )

        summary_calls: List[Dict[str, Any]] = []
        for result in flush_results:
            cmd = result.get("command", {})
            summary_calls.append(
                {
                    "tool_name": cmd.get("action", ""),
                    "arguments": cmd,
                    "reasoning": cmd.get("reason", ""),
                }
            )
            self._pause()

        self.tools.update_rolling_summary(self._tick, summary_calls)

        # -- Check for mission success to attach final metrics to mission_log.json --
        if not getattr(self, "_mission_successful_logged", False):
            # Force metrics update so snap is perfectly current
            if hasattr(self.tools.model, "_update_metrics_state"):
                self.tools.model._update_metrics_state()
                
            snap = self.tools.model.get_metrics_snapshot()
            found = int(snap.get("survivors_found", 0))
            total = int(snap.get("survivors_total", 0))
            hidden_total = int(ms.get("actual_survivors_total_hidden", 999))
            
            # Use real hidden total to determine absolute success
            if found >= hidden_total and hidden_total > 0:
                self._mission_successful_logged = True
                fhc_t = snap.get("fhc_ticks", -1)
                fhc_s = snap.get("fhc_seconds", -1.0)
                fhc_str = f"{fhc_t} ticks / {round(float(fhc_s), 2)}s" if fhc_t >= 0 else "pending"
                
                self._mission_log_data["mission_successful"] = True
                self._mission_log_data["metrics"] = {
                    "first_human_contact": fhc_str,
                    "coverage": f"{snap.get('coverage_pct', 0.0)}%",
                    "area_coverage_rate": snap.get("area_coverage_rate", 0.0),
                    "survivor_detected": f"{snap.get('survivor_detected_rate', 0.0)}%",
                    "tool_accuracy": f"{snap.get('tool_call_accuracy', 0.0)}%",
                    "discovery_latency_ticks": snap.get("discovery_latency_ticks", 0.0),
                    "discovery_latency_seconds": snap.get("discovery_latency_seconds", 0.0),
                    "average_drone_wait_idle": snap.get("drone_wait_count", 0)
                }
                self._dump_mission_log()
                print("\n[MISSION SUCCESS] Final metrics appended to mission_log.json!")
