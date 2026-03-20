"""
LangGraph backed controller for Mesa UI Integration.
Uses Commander -> Operator -> Tool execution loop with Send API for parallel processing.
"""

import os
import sys
# Ensure parent directory (V-Hack) is in sys.path so 'core' and others can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated, Callable
import asyncio
import threading
import operator
import os
import time
import json

from pydantic import BaseModel, Field

# We import the tool server and base controller logic
import core.mesa_drone_rescue_mcp as base

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, START, END
    from langgraph.constants import Send
except ImportError:
    # Will fail gracefully if requirements aren't installed
    pass

# Optional adapter import (not required for current direct ClientSession MCP flow)
try:
    from langchain_mcp_adapters.tools import load_mcp_tools
except Exception:
    load_mcp_tools = None  # type: ignore[assignment]
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the actual MCP server script you wrote
server_params = StdioServerParameters(
    command="python",
    args=["mcp_drone_server.py"]
)


# ═══════════════════════════════════════════════════════════════════════════
#  Pydantic & State Models
# ═══════════════════════════════════════════════════════════════════════════

class CommanderAssignments(BaseModel):
    assignments: Dict[str, int] = Field(
        description="Map of drone_id to sector_id. Example: {'d_0': 1, 'd_1': 2}"
    )

class DroneTickState(TypedDict):
    """Global state for a single simulation tick."""
    tick: int
    global_mission_state: Dict[str, Any]
    active_drone_telemetry: Dict[str, Any]
    available_drones: List[str]
    current_commander_order: Dict[str, int]
    
    # Reducers for parallel Operator execution
    staged_commands: Annotated[List[Dict[str, Any]], operator.add]
    fallback_actions: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]


class OperatorState(TypedDict):
    """State for an individual Operator node executing in parallel via Send API."""
    drone_id: str
    global_mission_state: Dict[str, Any]
    drone_telemetry: Dict[str, Any]
    assigned_sector_id: Optional[int]


# ═══════════════════════════════════════════════════════════════════════════
#  LangChain Tools (Loaded via MCP)
# ═══════════════════════════════════════════════════════════════════════════

async def build_mcp_graph_tools():
    if load_mcp_tools is None:
        return []
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # This dynamically pulls the tools over the official MCP protocol!
            tools = await load_mcp_tools(session) 
            return tools


# ═══════════════════════════════════════════════════════════════════════════
#  Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

def build_drone_graph(
    commander_llm: "ChatOpenAI",
    operator_llm: "ChatOpenAI",
    stage_command_fn: Callable[..., Dict[str, Any]],
):
    """Builds and returns the compiled LangGraph for the simulation tick."""

    def fallback_assignments(ms: Dict[str, Any], drones: List[str]) -> Dict[str, int]:
        sectors = ms.get("sectors", [])
        incomplete = [
            sector for sector in sectors
            if float(sector.get("coverage_pct", 0)) < 99.0 and sector.get("id") is not None
        ]
        ordered = sorted(
            incomplete,
            key=lambda sector: (float(sector.get("coverage_pct", 0)), int(sector.get("id", 0))),
        )
        assignments: Dict[str, int] = {}
        for drone_id, sector in zip(drones, ordered):
            assignments[drone_id] = int(sector["id"])
        return assignments
    
    def commander_node(state: DroneTickState) -> Dict[str, Any]:
        """Strategic node: looks at mission state and assigns drones to sectors."""
        ms = state["global_mission_state"]
        drones = state["available_drones"]
        telemetry = state["active_drone_telemetry"]
        
        if not drones:
            return {"current_commander_order": {}}
            
        sys_msg = SystemMessage(content=(
            "You are the Mission Commander. Your job is to assign drones to sectors.\n"
            "Review the sector coverage_pct. Focus on sectors less than 100% scanned.\n"
            "Do NOT overlap multiple drones in the same sector.\n"
            "Return a dict mapping drone_id to sector_id (integer) for all available drones.\n"
            "Respond ONLY with a valid JSON object matching this schema: {\"assignments\": {\"d_0\": 1, \"d_1\": 2}}"
        ))
        
        user_msg = HumanMessage(content=(
            f"Available Drones: {drones}\n"
            f"Drone Telemetry: {telemetry}\n"
            f"Mission State (Sectors): {ms.get('sectors')}\n"
        ))
        
        try:
            response = commander_llm.invoke([sys_msg, user_msg])
            content = str(response.content)
            
            import json, re
            content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'```\s*', '', content)
            start = content.find('{')
            end = content.rfind('}')
            
            if start != -1 and end != -1 and end >= start:
                data = json.loads(content[start:end+1])
                assignments = data.get("assignments", {})
                if isinstance(assignments, dict) and assignments:
                    return {"current_commander_order": assignments}
                fallback = fallback_assignments(ms, drones)
                return {
                    "errors": ["Commander returned JSON without usable assignments; using deterministic fallback."],
                    "current_commander_order": fallback,
                }
            else:
                fallback = fallback_assignments(ms, drones)
                preview = content[:200].replace("\n", " ")
                return {
                    "errors": [f"No JSON object found in Commander response. Preview: {preview}"],
                    "current_commander_order": fallback,
                }
                
        except Exception as e:
            fallback = fallback_assignments(ms, drones)
            return {
                "errors": [f"Commander failed manual parse: {str(e)}. Using deterministic fallback."],
                "current_commander_order": fallback,
            }

    def map_drones(state: DroneTickState) -> List[Send]:
        sends = []
        for d_id in state["available_drones"]:
            sector_id = state["current_commander_order"].get(d_id)
            sends.append(Send("operator_node", {
                "drone_id": d_id,
                "global_mission_state": state["global_mission_state"],
                "drone_telemetry": state["active_drone_telemetry"].get(d_id, {}),
                "assigned_sector_id": sector_id,
            }))
        return sends

    def operator_node(state: OperatorState) -> Dict[str, Any]:
        """Tactical node for a single drone. Extracts command via manual JSON parse."""
        d_id = state["drone_id"]
        t = state["drone_telemetry"]
        assigned = state["assigned_sector_id"]
        ms = state["global_mission_state"]
        
        # 1. Extract the specific sector information
        sector_info = "Unknown (free roam)"
        if assigned is not None:
            for s in ms.get("sectors", []):
                if s["id"] == assigned:
                    origin = s["origin"]
                    size = s["size"]
                    sector_info = f"Sector {assigned}: x-range [{origin[0]} to {origin[0]+size[0]-1}], y-range [{origin[1]} to {origin[1]+size[1]-1}]"
                    break
        
        # Hard rule: Battery check first
        battery = t.get("battery", 0)
        at_base = t.get("at_base", False)

        if assigned is None:
            res = stage_command_fn(d_id, "wait", reason="No commander sector assignment available")
            return {
                "fallback_actions": [res],
                "errors": [f"{d_id} had no commander assignment; staged wait."],
            }
        
        # FIX: Only force charge if actually low or returning from a trip.
        if battery < 30 and not at_base:
            res = stage_command_fn(d_id, "recall_to_base", reason="Low battery fallback")
            return {"fallback_actions": [res]}
            
        if at_base and battery < 80:
            res = stage_command_fn(d_id, "charge_drone", reason="Replenishing battery at base")
            return {"fallback_actions": [res]}
            
        sys_msg = SystemMessage(content=(
            "You are an automated Drone Operator AI.\n"
            "Valid actions are: 'move_and_scan', 'recall_to_base', 'charge_drone', 'wait'.\n"
            "If assigning 'move_and_scan', you MUST provide 'x' and 'y' integer coordinates.\n"
            "You MUST output exactly ONE valid JSON object and absolutely NOTHING else. No markdown, no greetings, no explanations.\n"
            "Example of EXACT expected output:\n"
            "{\"action\": \"move_and_scan\", \"x\": 10, \"y\": 15, \"reason\": \"Scanning assigned sector\"}"
        ))
        
        obstacle_positions = ms.get("obstacle_positions", [])
        grid_bounds = ms.get("grid", [24, 16]) # [width, height]
        
        user_msg = HumanMessage(content=(
            f"Drone ID: {d_id}\n"
            f"Battery: {battery}%\n"
            f"Current Position: {t.get('pos')}\n"
            f"Assigned Sector Boundaries: {sector_info}\n"
            f"Grid Bounds: x=[0 to {grid_bounds[0]-1}], y=[0 to {grid_bounds[1]-1}]. You MUST stay within these bounds.\n"
            f"High-Building Obstacles to AVOID: {obstacle_positions}\n"
            "Output your JSON object now:"
        ))
        
        try:
            res = operator_llm.invoke([sys_msg, user_msg])
            content = str(res.content)
            
            import re
            # Clean up potential markdown blocks
            clean_content = re.sub(r'```json\s*', '', content, flags=re.IGNORECASE)
            clean_content = re.sub(r'```\s*', '', clean_content)
            
            start = clean_content.find('{')
            end = clean_content.rfind('}')
            
            if start != -1 and end != -1 and end >= start:
                cmd_data = json.loads(clean_content[start:end+1])
                action = cmd_data.get("action", "wait")
                x = cmd_data.get("x")
                y = cmd_data.get("y")
                reason = cmd_data.get("reason", "Operator LLM")
                
                # Execute the staging validation locally
                out = stage_command_fn(d_id, action, x, y, reason)
                
                # FIX: Check if it actually staged successfully
                if out.get("staged", False):
                    return {"staged_commands": [out]}
                else:
                    # Staging failed (e.g. invalid bounds / obstacle)
                    fallback = stage_command_fn(d_id, "wait", reason=f"Validation failed: {out.get('reason')}")
                    return {"fallback_actions": [fallback], "errors": [f"{d_id} Validation Error: {out.get('reason')}"]}
            else:
                fallback = stage_command_fn(d_id, "wait", reason="Operator returned no valid JSON.")
                return {"fallback_actions": [fallback], "errors": [f"{d_id} No JSON structure. Raw response: {content}"]}
                
        except Exception as e:
            fallback = stage_command_fn(d_id, "wait", reason=f"Operator parsing error.")
            return {"fallback_actions": [fallback], "errors": [f"{d_id} Operator Error ({str(e)}). Raw response: {content}"]}

    builder = StateGraph(DroneTickState)
    builder.add_node("commander_node", commander_node)
    builder.add_node("operator_node", operator_node)
    builder.add_edge(START, "commander_node")
    builder.add_conditional_edges("commander_node", map_drones, ["operator_node"])
    builder.add_edge("operator_node", END)
    
    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
#  Mesa AI Controller
# ═══════════════════════════════════════════════════════════════════════════

class LangGraphOpenRouterAiController(base.GeminiAiController):
    """LangGraph backed controller executing Commander -> Operator in parallel."""

    def __init__(
        self,
        tools: base.InUiToolServer,
        model_name: str = "qwen/qwen3-14b", # Default heavy commander
        operator_model_name: str = "qwen/qwen-2.5-7b-instruct", # Default fast operator
        action_delay_s: float = 0.0,
    ):
        super().__init__(tools, model_name, action_delay_s, 100)
        self.tools = tools
        self.model_name = model_name
        self.operator_model_name = operator_model_name
        self._tick = 0
        self._mcp_langchain_tools: Dict[str, Any] = {}
        self._pending_stage_commands: List[Dict[str, Any]] = []
        self._load_langchain_mcp_tools()
        
        # Instantiate dual clients
        self._commander_client = self._maybe_client(self.model_name)
        self._operator_client = self._maybe_client(self.operator_model_name)
        
        if self._commander_client and self._operator_client:
            self.graph = build_drone_graph(self._commander_client, self._operator_client, self._stage_command_via_mcp)
        else:
            self.graph = None

    def _load_langchain_mcp_tools(self) -> None:
        if load_mcp_tools is None:
            return

        def _load_tools_sync() -> List[Any]:
            try:
                asyncio.get_running_loop()
                has_running_loop = True
            except RuntimeError:
                has_running_loop = False

            if not has_running_loop:
                return asyncio.run(build_mcp_graph_tools())

            holder: Dict[str, Any] = {"tools": []}

            def _runner() -> None:
                try:
                    holder["tools"] = asyncio.run(build_mcp_graph_tools())
                except Exception as exc:
                    holder["error"] = exc

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()

            if "error" in holder:
                raise holder["error"]
            return holder.get("tools", [])

        try:
            tools = _load_tools_sync()
        except Exception:
            return

        self._mcp_langchain_tools = {
            str(getattr(tool_obj, "name", "")): tool_obj
            for tool_obj in tools
            if getattr(tool_obj, "name", None)
        }

    @staticmethod
    def _coerce_dict(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                out = json.loads(raw)
                if isinstance(out, dict):
                    return out
            except Exception:
                return {"ok": True, "raw": raw}
        if isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], dict):
                return raw[0]
            return {"ok": True, "raw": str(raw)}
        content = getattr(raw, "content", None)
        if content is not None:
            return LangGraphOpenRouterAiController._coerce_dict(content)
        return {"ok": True, "raw": str(raw)}

    @staticmethod
    def _text_content(result: Any) -> str:
        if not getattr(result, "content", None):
            return ""
        parts: List[str] = []
        for c in result.content:
            if getattr(c, "type", None) == "text":
                parts.append(getattr(c, "text", ""))
        return "\n".join(parts).strip()

    @classmethod
    def _structured(cls, result: Any) -> Dict[str, Any]:
        sc = getattr(result, "structuredContent", None)
        if isinstance(sc, dict):
            if "result" in sc and isinstance(sc["result"], dict):
                return sc["result"]
            return sc
        txt = cls._text_content(result)
        if txt:
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    if "result" in obj and isinstance(obj["result"], dict):
                        return obj["result"]
                    return obj
            except Exception:
                pass
        return {}

    async def _mcp_call_async(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        arguments = arguments or {}
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name=name, arguments=arguments)
                if getattr(result, "isError", False):
                    detail = self._text_content(result)
                    return {"ok": False, "reason": detail or "MCP tool error"}
                structured = self._structured(result)
                if isinstance(structured, dict):
                    return structured
                return {"ok": True}

    async def _stage_and_flush_async(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                for cmd in commands:
                    await session.call_tool(name="stage_command", arguments=cmd)

                flush_result = await session.call_tool(name="flush_commands", arguments={})
                structured = self._structured(flush_result)
                if isinstance(structured, dict):
                    return structured.get("results", [])
                return []

    def _stage_and_flush(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _call_sync() -> List[Dict[str, Any]]:
            try:
                asyncio.get_running_loop()
                has_running_loop = True
            except RuntimeError:
                has_running_loop = False

            if not has_running_loop:
                return asyncio.run(self._stage_and_flush_async(commands))

            holder: Dict[str, Any] = {"result": []}

            def _runner() -> None:
                try:
                    holder["result"] = asyncio.run(self._stage_and_flush_async(commands))
                except Exception as exc:
                    holder["result"] = [{"error": str(exc), "command": {"action": "flush_commands", "drone_id": ""}}]

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            return holder.get("result", [])

        return _call_sync()

    def _mcp_call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        adapter_tool = self._mcp_langchain_tools.get(name)
        if adapter_tool is not None:
            try:
                raw = adapter_tool.invoke(arguments or {})
                return self._coerce_dict(raw)
            except Exception:
                pass

        def _call_sync() -> Dict[str, Any]:
            try:
                asyncio.get_running_loop()
                has_running_loop = True
            except RuntimeError:
                has_running_loop = False

            if not has_running_loop:
                return asyncio.run(self._mcp_call_async(name, arguments))

            holder: Dict[str, Any] = {"result": {"ok": False, "reason": "MCP call thread did not return"}}

            def _runner() -> None:
                try:
                    holder["result"] = asyncio.run(self._mcp_call_async(name, arguments))
                except Exception as exc:
                    holder["result"] = {"ok": False, "reason": str(exc)}

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()
            return holder["result"]

        try:
            return _call_sync()
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    def _stage_command_via_mcp(
        self,
        drone_id: str,
        action: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        # Route directly through the in-process InUiToolServer so the Mesa UI
        # model's DroneAgents actually move (not a detached MCP subprocess).
        return self.tools.stage_drone_command(drone_id, action, x, y, reason)

    def _maybe_client(self, model_string: Optional[str] = None):
        if model_string is None:
            # Safely return None when called by base GeminiAiController.__init__
            return None
            
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
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
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "MesaDroneRescue")
            }
        )

    def think_and_act(self) -> None:
        self._tick += 1
        print("\n" + "═" * 72)
        print(f"LANGGRAPH PARALLEL CONTROLLER — tick {self._tick} ({self.model_name})")
        print("═" * 72)

        if not self.graph:
            print("[WARN] LangGraph or ChatOpenAI not available. Is OPENROUTER_API_KEY set?")
            return

        # 1. Hydrate state
        ms = self.tools.get_mission_state()
        drones_info = self.tools.discover_drones()["drones"]
        telemetry = {d["id"]: self.tools.get_drone_status(d["id"]) for d in drones_info}
        available = [d["id"] for d in drones_info if not d.get("disabled")]
        
        survivors_total_val = ms.get("survivors_total", 0)
        try:
            survivors_total = int(survivors_total_val) if isinstance(survivors_total_val, (int, float)) or (isinstance(survivors_total_val, str) and survivors_total_val.isdigit()) else 0
        except ValueError:
            survivors_total = 0
        survivors_found = int(ms.get("survivors_found", 0))
        sectors = ms.get("sectors", [])
        all_sectors_scanned = all(s.get("coverage_pct", 0) >= 99.0 for s in sectors)

        # Win condition check
        if all_sectors_scanned:
             print("[INFO] Mission Complete. Recalling all remaining drones.")
             for d_id in available:
                 self.tools.recall_to_base(d_id)
                 self._pause()
             return

        initial_state: DroneTickState = {
            "tick": self._tick,
            "global_mission_state": ms,
            "active_drone_telemetry": telemetry,
            "available_drones": available,
            "current_commander_order": {},
            "staged_commands": [],
            "fallback_actions": [],
            "errors": [],
        }

        # 2. Invoke Graph
        try:
            print("[LANGGRAPH] Starting Parallel Execution...")
            start_time = time.time()
            # We use invoke on the graph which handles the Map-Reduce internally
            final_state = self.graph.invoke(initial_state)
            elapsed = time.time() - start_time
            print(f"[LANGGRAPH] Execution Complete in {elapsed:.2f}s")
            
            # Print Commander assignments
            assignments = final_state.get("current_commander_order", {})
            if assignments:
                print(f"  [COMMANDER] Assignments: {assignments}")
                
            errors = final_state.get("errors", [])
            for err in errors:
                print(f"  [ERROR] {err}")
                
        except Exception as e:
            print(f"[ERROR] LangGraph execution failed: {e}")
            return

        # 3. Flush Staged Commands (in-process — directly updates Mesa UI model)
        print("\n[FLUSHING STAGED COMMANDS]")
        results = self.tools.flush_staged_commands()
        if not results:
            print("  [INFO] No staged commands were flushed.")
        
        summary_calls = []
        for r in results:
            cmd = r.get("command", {})
            res = r.get("result", {})
            err = r.get("error")
            action = cmd.get("action", "")
            d_id = cmd.get("drone_id", "")
            
            if err:
                print(f"  [FAIL] {d_id} {action}: {err}")
            else:
                st = str(res)[:200]
                print(f"  [OK] {d_id} {action} -> {st}")
                
            # Convert to old format for rolling summary
            summary_calls.append({
                "tool_name": action,
                "arguments": cmd,
                "reasoning": cmd.get("reason", "")
            })
            self._pause()

        self.tools.update_rolling_summary(self._tick, summary_calls)
