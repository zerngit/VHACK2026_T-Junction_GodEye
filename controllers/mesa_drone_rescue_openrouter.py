"""
OpenRouter-hosted Qwen variant for the Mesa drone rescue UI.

This file leaves the original mesa_drone_rescue_mcp.py untouched and provides
an alternate launcher that routes the in-UI AI controller through OpenRouter.

Usage:
    set OPENROUTER_API_KEY=...
    python mesa_drone_rescue_openrouter.py
"""

from __future__ import annotations

import os
import sys
# Ensure parent directory (V-Hack) is in sys.path so 'core' and others can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import os
import socket
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import core.mesa_drone_rescue_mcp as base

try:
    from controllers.langgraph_drone_controller import LangGraphOpenRouterAiController
    LANGGRAPH_IMPORT_ERROR: Optional[str] = None
except Exception as exc:
    LangGraphOpenRouterAiController = None  # type: ignore
    LANGGRAPH_IMPORT_ERROR = str(exc)

try:
    from controllers.langgraph_mesa_trace_bridge import LangGraphMesaTraceController
    LANGGRAPH_TRACE_IMPORT_ERROR: Optional[str] = None
except Exception as exc:
    LangGraphMesaTraceController = None  # type: ignore
    LANGGRAPH_TRACE_IMPORT_ERROR = str(exc)

try:
    from openrouter import OpenRouter  # type: ignore
except Exception:
    OpenRouter = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    import tornado.web  # type: ignore
except Exception:
    tornado = None  # type: ignore


class OpenRouterAiController(base.OllamaAiController):
    """OpenRouter-backed controller using the official Python SDK."""

    def __init__(
        self,
        tools: base.InUiToolServer,
        model_name: str = "qwen/qwen3-14b",
        action_delay_s: float = 0.0,
        max_calls_per_tick: int = 8,
    ):
        super().__init__(tools, model_name, action_delay_s, max_calls_per_tick)
        self._client_mode = "none"
        self._client = self._maybe_client()

    def _maybe_client(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return None

        headers = {
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            "X-OpenRouter-Title": os.environ.get(
                "OPENROUTER_APP_TITLE",
                "V-Hack Mesa OpenRouter",
            ),
        }

        # Preferred: official OpenRouter Python SDK
        if OpenRouter is not None:
            try:
                client = OpenRouter(api_key=api_key, default_headers=headers)
                self._client_mode = "openrouter-sdk"
                return client
            except Exception:
                pass

        # Fallback: OpenAI-compatible client against OpenRouter base URL
        if OpenAI is None:
            return None
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers=headers,
            )
            self._client_mode = "openai-compatible"
            return client
        except Exception:
            return None

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    parts.append(str(getattr(item, "text", "")))
            return "\n".join(p for p in parts if p).strip()
        return str(content or "").strip()

    def think_and_act(self) -> None:
        self._tick += 1
        base.log_to_file(
            f"**Tick {self._tick} (OpenRouter)**: Starting reasoning cycle. Model: `{self.model_name}`"
        )

        print("\n" + "═" * 72)
        print(f"OPENROUTER AI CONTROLLER — tick {self._tick} ({self.model_name})")
        print("═" * 72)

        if self._client is None:
            if not self._warned_unavailable:
                missing = []
                if OpenRouter is None and OpenAI is None:
                    missing.append("no OpenRouter/OpenAI client package installed")
                if not os.environ.get("OPENROUTER_API_KEY"):
                    missing.append("OPENROUTER_API_KEY not set")
                msg = ", ".join(missing) if missing else "unknown reason"
                print(f"[WARN] OpenRouter unavailable ({msg}).")
                self._warned_unavailable = True
            return

        ms = self.tools.get_mission_state()
        drones = self.tools.discover_drones()["drones"]
        summarize_tick = self._should_summarize_tick()
        prompt = self._build_prompt(ms, drones, summarize_tick=summarize_tick)
        prompt = "\n".join(line.rstrip() for line in prompt.splitlines()).strip()

        if "qwen3" in self.model_name.lower():
            prompt = "/no_think\n" + prompt

        calls: List[Dict[str, Any]] = []
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                print(f"\n[LLM PROMPT SENT] (attempt {attempt + 1})")
                if prompt:
                    self._print_wrapped("  ", prompt)
                else:
                    print("  [empty]")

                if self._client_mode == "openrouter-sdk":
                    response = self._client.chat.send(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = self._extract_message_text(response.choices[0].message)
                else:
                    response = self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = self._extract_message_text(response.choices[0].message)

                print(f"\n[LLM RAW RESPONSE] (attempt {attempt + 1})")
                if text:
                    self._print_wrapped("  ", text)
                else:
                    print("  [empty]")

                calls = self._parse_tool_calls(text)
                if calls:
                    break

                print(
                    f"[DEBUG] Attempt {attempt + 1}: OpenRouter returned raw text: '{text[:300]}'"
                )
            except Exception as exc:
                print(f"[WARN] OpenRouter call failed (attempt {attempt + 1}): {exc}.")
                base.log_to_file(f"⚠️ **OpenRouter API Error**: {exc}")

        if not calls:
            print("[WARN] No valid calls generated this tick.")
            return

        for call in calls:
            reasoning = call.get("reasoning") or ""
            if reasoning:
                print("\n[CHAIN OF THOUGHT]")
                self._print_wrapped("  ", reasoning)
                base.log_to_file(f"**Reasoning**: {reasoning}")

            name = str(call["tool_name"])
            args = dict(call["arguments"])
            print(f"\n[MCP TOOL CALL] {name}({json.dumps(args)})")
            result = self._exec(name, args)
            preview = str(json.dumps(result, default=str))[:260]
            print(f"  [OK] {preview}" if preview else "  [OK]")
            base.log_to_file(f"✅ **Execution (`{name}`)**: Success. {preview}")
            self._recent_log += f"\nTOOL {name}({json.dumps(args)}) -> {preview or '[ok]'}"
            self._pause()

        if summarize_tick:
            self._recent_log = self._build_status_summary(ms, drones)

        self.tools.update_rolling_summary(self._tick, calls)


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO EXPORT UI ELEMENT
# ═══════════════════════════════════════════════════════════════════════════

class VideoExportElement(base.TextElement):
    """Mesa TextElement that shows recording status and Export/Clear buttons."""

    def render(self, model):
        fc = model._frame_capture
        frame_count = len(fc.frames)
        is_capturing = fc.capturing

        status_color = "#00cc44" if is_capturing else "#888888"
        status_text = f"🔴 Recording: {frame_count} frames" if is_capturing else f"⏸ Stopped: {frame_count} frames"

        return f"""
        <div style="font-family:Arial;padding:8px;margin-bottom:10px;
                    background:#1a1a2e;border-radius:8px;border:1px solid #333;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="color:{status_color};font-weight:bold;font-size:14px;">
              {status_text}
            </span>
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button onclick="fetch('/export_video', {{method:'POST'}})
                            .then(r=>r.json())
                            .then(d=>{{alert(d.message||d.error)}})
                            .catch(e=>alert('Export failed: '+e));"
                    style="padding:6px 14px;background:#0078d4;color:white;border:none;
                           border-radius:4px;cursor:pointer;font-weight:bold;">
              🎬 Export MP4
            </button>
            <button onclick="fetch('/clear_frames', {{method:'POST'}})
                            .then(r=>r.json())
                            .then(d=>alert(d.message||'Cleared'))
                            .catch(e=>alert('Error: '+e));"
                    style="padding:6px 14px;background:#cc4400;color:white;border:none;
                           border-radius:4px;cursor:pointer;">
              🗑️ Clear Frames
            </button>
            <button onclick="fetch('/toggle_capture', {{method:'POST'}})
                            .then(r=>r.json())
                            .then(d=>alert(d.message))
                            .catch(e=>alert('Error: '+e));"
                    style="padding:6px 14px;background:#6b21a8;color:white;border:none;
                           border-radius:4px;cursor:pointer;">
              ⏯ Toggle Capture
            </button>
          </div>
        </div>
        """


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL — extends base with OpenRouter + frame capture
# ═══════════════════════════════════════════════════════════════════════════

class OpenRouterDroneRescueModel(base.DroneRescueModel):
    """Base model plus a hosted OpenRouter controller option."""

    def __init__(
        self,
        width: int = 24,
        height: int = 16,
        num_drones: int = 4,
        num_survivors: int = 12,
        scenario: str = "A: Palu city",
        simulate_ai: bool = True,
        ai_delay_s: float = 0.15,
        use_gemini_ai: bool = False,
        gemini_model: str = "gemini-2.5-flash",
        use_ollama_ai: bool = False,
        ollama_model: str = "llama3.1",
        use_crew_ai: bool = False,
        crew_ai_model: str = "ollama/llama3.1",
        use_openrouter_ai: bool = False,
        openrouter_model: str = "qwen/qwen3-14b",
        use_langgraph_ai: bool = True,
        langgraph_model: str = "qwen/qwen3-14b",
        use_langgraph_trace_ai: bool = False,
        langgraph_trace_operator_model: str = "qwen/qwen-2.5-7b-instruct",
    ):
        super().__init__(
            width=width,
            height=height,
            num_drones=num_drones,
            num_survivors=num_survivors,
            scenario=scenario,
            simulate_ai=simulate_ai,
            ai_delay_s=ai_delay_s,
            use_gemini_ai=use_gemini_ai,
            gemini_model=gemini_model,
            use_ollama_ai=use_ollama_ai,
            ollama_model=ollama_model,
            use_crew_ai=use_crew_ai,
            crew_ai_model=crew_ai_model,
        )
        self.use_openrouter_ai = bool(use_openrouter_ai)
        self.openrouter_model = str(openrouter_model or "qwen/qwen3-14b")
        self.use_langgraph_ai = bool(use_langgraph_ai)
        self.langgraph_model = str(langgraph_model or "qwen/qwen3-14b")
        self.use_langgraph_trace_ai = bool(use_langgraph_trace_ai)
        self.langgraph_trace_operator_model = str(
            langgraph_trace_operator_model or "qwen/qwen-2.5-7b-instruct"
        )
        self._openrouter_ai: Optional[OpenRouterAiController] = None
        self._langgraph_ai: Optional[Any] = None
        self._langgraph_trace_ai: Optional[Any] = None

        if self.simulate_ai:
            if self.use_langgraph_trace_ai and LangGraphMesaTraceController is not None:
                self._langgraph_trace_ai = LangGraphMesaTraceController(
                    self._tools,
                    model_name=self.langgraph_model,
                    operator_model_name=self.langgraph_trace_operator_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._langgraph_ai = None
                self._openrouter_ai = None
                self._crew_ai = None
                self._gemini_ai = None
                self._ollama_ai = None
                self._ai = None
                print(
                    "[AI MODE] LangGraph TRACE bridge enabled "
                    f"(commander={self.langgraph_model}, operator={self.langgraph_trace_operator_model})."
                )
            elif self.use_langgraph_trace_ai and LangGraphMesaTraceController is None:
                reason = LANGGRAPH_TRACE_IMPORT_ERROR or "Unknown import error"
                print(f"[WARN] LangGraph trace bridge requested but unavailable: {reason}")
                print("[WARN] Falling back to regular LangGraph/OpenRouter modes.")
                if self.use_langgraph_ai and LangGraphOpenRouterAiController is not None:
                    self._langgraph_ai = LangGraphOpenRouterAiController(
                        self._tools,
                        model_name=self.langgraph_model,
                        action_delay_s=self.ai_delay_s,
                    )
                    self._openrouter_ai = None
                    self._crew_ai = None
                    self._gemini_ai = None
                    self._ollama_ai = None
                    self._ai = None
            elif self.use_langgraph_ai and LangGraphOpenRouterAiController is not None:
                self._langgraph_ai = LangGraphOpenRouterAiController(
                    self._tools,
                    model_name=self.langgraph_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._langgraph_trace_ai = None
                self._openrouter_ai = None
                self._crew_ai = None
                self._gemini_ai = None
                self._ollama_ai = None
                self._ai = None
                print(f"[AI MODE] LangGraph controller enabled (commander={self.langgraph_model}).")
            elif self.use_langgraph_ai and LangGraphOpenRouterAiController is None:
                reason = LANGGRAPH_IMPORT_ERROR or "Unknown import error"
                print(f"[WARN] LangGraph requested but unavailable: {reason}")
                print("[WARN] Falling back to OpenRouterAiController when enabled.")
                if self.use_openrouter_ai:
                    self._openrouter_ai = OpenRouterAiController(
                        self._tools,
                        model_name=self.openrouter_model,
                        action_delay_s=self.ai_delay_s,
                    )
                    self._crew_ai = None
                    self._gemini_ai = None
                    self._ollama_ai = None
                    self._ai = None
                    self._langgraph_trace_ai = None
                    print(f"[AI MODE] OpenRouter controller enabled (model={self.openrouter_model}).")
            elif self.use_openrouter_ai:
                self._openrouter_ai = OpenRouterAiController(
                    self._tools,
                    model_name=self.openrouter_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._crew_ai = None
                self._gemini_ai = None
                self._ollama_ai = None
                self._ai = None
                self._langgraph_trace_ai = None
                print(f"[AI MODE] OpenRouter controller enabled (model={self.openrouter_model}).")

        # Auto-enable frame capture when simulation starts
        self._frame_capture.capturing = True

    def step(self):
        self._tools.reset_tick_state()
        langgraph_trace_ai = getattr(self, "_langgraph_trace_ai", None)
        langgraph_ai = getattr(self, "_langgraph_ai", None)
        openrouter_ai = getattr(self, "_openrouter_ai", None)
        crew_ai = self._crew_ai
        gemini_ai = self._gemini_ai
        ollama_ai = self._ollama_ai
        ai = self._ai

        if langgraph_trace_ai is not None:
            langgraph_trace_ai.think_and_act()
        elif langgraph_ai is not None:
            langgraph_ai.think_and_act()
        elif openrouter_ai is not None:
            openrouter_ai.think_and_act()
        elif crew_ai is not None:
            crew_ai.think_and_act()
        elif gemini_ai is not None:
            gemini_ai.think_and_act()
        elif ollama_ai is not None:
            ollama_ai.think_and_act()
        elif ai is not None:
            ai.think_and_act()

        self.datacollector.collect(self)
        self.schedule.step()
        # ── Capture frame for video export (after all agents have moved) ──
        self._frame_capture.capture_frame()


# ═══════════════════════════════════════════════════════════════════════════
#  TORNADO HANDLERS for video export API
# ═══════════════════════════════════════════════════════════════════════════

class ExportVideoHandler(tornado.web.RequestHandler):  # type: ignore[misc]
    """POST /export_video — exports captured frames to MP4."""

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


# ═══════════════════════════════════════════════════════════════════════════
#  SERVER SETUP
# ═══════════════════════════════════════════════════════════════════════════

grid = base.CanvasGrid(base.portrayal, 24, 16, 980, 650)
chart = base.ChartModule(
    [
        {"Label": "SurvivorsFound", "Color": "Green"},
        {"Label": "SectorsDone", "Color": "Blue"},
        {"Label": "AvgBattery", "Color": "Orange"},
    ],
    data_collector_name="datacollector",
)
legend = base.Legend()
movement_board = base.MovementDashboard()
video_export = VideoExportElement()

server = base.ModularServer(
    OpenRouterDroneRescueModel,
    [video_export, legend, movement_board, grid, chart],
    "Drone Fleet Search & Rescue — OpenRouter Qwen 3 14B",
    {
        "width": 24,
        "height": 16,
        "scenario": base.Choice(
            "Scenario",
            value="A: Palu city",
            choices=list(base.SCENARIOS.keys()),
            description="Pick a scenario. A/B/C/E = original layouts. D = city with buildings (applies on Reset).",
        ),
        "num_drones": base.Slider("Drones", value=4, min_value=3, max_value=5, step=1),
        "num_survivors": base.Slider("Survivors", value=12, min_value=5, max_value=20, step=1),
        "simulate_ai": base.Checkbox("Simulate (AI drives drones)", value=True),
        "use_gemini_ai": base.Checkbox("Use Gemini (real LLM agent)", value=False),
        "gemini_model": base.Choice(
            "Gemini model",
            value="gemini-2.5-flash",
            choices=["gemini-2.5-flash", "gemini-2.5-pro"],
            description="Requires GEMINI_API_KEY in environment (.env supported).",
        ),
        "use_ollama_ai": base.Checkbox("Use Ollama (local Edge AI)", value=False),
        "ollama_model": base.Choice(
            "Ollama model",
            value="llama3.1",
            choices=["llama3.1", "qwen2.5:3b", "qwen2.5:14b", "qwen3:14b"],
            description="Requires Ollama to be running locally.",
        ),
        "use_crew_ai": base.Checkbox("Use CrewAI (hierarchical agents)", value=False),
        "crew_ai_model": base.Choice(
            "CrewAI model",
            value="ollama/hierarchical (14B/3B)",
            choices=[
                "ollama/hierarchical (14B/3B)",
                "ollama/qwen2.5:3b",
                "ollama/qwen2.5:14b",
                "ollama/qwen3:14b",
                "openai/gpt-4o-mini",
            ],
            description="Model for CrewAI agents. 'ollama/' prefix = local, 'openai/' = cloud.",
        ),
        "use_openrouter_ai": base.Checkbox("Use OpenRouter (hosted API)", value=True),
        "openrouter_model": base.Choice(
            "OpenRouter model",
            value="qwen/qwen3-14b",
            choices=["qwen/qwen3-14b"],
            description="Requires OPENROUTER_API_KEY. Uses the official OpenRouter Python SDK.",
        ),
        "use_langgraph_ai": base.Checkbox("Use LangGraph AI", value=True),
        "langgraph_model": base.Choice(
            "LangGraph commander model",
            value="qwen/qwen3-14b",
            choices=["qwen/qwen3-14b"],
            description="Requires langgraph stack + OPENROUTER_API_KEY. Uses Commander->Operator routing.",
        ),
        "use_langgraph_trace_ai": base.Checkbox("Use LangGraph TRACE bridge", value=False),
        "langgraph_trace_operator_model": base.Choice(
            "LangGraph TRACE operator model",
            value="qwen/qwen-2.5-7b-instruct",
            choices=["qwen/qwen-2.5-7b-instruct", "qwen/qwen3-14b"],
            description="Used only in TRACE bridge mode. Full tick logs are written to langgraph_tick_trace_log.jsonl.",
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
    server.port = _pick_free_port([8534, 8535, 8536, 8537])

    # ── Register custom Tornado handlers for video export ──
    # ModularServer inherits from tornado.web.Application
    extra_handlers = [
        (r"/export_video", ExportVideoHandler, {"server": server}),
        (r"/clear_frames", ClearFramesHandler, {"server": server}),
        (r"/toggle_capture", ToggleCaptureHandler, {"server": server}),
    ]
    # Add handlers to the Tornado application
    server.add_handlers(r".*", extra_handlers)

    print(f"Launching OpenRouter drone fleet server at http://127.0.0.1:{server.port}")
    print("  🎬 Video export endpoints: /export_video, /clear_frames, /toggle_capture")

    server.launch()