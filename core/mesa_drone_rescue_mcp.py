"""
Drone Fleet Search & Rescue — Mesa Simulation Core
=================================================

This module contains the **simulation-only** components for a drone fleet
search-and-rescue scenario:

- Mesa agents (`DroneAgent`, `SurvivorAgent`, `ChargingStationAgent`, etc.)
- The Mesa model (`DroneRescueModel`)
- Visual portrayal + optional Mesa UI server (for local visualization)

The **official MCP (Model Context Protocol)** server and the orchestration
agent live in separate scripts (see `mcp_drone_server.py` and
`mcp_drone_orchestrator.py`). This keeps the simulation core clean and allows
all Agent↔Drone communication to happen via real MCP tool calls.
"""

import sys
import io

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import random
from dataclasses import dataclass
import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple, cast

from mesa import Agent, Model  # type: ignore
from mesa.space import MultiGrid  # type: ignore
from mesa.time import RandomActivation  # type: ignore
from mesa.datacollection import DataCollector  # type: ignore

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement  # type: ignore
from mesa.visualization.ModularVisualization import ModularServer  # type: ignore
from mesa.visualization.UserParam import Slider, Checkbox, Choice  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

try:
    from google import genai  # type: ignore
except Exception:  # optional dependency at runtime
    genai = None

try:
    import ollama  # type: ignore
except Exception:
    ollama = None

try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew, Process  # type: ignore
    from crewai.tools import tool as crewai_tool  # type: ignore
except Exception:
    CrewAgent = None  # type: ignore
    CrewTask = None   # type: ignore
    Crew = None       # type: ignore
    Process = None    # type: ignore
    crewai_tool = None  # type: ignore

try:
    from controllers.langgraph_drone_controller import LangGraphOpenRouterAiController  # type: ignore
except Exception:
    LangGraphOpenRouterAiController = None  # type: ignore

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

try:
    import imageio.v3 as iio  # type: ignore
except Exception:
    iio = None  # type: ignore

# Load environment variables from .env when available (e.g., GEMINI_API_KEY)
if load_dotenv is not None:
    try:
        load_dotenv()
    except Exception:
        pass

def log_to_file(msg: str) -> None:
    try:
        with open("Varsity_Hackathon_Mission_Log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS — sectors, colours, bases
# ═══════════════════════════════════════════════════════════════════════════

SECTOR_DEFS: Dict[int, Dict[str, Any]] = {
    1: {"name": "Sector 1 (NW)", "origin": (0, 8),  "size": (8, 8)},
    2: {"name": "Sector 2 (N)",  "origin": (8, 8),  "size": (8, 8)},
    3: {"name": "Sector 3 (NE)", "origin": (16, 8), "size": (8, 8)},
    4: {"name": "Sector 4 (SW)", "origin": (0, 0),  "size": (8, 8)},
    5: {"name": "Sector 5 (S)",  "origin": (8, 0),  "size": (8, 8)},
    6: {"name": "Sector 6 (SE)", "origin": (16, 0), "size": (8, 8)},
}

SECTOR_COLORS = {
    1: ("#e8f0fe", "#b3cde8"),
    2: ("#e6f4ea", "#a5d6b7"),
    3: ("#fef7e0", "#f5e0a0"),
    4: ("#fce8e6", "#f2b8b5"),
    5: ("#f3e8fd", "#d4b5f0"),
    6: ("#fff3e0", "#f5cfa0"),
}

BASE_POSITIONS = [(0, 0), (23, 0), (0, 15), (23, 15)]

BATTERY_COST_MOVE = 1       # per cell moved
BATTERY_COST_SCAN = 5       # per thermal_scan call
BATTERY_CHARGE_RATE = 25    # per charge_drone call
BATTERY_CRITICAL = 20       # recall threshold
BATTERY_FULL = 90           # considered fully charged
SCAN_RADIUS = 2


def _sector_waypoints(origin: Tuple[int, int], size: Tuple[int, int]):
    sx, sy = origin
    return [(sx + 2, sy + 2), (sx + 5, sy + 2),
            (sx + 2, sy + 5), (sx + 5, sy + 5)]


def pos_to_sector_id(x: int, y: int):
    for sid, s in SECTOR_DEFS.items():
        origin: Tuple[int, int] = s["origin"]
        ox, oy = origin
        size: Tuple[int, int] = s["size"]
        w, h = size
        if ox <= x < ox + w and oy <= y < oy + h:
            return sid
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIOS (3 presets)
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "A: Palu city": {
        "seed": 1337,
        "has_buildings": True,
        "all_low_buildings": True,
    },
    "B: Two hotspots": {
        "seed": 2026,
        "survivor_positions": [
            (4, 12), (5, 12), (4, 13), (6, 11), (3, 11), (6, 13),
            (18, 3), (19, 3), (18, 4), (20, 2), (17, 2), (20, 4),
        ],
    },
    "C: Perimeter scattered": {
        "seed": 7,
        "survivor_positions": [
            (1, 1), (22, 1), (1, 14), (22, 14),
            (6, 1), (17, 1), (6, 14), (17, 14),
            (1, 6), (22, 6), (1, 9), (22, 9),
        ],
    },
    "D: City with high buildings": {
        "seed": 42,
        "has_buildings": True,
        "survivor_positions": [],  # survivors placed randomly inside buildings
    },
    
    "E: Center quake (clustered)": {
        "seed": 1337,
        "survivor_positions": [
            (11, 7), (12, 7), (13, 7),
            (11, 8), (12, 8), (13, 8),
            (10, 6), (14, 6), (10, 9), (14, 9),
            (9, 8), (15, 8),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  "IN-UI MCP" tools + simple reasoning agent
#  (This is what makes drones move in the UI.)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════════
#  VOTING SIMULATOR — Mock drone voting on idle drone task
# ═══════════════════════════════════════════════════════════════════════════

class VotingSimulator:
    """
    Orchestrates a mock voting round where drones vote on what an idle/failed drone should do.
    Entirely deterministic — no LLM involved.
    """

    def __init__(self, model: "DroneRescueModel"):
        self.model = model
        self.state: str = "IDLE"  # IDLE | VOTING | VOTING_COMPLETE | EXECUTING
        self.idle_drone_id: Optional[str] = None
        self.all_votes: Dict[str, Dict[str, Any]] = {}
        self.vote_tally: Dict[int, int] = {}
        self.winning_sector: Optional[int] = None
        self.winning_action: Optional[Dict[str, Any]] = None

    def analyze_sector_coverage(self) -> Dict[int, float]:
        """Return {sector_id: coverage_pct} for all sectors."""
        coverage: Dict[int, float] = {}
        for sid, sdef in SECTOR_DEFS.items():
            origin: Tuple[int, int] = cast(Tuple[int, int], sdef["origin"])
            ox, oy = origin
            size: Tuple[int, int] = cast(Tuple[int, int], sdef["size"])
            w, h = size
            total = w * h
            scanned = sum(
                1
                for xi in range(ox, ox + w)
                for yi in range(oy, oy + h)
                if self.model.tile_map.get((xi, yi)) and self.model.tile_map[(xi, yi)].scanned
            )
            pct = round((scanned / max(1, total)) * 100, 1)
            coverage[sid] = pct
        return coverage

    def get_drone_vote(self, voter_drone_id: str, idle_drone_id: str) -> Dict[str, Any]:
        """Each drone votes for the least-covered sector."""
        coverage = self.analyze_sector_coverage()
        best_sector = min(coverage.keys(), key=lambda s: (coverage[s], s))
        best_coverage_pct = coverage[best_sector]
        reasoning = f"Sector {best_sector} has lowest coverage ({best_coverage_pct}%)"
        
        return {
            "voter_id": voter_drone_id,
            "target_sector": best_sector,
            "reasoning": reasoning,
        }

    def tally_votes(self, votes_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate votes using majority rule."""
        self.vote_tally.clear()
        for voter_id, vote_info in votes_dict.items():
            sector = vote_info["target_sector"]
            self.vote_tally[sector] = self.vote_tally.get(sector, 0) + 1
        
        if not self.vote_tally:
            return {"error": "No votes recorded"}
        
        max_votes = max(self.vote_tally.values())
        candidates = [s for s, count in self.vote_tally.items() if count == max_votes]
        self.winning_sector = min(candidates)
        
        tally_str = " | ".join([f"Sector {s}: {c}" for s, c in sorted(self.vote_tally.items())])
        result_reasoning = f"Majority rule: Sector {self.winning_sector} wins ({max_votes} votes). [{tally_str}]"
        
        return {
            "winning_sector": self.winning_sector,
            "vote_tally": self.vote_tally.copy(),
            "result_reasoning": result_reasoning,
        }

    def execute_voting_round(self, idle_drone_id: str) -> Dict[str, Any]:
        """Execute full voting round and return results."""
        self.state = "VOTING"
        self.idle_drone_id = idle_drone_id
        self.all_votes.clear()
        
        all_drones = [a for a in self.model.schedule.agents if isinstance(a, DroneAgent)]
        voter_drones = [d for d in all_drones if d.unique_id != idle_drone_id]
        
        if not voter_drones:
            self.state = "IDLE"
            return {"error": f"No voter drones available"}
        
        for voter_drone in voter_drones:
            vote = self.get_drone_vote(voter_drone.unique_id, idle_drone_id)
            self.all_votes[voter_drone.unique_id] = vote
        
        tally_result = self.tally_votes(self.all_votes)
        if "error" in tally_result:
            self.state = "IDLE"
            return tally_result
        
        if self.winning_sector is None:
            self.state = "IDLE"
            return {"error": "No winning sector determined"}
        
        winning_sector_def = SECTOR_DEFS.get(self.winning_sector)
        if not winning_sector_def:
            self.state = "IDLE"
            return {"error": f"Invalid sector {self.winning_sector}"}
        
        origin: Tuple[int, int] = cast(Tuple[int, int], winning_sector_def["origin"])
        size: Tuple[int, int] = cast(Tuple[int, int], winning_sector_def["size"])
        waypoints = _sector_waypoints(origin, size)
        if not waypoints:
            self.state = "IDLE"
            return {"error": f"No waypoints in sector {self.winning_sector}"}
        
        target_wp = waypoints[0]
        
        self.winning_action = {
            "type": "move_and_scan",
            "drone_id": idle_drone_id,
            "x": target_wp[0],
            "y": target_wp[1],
            "sector": self.winning_sector,
            "reason": f"Voted action: scan Sector {self.winning_sector}",
        }
        
        self.state = "VOTING_COMPLETE"
        
        return {
            "voting_complete": True,
            "idle_drone_id": idle_drone_id,
            "voter_count": len(voter_drones),
            "all_votes": self.all_votes.copy(),
            "vote_tally": self.vote_tally.copy(),
            "winning_sector": self.winning_sector,
            "winning_action": self.winning_action.copy(),
            "result_reasoning": tally_result["result_reasoning"],
        }

    def get_voting_state(self) -> Dict[str, Any]:
        """Return current voting state for display."""
        votes_formatted = []
        for voter_id, vote_info in self.all_votes.items():
            votes_formatted.append({
                "voter_id": voter_id,
                "target_sector": vote_info["target_sector"],
                "reasoning": vote_info["reasoning"],
            })
        
        return {
            "state": self.state,
            "idle_drone_id": self.idle_drone_id,
            "votes": votes_formatted,
            "vote_tally": self.vote_tally.copy(),
            "winning_sector": self.winning_sector,
            "winning_action": self.winning_action.copy() if self.winning_action else {},
        }

    def reset_voting_state(self) -> None:
        """Reset for next round."""
        self.state = "IDLE"
        self.idle_drone_id = None
        self.all_votes.clear()
        self.vote_tally.clear()
        self.winning_sector = None
        self.winning_action = None


class InUiToolServer:
    """Tiny tool registry so the AI can act like MCP, but in-process."""

    def __init__(self, model: "DroneRescueModel"):
        self.model = model
        # ── Sector assignment & waypoint tracking (used by get_drone_orders) ──
        self._sector_assignments: Dict[str, List[int]] = {}
        self._waypoint_queue: Dict[str, List[Tuple[int, int]]] = {}
        self._resume_position: Dict[str, Tuple[int, int]] = {}
        # ── Movement arrow tracking (trail of arrows per drone) ──
        self._arrow_agents: Dict[str, List["MovementArrowAgent"]] = {}
        self._arrow_id_counter: int = 0
        # ── One-move-per-drone-per-tick limiter ──
        self._moved_this_tick: set = set()
        # ── Rolling Mission Summary (Phase 3: Spatial Memory) ──
        self._rolling_summary: str = "Mission just started. No actions taken yet."
        self._tick_history: List[str] = []  # Last N tick summaries
        # ── LangGraph Staging Buffer ──
        self._staged_commands: List[Dict[str, Any]] = []
        # ── Voting Simulator ──
        self.voting_simulator: VotingSimulator = VotingSimulator(model)

    def reset_tick_state(self) -> None:
        """Called at the start of each model step to reset per-tick limits."""
        self._moved_this_tick.clear()
        self._staged_commands.clear()

    def _place_movement_arrows(self, drone_id: str, path: List[Tuple[int, int]]) -> None:
        """Remove old arrows for this drone, then place arrows along the full path.

        `path` is a list of cells the drone traversed, e.g. [(3,3), (4,4), (5,5), (6,6)].
        Arrows are placed on every cell EXCEPT the last one (the drone's current pos).
        Each arrow points toward the next cell in the path.
        """
        # Remove old arrows
        for old_arrow in self._arrow_agents.pop(drone_id, []):
            self.model.grid.remove_agent(old_arrow)
            self.model.schedule.remove(old_arrow)
        # Need at least 2 cells (start + end) to draw a path
        if len(path) < 2:
            return
        new_arrows: List["MovementArrowAgent"] = []
        # Place an arrow on every cell except the last (drone's current position)
        for i in range(len(path) - 1):
            cell = path[i]
            next_cell = path[i + 1]
            dx = 1 if next_cell[0] > cell[0] else (-1 if next_cell[0] < cell[0] else 0)
            dy = 1 if next_cell[1] > cell[1] else (-1 if next_cell[1] < cell[1] else 0)
            uid = f"arrow_{self._arrow_id_counter}"
            self._arrow_id_counter += 1
            arrow = MovementArrowAgent(uid, self.model, drone_id, (dx, dy))
            self.model.schedule.add(arrow)
            self.model.grid.place_agent(arrow, cell)
            new_arrows.append(arrow)
        self._arrow_agents[drone_id] = new_arrows

    def _assign_sectors(self) -> None:
        """Distribute sectors among drones, no overlap."""
        if self._sector_assignments:
            return  # already assigned
        drones = sorted(
            [a for a in self.model.schedule.agents if isinstance(a, DroneAgent)],
            key=lambda d: d.unique_id,
        )
        if not drones:
            return
        n = len(drones)
        sector_ids = sorted(SECTOR_DEFS.keys())
        for i, sid in enumerate(sector_ids):
            did = drones[i % n].unique_id
            self._sector_assignments.setdefault(did, []).append(sid)
        for d in drones:
            wps: List[Tuple[int, int]] = []
            for sid in self._sector_assignments.get(d.unique_id, []):
                sdef = SECTOR_DEFS[sid]
                origin: Tuple[int, int] = cast(Tuple[int, int], sdef["origin"])
                size: Tuple[int, int] = cast(Tuple[int, int], sdef["size"])
                wps.extend(_sector_waypoints(origin, size))
            self._waypoint_queue[d.unique_id] = wps

    def discover_drones(self) -> Dict[str, Any]:
        drones = [a for a in self.model.schedule.agents if isinstance(a, DroneAgent)]
        return {
            "drones": [
                {"id": d.unique_id, "pos": list(d.pos), "battery": d.battery, "disabled": d.disabled}
                for d in drones
            ],
            "count": len(drones),
        }

    def get_mission_state(self) -> Dict[str, Any]:
        survivors = [a for a in self.model.schedule.agents if isinstance(a, SurvivorAgent)]
        found = [s for s in survivors if s.detected]
        drones = [a for a in self.model.schedule.agents if isinstance(a, DroneAgent)]
        sectors = []
        for sid, sdef in SECTOR_DEFS.items():
            origin: Tuple[int, int] = cast(Tuple[int, int], sdef["origin"])
            ox, oy = origin
            size: Tuple[int, int] = cast(Tuple[int, int], sdef["size"])
            w, h = size
            total = w * h
            scanned = sum(
                1
                for xi in range(ox, ox + w)
                for yi in range(oy, oy + h)
                if self.model.tile_map.get((xi, yi)) and self.model.tile_map[(xi, yi)].scanned
            )
            pct = float(f"{scanned / max(1, total) * 100:.1f}")
            sectors.append(
                {
                    "id": sid,
                    "name": cast(str, sdef["name"]),
                    "origin": [ox, oy],
                    "size": [w, h],
                    "coverage_pct": pct,
                }
            )
            
        scanned_count = sum(1 for t in self.model.tile_map.values() if t.scanned)
        
        unscanned_grids = []
        if self.model.schedule.steps > 10:
            for (x, y), t in self.model.tile_map.items():
                if not t.scanned:
                    # Find the sector for this grid
                    sector_id = None
                    for sid, sdef in SECTOR_DEFS.items():
                        ox, oy = sdef["origin"]
                        w, h = sdef["size"]
                        if ox <= x < ox + w and oy <= y < oy + h:
                            sector_id = sid
                            break
                    unscanned_grids.append({"pos": [x, y], "sector": sector_id})
                    if len(unscanned_grids) >= 30: # Limit to 30 to avoid prompt overflow
                        break

        return {
            "tick": self.model.schedule.steps,
            "tool_calls": getattr(self, "_tool_call_count", 0),
            "grid": [self.model.grid.width, self.model.grid.height],
            "bases": [list(b) for b in BASE_POSITIONS],
            "drones": {d.unique_id: {"pos": list(d.pos), "battery": d.battery, "disabled": d.disabled} for d in drones},
            "survivors_found": len(found),
            "survivors_total": "Unknown (Find all possible)",
            "actual_survivors_total_hidden": len(survivors), # Hidden from AI, used internally
            "survivor_locations": [{"id": s.unique_id, "pos": list(s.pos)} for s in found],
            "scanned_cell_count": scanned_count,
            "unscanned_grids": unscanned_grids,
            "sectors": sectors,
            # ── building / obstacle data (Compressed: only send obstacles in incomplete sectors) ──
            "obstacle_positions": [
                list(b.pos) for b in self.model.building_list 
                if b.is_obstacle and any(
                    # Check if this building is in a sector that is not yet fully scanned
                    (sum(1 for xi in range(sdef["origin"][0], sdef["origin"][0] + sdef["size"][0])
                         for yi in range(sdef["origin"][1], sdef["origin"][1] + sdef["size"][1])
                         if self.model.tile_map.get((xi, yi)) and self.model.tile_map[(xi, yi)].scanned)
                     / (sdef["size"][0] * sdef["size"][1])) < 1.0
                    and sdef["origin"][0] <= b.pos[0] < sdef["origin"][0] + sdef["size"][0]
                    and sdef["origin"][1] <= b.pos[1] < sdef["origin"][1] + sdef["size"][1]
                    for sdef in SECTOR_DEFS.values()
                )
            ],
        }

    def move_to(self, drone_id: str, x: int, y: int, reason: str = "Navigating") -> Dict[str, Any]:
        # ── One-move-per-tick guard ──
        if drone_id in self._moved_this_tick:
            return {"moved": False, "reason": "already moved this tick"}
        d = self.model.get_drone(drone_id)
        if d.disabled:
            return {"moved": False, "reason": "drone disabled"}
        if d.battery <= 0:
            d.disabled = True
            return {"moved": False, "reason": "battery depleted"}

        tx = max(0, min(self.model.grid.width - 1, int(x)))
        ty = max(0, min(self.model.grid.height - 1, int(y)))
        pos: Tuple[int, int] = cast(Tuple[int, int], d.pos)
        cx, cy = pos
        # Track the full path for arrow placement
        path: List[Tuple[int, int]] = [(cx, cy)]
        steps = 0
        blocked = False
        while steps < d.speed and (cx, cy) != (tx, ty) and d.battery > 0:
            # Ideal step towards target
            dx_ideal = (1 if tx > cx else -1) if tx != cx else 0
            dy_ideal = (1 if ty > cy else -1) if ty != cy else 0
            
            # ── Obstacle Avoidance / Sliding Logic ──
            # Try directions in order of preference:
            # 1. Direct diagonal (if applicable)
            # 2. Straight horizontal/vertical component
            # 3. Sidestepping (diagonal moves that keep some progress)
            # 4. Pure perpendicular sidestep (to escape corridors)
            # 5. Backward diagonal / backward (last resort)
            
            candidates = []
            # Primary: direct diagonal
            if dx_ideal != 0 and dy_ideal != 0:
                candidates.append((dx_ideal, dy_ideal))
            # Secondary: axis-aligned towards target
            if dx_ideal != 0:
                candidates.append((dx_ideal, 0))
            if dy_ideal != 0:
                candidates.append((0, dy_ideal))
            
            # Tertiary: sidestepping (diagonal with one component towards target)
            if dx_ideal == 0:  # Moving vertically — try diagonal sidestep
                candidates.extend([(1, dy_ideal), (-1, dy_ideal)])
            if dy_ideal == 0:  # Moving horizontally — try diagonal sidestep
                candidates.extend([(dx_ideal, 1), (dx_ideal, -1)])
            
            # Quaternary: pure perpendicular sidestep (escape narrow corridors)
            if dx_ideal == 0:  # Moving vertically — try pure left/right
                candidates.extend([(1, 0), (-1, 0)])
            if dy_ideal == 0:  # Moving horizontally — try pure up/down
                candidates.extend([(0, 1), (0, -1)])
            # For diagonal movement, also add pure perpendicular
            if dx_ideal != 0 and dy_ideal != 0:
                candidates.extend([(0, -dy_ideal), (0, dy_ideal), (-dx_ideal, 0), (dx_ideal, 0)])
            
            # Last resort: backward movement (to back out of dead ends)
            if dx_ideal != 0 and dy_ideal != 0:
                candidates.extend([(-dx_ideal, dy_ideal), (dx_ideal, -dy_ideal), (-dx_ideal, -dy_ideal)])
            elif dx_ideal == 0:
                candidates.extend([(1, -dy_ideal), (-1, -dy_ideal), (0, -dy_ideal)])
            elif dy_ideal == 0:
                candidates.extend([(-dx_ideal, 1), (-dx_ideal, -1), (-dx_ideal, 0)])
            
            # De-duplicate while preserving order
            seen = set()
            unique_candidates = []
            for c in candidates:
                if c not in seen and c != (0, 0):
                    seen.add(c)
                    unique_candidates.append(c)
            candidates = unique_candidates
                
            best_nx, best_ny = cx, cy
            found_move = False
            
            for cdx, cdy in candidates:
                nx, ny = cx + cdx, cy + cdy
                
                # Bounds check
                if not (0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height):
                    continue
                # Obstacle check
                if self.model.is_high_building((nx, ny)):
                    continue
                # Stacking check
                stack_blocked = False
                if (nx, ny) not in BASE_POSITIONS:
                    for agent in self.model.grid.get_cell_list_contents((nx, ny)):
                        if agent.__class__.__name__ == "DroneAgent" and getattr(agent, "unique_id", None) != d.unique_id:
                            stack_blocked = True
                            break
                if stack_blocked:
                    continue
                
                # Valid move found
                best_nx, best_ny = nx, ny
                found_move = True
                break  # Best available candidate based on preference order
            
            if not found_move:
                blocked = True
                break
                
            cx, cy = best_nx, best_ny
            path.append((cx, cy))
            d.battery = max(0.0, float(d.battery) - BATTERY_COST_MOVE)
            steps = steps + 1  # type: ignore[operator]
        self.model.grid.move_agent(d, (cx, cy))
        # ── Place movement arrows along the full path ──
        self._place_movement_arrows(drone_id, path)
        # ── Append to model history ──
        if pos != (cx, cy):
            self.model.movement_history.append((self.model.schedule.steps, drone_id, pos, (cx, cy), reason))
        # ── Mark this drone as moved for this tick ──
        self._moved_this_tick.add(drone_id)
        if d.battery <= 0:
            d.disabled = True
        result = {
            "new_pos": [cx, cy],
            "battery": d.battery,
            "arrived": (cx, cy) == (tx, ty),
            "steps_taken": steps,
        }
        if blocked:
            result["blocked_by_building"] = True
            result["blocked_pos"] = [cx + dx_ideal, cy + dy_ideal]
        return result

    def move_and_scan(self, drone_id: str, x: int, y: int, reason: str = "Scanning sector") -> Dict[str, Any]:
        res = self.move_to(drone_id, x, y, reason=reason)
        if "moved" in res and not res["moved"]:
            return res
        s_res = self.thermal_scan(drone_id)
        if "scanned" in s_res and not s_res["scanned"]:
            res["scan_reason"] = s_res["reason"]
            return res

        coord = s_res.get("coordinate") or res.get("new_pos") or [0, 0]
        vx = int(coord[0]) if isinstance(coord, list) and len(coord) >= 2 else 0
        vy = int(coord[1]) if isinstance(coord, list) and len(coord) >= 2 else 0
        filter_result = self._verify_signature_internal(vx, vy)

        res["scanned_cells"] = s_res.get("scanned_cells")
        res["coordinate"] = s_res.get("coordinate")
        res["thermal_intensity"] = s_res.get("thermal_intensity")
        res["survivors_found"] = filter_result.get("confirmed_survivors", [])
        res["filter_result"] = filter_result
        res["battery"] = s_res.get("battery")
        return res

    def _thermal_intensity_seed(self, coord: Tuple[int, int]) -> float:
        for a in self.model.grid.get_cell_list_contents([coord]):
            if isinstance(a, SurvivorAgent) and not a.detected:
                return round(max(0.0, min(1.0, 0.85 + random.uniform(-0.05, 0.10))), 2)
        if random.random() < 0.15:
            return round(max(0.0, min(1.0, 0.70 + random.uniform(-0.08, 0.08))), 2)
        return round(max(0.0, min(1.0, 0.30 + random.uniform(-0.08, 0.08))), 2)

    def _verify_signature_internal(self, x: int, y: int) -> Dict[str, Any]:
        threshold = 0.80
        tx = max(0, min(self.model.grid.width - 1, int(x)))
        ty = max(0, min(self.model.grid.height - 1, int(y)))
        intensity = self._thermal_intensity_seed((tx, ty))

        neighborhood = self.model.grid.get_neighborhood(
            (tx, ty), moore=True, include_center=True, radius=SCAN_RADIUS
        )
        candidate_survivors: List[Tuple[SurvivorAgent, Tuple[int, int]]] = []
        for cell in neighborhood:
            for a in self.model.grid.get_cell_list_contents([cell]):
                if isinstance(a, SurvivorAgent) and not a.detected:
                    candidate_survivors.append((a, cast(Tuple[int, int], cell)))

        if candidate_survivors:
            intensity = max(intensity, threshold)

        if intensity >= threshold and candidate_survivors:
            confirmed_survivors: List[Dict[str, Any]] = []
            for survivor, spos in candidate_survivors:
                survivor.detected = True
                confirmed_survivors.append(
                    {
                        "id": survivor.unique_id,
                        "pos": [spos[0], spos[1]],
                    }
                )
            return {
                "verification_status": "CONFIRMED",
                "human_detected": True,
                "confirmed_count": len(confirmed_survivors),
                "confirmed_survivors": confirmed_survivors,
                "threshold": threshold,
                "intensity": intensity,
                "coordinate": [tx, ty],
                "scan_center": [tx, ty],
            }

        return {
            "verification_status": "FALSE_POSITIVE",
            "human_detected": False,
            "confirmed_count": 0,
            "confirmed_survivors": [],
            "threshold": threshold,
            "intensity": intensity,
            "coordinate": [tx, ty],
            "scan_center": [tx, ty],
        }

    def thermal_scan(self, drone_id: str) -> Dict[str, Any]:
        d = self.model.get_drone(drone_id)
        if d.disabled:
            return {"scanned": False, "reason": "drone disabled"}
        if d.battery < BATTERY_COST_SCAN:
            return {"scanned": False, "reason": "insufficient battery for scan"}

        d.battery = max(0.0, float(d.battery) - BATTERY_COST_SCAN)
        cells = self.model.grid.get_neighborhood(d.pos, moore=True, include_center=True, radius=SCAN_RADIUS)

        for cell in cells:
            tile = self.model.tile_map.get(cell)
            if tile:
                tile.scanned = True

        coord = cast(Tuple[int, int], d.pos)
        raw_intensity = self._thermal_intensity_seed(coord)

        if d.battery <= 0:
            d.disabled = True

        return {
            "scanned": True,
            "scanned_cells": len(cells),
            "coordinate": [coord[0], coord[1]],
            "thermal_intensity": raw_intensity,
            "status": "Potential Signature Detected",
            "battery": d.battery,
            "drone_pos": list(d.pos),
        }

    def recall_to_base(self, drone_id: str) -> Dict[str, Any]:
        d = self.model.get_drone(drone_id)
        if d.disabled:
            return {"moved": False, "reason": "drone disabled"}
        nearest = min(
            BASE_POSITIONS,
            key=lambda b: abs(b[0] - d.pos[0]) + abs(b[1] - d.pos[1]),
        )
        return self.move_to(drone_id, nearest[0], nearest[1], reason="Low battery RTS") | {"base_target": list(nearest)}

    def charge_drone(self, drone_id: str) -> Dict[str, Any]:
        d = self.model.get_drone(drone_id)
        if d.pos not in BASE_POSITIONS:
            return {"charged": False, "reason": f"Not at a station (drone at {list(d.pos)})"}
        # Instant charge: one tick = full battery
        d.battery = 100.0
        d.disabled = False
        return {"charged": True, "battery": d.battery, "full": True}

    def get_drone_orders(self) -> Dict[str, Any]:
        """[DELETED: UI now uses True Swarm Autonomy]"""
        return {}

    def get_nearby_obstacles(self, drone_id: str, radius: int = 5) -> Dict[str, Any]:
        """Return obstacles within `radius` cells of the drone's current position.
        
        This is the Phase 3 "Localized Obstacle Tool" — lets the LLM "look around"
        without needing the entire global obstacle list in the prompt.
        """
        d = self.model.get_drone(drone_id)
        dx, dy = d.pos
        nearby: List[List[int]] = []
        for b in self.model.building_list:
            if b.is_obstacle:
                bx, by = b.pos
                if abs(bx - dx) <= radius and abs(by - dy) <= radius:
                    nearby.append([bx, by])
        return {
            "drone_id": drone_id,
            "drone_pos": [dx, dy],
            "radius": radius,
            "nearby_obstacles": nearby,
            "count": len(nearby),
        }

    def update_rolling_summary(self, tick: int, actions_taken: List[Dict[str, Any]]) -> None:
        """Update the rolling mission summary with a compressed tick report.
        
        Phase 3: Rolling Mission Summary — keeps a sliding window of the last
        5 tick summaries so the LLM has short-term memory without context bloat.
        """
        if not actions_taken:
            entry = f"Tick {tick}: No actions (model returned empty)."
        else:
            parts = []
            for a in actions_taken:
                tool = a.get("tool_name", "?")
                drone = a.get("arguments", {}).get("drone_id", "?")
                reasoning = a.get("reasoning", "")
                if tool == "move_and_scan":
                    x = a.get("arguments", {}).get("x", "?")
                    y = a.get("arguments", {}).get("y", "?")
                    parts.append(f"{drone}→({x},{y}) scan")
                elif tool == "recall_to_base":
                    parts.append(f"{drone}→base")
                elif tool == "charge_drone":
                    parts.append(f"{drone} charging")
                else:
                    parts.append(f"{drone}:{tool}")
            entry = f"Tick {tick}: {'; '.join(parts)}"
        
        self._tick_history.append(entry)
        # Keep only the last 5 tick summaries (sliding window)
        if len(self._tick_history) > 5:
            self._tick_history = self._tick_history[-5:]
        
        # Build the rolling summary string
        self._rolling_summary = "\n".join(self._tick_history)

    def get_rolling_summary(self) -> str:
        """Return the current rolling mission summary."""
        return self._rolling_summary

    # ── LangGraph staging helpers ─────────────────────────────────────

    def get_sector_status(self, sector_id: int) -> Dict[str, Any]:
        """Return coverage %, list of drone IDs present, and unscanned cell count for one sector."""
        sdef = SECTOR_DEFS.get(sector_id)
        if sdef is None:
            return {"error": f"Unknown sector {sector_id}"}
        origin: Tuple[int, int] = cast(Tuple[int, int], sdef["origin"])
        ox, oy = origin
        size: Tuple[int, int] = cast(Tuple[int, int], sdef["size"])
        w, h = size
        total = w * h
        scanned = sum(
            1
            for xi in range(ox, ox + w)
            for yi in range(oy, oy + h)
            if self.model.tile_map.get((xi, yi)) and self.model.tile_map[(xi, yi)].scanned
        )
        pct = round(scanned / max(1, total) * 100, 1)
        # Drones currently in this sector
        drones_here: List[str] = []
        for a in self.model.schedule.agents:
            if isinstance(a, DroneAgent) and not a.disabled:
                dx, dy = a.pos
                if ox <= dx < ox + w and oy <= dy < oy + h:
                    drones_here.append(a.unique_id)
        return {
            "sector_id": sector_id,
            "name": cast(str, sdef["name"]),
            "origin": [ox, oy],
            "size": [w, h],
            "coverage_pct": pct,
            "unscanned_cells": total - scanned,
            "drones_in_sector": drones_here,
        }

    def get_drone_status(self, drone_id: str) -> Dict[str, Any]:
        """Return position, battery, disabled state, and current sector for one drone."""
        try:
            d = self.model.get_drone(drone_id)
        except ValueError:
            return {"error": f"Drone not found: {drone_id}"}
        dx, dy = d.pos
        sector = pos_to_sector_id(dx, dy)
        at_base = (dx, dy) in BASE_POSITIONS
        return {
            "drone_id": drone_id,
            "pos": [dx, dy],
            "battery": d.battery,
            "disabled": d.disabled,
            "sector_id": sector,
            "at_base": at_base,
        }

    def validate_coordinate(self, x: int, y: int) -> Dict[str, Any]:
        """Check whether (x, y) is a valid drone destination (in-bounds, not a high building)."""
        if not (0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height):
            return {"valid": False, "reason": f"Out of bounds: ({x}, {y}). Grid is {self.model.grid.width}x{self.model.grid.height}."}
        if self.model.is_high_building((x, y)):
            return {"valid": False, "reason": f"High building obstacle at ({x}, {y}). Choose an adjacent cell."}
        return {"valid": True, "reason": "ok"}

    def stage_drone_command(
        self,
        drone_id: str,
        action: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Append a structured command dict without executing it.
        
        Valid actions: move_and_scan, recall_to_base, charge_drone, wait.
        """
        VALID_ACTIONS = {"move_and_scan", "recall_to_base", "charge_drone", "wait"}
        if action not in VALID_ACTIONS:
            return {"staged": False, "reason": f"Unknown action '{action}'. Use one of {sorted(VALID_ACTIONS)}."}
        if action == "move_and_scan" and (x is None or y is None):
            return {"staged": False, "reason": "move_and_scan requires x and y coordinates."}
        if action == "move_and_scan":
            # Pass directly to pathfinder. If target is blocked, it will path as close as possible.
            # We only bounds-check roughly here or let move_to clamp it.
            pass
        cmd: Dict[str, Any] = {
            "drone_id": drone_id,
            "action": action,
            "reason": reason,
        }
        if x is not None:
            cmd["x"] = int(x)
        if y is not None:
            cmd["y"] = int(y)
        self._staged_commands.append(cmd)
        return {"staged": True, "command": cmd}

    def flush_staged_commands(self) -> List[Dict[str, Any]]:
        """Execute all staged commands, return results, and clear the buffer."""
        results: List[Dict[str, Any]] = []
        for cmd in self._staged_commands:
            drone_id = cmd["drone_id"]
            action = cmd["action"]
            reason = cmd.get("reason", "")
            try:
                if action == "move_and_scan":
                    res = self.move_and_scan(drone_id, cmd["x"], cmd["y"], reason=reason or "LangGraph scan")
                elif action == "recall_to_base":
                    res = self.recall_to_base(drone_id)
                elif action == "charge_drone":
                    res = self.charge_drone(drone_id)
                elif action == "wait":
                    res = {"action": "wait", "drone_id": drone_id, "reason": reason}
                else:
                    res = {"error": f"Unknown action: {action}"}
                results.append({"command": cmd, "result": res})
            except Exception as exc:
                results.append({"command": cmd, "error": str(exc)})
        self._staged_commands.clear()
        return results


class SimpleAiController:
    """
    Deterministic AI that:
    - reasons/plans (printed to console)
    - calls the in-UI tool server to move + scan
    This makes the UI match "AI drives drones".
    """

    def __init__(self, tools: InUiToolServer, action_delay_s: float = 0.0):
        self.tools = tools
        self.action_delay_s = max(0.0, float(action_delay_s or 0.0))
        self._tick = 0
        self._next_waypoint: Dict[str, Tuple[int, int]] = {}
        self._waypoint_queue: List[Tuple[int, int]] = []

    def _pause(self) -> None:
        if self.action_delay_s > 0:
            time.sleep(self.action_delay_s)

    def _build_waypoint_queue(self) -> List[Tuple[int, int]]:
        ms = self.tools.get_mission_state()
        wps: List[Tuple[int, int]] = []
        # Prefer sectors with lowest coverage first.
        sectors = sorted(cast(List[Dict[str, Any]], ms["sectors"]), key=lambda s: float(s.get("coverage_pct", 0.0)))
        for s in sectors:
            for wp in cast(List[List[int]], s.get("waypoints", [])):
                wps.append((int(wp[0]), int(wp[1])))
        return wps

    def think_and_act(self) -> None:
        self._tick += 1
        ms = self.tools.get_mission_state()
        drones = cast(List[Dict[str, Any]], self.tools.discover_drones()["drones"])

        if not self._waypoint_queue:
            self._waypoint_queue = self._build_waypoint_queue()

        survivors_found = int(ms.get("survivors_found", 0))
        survivors_total = ms.get("survivors_total", "Unknown")

        # ── Check if all sectors fully scanned ──
        sectors = cast(List[Dict[str, Any]], ms.get("sectors", []))
        all_sectors_scanned = len(sectors) > 0 and all(
            float(s.get("coverage_pct", 0)) >= 99.0 for s in sectors
        )
        all_found = all_sectors_scanned

        print("\n" + "═" * 72)
        print(f"AI CONTROLLER — tick {self._tick}")
        print("═" * 72)
        print(f"Mission: survivors {survivors_found}/{survivors_total}")
        if all_found:
            print("[MISSION COMPLETE] All sectors scanned!")

        for d in drones:
            did = cast(str, d["id"])
            pos = cast(Tuple[int, int], tuple(d["pos"]))
            battery = float(d["battery"])
            disabled = bool(d["disabled"])

            reasoning_lines = [
                f"Drone {did} at {pos} battery={battery:.0f}% disabled={disabled}."
            ]

            if disabled:
                reasoning_lines.append("Drone is disabled; skip actions this tick.")
                print("[REASON]", " ".join(reasoning_lines))
                continue

            # ── Mission complete: return all drones to base ──
            if all_found:
                at_base = pos in BASE_POSITIONS
                if at_base:
                    reasoning_lines.append("All survivors found. Already at base. Mission complete.")
                    print("[REASON]", " ".join(reasoning_lines))
                else:
                    reasoning_lines.append("All survivors found. Returning to base.")
                    print("[REASON]", " ".join(reasoning_lines))
                    print("[CALL] recall_to_base", {"drone_id": did})
                    self.tools.recall_to_base(did)
                    self._pause()
                    # Charge if we actually reached a station this tick.
                    if tuple(self.tools.model.get_drone(did).pos) in BASE_POSITIONS:
                        print("[CALL] charge_drone", {"drone_id": did})
                        self.tools.charge_drone(did)
                        self._pause()
                continue

            if battery <= BATTERY_CRITICAL:
                reasoning_lines.append("Battery is critical. Recall to base, then charge.")
                print("[REASON]", " ".join(reasoning_lines))
                print("[CALL] recall_to_base", {"drone_id": did})
                self.tools.recall_to_base(did)
                self._pause()
                # Charge if we actually reached a station this tick.
                if tuple(self.tools.model.get_drone(did).pos) in BASE_POSITIONS:
                    print("[CALL] charge_drone", {"drone_id": did})
                    self.tools.charge_drone(did)
                    self._pause()
                continue

            # Always scan opportunistically (this is what reveals survivors in UI).
            reasoning_lines.append("Scan nearby cells for survivors and mark coverage.")
            print("[REASON]", " ".join(reasoning_lines))
            print("[CALL] thermal_scan", {"drone_id": did})
            self.tools.thermal_scan(did)
            self._pause()

            # Move towards next waypoint.
            if not self._next_waypoint.get(did):
                if self._waypoint_queue:
                    self._next_waypoint[did] = self._waypoint_queue.pop(0)

            target = self._next_waypoint.get(did)
            if target:
                if pos == target:
                    self._next_waypoint.pop(did, None)
                else:
                    print("[CALL] move_to", {"drone_id": did, "x": target[0], "y": target[1]})
                    result = self.tools.move_to(did, target[0], target[1], reason="Auto-routing to waypoint")
                    self._pause()
                    # If blocked by a high building, skip this waypoint
                    if result.get("blocked_by_building"):
                        print(f"[OBSTACLE] Blocked by high building at {result.get('blocked_pos')}. Skipping waypoint.")
                        self._next_waypoint.pop(did, None)


class GeminiAiController:
    """
    LLM-backed controller (Gemini).

    It sends a mission snapshot + tool schema to Gemini and expects strict JSON:
      {"tool_calls":[{"tool_name":"...", "arguments":{...}, "reasoning":"..."}]}
    Then executes those tool calls against the in-process tool server.
    """

    def __init__(
        self,
        tools: InUiToolServer,
        model_name: str = "gemini-2.5-flash",
        action_delay_s: float = 0.0,
        max_calls_per_tick: int = 5,
    ):
        self.tools = tools
        self.model_name = str(model_name or "gemini-2.5-flash")
        self.action_delay_s = max(0.0, float(action_delay_s or 0.0))
        self.max_calls_per_tick = max(1, int(max_calls_per_tick))
        self._tick = 0
        self._recent_log: str = ""
        self._summarize_every_ticks = 10
        self._warned_unavailable = False
        self._client = self._maybe_client()

    def _maybe_client(self):
        if genai is None:
            return None
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        try:
            return genai.Client(api_key=api_key)
        except Exception:
            return None

    def _pause(self) -> None:
        if self.action_delay_s > 0:
            time.sleep(self.action_delay_s)

    def _print_wrapped(self, prefix: str, text: str) -> None:
        for ln in textwrap.wrap(text, width=92):
            print(f"{prefix}{ln}")

    def _should_summarize_tick(self) -> bool:
        return self._tick > 0 and (self._tick % self._summarize_every_ticks == 0)

    def _build_status_summary(self, mission_state: Dict[str, Any], drones: List[Dict[str, Any]]) -> str:
        survivors_found = int(mission_state.get("survivors_found", 0))
        survivors_total = mission_state.get("survivors_total", "Unknown")
        scanned_cells = int(mission_state.get("scanned_cell_count", 0))
        lines = [
            f"Tick {self._tick}: survivors {survivors_found}/{survivors_total}, scanned cells={scanned_cells}",
        ]
        for drone in drones:
            did = str(drone.get("id", "?"))
            pos = drone.get("pos", [0, 0])
            battery = float(drone.get("battery", 0))
            disabled = bool(drone.get("disabled", False))
            state = "disabled" if disabled else "active"
            lines.append(f"- {did}: pos={pos}, battery={battery:.1f}%, {state}")
        return "\n".join(lines)

    def _tool_schema(self) -> List[Dict[str, Any]]:
        # Keep it simple + stable: only tools we actually support here.
        return [
            {
                "name": "discover_drones",
                "description": "List drones with id/pos/battery/disabled.",
                "parameters": [],
            },
            {
                "name": "get_mission_state",
                "description": "Mission snapshot: survivors found/total and sector coverage/waypoints.",
                "parameters": [],
            },
            {
                "name": "move_to",
                "description": "Move a drone toward (x,y) up to its speed; costs battery per cell.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                    {"name": "x", "type": "integer", "required": True},
                    {"name": "y", "type": "integer", "required": True},
                ],
            },
            {
                "name": "move_and_scan",
                "description": "Move drone toward (x,y) and perform a thermal scan upon arrival.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                    {"name": "x", "type": "integer", "required": True},
                    {"name": "y", "type": "integer", "required": True},
                ],
            },
            {
                "name": "thermal_scan",
                "description": "Scan radius-2 around drone; marks tiles scanned and detects survivors; costs battery.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                ],
            },
            {
                "name": "recall_to_base",
                "description": "Move drone toward nearest charging station.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                ],
            },
            {
                "name": "charge_drone",
                "description": "Instant charge drone battery to 100% if on a base.",
                "parameters": [
                    {"name": "drone_id", "type": "string", "required": True},
                ],
            },
            {
                "name": "get_drone_orders",
                "description": "Get pre-computed optimal actions for all drones. Sectors auto-assigned, no overlap.",
                "parameters": [],
            },
        ]

    def _build_prompt(self, mission_state: Dict[str, Any], drones: List[Dict[str, Any]], summarize_tick: bool = False) -> str:
        tools_json = json.dumps(self._tool_schema(), default=str, separators=(",", ":"))
        ms_json = str(
            json.dumps({"mission_state": mission_state, "drones": drones}, default=str, separators=(",", ":"))
        )[:3500]  # type: ignore[index]
        recent = self._recent_log[-2000:] if self._recent_log else ""
        num_drones = len(drones)
        drone_ids_str = ", ".join(d["id"] for d in drones)

        # ── Extract obstacle data for the prompt ──
        obstacle_positions = cast(List[List[int]], mission_state.get("obstacle_positions", []))
        obstacle_text = str(json.dumps(obstacle_positions, default=str, separators=(",", ":")))[:1200]  # type: ignore[index]
        summarize_instruction = (
            "\nSPECIAL INSTRUCTION FOR THIS TICK:\n"
            '"Summarize the current status of all drones and remove all previous turn logs."\n'
            "Include that summary in your reasoning for each tool call, then proceed with normal actions.\n\n"
            if summarize_tick
            else ""
        )

        unscanned = mission_state.get("unscanned_grids", [])
        unscanned_text = ""
        if unscanned:
            unscanned_text = f"\n  UNSCANNED GRIDS OUTSTANDING: {json.dumps(unscanned, separators=(',', ':'))[:1000]}\n"

        return (
            "You are the Command Agent for a simulated fleet of drones in an earthquake search-and-rescue mission.\n"
            "Your MAIN objective is meticulously scanning all grids to find unknown survivors FAST while maintaining the battery.\n"
            "You do not know how many total survivors there are. You must scan 100% of the map.\n"
            "You control ONLY the tools listed below. You must output STRICT JSON only.\n\n"
            "Constraints:\n"
            f"- MULTI-TASKING: You MUST output EXACTLY {num_drones} tool_calls per tick, one for each drone ({drone_ids_str})!\n"
            "- CRITICAL SURVIVAL PROTOCOL:\n"
            "  1. Moving costs 1% battery per cell. Scanning costs 5%.\n"
            "  2. If drone's battery is < 30%, you MUST call recall_to_base IMMEDIATELY.\n"
            "  3. DO NOT use thermal_scan or move_to for search tasks when battery is < 30%.\n"
            "  4. Once at base, call charge_drone ONCE — it instantly restores battery to 100%. Then leave.\n"
            "- BUILDING OBSTACLE PROTOCOL:\n"
            "  The city has buildings of varying heights on the grid.\n"
            "  • LOW buildings (green, height < 6 floors): drones CAN fly over these.\n"
            "  • HIGH buildings (red, height >= 6 floors): drones CANNOT enter these cells!\n"
            "    If a drone tries to move through a HIGH building cell, it will be BLOCKED.\n"
            "  • Survivors are located INSIDE buildings. Scan from adjacent or overhead cells.\n"
            "  • You MUST plan movement paths that AVOID high-building obstacle cells.\n"
            "  • If move_and_scan returns 'blocked_by_building', choose a different target.\n"
            f"  HIGH-BUILDING OBSTACLE POSITIONS (avoid these): {obstacle_text}\n"
            f"{unscanned_text}"
            "- SEARCH PROTOCOL:\n"
            "  1. Look at the `sectors` array in the state JSON. Focus on sectors where `coverage_pct` is less than 100.\n"
            "  2. Pick random `(x, y)` coordinates within the boundaries of those incomplete sectors (using their `origin` and `size`).\n"
            "  3. ALWAYS use `move_and_scan` — it moves AND thermal scans in one tick. Provide `x` and `y`.\n"
            "  4. DO NOT overlap multiple drones on the exact same coordinate.\n"
            "  5. DO NOT call thermal_scan or move_to separately.\n"
            "  6. Every drone MUST move at least one step in each tick to navigate around obstacles or reach targets. Do not stay idle unless charging or mission is complete.\n"
            "  7. After charging, return to where you left off and continue.\n"
            "  8. When all survivors found, all drones recall_to_base.\n\n"
            f"Available tools (JSON schema):\n{tools_json}\n\n"
            f"Current state (JSON):\n{ms_json}\n\n"
            f"Recent tool log (may be empty):\n{recent}\n\n"
            f"{summarize_instruction}"
            "Respond with ONLY this JSON shape (no prose outside JSON):\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "tool_name": "move_and_scan",\n'
            '      "arguments": {"drone_id": "d_0", "x": 2, "y": 10},\n'
            '      "reasoning": "Why this is the next call."\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"You MUST output EXACTLY {num_drones} tool_calls per tick, assigning exactly 1 action to each of the drones ({drone_ids_str}). Do not leave any drone idle!\n"
        )


    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        text = text.strip()
        
        # Robust Regex to extract JSON object from markdown, <think> tags, or conversational filler
        import re
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            text = match.group(1)
            
        try:
            obj = json.loads(text)
        except Exception:
            return []
            
        calls = obj.get("tool_calls", [])
        if not isinstance(calls, list):
            return []
        out: List[Dict[str, Any]] = []
        for c in cast(List[Any], calls)[: self.max_calls_per_tick]:  # type: ignore[index]
            if not isinstance(c, dict):
                continue
            name = c.get("tool_name")
            args = c.get("arguments") or {}
            reasoning = str(c.get("reasoning") or "").strip()
            if not name or not isinstance(args, dict):
                continue
            out.append({"tool_name": str(name), "arguments": args, "reasoning": reasoning})
        return out

    def _exec(self, name: str, args: Dict[str, Any], reasoning: str = "") -> Dict[str, Any]:
        # Execute against our in-process tool server.
        if name == "discover_drones":
            return self.tools.discover_drones()
        if name == "get_mission_state":
            return self.tools.get_mission_state()
        if name == "move_to":
            return self.tools.move_to(str(args.get("drone_id", "")), int(args.get("x", 0)), int(args.get("y", 0)), reason=reasoning or "Moving via move_to")
        if name == "move_and_scan":
            return self.tools.move_and_scan(str(args.get("drone_id", "")), int(args.get("x", 0)), int(args.get("y", 0)), reason=reasoning or "Scanning area")
        if name == "thermal_scan":
            return self.tools.thermal_scan(str(args.get("drone_id", "")))
        if name == "recall_to_base":
            return self.tools.recall_to_base(str(args.get("drone_id", "")))
        if name == "charge_drone":
            return self.tools.charge_drone(str(args.get("drone_id", "")))
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def think_and_act(self) -> None:
        self._tick += 1
        ms = self.tools.get_mission_state()
        drones = cast(List[Dict[str, Any]], self.tools.discover_drones()["drones"])
        summarize_tick = self._should_summarize_tick()

        print("\n" + "═" * 72)
        print(f"GEMINI AI CONTROLLER — tick {self._tick} ({self.model_name})")
        print("═" * 72)

        if self._client is None:
            if not self._warned_unavailable:
                missing = []
                if genai is None:
                    missing.append("google-genai import failed")
                if not os.environ.get("GEMINI_API_KEY"):
                    missing.append("GEMINI_API_KEY not set (check .env)")
                msg = ", ".join(missing) if missing else "unknown reason"
                print(f"[WARN] Gemini AI unavailable ({msg}). Falling back.")
                self._warned_unavailable = True
            return

        prompt = self._build_prompt(mission_state=ms, drones=drones, summarize_tick=summarize_tick)
        print("\n[LLM PROMPT SENT]")
        if prompt:
            self._print_wrapped("  ", prompt)
        else:
            print("  [empty]")
        client = self._client
        try:
            resp = client.models.generate_content(model=self.model_name, contents=prompt)
            text = (getattr(resp, "text", None) or "").strip()
        except Exception as exc:
            print(f"[WARN] Gemini call failed: {exc}. Falling back this tick.")
            return

        print("\n[LLM RAW RESPONSE]")
        if text:
            self._print_wrapped("  ", text)
        else:
            print("  [empty]")

        calls = self._parse_tool_calls(text)
        if not calls:
            print("[WARN] Gemini returned no valid tool_calls this tick.")
            return

        for call in calls:
            reasoning = call.get("reasoning") or ""
            if reasoning:
                print("\n[CHAIN OF THOUGHT]")
                self._print_wrapped("  ", reasoning)
            name = cast(str, call["tool_name"])
            args = cast(Dict[str, Any], call["arguments"])
            print(f"\n[MCP TOOL CALL] {name}({json.dumps(args)})")
            result = self._exec(name, args, reasoning=reasoning)
            preview = str(json.dumps(result, default=str))[:260]  # type: ignore[index]
            print(f"  [OK] {preview}" if preview else "  [OK]")
            self._recent_log += f"\nTOOL {name}({json.dumps(args)}) -> {preview or '[ok]'}"
            self._pause()

        if summarize_tick:
            self._recent_log = self._build_status_summary(ms, drones)


class OllamaAiController(GeminiAiController):
    def __init__(self, tools: InUiToolServer, model_name: str = "llama3.1", action_delay_s: float = 0.0, max_calls_per_tick: int = 8):
        super().__init__(tools, model_name, action_delay_s, max_calls_per_tick)

    def _build_prompt(self, mission_state: Dict[str, Any], drones: List[Dict[str, Any]], summarize_tick: bool = False) -> str:
        import textwrap
        tools_json = json.dumps(self._tool_schema(), default=str, separators=(",", ":"))
        
        # ── Phase 3: Build per-drone local obstacle maps instead of global list ──
        local_obstacles: Dict[str, List[List[int]]] = {}
        for drone_info in drones:
            did = drone_info.get("id", "")
            if did:
                nearby = self.tools.get_nearby_obstacles(did, radius=5)
                local_obstacles[did] = nearby.get("nearby_obstacles", [])
        
        # ── Phase 3: Get rolling summary for short-term memory ──
        rolling_summary = self.tools.get_rolling_summary()
        
        # ── Compact mission state (exclude global obstacles — they're now per-drone) ──
        compact_state = {
            "tick": mission_state.get("tick", 0),
            "grid": mission_state.get("grid", [24, 16]),
            "bases": mission_state.get("bases", []),
            "sectors": mission_state.get("sectors", []),
            "survivors_found": mission_state.get("survivors_found", 0),
            "survivors_total": "Unknown (Find all possible)",
            "scanned_cell_count": mission_state.get("scanned_cell_count", 0),
        }
        
        # Add unscanned grids if present (tick > 10)
        unscanned = mission_state.get("unscanned_grids", [])
        if unscanned:
            compact_state["unscanned_grids"] = unscanned

        ms_json = json.dumps({"mission_state": compact_state, "drones": drones}, default=str, separators=(",", ":"))
        
        # ── Per-drone obstacle context (only nearby obstacles within 5 cells) ──
        obstacle_context = ""
        for did, obs in local_obstacles.items():
            if obs:
                obstacle_context += f"  {did} nearby obstacles: {obs}\n"
            else:
                obstacle_context += f"  {did}: clear (no obstacles within 5 cells)\n"

        summarize_instruction = (
            'SPECIAL INSTRUCTION FOR THIS TICK: "Summarize the current status of all drones and remove all previous turn logs."\n'
            "Include this concise summary in reasoning, then continue with standard tool calls.\n"
            if summarize_tick
            else ""
        )

        prompt = textwrap.dedent(f"""\
            You are the Swarm Commander for a fleet of drones. Your objective is to map 100% of all sectors and meticulously search all grids. You do not know how many survivors there are.
            You MUST output EXACTLY one tool call for EACH drone in the fleet. All drones must move simultaneously.

            CRITICAL STANDARD OPERATING PROCEDURE (Follow in exact order for EACH drone):

            STEP 1: BATTERY & SURVIVAL
            - If drone battery < 30%: You MUST output `recall_to_base`.
            - If drone is at a base (charging station) and battery is < 100%: You MUST output `charge_drone`.

            STEP 2: SECTOR ASSIGNMENT (NO OVERLAP)
            - Identify sectors where `coverage_pct` < 100.
            - Assign each available, charged drone to a DIFFERENT incomplete sector. 
            - NEVER assign two drones to the same sector. Do not overlap.

            STEP 3: SYSTEMATIC SEARCH & OBSTACLES
            - For the assigned sector, look at its `origin` [x, y] and `size` [width, height].
            - Pick a specific `(x, y)` coordinate inside that sector's boundaries. 
            - GRID LIMITS: You MUST ensure that 0 <= x <= 23 and 0 <= y <= 15.
            - CRITICAL: Check your drone's NEARBY OBSTACLES below. Your chosen `(x, y)` MUST NOT overlap with any obstacle position!
            - Output the `move_and_scan` tool with your chosen `(x, y)`.

            STEP 4: MISSION COMPLETE ENDGAME
            - If ALL sectors are at 100% coverage, output `recall_to_base` for ALL drones to end the mission.

            AVAILABLE TOOLS:
            {tools_json}

            CURRENT MISSION STATE:
            {ms_json}

            PER-DRONE NEARBY OBSTACLES (within 5 cells):
            {obstacle_context}
            RECENT MISSION HISTORY (rolling summary):
            {rolling_summary}
            {summarize_instruction}

            OUTPUT FORMAT EXAMPLE:
            You MUST respond with a valid JSON block. You may include reasoning or "thinking" tags before the JSON, but the final JSON MUST match this shape EXACTLY:
            {{
              "tool_calls": [
                {{
                  "tool_name": "move_and_scan",
                  "arguments": {{"drone_id": "d_0", "x": 5, "y": 10}},
                  "reasoning": "Sector 2 incomplete. Routing Drone 0."
                }}
              ]
            }}

            OUTPUT YOUR JSON RESPONSE BELOW:
        """)
        return prompt

    def think_and_act(self) -> None:
        self._tick += 1
        log_to_file(f"**Tick {self._tick} (Ollama)**: Starting reasoning cycle. Model: `{self.model_name}`")

        print("\n" + "═" * 72)
        print(f"OLLAMA AI CONTROLLER — tick {self._tick} ({self.model_name})")
        print("═" * 72)

        if ollama is None:
            if not self._warned_unavailable:
                print("[WARN] Ollama AI unavailable (import failed). Is `ollama` package installed?")
                self._warned_unavailable = True
            return

        ms = self.tools.get_mission_state()
        drones = self.tools.discover_drones()["drones"]
        summarize_tick = self._should_summarize_tick()

        prompt = self._build_prompt(ms, drones, summarize_tick=summarize_tick)
        
        # ── Qwen3 specific: prepend /no_think to skip <think> phase ──
        # This saves ~500-800 tokens of reasoning, leaving room for JSON output.
        if "qwen3" in self.model_name.lower():
            prompt = "/no_think\n" + prompt

        calls = []
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                if attempt == 0:
                    messages = [{"role": "user", "content": prompt}]
                else:
                    # Retry with a nudge: shorter, more direct instruction
                    print("[RETRY] First attempt returned truncated/empty JSON. Retrying with nudge...")
                    log_to_file(f"⚠️ **Retry** tick {self._tick}: first attempt was truncated.")
                    nudge = (
                        "Your previous response was truncated. Output ONLY the JSON object now, "
                        "no reasoning, no markdown. Start with { and end with }.\n"
                        + prompt
                    )
                    if "qwen3" in self.model_name.lower():
                        nudge = "/no_think\n" + nudge
                    messages = [{"role": "user", "content": nudge}]

                sent_prompt = str(messages[0].get("content", "")) if messages else ""
                print(f"\n[LLM PROMPT SENT] (attempt {attempt + 1})")
                if sent_prompt:
                    self._print_wrapped("  ", sent_prompt)
                else:
                    print("  [empty]")

                resp = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": 0.6,
                        "num_predict": 4096,
                        "num_ctx": 16384,
                        "keep_alive": "5m"
                    },
                )
                text = resp.get("message", {}).get("content", "").strip()
                print(f"\n[LLM RAW RESPONSE] (attempt {attempt + 1})")
                if text:
                    self._print_wrapped("  ", text)
                else:
                    print("  [empty]")
                calls = self._parse_tool_calls(text)
                if calls:
                    break  # Success — exit retry loop
                else:
                    print(f"[DEBUG] Attempt {attempt+1}: Qwen returned this raw text: '{text[:300]}'")
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Brief pause before retry
            except Exception as exc:
                print(f"[WARN] Ollama call failed (attempt {attempt+1}): {exc}.")
                log_to_file(f"⚠️ **Ollama API Error**: {exc}")

        if not calls:
            print("[WARN] No valid calls generated this tick (after retries).")
            return

        for call in calls:
            reasoning = call.get("reasoning") or ""
            if reasoning:
                print("\n[CHAIN OF THOUGHT]")
                self._print_wrapped("  ", reasoning)
                log_to_file(f"**Reasoning**: {reasoning}")
            name = str(call["tool_name"])
            args = dict(call["arguments"])
            print(f"\n[MCP TOOL CALL] {name}({json.dumps(args)})")
            result = self._exec(name, args)
            preview = str(json.dumps(result, default=str))[:260]
            print(f"  [OK] {preview}" if preview else "  [OK]")
            log_to_file(f"✅ **Execution (`{name}`)**: Success. {preview}")
            self._recent_log += f"\nTOOL {name}({json.dumps(args)}) -> {preview or '[ok]'}"
            self._pause()

        if summarize_tick:
            self._recent_log = self._build_status_summary(ms, drones)

        # ── Phase 3: Update rolling mission summary ──
        self.tools.update_rolling_summary(self._tick, calls)


# ═══════════════════════════════════════════════════════════════════════════
#  CREWAI CONTROLLER — Hierarchical Commander + Pilot agents
# ═══════════════════════════════════════════════════════════════════════════

class CrewAiController:
    """
    CrewAI-backed controller using a hierarchical agent process.

    Commander Agent  — strategic mission lead; decides which drones go where.
    Pilot Agent      — tactical execution; issues tool calls efficiently.

    Tool calls still go through the in-process InUiToolServer, so this
    controller cannot break simulation state integrity.
    """

    def __init__(
        self,
        tools: InUiToolServer,
        model_name: str = "ollama/llama3.1",
        action_delay_s: float = 0.0,
    ):
        self.tools = tools
        self.model_name = str(model_name or "ollama/llama3.1")
        self.action_delay_s = max(0.0, float(action_delay_s or 0.0))
        self._tick = 0
        self._recent_log: str = ""
        self._warned_unavailable = False

    def _pause(self) -> None:
        if self.action_delay_s > 0:
            time.sleep(self.action_delay_s)

    def _print_wrapped(self, prefix: str, text: str) -> None:
        for ln in textwrap.wrap(text, width=92):
            print(f"{prefix}{ln}")

    # ── CrewAI @tool wrappers ──────────────────────────────────────────
    # These closures capture `self` so CrewAI agents can call the simulation.

    def _build_crewai_tools(self) -> list:
        """Build CrewAI @tool-decorated functions bound to self.tools."""
        tool_server = self.tools

        @crewai_tool  # type: ignore
        def get_mission_state(dummy: str = "") -> str:
            """Get the current mission snapshot: survivors found/total, sector coverage, drone positions, and battery levels. Pass any string or empty string."""
            result = tool_server.get_mission_state()
            return json.dumps(result, default=str)

        @crewai_tool  # type: ignore
        def move_and_scan(drone_id: str, x: int, y: int) -> str:
            """Move a drone toward (x,y) and perform a thermal scan. Input: drone_id string, x integer, y integer."""
            result = tool_server.move_and_scan(str(drone_id), int(x), int(y))
            return json.dumps(result, default=str)

        @crewai_tool  # type: ignore
        def thermal_scan(drone_id: str) -> str:
            """Perform a thermal scan around a drone to detect survivors. Input: the drone_id string (e.g. 'drone_0')."""
            result = tool_server.thermal_scan(str(drone_id))
            return json.dumps(result, default=str)

        @crewai_tool  # type: ignore
        def recall_to_base(drone_id: str) -> str:
            """Recall a drone to the nearest charging base. Input: the drone_id string (e.g. 'drone_0')."""
            result = tool_server.recall_to_base(str(drone_id))
            return json.dumps(result, default=str)

        @crewai_tool  # type: ignore
        def charge_drone(drone_id: str) -> str:
            """Charge a drone's battery to 100%% if it is on a base. Input: the drone_id string (e.g. 'drone_0')."""
            result = tool_server.charge_drone(str(drone_id))
            return json.dumps(result, default=str)

        return [
            get_mission_state,
            move_and_scan,
            thermal_scan,
            recall_to_base,
            charge_drone,
        ]

    def think_and_act(self) -> None:
        self._tick += 1

        print("\n" + "═" * 72)
        print(f"CREWAI CONTROLLER — tick {self._tick} ({self.model_name})")
        print("═" * 72)

        # ── Guard: CrewAI must be importable ──────────────────────────────
        if CrewAgent is None or Crew is None or crewai_tool is None:
            if not self._warned_unavailable:
                print("[WARN] CrewAI unavailable (import failed). Is `crewai` installed?")
                self._warned_unavailable = True
            return

        # ── Gather state for the prompt context ──────────────────────────
        ms = self.tools.get_mission_state()
        drones = cast(List[Dict[str, Any]], self.tools.discover_drones()["drones"])

        survivors_found = int(ms.get("survivors_found", 0))
        survivors_total = ms.get("survivors_total", "Unknown")

        print(f"  Survivors: {survivors_found}/{survivors_total}")
        print(f"  Drones: {len(drones)}")

        # ── Check if mission complete ────────────────────────────────────
        sectors = ms.get("sectors", [])
        all_drones_returned = all(list(d["pos"]) in [list(base) for base in BASE_POSITIONS] for d in drones if not d.get("disabled"))
        all_sectors_scanned = all(s.get("coverage_pct", 0) >= 99.0 for s in sectors)
        
        if all_sectors_scanned and all_drones_returned:
            print("[INFO] Mission complete: all sectors scanned and drones returned to base.")
            return

        # ── Build context string ─────────────────────────────────────────
        drone_status = "\n".join(
            f'  {d["id"]}: pos={d["pos"]} bat={d["battery"]}% disabled={d["disabled"]}'
            for d in drones
        )

        context_block = (
            f"TICK {self._tick}\n"
            f"Survivors found: {survivors_found}/{survivors_total}\n"
            f"Drone status:\n{drone_status}\n"
        )

        # ── Build CrewAI tools ───────────────────────────────────────────
        lc_tools = self._build_crewai_tools()

        # ── Define models (support for hierarchical split) ───────────────
        manager_model = self.model_name
        agent_model = self.model_name

        if self.model_name == "ollama/hierarchical (14B/3B)":
            manager_model = "ollama/qwen3:14b"
            agent_model = "ollama/qwen2.5:3b"

        # ── Define CrewAI Agents ──────────────────────────────────────────
        commander = CrewAgent(
            role="Mission Commander",
            goal=(
                "Coordinate the drone fleet to find all survivors as quickly as "
                "possible. Assign drones to sectors, manage battery recalls, and "
                "ensure full map coverage."
            ),
            backstory=(
                "You are an experienced SAR (Search and Rescue) operations commander. "
                "You excel at reading mission state, distributing drones across sectors, "
                "and making smart trade-offs between exploration speed and battery "
                "management. You always think step-by-step before issuing orders."
            ),
            tools=lc_tools,
            llm=manager_model,
            verbose=True,
            allow_delegation=True,
        )

        pilot = CrewAgent(
            role="Drone Pilot",
            goal=(
                "Execute tactical drone operations efficiently: move drones, "
                "scan for survivors, recall low-battery drones, and charge them."
            ),
            backstory=(
                "You are a skilled drone pilot who follows the Commander's orders "
                "precisely. You use move_and_scan to navigate and detect survivors, "
                "recall_to_base when battery is low, and charge_drone at stations. "
                "You always explain your chain of thought before each tool call."
            ),
            tools=lc_tools,
            llm=agent_model,
            verbose=True,
            allow_delegation=False,
        )

        # ── Define Tasks ─────────────────────────────────────────────────
        strategy_task = CrewTask(
            description=(
                f"Analyze the current mission state and decide the optimal plan "
                f"for this tick.\n\n"
                f"CONTEXT:\n{context_block}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. First call get_mission_state to see live data.\n"
                f"2. Identify which drones need to charge (battery < 20%%).\n"
                f"3. Identify which drones should move to uncovered sectors.\n"
                f"4. MANDATE: Every drone MUST move at least one step this tick to avoid staying stuck behind buildings.\n"
                f"5. Provide a clear action plan for the Pilot to execute.\n"
                f"6. Think step-by-step before deciding."
            ),
            expected_output=(
                "A clear, structured action plan listing each drone_id and "
                "what tool to call with what arguments, plus reasoning."
            ),
            agent=commander,
        )

        execution_task = CrewTask(
            description=(
                f"Execute the Commander's action plan by calling the appropriate tools.\n\n"
                f"CONTEXT:\n{context_block}\n\n"
                f"RULES:\n"
                f"- For each drone that needs to move+scan: call move_and_scan with "
                f"  JSON arg: {{\"drone_id\": \"drone_X\", \"x\": N, \"y\": N}}\n"
                f"- For low-battery drones: call recall_to_base with the drone_id\n"
                f"- For drones at base: call charge_drone with the drone_id\n"
                f"- MANDATE: Ensure every drone moves at least one step to avoid getting stuck or staying idle.\n"
                f"- Think step-by-step before each tool call.\n"
                f"- Execute ALL drone actions for this tick."
            ),
            expected_output=(
                "A summary of all tool calls made and their results."
            ),
            agent=pilot,
        )

        # ── Run the Crew ─────────────────────────────────────────────────
        try:
            crew = Crew(
                agents=[commander, pilot],
                tasks=[strategy_task, execution_task],
                process=Process.sequential,
                verbose=True,
            )
            result = crew.kickoff()
            print("\n[CREWAI RESULT]")
            output_str = str(result.raw) if hasattr(result, "raw") else str(result)
            self._print_wrapped("  ", output_str[:500])
            
            # Local models might return raw JSON function calls instead of letting CrewAI execute them
            parsed_tools = False
            try:
                data = json.loads(output_str)
                tool_calls = data.get("tool_calls", [])
                if tool_calls:
                    print("[INFO] Local model returned raw tool calls. Executing manually...")
                    for call in tool_calls:
                        func = call.get("function", {})
                        t_name = func.get("name")
                        t_args = func.get("arguments", {})
                        if t_name == "move_and_scan" and "args_json" in t_args:
                            try:
                                t_args = json.loads(t_args["args_json"])
                            except Exception:
                                pass
                        if t_name and t_name != "get_mission_state":
                            print(f"  [MANUAL EXEC] {t_name}({t_args})")
                            self._exec_fallback(t_name, t_args)
                            parsed_tools = True
            except Exception:
                pass
            
            if not parsed_tools and "move_and_scan" not in output_str:
                raise Exception("Agent did not execute any movement tools.")
                
        except Exception as exc:
            print(f"[WARN] CrewAI execution failed or stalled: {exc}")

        self._pause()

    def _exec_fallback(self, name: str, args: Dict[str, Any]) -> None:
        """Direct tool execution for fallback mode."""
        try:
            if name == "move_and_scan":
                self.tools.move_and_scan(
                    str(args.get("drone_id", "")),
                    int(args.get("x", 0)),
                    int(args.get("y", 0)),
                )
            elif name == "thermal_scan":
                self.tools.thermal_scan(str(args.get("drone_id", "")))
            elif name == "recall_to_base":
                self.tools.recall_to_base(str(args.get("drone_id", "")))
            elif name == "charge_drone":
                self.tools.charge_drone(str(args.get("drone_id", "")))
        except Exception as exc:
            print(f"    [ERROR] {name} failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  FRAME CAPTURE MANAGER — renders Mesa grid → PIL images → MP4 video
# ═══════════════════════════════════════════════════════════════════════════

class FrameCaptureManager:
    """Captures each simulation tick as a layered PIL image.

    Layers (drawn in order):
        0  — Sector tile background (scanned / unscanned colour)
        1  — Buildings (pink = high obstacle, sky-blue = low flyable)
        1b — Charging stations (gold square + ⚡)
        2  — Survivors (red = detected, grey = hidden)
        2b — Movement arrows (orange dot + arrow char)
        3  — Drones (green/orange/red based on battery, + 🚁)

    Usage:
        mgr = FrameCaptureManager(model, cell_px=40)
        mgr.capturing = True        # start recording
        mgr.capture_frame()         # call each tick
        mgr.capturing = False       # stop recording
        mgr.export_to_mp4("out.mp4", fps=4)
    """

    def __init__(self, model: "DroneRescueModel", cell_px: int = 40):
        self.model = model
        self.cell_px = cell_px
        self.frames: list = []      # List[PIL.Image.Image]
        self.capturing: bool = False

    # ── colour helpers ────────────────────────────────────────────────

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
        h = hex_str.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    # ── main render ───────────────────────────────────────────────────

    def render_frame(self) -> Any:
        """Render the current grid state as a PIL Image (returns None if Pillow missing)."""
        if Image is None or ImageDraw is None:
            return None

        cp = self.cell_px
        w = self.model.grid.width
        h = self.model.grid.height
        img = Image.new("RGB", (w * cp, h * cp), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Try to get a reasonable font; fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", max(10, cp // 3))
            small_font = ImageFont.truetype("arial.ttf", max(8, cp // 4))
        except Exception:
            font = ImageFont.load_default()
            small_font = font

        # ── Layer 0: Sector tiles ────────────────────────────────────
        for (cx, cy), tile in self.model.tile_map.items():
            base_col, scanned_col = SECTOR_COLORS[tile.sector_id]
            colour = self._hex_to_rgb(scanned_col if tile.scanned else base_col)
            # Mesa grid: y=0 is bottom, PIL: y=0 is top → flip
            px = cx * cp
            py = (h - 1 - cy) * cp
            draw.rectangle([px, py, px + cp - 1, py + cp - 1], fill=colour)

        # ── Layer 1: Buildings ───────────────────────────────────────
        for b in self.model.building_list:
            bx, by = b.pos
            px = bx * cp
            py = (h - 1 - by) * cp
            margin = int(cp * 0.04)
            if b.is_obstacle:
                fill = self._hex_to_rgb("#FF69B4")
                draw.rectangle([px + margin, py + margin, px + cp - 1 - margin, py + cp - 1 - margin], fill=fill)
                txt = f"{b.height}F"
                draw.text((px + cp // 4, py + cp // 4), txt, fill=(255, 255, 255), font=small_font)
            else:
                fill = self._hex_to_rgb("#87CEEB")
                margin2 = int(cp * 0.10)
                draw.rectangle([px + margin2, py + margin2, px + cp - 1 - margin2, py + cp - 1 - margin2], fill=fill)
                txt = f"{b.height}F"
                draw.text((px + cp // 4, py + cp // 4), txt, fill=(51, 51, 51), font=small_font)

        # ── Layer 1b: Charging stations ──────────────────────────────
        for bp in BASE_POSITIONS:
            bx, by = bp
            px = bx * cp
            py = (h - 1 - by) * cp
            margin = int(cp * 0.08)
            draw.rectangle([px + margin, py + margin, px + cp - 1 - margin, py + cp - 1 - margin],
                           fill=self._hex_to_rgb("#ffd700"))
            draw.text((px + cp // 3, py + cp // 4), "E", fill=(0, 0, 0), font=font)

        # ── Layer 2: Survivors ───────────────────────────────────────
        for a in self.model.schedule.agents:
            if not isinstance(a, SurvivorAgent):
                continue
            sx, sy = a.pos
            px = sx * cp + cp // 2
            py = (h - 1 - sy) * cp + cp // 2
            if a.detected:
                r = int(cp * 0.45)
                draw.ellipse([px - r, py - r, px + r, py + r], fill=self._hex_to_rgb("#ff3333"))
                draw.text((px - r // 2, py - r // 2), "S", fill=(255, 255, 255), font=small_font)
            else:
                r = int(cp * 0.25)
                draw.ellipse([px - r, py - r, px + r, py + r], fill=self._hex_to_rgb("#9aa0a6"))

        # ── Layer 2b: Movement arrows ────────────────────────────────
        for a in self.model.schedule.agents:
            if not isinstance(a, MovementArrowAgent):
                continue
            ax, ay = a.pos
            px = ax * cp + cp // 2
            py = (h - 1 - ay) * cp + cp // 2
            r = int(cp * 0.30)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=self._hex_to_rgb("#ff8c00"))
            draw.text((px - r // 2, py - r // 2), a.arrow_char, fill=(255, 255, 255), font=small_font)

        # ── Layer 3: Drones ──────────────────────────────────────────
        for a in self.model.schedule.agents:
            if not isinstance(a, DroneAgent):
                continue
            dx, dy = a.pos
            px = dx * cp
            py = (h - 1 - dy) * cp
            sz = int(cp * 0.7)
            off = (cp - sz) // 2
            if a.battery > 50:
                fill = self._hex_to_rgb("#00aa00")
            elif a.battery > 20:
                fill = self._hex_to_rgb("#cc8800")
            else:
                fill = self._hex_to_rgb("#cc0000")
            draw.rectangle([px + off, py + off, px + off + sz, py + off + sz], fill=fill)
            draw.text((px + off + 2, py + off), a.unique_id, fill=(255, 255, 255), font=small_font)

        # ── HUD overlay: tick, survivors, battery ────────────────────
        tick = self.model.schedule.steps
        survivors = [a for a in self.model.schedule.agents if isinstance(a, SurvivorAgent)]
        found = len([s for s in survivors if s.detected])
        total = len(survivors)
        drones = [a for a in self.model.schedule.agents if isinstance(a, DroneAgent)]
        avg_bat = sum(d.battery for d in drones) / max(1, len(drones))
        hud = f"Tick {tick}  |  Survivors {found}/{total}  |  Avg Battery {avg_bat:.0f}%"
        draw.rectangle([0, 0, len(hud) * 8 + 16, 22], fill=(0, 0, 0, 180))
        draw.text((8, 4), hud, fill=(255, 255, 255), font=small_font)

        return img

    # ── capture / export ──────────────────────────────────────────────

    def capture_frame(self) -> None:
        """Capture current tick if recording is active."""
        if not self.capturing:
            return
        img = self.render_frame()
        if img is not None:
            self.frames.append(img)

    def clear_frames(self) -> None:
        """Discard all captured frames."""
        self.frames.clear()

    def export_to_mp4(self, output_path: str = "simulation_recording.mp4", fps: int = 4) -> str:
        """Concatenate captured frames into an MP4 video. Returns the output path."""
        if iio is None:
            raise RuntimeError("imageio is not installed. Run: pip install imageio av")
        if Image is None:
            raise RuntimeError("Pillow is not installed. Run: pip install Pillow")
        if not self.frames:
            raise RuntimeError("No frames captured. Run the simulation first.")

        import numpy as np  # type: ignore
        frame_arrays = [np.array(f.convert("RGB")) for f in self.frames]
        codec_attempts = [
            {"codec": "libx264", "out_pixel_format": "yuv420p"},
            {"codec": "mpeg4", "out_pixel_format": "yuv420p"},
        ]
        export_errors = []

        for codec_options in codec_attempts:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                iio.imwrite(output_path, frame_arrays, fps=fps, plugin="pyav", **codec_options)
                codec_name = codec_options["codec"]
                print(f"[VIDEO] Exported {len(self.frames)} frames → {output_path} ({fps} FPS, codec={codec_name})")
                return output_path
            except Exception as exc:
                export_errors.append(f"{codec_options['codec']}: {exc}")

        raise RuntimeError(
            "MP4 export failed for all supported codecs. "
            + " | ".join(export_errors)
        )


# ═══════════════════════════════════════════════════════════════════════════
#  MESA AGENTS  — only drones + minimal environment markers
# ═══════════════════════════════════════════════════════════════════════════

class SectorTileAgent(Agent):
    """Background tile — shows sector colour and whether it has been scanned."""
    def __init__(self, uid, model, sector_id):
        super().__init__(uid, model)  # type: ignore[call-arg]
        self.sector_id = sector_id
        self.scanned = False

    def step(self):
        pass


class SurvivorAgent(Agent):
    """Thermal signature placed in the disaster zone.  Hidden until scanned."""
    def __init__(self, uid, model):
        super().__init__(uid, model)  # type: ignore[call-arg]
        self.detected = False

    def step(self):
        pass


class ChargingStationAgent(Agent):
    """Charging pad at a grid corner."""
    def __init__(self, uid, model):
        super().__init__(uid, model)  # type: ignore[call-arg]

    def step(self):
        pass


class DroneAgent(Agent):
    """Physical drone — ALL behaviour is driven externally via MCP tools.
    step() is intentionally empty."""
    def __init__(self, uid, model, speed=3):
        super().__init__(uid, model)  # type: ignore[call-arg]
        self.speed = speed
        self.battery = 100.0
        self.disabled = False

    def step(self):
        pass


class BuildingAgent(Agent):
    """City building with a height category.

    height_category:
        'high'  — skyscrapers; drones CANNOT fly through these cells.
        'low'   — low-rise buildings; drones CAN fly over them.

    The height value (int, 1-10) determines the category:
        height >= HIGH_BUILDING_THRESHOLD  →  'high'
        height <  HIGH_BUILDING_THRESHOLD  →  'low'
    """
    HIGH_BUILDING_THRESHOLD = 6  # floors; >= this is a skyscraper

    def __init__(self, uid, model, height: int = 1):
        super().__init__(uid, model)  # type: ignore[call-arg]
        self.height = height  # 1-10 floors
        self.height_category = (
            "high" if height >= self.HIGH_BUILDING_THRESHOLD else "low"
        )

    @property
    def is_obstacle(self) -> bool:
        """True when drones must dodge this building."""
        return self.height_category == "high"

    def step(self):
        pass


class MovementArrowAgent(Agent):
    """Directional arrow placed at a drone's previous position after it moves."""

    # Direction vector → Unicode arrow
    ARROW_MAP = {
        (0, 1): "\u2191",    # ↑  (up)
        (0, -1): "\u2193",   # ↓  (down)
        (-1, 0): "\u2190",   # ←  (left)
        (1, 0): "\u2192",    # →  (right)
        (1, 1): "\u2197",    # ↗  (up-right)
        (1, -1): "\u2198",   # ↘  (down-right)
        (-1, -1): "\u2199",  # ↙  (down-left)
        (-1, 1): "\u2196",   # ↖  (up-left)
    }

    def __init__(self, uid, model, drone_id: str, direction: Tuple[int, int]):
        super().__init__(uid, model)  # type: ignore[call-arg]
        self.drone_id = drone_id
        self.direction = direction  # (dx, dy) normalised to -1/0/1

    @property
    def arrow_char(self) -> str:
        return self.ARROW_MAP.get(self.direction, "·")

    def step(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  UI LEGEND
# ═══════════════════════════════════════════════════════════════════════════

class Legend(TextElement):
    def render(self, model):
        drones = [a for a in model.schedule.agents
                  if isinstance(a, DroneAgent)]
        survivors = [a for a in model.schedule.agents
                     if isinstance(a, SurvivorAgent)]
        found = len([s for s in survivors if s.detected])
        total_s = len(survivors)
        sectors_done = 0
        for sid, sdef in SECTOR_DEFS.items():
            origin: Tuple[int, int] = cast(Tuple[int, int], sdef["origin"])
            ox, oy = origin
            size: Tuple[int, int] = cast(Tuple[int, int], sdef["size"])
            w, h = size
            total = w * h
            scanned = sum(
                1
                for xi in range(ox, ox + w)
                for yi in range(oy, oy + h)
                if model.tile_map.get((xi, yi)) and model.tile_map[(xi, yi)].scanned
            )
            if total > 0 and scanned / total >= 0.999:
                sectors_done += 1

        # Building stats
        high_count = len([b for b in model.building_list if b.is_obstacle])
        low_count = len([b for b in model.building_list if not b.is_obstacle])

        drone_rows = ""
        for d in drones:
            bc = ("#00aa00" if d.battery > 50
                  else "#cc8800" if d.battery > 20 else "#cc0000")
            drone_rows += (
                f"<div>🚁 {d.unique_id}: "
                f"<span style='color:{bc};font-weight:bold;'>"
                f"{d.battery:.0f}%</span></div>"
            )

        return f"""
        <div style="font-family:Arial;line-height:1.6;padding-left:10px;">
          <h3>🤖 MCP Drone Command</h3>
          <div style="display:flex;gap:10px;">
            <div style="flex:1;font-size:13px;">
                <div style="padding:8px;background:#e8f4fd;border-radius:5px;
                            border-left:4px solid #0078d4;margin-bottom:8px;">
                  <strong>Fleet</strong><br/>{drone_rows}
                </div>
                <div style="padding:8px;background:#f0f0f0;border-radius:5px;
                            margin-bottom:8px;">
                  <strong>Mission</strong><br/>
                  Sectors: {sectors_done}/6<br/>
                  Survivors: {found}/{total_s}
                </div>
                <div style="padding:8px;background:#fff3e0;border-radius:5px;
                            border-left:4px solid #ff6600;margin-bottom:8px;">
                  <strong>🏙️ City Buildings</strong><br/>
                  <span style="color:#FF69B4;font-weight:bold;">■</span> High (obstacle): {high_count}<br/>
                  <span style="color:#87CEEB;font-weight:bold;">■</span> Low (flyable): {low_count}
                </div>
            </div>
            <div style="flex:1;padding:8px;background:#fafafa;border-radius:5px;
                        border-left:4px solid #ccc;margin-bottom:8px;">
                <h4 style="margin-top:0;">Sector Colours</h4>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                            gap:2px;font-size:12px;text-align:center;">
                  <div style="background:#e8f0fe;padding:2px 4px;font-weight:bold;">NW</div>
                  <div style="background:#e6f4ea;padding:2px 4px;font-weight:bold;">N</div>
                  <div style="background:#fef7e0;padding:2px 4px;font-weight:bold;">NE</div>
                  <div style="background:#fce8e6;padding:2px 4px;font-weight:bold;">SW</div>
                  <div style="background:#f3e8fd;padding:2px 4px;font-weight:bold;">S</div>
                  <div style="background:#fff3e0;padding:2px 4px;font-weight:bold;">SE</div>
                </div>
                <div style="margin-top:12px;font-size:13px;line-height:1.8;">
                  ⚡ Charging station<br/>
                  🆘 Survivor (detected)<br/>
                  ⚪ Survivor (undetected)<br/>
                  🚁 Drone (colour = battery)<br/>
                  <span style="color:#FF69B4;">🏙️</span> High building (PINK = blocked)<br/>
                  <span style="color:#87CEEB;">🏠</span> Low building (SKY BLUE = flyable)
                </div>
            </div>
          </div>
        </div>
        """

class MovementDashboard(TextElement):
    def render(self, model):
        logs = getattr(model, 'movement_history', [])
        log_rows = ""
        # Display the most recent 100 movements to keep the UI smooth but scrollable
        recent_logs = logs[-100:]
        
        for entry in reversed(recent_logs):
            if len(entry) == 5:
                tick, d_id, p_from, p_to, reason = entry
            else:
                tick, d_id, p_from, p_to = entry
                reason = "Moving"
                
            short_reason = reason if len(reason) < 120 else reason[:117] + "..."

            log_rows += (
                f"<div style='border-bottom: 1px solid #e0e0e0; padding: 6px 0; margin: 0; font-size: 13px; display: flex; justify-content: space-between;'>"
                f"<div>"
                f"<span style='color:#a0a0a0; font-family: monospace; font-size: 11px; margin-right: 8px;'>[T:{tick:03d}]</span> "
                f"<strong style='color:#333; margin-right: 4px;'>{d_id}</strong> <span style='color:#666;'>moved</span> <span style='color:#0078d4;'>{p_from}</span> &rarr; <span style='color:#28a745;'>{p_to}</span>"
                f"</div>"
                f"<div style='color: #6c757d; font-style: italic; font-size: 12px; max-width: 150px; text-align: right;' title='{reason}'>{short_reason}</div>"
                f"</div>"
            )
            
        if not log_rows:
            log_rows = "<div style='color: #888; font-style: italic; font-size: 13px; padding: 8px; text-align: center;'>Awaiting drone deployment...</div>"
            
        return f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-left: 10px; margin-top: 10px; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; height: 350px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background: white;">
            <div style="background: #2c3e50; color: white; padding: 10px 15px; font-weight: 600; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
                <span>🛰️ Drone Movement Log</span>
                <span style="font-size: 11px; font-weight: normal; opacity: 0.8; background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 10px;">Live Updates</span>
            </div>
            <div style="padding: 0 15px; overflow-y: auto; flex: 1; background: #fafafa;">
                {log_rows}
            </div>
        </div>
        """

# ═══════════════════════════════════════════════════════════════════════════
#  MESA MODEL
# ═══════════════════════════════════════════════════════════════════════════

class DroneRescueModel(Model):
    def __init__(self, width=24, height=16,
                 num_drones=4, num_survivors=12,
                 scenario: str = "A: Palu city",
                 simulate_ai: bool = True,
                 ai_delay_s: float = 0.15,
                 use_gemini_ai: bool = False,
                 gemini_model: str = "gemini-2.5-flash",
                 use_ollama_ai: bool = False,
                 ollama_model: str = "llama3.1",
                 use_crew_ai: bool = False,
                 crew_ai_model: str = "ollama/llama3.1",
                 use_langgraph_ai: bool = False,
                 langgraph_model: str = "qwen/qwen3-14b"):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.num_drones = num_drones
        self.num_survivors = num_survivors
        self.scenario = scenario
        self.simulate_ai = simulate_ai
        self.ai_delay_s = float(ai_delay_s or 0.0)
        self.use_gemini_ai = bool(use_gemini_ai)
        self.gemini_model = str(gemini_model or "gemini-2.5-flash")
        self.use_ollama_ai = bool(use_ollama_ai)
        self.ollama_model = str(ollama_model or "llama3.1")
        self.use_crew_ai = bool(use_crew_ai)
        self.crew_ai_model = str(crew_ai_model or "ollama/llama3.1")
        self.use_langgraph_ai = bool(use_langgraph_ai)
        self.langgraph_model = str(langgraph_model or "qwen/qwen3-14b")
        self.movement_history: List[str] = []
        self._tools: InUiToolServer = InUiToolServer(self)
        self._ai: Optional[SimpleAiController] = None
        self._gemini_ai: Optional[GeminiAiController] = None
        self._ollama_ai: Optional[OllamaAiController] = None
        self._crew_ai: Optional[CrewAiController] = None
        self._langgraph_ai = None

        # ── sector background tiles ──
        self.tile_map: Dict[Tuple, SectorTileAgent] = {}
        tid: int = 0
        for x in range(width):
            for y in range(height):
                sid = pos_to_sector_id(x, y)
                if sid is None:
                    continue
                t = SectorTileAgent(f"tile_{tid}", self, sid)
                self.schedule.add(t)
                self.grid.place_agent(t, (x, y))
                self.tile_map[(x, y)] = t
                tid = tid + 1  # type: ignore[operator]

        # ── charging stations (4 corners) ──
        for i, bp in enumerate(BASE_POSITIONS):
            cs = ChargingStationAgent(f"base_{i}", self)
            self.schedule.add(cs)
            self.grid.place_agent(cs, bp)

        # ── scenario config ──
        spec = SCENARIOS.get(self.scenario, None)
        if spec and isinstance(spec.get("seed"), int):
            rng = random.Random(int(spec["seed"]))
        else:
            rng = random.Random(0)

        has_buildings = bool(spec and spec.get("has_buildings", False))

        # ── city buildings (only for building-enabled scenarios) ──
        self.building_map: Dict[Tuple[int, int], BuildingAgent] = {}
        self.building_list: List[BuildingAgent] = []

        if has_buildings:
            bid: int = 0
            # ── Centered 12-block city cluster ──
            # 4 columns of 3-wide blocks, 3 rows of 2-tall blocks
            # separated by 2-cell roads.
            # Total X = 4*3 + 3*2 = 18, centered in 24 → start at 3
            # Total Y = 3*2 + 2*2 = 10, centered in 16 → start at 3
            CLUSTER_X_MIN, CLUSTER_X_MAX = 3, 20   # inclusive
            CLUSTER_Y_MIN, CLUSTER_Y_MAX = 3, 12   # inclusive
            BLOCK_W, ROAD_W = 3, 2   # block width and road width
            BLOCK_H, ROAD_H = 2, 2   # block height and road height
            PERIOD_X = BLOCK_W + ROAD_W   # 5
            PERIOD_H = BLOCK_H + ROAD_H   # 4

            for x in range(width):
                for y in range(height):
                    # Only place buildings inside the cluster bounds
                    if not (CLUSTER_X_MIN <= x <= CLUSTER_X_MAX and
                            CLUSTER_Y_MIN <= y <= CLUSTER_Y_MAX):
                        continue
                    # Skip charging-station corners
                    if (x, y) in BASE_POSITIONS:
                        continue
                    # Roads inside the cluster (2-cell wide)
                    lx = x - CLUSTER_X_MIN   # local x: 0..17
                    ly = y - CLUSTER_Y_MIN   # local y: 0..9
                    if lx % PERIOD_X >= BLOCK_W:   # road columns
                        continue
                    if ly % PERIOD_H >= BLOCK_H:   # road rows
                        continue

                    # Determine which block (col, row) this cell belongs to
                    bcol = lx // PERIOD_X   # 0..3
                    brow = ly // PERIOD_H   # 0..2

                    # Distance from center of the 4x3 grid (1.5, 1.0)
                    dist = abs(bcol - 1.5) + abs(brow - 1.0)
                    # max dist = 1.5+1.0 = 2.5; inner blocks dist ~0.5

                    # Inner blocks → mostly High (Pink), outer → mostly Low (Sky Blue)
                    high_prob = max(0.2, min(0.8, 0.8 - dist * 0.24))

                    if spec and spec.get("all_low_buildings", False):
                        h = rng.choice([1, 2, 3, 4, 5])     # ALL Low building
                    elif rng.random() < high_prob:
                        h = rng.choice([6, 7, 8, 9, 10])   # High building
                    else:
                        h = rng.choice([1, 2, 3, 4, 5])     # Low building

                    b = BuildingAgent(f"bld_{bid}", self, height=h)
                    self.schedule.add(b)
                    self.grid.place_agent(b, (x, y))
                    self.building_map[(x, y)] = b
                    self.building_list.append(b)
                    bid = bid + 1  # type: ignore[operator]

        # ── survivors ──
        if has_buildings:
            # City scenario: survivors go INSIDE buildings
            final_positions: List[Tuple[int, int]] = []
            if spec and spec.get("survivor_positions"):
                # Use specified positions if provided
                final_positions = [
                    tuple(p) for p in spec["survivor_positions"]
                    if isinstance(p, (list, tuple)) and len(p) == 2
                ]
            else:
                building_cells = list(self.building_map.keys())
                rng.shuffle(building_cells)
                used: set = set()
                for bc in building_cells:
                    if len(final_positions) >= int(num_survivors):
                        break
                    if bc not in used:
                        final_positions.append(bc)
                        used.add(bc)
            for i in range(min(int(num_survivors), len(final_positions))):
                s = SurvivorAgent(f"surv_{i}", self)
                self.schedule.add(s)
                self.grid.place_agent(s, final_positions[i])
        else:
            # Original scenarios A/B/C: survivors on open tiles
            positions: List[Tuple[int, int]] = []
            if spec and isinstance(spec.get("survivor_positions"), list):
                positions = [
                    tuple(p)
                    for p in spec["survivor_positions"]
                    if isinstance(p, (list, tuple)) and len(p) == 2
                ]
            if not positions:
                all_pos = [
                    (x, y)
                    for x in range(width)
                    for y in range(height)
                    if (x, y) not in BASE_POSITIONS
                ]
                rng.shuffle(all_pos)
                positions = cast(List[Tuple[int, int]], all_pos)[: int(num_survivors)]  # type: ignore[index]

            # Clamp, de-dupe, and avoid bases.
            uniq: List[Tuple[int, int]] = []
            seen: set = set()
            for (x, y) in positions:
                cx = max(0, min(width - 1, int(x)))
                cy = max(0, min(height - 1, int(y)))
                if (cx, cy) in BASE_POSITIONS:
                    continue
                if (cx, cy) in seen:
                    continue
                seen.add((cx, cy))
                uniq.append((cx, cy))

            for i in range(min(int(num_survivors), len(uniq))):
                s = SurvivorAgent(f"surv_{i}", self)
                self.schedule.add(s)
                self.grid.place_agent(s, uniq[i])

        # ── drones (start at charging stations) ──
        for i in range(num_drones):
            bp = BASE_POSITIONS[i % len(BASE_POSITIONS)]
            d = DroneAgent(f"d_{i}", self, speed=3)
            self.schedule.add(d)
            self.grid.place_agent(d, bp)

        # ── data collector ──
        self.datacollector = DataCollector(
            model_reporters={
                "SurvivorsFound": lambda m: len([
                    a for a in m.schedule.agents
                    if isinstance(a, SurvivorAgent) and a.detected]),
                "SectorsDone": lambda m: sum(
                    1
                    for sid, sdef in SECTOR_DEFS.items()
                    if (
                        sum(
                            1
                            for xi in range(cast(Tuple[int, int], sdef["origin"])[0], cast(Tuple[int, int], sdef["origin"])[0] + cast(Tuple[int, int], sdef["size"])[0])
                            for yi in range(cast(Tuple[int, int], sdef["origin"])[1], cast(Tuple[int, int], sdef["origin"])[1] + cast(Tuple[int, int], sdef["size"])[1])
                            if m.tile_map.get((xi, yi)) and m.tile_map[(xi, yi)].scanned
                        )
                        / max(1, (cast(Tuple[int, int], sdef["size"])[0] * cast(Tuple[int, int], sdef["size"])[1]))
                    )
                    >= 0.999
                ),
                "AvgBattery": lambda m: round(
                    sum(a.battery for a in m.schedule.agents
                        if isinstance(a, DroneAgent))
                    / max(1.0, float(m.num_drones)), 1),  # type: ignore[call-overload]
            }
        )

        self.running = True

        # ── Frame capture manager (for video export) ──
        self._frame_capture: FrameCaptureManager = FrameCaptureManager(self)

        # In-UI AI controller (keeps UI and "AI actions" in the same process)
        self._tools = InUiToolServer(self)
        if self.simulate_ai:
            if self.use_langgraph_ai and LangGraphOpenRouterAiController is not None:
                self._langgraph_ai = LangGraphOpenRouterAiController(
                    self._tools,
                    model_name=self.langgraph_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._crew_ai = None
                self._gemini_ai = None
                self._ollama_ai = None
                self._ai = None
            elif self.use_crew_ai:
                self._crew_ai = CrewAiController(
                    self._tools,
                    model_name=self.crew_ai_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._gemini_ai = None
                self._ollama_ai = None
                self._ai = None
            elif self.use_gemini_ai:
                self._gemini_ai = GeminiAiController(
                    self._tools,
                    model_name=self.gemini_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._ai = None
                self._ollama_ai = None
                self._crew_ai = None
            elif self.use_ollama_ai:
                self._ollama_ai = OllamaAiController(
                    self._tools,
                    model_name=self.ollama_model,
                    action_delay_s=self.ai_delay_s,
                )
                self._gemini_ai = None
                self._ai = None
                self._crew_ai = None
            else:
                self._ai = SimpleAiController(self._tools, action_delay_s=self.ai_delay_s)
                self._gemini_ai = None
                self._ollama_ai = None
                self._crew_ai = None
        else:
            self._ai = None
            self._gemini_ai = None
            self._ollama_ai = None
            self._crew_ai = None
            self._langgraph_ai = None

    def get_drone(self, drone_id: str) -> DroneAgent:
        for a in self.schedule.agents:
            if isinstance(a, DroneAgent) and a.unique_id == drone_id:
                return a
        raise ValueError(f"Drone not found: {drone_id}")

    def is_high_building(self, pos: Tuple[int, int]) -> bool:
        """Return True if the cell contains a HIGH building (obstacle)."""
        b = self.building_map.get(pos)
        return b is not None and b.is_obstacle

    def get_buildings_info(self) -> List[Dict[str, Any]]:
        """Return a compact list of all buildings for the AI controller."""
        return [
            {
                "id": b.unique_id,
                "pos": list(b.pos),
                "height": b.height,
                "category": b.height_category,
                "obstacle": b.is_obstacle,
            }
            for b in self.building_list
        ]

    def trigger_voting(self, drone_id: str) -> Dict[str, Any]:
        """Programmatic method to trigger a voting round for a specific drone."""
        drones = [a for a in self.schedule.agents if isinstance(a, DroneAgent)]
        drone_ids = [d.unique_id for d in drones]
        
        if drone_id not in drone_ids:
            return {"error": f"Drone {drone_id} not found. Available: {drone_ids}"}
        
        voting_result = self._tools.voting_simulator.execute_voting_round(drone_id)
        return voting_result

    def step(self):
        # Reset per-tick state so each drone can move once this tick
        self._tools.reset_tick_state()
        
        # ── Handle voting execution if in progress ──
        voting_sim = self._tools.voting_simulator
        if voting_sim.state == "VOTING_COMPLETE":
            if voting_sim.winning_action and voting_sim.idle_drone_id:
                try:
                    action = voting_sim.winning_action
                    drone_id = action["drone_id"]
                    x = action["x"]
                    y = action["y"]
                    reason = action["reason"]
                    result = self._tools.move_and_scan(drone_id, x, y, reason=reason)
                    log_to_file(
                        f"[T:{self.schedule.steps:03d}] VOTING: {voting_sim.idle_drone_id} → Action: {reason} | "
                        f"Tally: {voting_sim.vote_tally} | Winner: Sector {voting_sim.winning_sector}"
                    )
                except Exception as exc:
                    log_to_file(f"[T:{self.schedule.steps:03d}] VOTING ERROR: {str(exc)}")
            voting_sim.reset_voting_state()
        
        langgraph_ai = self._langgraph_ai
        crew_ai = self._crew_ai
        gemini_ai = self._gemini_ai
        ollama_ai = self._ollama_ai
        ai = self._ai
        if langgraph_ai is not None:
            langgraph_ai.think_and_act()
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
        
        # ── Check for completion (All sectors scanned) ──
        ms = self._tools.get_mission_state()
        sectors = ms.get("sectors", [])
        all_sectors_scanned = len(sectors) > 0 and all(float(s.get("coverage_pct", 0)) >= 99.0 for s in sectors)
        
        print(f"[DEBUG step] all_sectors_scanned: {all_sectors_scanned}, running: {self.running}, sectors_len: {len(sectors)}")

        if all_sectors_scanned and self.running:
            self.running = False
            surv_found = ms.get("survivors_found", 0)
            surv_total = ms.get("actual_survivors_total_hidden", 0)
            accuracy = (surv_found / max(1, surv_total)) * 100
            print("\n" + "═" * 72)
            print("🚀 MISSION ACCOMPLISHED: All sectors have been 100% scanned!")
            print(f"📊 Accuracy: Found {surv_found}/{surv_total} survivors ({accuracy:.1f}%)")
            print("═" * 72 + "\n")
            log_to_file(f"[MISSION COMPLETE] All sectors scanned! Found: {surv_found}/{surv_total} ({accuracy:.1f}%)")

            # Show a pop-up message box
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw() # Hide the main window
                root.attributes('-topmost', True) # Bring to front
                messagebox.showinfo(
                    "Mission Accomplished",
                    f"All grids have been searched!\n\n"
                    f"Survivors Found: {surv_found}\n"
                    f"Actual Survivors: {surv_total}\n"
                    f"Accuracy: {accuracy:.1f}%"
                )
                root.destroy()
            except Exception as e:
                print(f"Could not show pop-up: {e}")

        # ── Capture frame for video export (after all agents have moved) ──
        self._frame_capture.capture_frame()


# ═══════════════════════════════════════════════════════════════════════════
#  PORTRAYAL
# ═══════════════════════════════════════════════════════════════════════════

def portrayal(agent):
    if agent is None:
        return None

    if isinstance(agent, SectorTileAgent):
        base, scanned = SECTOR_COLORS[agent.sector_id]
        return {"Shape": "rect", "w": 1, "h": 1, "Filled": True,
                "Layer": 0, "Color": scanned if agent.scanned else base}

    if isinstance(agent, MovementArrowAgent):
        return {"Shape": "circle", "r": 0.35, "Filled": True,
                "Layer": 2, "Color": "#ff8c00",
                "text": agent.arrow_char, "text_color": "white"}

    if isinstance(agent, BuildingAgent):
        # High buildings (skyscrapers) = BUBBLEGUM PINK, Low buildings = SKY BLUE
        if agent.is_obstacle:
            return {"Shape": "rect", "w": 0.92, "h": 0.92, "Filled": True,
                    "Layer": 1, "Color": "#FF69B4",
                    "text": f"🏙️{agent.height}F", "text_color": "white"}
        else:
            return {"Shape": "rect", "w": 0.80, "h": 0.80, "Filled": True,
                    "Layer": 1, "Color": "#87CEEB",
                    "text": f"🏠{agent.height}F", "text_color": "#333333"}

    if isinstance(agent, ChargingStationAgent):
        return {"Shape": "rect", "w": 0.85, "h": 0.85, "Filled": True,
                "Layer": 1, "Color": "#ffd700", "text": "⚡",
                "text_color": "black"}

    if isinstance(agent, SurvivorAgent):
        if agent.detected:
            return {"Shape": "circle", "r": 0.45, "Filled": True,
                    "Layer": 2, "Color": "#ff3333", "text": "🆘",
                    "text_color": "white"}
        # Hidden survivors must still be visible enough to debug scenarios.
        return {"Shape": "circle", "r": 0.25, "Filled": True,
                "Layer": 2, "Color": "#9aa0a6"}

    if isinstance(agent, DroneAgent):
        if agent.battery > 50:
            c = "#00aa00"
        elif agent.battery > 20:
            c = "#cc8800"
        else:
            c = "#cc0000"
        return {"Shape": "rect", "w": 0.7, "h": 0.7, "Filled": True,
                "Layer": 3, "Color": c, "text": "🚁",
                "text_color": "white"}

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  SERVER & LAUNCH
# ═══════════════════════════════════════════════════════════════════════════

# Larger canvas so sectors + icons are readable.
grid = CanvasGrid(portrayal, 24, 16, 980, 650)
chart = ChartModule(
    [{"Label": "SurvivorsFound", "Color": "Green"},
     {"Label": "SectorsDone", "Color": "Blue"},
     {"Label": "AvgBattery", "Color": "Orange"}],
    data_collector_name="datacollector",
)
legend = Legend()
movement_board = MovementDashboard()

server = ModularServer(
    DroneRescueModel,
    [legend, movement_board, grid, chart],
    "Drone Fleet Search & Rescue — MCP + Chain-of-Thought 🤖",
    {
        "width": 24,
        "height": 16,
        "scenario": Choice(
            "Scenario",
            value="A: Palu city",
            choices=list(SCENARIOS.keys()),
            description="Pick a scenario.",
        ),
        "num_drones": Slider("Drones", value=4, min_value=3, max_value=5, step=1),
        "num_survivors": Slider("Survivors", value=12, min_value=5, max_value=20, step=1),
        "simulate_ai": Checkbox("Simulate (AI drives drones)", value=True),
        "use_gemini_ai": Checkbox("Use Gemini (real LLM agent)", value=False),
        "gemini_model": Choice(
            "Gemini model",
            value="gemini-2.5-flash",
            choices=["gemini-2.5-flash", "gemini-2.5-pro"],
            description="Requires GEMINI_API_KEY in environment (.env supported).",
        ),
        "use_ollama_ai": Checkbox("Use Ollama (local Edge AI)", value=False),
        "ollama_model": Choice(
            "Ollama model",
            value="llama3.1",
            choices=["llama3.1", "qwen2.5:3b", "qwen2.5:14b", "qwen3:14b"],
            description="Requires Ollama to be running locally.",
        ),
        "use_crew_ai": Checkbox("Use CrewAI (hierarchical agents)", value=False),
        "crew_ai_model": Choice(
            "CrewAI model",
            value="ollama/hierarchical (14B/3B)",
            choices=["ollama/hierarchical (14B/3B)", "ollama/qwen2.5:3b", "ollama/qwen2.5:14b", "ollama/qwen3:14b", "openai/gpt-4o-mini"],
            description="Model for CrewAI agents. 'ollama/' prefix = local, 'openai/' = cloud.",
        ),
        "use_langgraph_ai": Checkbox("Use LangGraph (parallel Commander+Operator)", value=False),
        "langgraph_model": Choice(
            "LangGraph model",
            value="qwen/qwen3-14b",
            choices=["qwen/qwen3-14b", "qwen/qwen-2.5-7b-instruct", "openai/gpt-4o-mini", "google/gemini-2.0-flash-001"],
            description="Commander model for LangGraph. Requires OPENROUTER_API_KEY in environment.",
        ),
        "ai_delay_s": Slider("AI delay (sec)", value=0.15, min_value=0.0, max_value=1.5, step=0.05),
    },
)

if __name__ == "__main__":
    # Avoid WinError 10048 when a previous server is still running.
    import socket

    def _pick_free_port(preferred: List[int]) -> int:
        # First try a few well-known ports.
        for p in preferred:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # Bind to all interfaces (matches Tornado behavior).
                    s.bind(("", p))
                return p
            except OSError:
                continue
        # Fall back to an ephemeral free port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return int(s.getsockname()[1])

    server.port = _pick_free_port([8524, 8525, 8526, 8527])
    print(f"Launching MCP drone fleet server at http://127.0.0.1:{server.port}")
    server.launch()
