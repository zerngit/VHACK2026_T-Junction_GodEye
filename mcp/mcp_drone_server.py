"""
Official MCP server for the Drone Rescue Mesa simulation.

This server exposes the drone fleet capabilities as MCP tools using the
official `mcp` Python SDK (`mcp.server.fastmcp.FastMCP`).

The simulation state (a live `DroneRescueModel`) is kept server-side, and all
Agent↔Drone communication happens via MCP tool calls.

Phase 2 — External State Tracking
──────────────────────────────────
A **global state tracker** (`_state_tracker` dict) is maintained server-side.
Every mutating tool (`move_to`, `thermal_scan`, `recall_to_base`, `charge_drone`)
writes back into this tracker so the AI never needs to memorise grid history.
The `get_mission_state` tool returns a highly-compressed, flat JSON snapshot
that any LLM (cloud or edge) can parse with zero memory overhead.
"""

from __future__ import annotations

import os
import sys
# Ensure parent directory (V-Hack) is in sys.path so 'core' and others can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, cast

from mcp.server.fastmcp import Context, FastMCP  # type: ignore
from mcp.server.session import ServerSession  # type: ignore

from core.mesa_drone_rescue_mcp import (  # type: ignore
    BASE_POSITIONS,
    BATTERY_CHARGE_RATE,
    BATTERY_COST_MOVE,
    BATTERY_COST_SCAN,
    BATTERY_CRITICAL,
    BATTERY_FULL,
    SCAN_RADIUS,
    SECTOR_DEFS,
    DroneAgent,
    DroneRescueModel,
    SurvivorAgent,
    _sector_waypoints,
)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Global State Tracker  (memory lives on the MCP server)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MissionStateTracker:
    """
    Centralised, server-side state tracker that offloads memory from the LLM.

    Updated by every mutating tool call so the AI can query a single
    compressed snapshot via `get_mission_state()` instead of keeping its
    own history.
    """
    # Set of (x, y) tuples that have been scanned at least once.
    scanned_coords: Set[Tuple[int, int]] = field(default_factory=set)

    # Per-drone status: drone_id -> {pos, battery, disabled, speed}
    drone_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Survivor tracking
    total_survivors: int = 0
    survivors_found: int = 0
    survivor_locations: List[Dict[str, Any]] = field(default_factory=list)

    # Cumulative tool call counter (useful for the judges / analytics)
    tool_calls: int = 0

    # ── Sector assignment and waypoint progress ──
    # drone_id -> list of sector IDs assigned to this drone
    sector_assignments: Dict[str, List[int]] = field(default_factory=dict)
    # drone_id -> list of (x,y) waypoints remaining
    waypoint_queue: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    # drone_id -> (x,y) position the drone was at before returning to charge
    resume_position: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def sync_drone(self, drone: Any) -> None:
        """Refresh the registry entry for a single drone agent."""
        self.drone_registry[drone.unique_id] = {
            "pos": list(drone.pos),
            "battery": drone.battery,
            "disabled": drone.disabled,
        }

    def sync_all_drones(self, model: DroneRescueModel) -> None:
        """Full fleet refresh from the live Mesa model."""
        for a in model.schedule.agents:
            if isinstance(a, DroneAgent):
                self.sync_drone(cast(Any, a))

    def record_scan(self, cells: list, found_survivors: List[Dict]) -> None:
        """Mark cells as scanned and register newly detected survivors."""
        for cell in cells:
            self.scanned_coords.add(tuple(cell))
        for s in found_survivors:
            self.survivors_found += 1
            self.survivor_locations.append(s)

    def assign_sectors(self, num_drones: int) -> None:
        """
        Distribute sectors evenly among drones so they don't overlap.
        Must be called once after drones are discovered.
        """
        if self.sector_assignments:
            return  # already assigned

        drone_ids = sorted(self.drone_registry.keys())
        if not drone_ids:
            return

        # Clamp to actual drone count
        n = min(num_drones, len(drone_ids))
        sector_ids = sorted(SECTOR_DEFS.keys())  # [1,2,3,4,5,6]

        # Round-robin assign sectors to drones
        for i, sid in enumerate(sector_ids):
            did = drone_ids[i % n]
            if did not in self.sector_assignments:
                self.sector_assignments[did] = []
            self.sector_assignments[did].append(sid)

        # Build per-drone waypoint queues from assigned sectors
        for did in drone_ids[:n]:
            wps: List[Tuple[int, int]] = []
            for sid in self.sector_assignments.get(did, []):
                sdef = SECTOR_DEFS[sid]
                wps.extend(_sector_waypoints(sdef["origin"], sdef["size"]))
            self.waypoint_queue[did] = wps

    def get_next_waypoint(self, drone_id: str) -> Tuple[int, int] | None:
        """Return the next unvisited waypoint for this drone, or None."""
        q = self.waypoint_queue.get(drone_id, [])
        if q:
            return q[0]
        return None

    def pop_waypoint(self, drone_id: str) -> None:
        """Remove the current waypoint (drone has reached/scanned it)."""
        q = self.waypoint_queue.get(drone_id, [])
        if q:
            q.pop(0)

    def to_compressed_dict(self, model: DroneRescueModel) -> Dict[str, Any]:
        """
        Build a highly-compressed, flat JSON dict of the entire mission state.

        Designed for low-VRAM edge devices: no nested nesting deeper than 2
        levels, short keys, minimal footprint.
        """
        # Sector coverage (computed live from the Mesa model's tile_map)
        sectors = []
        for sid, sdef in SECTOR_DEFS.items():
            ox, oy = sdef["origin"]
            w, h = sdef["size"]
            total = w * h
            scanned = sum(
                1
                for xi in range(ox, ox + w)
                for yi in range(oy, oy + h)
                if model.tile_map.get((xi, yi)) and model.tile_map[(xi, yi)].scanned
            )
            pct = float(f"{scanned / total * 100:.1f}") if total > 0 else 0.0
            sectors.append({
                "id": sid,
                "name": sdef["name"],
                "origin": [ox, oy],
                "size": [w, h],
                "coverage_pct": pct,
            })

        return {
            "tick": model.schedule.steps,
            "tool_calls": self.tool_calls,
            "grid": [model.grid.width, model.grid.height],
            "bases": [list(b) for b in BASE_POSITIONS],
            "drones": self.drone_registry,
            "survivors_found": self.survivors_found,
            "survivors_total": self.total_survivors,
            "survivor_locations": self.survivor_locations,
            "scanned_cell_count": len(self.scanned_coords),
            "sectors": sectors,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  App State  (holds both the Mesa model and the mission tracker)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AppState:
    model: DroneRescueModel
    tracker: MissionStateTracker


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[AppState]:
    import os as _os
    _num_drones = int(_os.environ.get("NUM_DRONES", "4"))
    model = DroneRescueModel(width=24, height=16, num_drones=_num_drones, num_survivors=12)

    # Bootstrap the state tracker from the live model.
    tracker = MissionStateTracker()
    tracker.total_survivors = sum(
        1 for a in model.schedule.agents if isinstance(a, SurvivorAgent)
    )
    tracker.sync_all_drones(model)

    try:
        yield AppState(model=model, tracker=tracker)
    finally:
        pass


mcp = FastMCP(
    "MESA Drone Rescue (Official MCP)",
    instructions=(
        "This server simulates a drone fleet search-and-rescue mission on a 2D grid. "
        "Use tool discovery to find available drone tools. "
        "Do not assume drone IDs; call discover_drones first. "
        "Call get_mission_state for a compressed snapshot of the entire world."
    ),
    lifespan=lifespan,
    json_response=True,
)


# ── Convenience accessors ────────────────────────────────────────────────

def _model(ctx: Context[ServerSession, AppState]) -> DroneRescueModel:
    return ctx.request_context.lifespan_context.model


def _tracker(ctx: Context[ServerSession, AppState]) -> MissionStateTracker:
    return ctx.request_context.lifespan_context.tracker


def _drone(ctx: Context[ServerSession, AppState], drone_id: str) -> Any:
    model = _model(ctx)
    for a in model.schedule.agents:
        if isinstance(a, DroneAgent) and cast(Any, a).unique_id == drone_id:
            return cast(Any, a)
    raise ValueError(f"Drone not found: {drone_id}")


def _thermal_intensity_seed(model: DroneRescueModel, coord: Tuple[int, int]) -> float:
    """Generate noisy raw thermal intensity at a coordinate."""
    for a in model.grid.get_cell_list_contents([coord]):
        if isinstance(a, SurvivorAgent) and not cast(Any, a).detected:
            return round(max(0.0, min(1.0, 0.85 + random.uniform(-0.05, 0.10))), 2)
    if random.random() < 0.15:
        return round(max(0.0, min(1.0, 0.70 + random.uniform(-0.08, 0.08))), 2)
    return round(max(0.0, min(1.0, 0.30 + random.uniform(-0.08, 0.08))), 2)


def verify_signature(x: int, y: int, ctx: Context[ServerSession, AppState]) -> Dict:
    """Internal preprocessing/cleaning check (NOT exposed as an MCP tool)."""
    model = _model(ctx)
    tracker = _tracker(ctx)

    tx = max(0, min(model.grid.width - 1, int(x)))
    ty = max(0, min(model.grid.height - 1, int(y)))
    threshold = 0.80
    intensity = _thermal_intensity_seed(model, (tx, ty))

    neighborhood = model.grid.get_neighborhood((tx, ty), moore=True, include_center=True, radius=SCAN_RADIUS)
    candidate_survivors: List[Dict[str, Any]] = []
    for cell in neighborhood:
        for a in model.grid.get_cell_list_contents([cell]):
            if isinstance(a, SurvivorAgent) and not cast(Any, a).detected:
                candidate_survivors.append(
                    {
                        "agent": cast(Any, a),
                        "pos": cast(Tuple[int, int], cell),
                    }
                )

    # Hard rule: if scan range contains a survivor, force intensity >= threshold.
    if candidate_survivors:
        intensity = max(intensity, threshold)

    if intensity >= threshold and candidate_survivors:
        confirmed_list: List[Dict[str, Any]] = []
        for entry in candidate_survivors:
            survivor = cast(Any, entry["agent"])
            pos = cast(Tuple[int, int], entry["pos"])
            survivor.detected = True
            confirmed_list.append({"id": survivor.unique_id, "pos": [pos[0], pos[1]]})

        tracker.record_scan([], confirmed_list)
        return {
            "verification_status": "CONFIRMED",
            "message": f"{len(confirmed_list)} survivor(s) detected in scan range. Data cleaned and verified.",
            "human_detected": True,
            "confirmed_count": len(confirmed_list),
            "confirmed_survivors": confirmed_list,
            "threshold": threshold,
            "intensity": intensity,
            "coordinate": [tx, ty],
            "scan_center": [tx, ty],
        }

    return {
        "verification_status": "FALSE_POSITIVE",
        "message": "Thermal signature below rescue threshold or no life signs found.",
        "human_detected": False,
        "confirmed_count": 0,
        "confirmed_survivors": [],
        "threshold": threshold,
        "intensity": intensity,
        "coordinate": [tx, ty],
        "scan_center": [tx, ty],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MCP TOOLS — each mutating tool updates the state tracker
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def discover_drones(ctx: Context[ServerSession, AppState]) -> Dict:
    """Discover all drones currently active on the network."""
    model = _model(ctx)
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    tracker.sync_all_drones(model)  # refresh registry

    drones = [cast(Any, a) for a in model.schedule.agents if isinstance(a, DroneAgent)]
    return {
        "drones": [
            {"id": d.unique_id, "pos": list(d.pos), "battery": d.battery, "disabled": d.disabled}
            for d in drones
        ],
        "count": len(drones),
    }


@mcp.tool()
def get_drone_status(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Get detailed status of a specific drone."""
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    d = _drone(ctx, drone_id)
    tracker.sync_drone(d)
    return {
        "id": d.unique_id,
        "pos": list(d.pos),
        "battery": d.battery,
        "speed": d.speed,
        "disabled": d.disabled,
    }


@mcp.tool()
def get_battery_status(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Get battery percentage; includes critical flag (≤ 20%)."""
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    d = _drone(ctx, drone_id)
    tracker.sync_drone(d)
    return {"drone_id": d.unique_id, "battery_pct": d.battery, "critical": d.battery <= BATTERY_CRITICAL}


@mcp.tool()
def move_to(drone_id: str, x: int, y: int, ctx: Context[ServerSession, AppState]) -> Dict:
    """Move drone toward (x,y). Moves up to drone speed cells. Costs 1% battery per cell."""
    model = _model(ctx)
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    d = _drone(ctx, drone_id)

    if d.disabled:
        return {"moved": False, "reason": "drone disabled"}
    if d.battery <= 0:
        d.disabled = True
        tracker.sync_drone(d)
        return {"moved": False, "reason": "battery depleted"}

    tx = max(0, min(model.grid.width - 1, int(x)))
    ty = max(0, min(model.grid.height - 1, int(y)))
    pos: Tuple[int, int] = cast(Tuple[int, int], d.pos)
    cx, cy = pos
    steps = 0
    blocked = False
    while steps < d.speed and (cx, cy) != (tx, ty) and d.battery > 0:
        dx_ideal = (1 if tx > cx else -1) if tx != cx else 0
        dy_ideal = (1 if ty > cy else -1) if ty != cy else 0

        # ── Obstacle Avoidance / Sliding Logic ──
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
        if dx_ideal == 0:
            candidates.extend([(1, dy_ideal), (-1, dy_ideal)])
        if dy_ideal == 0:
            candidates.extend([(dx_ideal, 1), (dx_ideal, -1)])
        # Quaternary: pure perpendicular sidestep
        if dx_ideal == 0:
            candidates.extend([(1, 0), (-1, 0)])
        if dy_ideal == 0:
            candidates.extend([(0, 1), (0, -1)])
        if dx_ideal != 0 and dy_ideal != 0:
            candidates.extend([(0, -dy_ideal), (0, dy_ideal), (-dx_ideal, 0), (dx_ideal, 0)])
        # Last resort: backward movement
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

        found_move = False
        for cdx, cdy in candidates:
            nx, ny = cx + cdx, cy + cdy
            # Bounds check
            if not (0 <= nx < model.grid.width and 0 <= ny < model.grid.height):
                continue
            # HIGH-BUILDING OBSTACLE CHECK
            if hasattr(model, 'is_high_building') and model.is_high_building((nx, ny)):
                continue
            # STACKING/COLLISION AVOIDANCE
            stack_blocked = False
            if (nx, ny) not in BASE_POSITIONS:
                for agent in model.grid.get_cell_list_contents((nx, ny)):
                    if agent.__class__.__name__ == "DroneAgent" and getattr(agent, "unique_id", None) != d.unique_id:
                        stack_blocked = True
                        break
            if stack_blocked:
                continue
            # Valid move found
            cx, cy = nx, ny
            found_move = True
            break

        if not found_move:
            blocked = True
            break

        d.battery = max(0, d.battery - BATTERY_COST_MOVE)
        steps += 1

    model.grid.move_agent(d, (cx, cy))
    if d.battery <= 0:
        d.disabled = True

    # ── Phase 2: update tracker ──
    tracker.sync_drone(d)

    result = {"new_pos": [cx, cy], "battery": d.battery, "arrived": (cx, cy) == (tx, ty), "steps_taken": steps}
    if blocked:
        result["blocked_by_building"] = True
    return result


@mcp.tool()
def move_and_scan(drone_id: str, x: int, y: int, ctx: Context[ServerSession, AppState]) -> Dict:
    """Move drone toward (x,y) up to its speed, then thermal scan radius-2. Costs 1%/cell + 5% for scan."""
    m_res = move_to(drone_id, x, y, ctx)
    if "moved" in m_res and not m_res["moved"]:
        return m_res
    s_res = thermal_scan(drone_id, ctx)
    if "scanned" in s_res and not s_res["scanned"]:
        m_res["scan_reason"] = s_res["reason"]
        return m_res

    coord = s_res.get("coordinate") or m_res.get("new_pos")
    vx = int(coord[0]) if isinstance(coord, list) and len(coord) >= 2 else int(m_res["new_pos"][0])
    vy = int(coord[1]) if isinstance(coord, list) and len(coord) >= 2 else int(m_res["new_pos"][1])
    filter_result = verify_signature(vx, vy, ctx)
    
    m_res.update({
        "scanned_cells": s_res.get("scanned_cells"),
        "coordinate": s_res.get("coordinate"),
        "thermal_intensity": s_res.get("thermal_intensity"),
        "survivors_found": filter_result.get("confirmed_survivors", []),
        "battery": s_res.get("battery"),
        "filter_result": filter_result,
    })
    return m_res


@mcp.tool()
def thermal_scan(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Returns RAW thermal intensity from current drone coordinate (noisy environmental signal)."""
    model = _model(ctx)
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    d = _drone(ctx, drone_id)

    if d.disabled:
        return {"scanned": False, "reason": "drone disabled"}
    if d.battery < BATTERY_COST_SCAN:
        return {"scanned": False, "reason": "insufficient battery for scan"}

    d.battery = max(0, d.battery - BATTERY_COST_SCAN)
    cells = model.grid.get_neighborhood(d.pos, moore=True, include_center=True, radius=SCAN_RADIUS)

    # mark tiles as scanned (for UI coverage)
    for cell in cells:
        tile = model.tile_map.get(cell)
        if tile:
            tile.scanned = True

    coord = cast(Tuple[int, int], d.pos)
    raw_intensity = _thermal_intensity_seed(model, coord)

    if d.battery <= 0:
        d.disabled = True

    # ── Phase 2: update tracker ──
    tracker.sync_drone(d)
    tracker.record_scan(cells, [])

    return {
        "scanned": True,
        "scanned_cells": len(cells),
        "coordinate": [coord[0], coord[1]],
        "thermal_intensity": raw_intensity,
        "status": "Potential Signature Detected",
        "battery": d.battery,
        "drone_pos": list(d.pos),
    }


@mcp.tool()
def recall_to_base(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Move drone toward the nearest charging station. Costs 1% battery per cell."""
    d = _drone(ctx, drone_id)

    if d.disabled:
        return {"moved": False, "reason": "drone disabled"}

    nearest = min(
        BASE_POSITIONS,
        key=lambda b: abs(b[0] - d.pos[0]) + abs(b[1] - d.pos[1]),
    )
    
    res = move_to(drone_id, nearest[0], nearest[1], ctx)
    res["base_target"] = list(nearest)
    return res


@mcp.tool()
def charge_drone(drone_id: str, ctx: Context[ServerSession, AppState]) -> Dict:
    """Instant charge: restores battery to 100%. Drone must be on a charging station."""
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    d = _drone(ctx, drone_id)
    if d.pos not in BASE_POSITIONS:
        return {"charged": False, "reason": f"Not at a station (drone at {list(d.pos)})"}
    # Instant charge: one tick = full battery
    d.battery = 100.0
    d.disabled = False

    # ── Phase 2: update tracker ──
    tracker.sync_drone(d)

    return {"charged": True, "battery": d.battery, "full": True}


@mcp.tool()
def assign_drone_to_sector(drone_id: str, sector_id: int, ctx: Context[ServerSession, AppState]) -> Dict:
    """
    High-level command: assign a drone to search a sector.

    Looks up the waypoints for `sector_id` (1–6), finds the next unscanned
    waypoint by checking the simulation tile_map, then internally calls
    `move_and_scan` to navigate the drone there and scan on arrival.

    This is the PREFERRED tool for searching — it removes the need for the
    LLM to calculate (x, y) coordinates.
    """
    model = _model(ctx)
    tracker = _tracker(ctx)
    tracker.record_tool_call()

    # Validate sector_id
    if sector_id not in SECTOR_DEFS:
        return {"moved": False, "reason": f"Invalid sector_id {sector_id}. Must be 1–6."}

    # Ensure sectors are assigned in the tracker
    tracker.assign_sectors(model.num_drones)

    sdef = SECTOR_DEFS[sector_id]
    origin: Tuple[int, int] = sdef["origin"]
    size: Tuple[int, int] = sdef["size"]
    wps = _sector_waypoints(origin, size)

    # Find the first unscanned waypoint by checking the tile_map
    # A waypoint is "unscanned" if the tile at that position has NOT been scanned
    next_wp: Tuple[int, int] | None = None
    for wp in wps:
        tile = model.tile_map.get(wp)
        if tile is None or not tile.scanned:
            next_wp = wp
            break

    if next_wp is None:
        # All waypoints in this sector have been scanned — sector is done
        return {
            "moved": False,
            "reason": f"Sector {sector_id} ({sdef['name']}) is fully scanned. No unscanned waypoints remain.",
            "sector_id": sector_id,
            "sector_name": sdef["name"],
            "sector_complete": True,
        }

    wx, wy = next_wp

    # Internally call the existing move_and_scan logic
    result = move_and_scan(drone_id, wx, wy, ctx)
    result["sector_id"] = sector_id
    result["sector_name"] = sdef["name"]
    result["target_waypoint"] = [wx, wy]
    result["sector_complete"] = False

    return result


@mcp.tool()
def get_sector_info(ctx: Context[ServerSession, AppState]) -> Dict:
    """Zone layout: sector names, origins, coverage %, and scan waypoints."""
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    model = _model(ctx)
    sectors: List[Dict] = []

    for sid, sdef in SECTOR_DEFS.items():
        ox, oy = sdef["origin"]
        w, h = sdef["size"]
        total = w * h
        scanned = sum(
            1
            for xi in range(ox, ox + w)
            for yi in range(oy, oy + h)
            if model.tile_map.get((xi, yi)) and model.tile_map[(xi, yi)].scanned
        )
        wps = _sector_waypoints(sdef["origin"], sdef["size"])
        sectors.append(
            {
                "id": sid,
                "name": sdef["name"],
                "origin": [ox, oy],
                "size": [w, h],
                "coverage_pct": float(f"{scanned / total * 100:.1f}") if total > 0 else 0.0,
                "waypoints": [list(wp) for wp in wps],
            }
        )

    return {"sectors": sectors}


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Compressed mission state (the AI's only "memory" source)
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_mission_state(ctx: Context[ServerSession, AppState]) -> Dict:
    """
    Compressed mission snapshot — the AI's single source of truth.

    Returns a flat JSON dict containing:
      • tick          – simulation step
      • tool_calls    – cumulative MCP calls made
      • grid          – [width, height]
      • bases         – charging station coordinates
      • drones        – per-drone status (pos, battery, disabled)
      • survivors_*   – found / total / locations
      • scanned_cell_count – int
      • sectors       – per-sector coverage % and waypoints

    This replaces any need for the LLM to memorise grid history.
    """
    model = _model(ctx)
    tracker = _tracker(ctx)
    tracker.record_tool_call()

    # Ensure drone registry is current before serialisation.
    tracker.sync_all_drones(model)

    return tracker.to_compressed_dict(model)



@mcp.tool()
def advance_simulation(steps: int = 1, ctx: Context[ServerSession, AppState] = None) -> Dict:  # type: ignore[assignment]
    """Advance the Mesa model by N steps (agents are passive; used for datacollection/UI parity)."""
    if ctx is None:
        raise ValueError("Context injection failed")
    tracker = _tracker(ctx)
    tracker.record_tool_call()
    model = _model(ctx)
    steps_i = max(1, int(steps))
    for _ in range(steps_i):
        model.step()
    return {"advanced": steps_i}


if __name__ == "__main__":
    # Default transport is STDIO, which works well for local orchestrators and MCP Inspector.
    mcp.run()
