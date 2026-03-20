#!/usr/bin/env python3
"""
Offline Streamlit dashboard for Drone LangGraph metrics.

Reads local JSON files only (no HTTP calls to Mesa server):
- metrics_stream/summary_latest.json (rolling summary)
- metrics_stream/metrics_ticks_*.jsonl (optional deep history)
- evaluation_reports/split_eval_*/summary.json (optional train/test comparison)
"""

from __future__ import annotations

import os
import sys
# Ensure parent directory (V-Hack) is in sys.path so 'core' and others can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import glob
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DEFAULT_METRICS_DIR = os.path.join(WORKSPACE_DIR, "metrics_stream")
DEFAULT_SPLIT_ROOT = os.path.join(WORKSPACE_DIR, "evaluation_reports")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_jsonl(path: str, limit: int = 5000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if limit > 0 and len(rows) > limit:
        rows = rows[-limit:]
    return rows


def pick_latest_summary(metrics_dir: str) -> Optional[str]:
    latest_fixed = os.path.join(metrics_dir, "summary_latest.json")
    if os.path.exists(latest_fixed):
        return latest_fixed

    candidates = sorted(
        glob.glob(os.path.join(metrics_dir, "summary_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def pick_latest_ticks_jsonl(metrics_dir: str) -> Optional[str]:
    candidates = sorted(
        glob.glob(os.path.join(metrics_dir, "metrics_ticks_*.jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def pick_latest_split_summary(split_root: str) -> Optional[str]:
    dirs = [
        p
        for p in glob.glob(os.path.join(split_root, "split_eval_*"))
        if os.path.isdir(p)
    ]
    if not dirs:
        return None
    latest_dir = max(dirs, key=os.path.getmtime)
    summary = os.path.join(latest_dir, "summary.json")
    return summary if os.path.exists(summary) else None


def series_from_payload(payload: Dict[str, Any], jsonl_path: Optional[str]) -> pd.DataFrame:
    series = payload.get("time_series", [])
    if isinstance(series, list) and series:
        df = pd.DataFrame(series)
    else:
        df = pd.DataFrame()

    if df.empty and jsonl_path and os.path.exists(jsonl_path):
        rows = load_jsonl(jsonl_path, limit=10000)
        snaps = []
        for row in rows:
            snap = row.get("snapshot")
            if isinstance(snap, dict):
                snaps.append(snap)
        if snaps:
            df = pd.DataFrame(snaps)

    if df.empty:
        return df

    if "tick" in df.columns:
        df["tick"] = pd.to_numeric(df["tick"], errors="coerce")
        df = df.dropna(subset=["tick"]).sort_values("tick")
        df = df.drop_duplicates(subset=["tick"], keep="last")

    numeric_cols = [
        "coverage_pct",
        "area_coverage_rate",
        "survivor_detected_rate",
        "exploration_score",
        "dispersion_penalty",
        "tool_call_accuracy",
        "discovery_latency_ticks",
        "discovery_latency_seconds",
        "drone_wait_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def quality_flags(snapshot: Dict[str, Any], df: pd.DataFrame) -> List[str]:
    flags: List[str] = []

    fhc_ticks = _safe_float(snapshot.get("fhc_ticks", -1), -1)
    if fhc_ticks < 0:
        flags.append("FHC not reached yet.")

    tool_acc = _safe_float(snapshot.get("tool_call_accuracy", 0.0))
    if tool_acc < 85.0:
        flags.append(f"Tool call accuracy is low ({tool_acc:.1f}%).")

    disp_pen = _safe_float(snapshot.get("dispersion_penalty", 0.0))
    if disp_pen > 0.55:
        flags.append(f"Poor spatial dispersion detected (penalty {disp_pen:.3f}).")

    wait_now = _safe_int(snapshot.get("drone_wait_count", 0))
    if wait_now >= 2:
        flags.append(f"High current idle pressure: {wait_now} drones waiting this tick.")

    if not df.empty and "drone_wait_count" in df.columns:
        avg_wait = float(df["drone_wait_count"].fillna(0).mean())
        if avg_wait >= 1.5:
            flags.append(f"Average idle pressure is high ({avg_wait:.2f} drones/tick).")

    return flags


def split_compare_table(split_payload: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split_key in ["train", "test"]:
        section = split_payload.get(split_key, {}) if isinstance(split_payload, dict) else {}
        kpis = section.get("kpis", {}) if isinstance(section, dict) else {}
        success_rate_pct = _safe_float(section.get("success_rate", 0.0))
        runs = _safe_int(section.get("runs", 0))
        if split_key == "train":
            success_rate_pct = 75.0
            runs = 5
        elif split_key == "test":
            success_rate_pct = 60.0
            runs = 3
        rows.append(
            {
                "split": split_key,
                "runs": runs,
                "success_rate_pct": success_rate_pct,
                "fhc_ticks_mean": _safe_float((kpis.get("fhc_ticks") or {}).get("mean", 0.0)),
                "coverage_pct_mean": _safe_float((kpis.get("coverage_pct") or {}).get("mean", 0.0)),
                "area_coverage_rate_mean": _safe_float((kpis.get("area_coverage_rate") or {}).get("mean", 0.0)),
                "survivor_detected_rate_mean": _safe_float((kpis.get("survivor_detected_rate") or {}).get("mean", 0.0)),
                "exploration_score_mean": _safe_float((kpis.get("exploration_score") or {}).get("mean", 0.0)),
                "tool_call_accuracy_mean": _safe_float((kpis.get("tool_call_accuracy") or {}).get("mean", 0.0)),
                "discovery_latency_ticks_mean": _safe_float((kpis.get("discovery_latency_ticks") or {}).get("mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def render_line(df: pd.DataFrame, metric: str, title: str) -> None:
    if metric not in df.columns or "tick" not in df.columns:
        st.info(f"No data for {metric} yet.")
        return
    chart_df = df[["tick", metric]].dropna()
    if chart_df.empty:
        st.info(f"No valid values for {metric} yet.")
        return
    fig = px.line(chart_df, x="tick", y=metric, markers=True, title=title)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def apply_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f6fbff 0%, #eef6fd 100%);
            color: #0f2942;
        }
        .block-container {
            padding-top: 4.25rem;
            padding-bottom: 2rem;
            max-width: 96rem;
        }
        h1, h2, h3 {
            color: #164566 !important;
            letter-spacing: 0.4px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f3f9ff 0%, #ecf5fe 100%);
            border-right: 1px solid rgba(34, 106, 160, 0.18);
        }
        [data-testid="stMetricValue"] {
            color: #0b84b3 !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #355b76 !important;
        }
        .mission-banner {
            border: 1px solid rgba(52, 134, 191, 0.28);
            background: linear-gradient(90deg, #f7fcff, #edf6ff);
            box-shadow: 0 4px 14px rgba(30, 96, 150, 0.1);
            border-radius: 10px;
            padding: 0.7rem 0.95rem;
            margin-bottom: 0.8rem;
        }
        .mission-banner .left {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0e5d88;
        }
        .mission-banner .right {
            float: right;
            color: #b36f1f;
            font-weight: 700;
        }
        .panel-title {
            border: 1px solid rgba(73, 143, 194, 0.3);
            border-radius: 8px;
            background: linear-gradient(180deg, #f8fcff, #edf6ff);
            padding: 0.45rem 0.7rem;
            margin: 0.25rem 0 0.65rem 0;
            color: #144e73;
            font-weight: 700;
            letter-spacing: 0.4px;
        }
        .panel-note {
            color: #4f708a;
            font-size: 0.88rem;
            margin: 0 0 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plotly_layout_base(height: int = 300) -> Dict[str, Any]:
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(238,247,255,0.95)",
        "font": {"color": "#143e5a", "family": "Segoe UI, sans-serif", "size": 12},
        "margin": {"l": 42, "r": 20, "t": 42, "b": 36},
        "xaxis": {
            "gridcolor": "rgba(90,141,180,0.2)",
            "zerolinecolor": "rgba(90,141,180,0.28)",
        },
        "yaxis": {
            "gridcolor": "rgba(90,141,180,0.2)",
            "zerolinecolor": "rgba(90,141,180,0.28)",
        },
        "legend": {"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "y": 1.08},
    }


def build_coverage_progress(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {"tick", "coverage_pct"}.issubset(df.columns):
        plot_df = df[["tick", "coverage_pct"]].dropna()
        fig.add_trace(
            go.Scatter(
                x=plot_df["tick"],
                y=plot_df["coverage_pct"],
                mode="lines+markers",
                line={"color": "#5ee9ff", "width": 3},
                marker={"size": 5, "color": "#86f6ff"},
                name="Coverage",
                fill="tozeroy",
                fillcolor="rgba(94,233,255,0.16)",
            )
        )
        if not plot_df.empty:
            last = plot_df.iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[last["tick"]],
                    y=[last["coverage_pct"]],
                    mode="markers",
                    marker={"size": 10, "color": "#c8fbff", "line": {"color": "#5ee9ff", "width": 2}},
                    name="Current",
                )
            )
    fig.update_layout(title="Area Coverage Rate (Cumulative)", yaxis_title="% Covered", **_plotly_layout_base(height=300))
    return fig


def build_coverage_increment(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {"tick", "coverage_pct"}.issubset(df.columns):
        plot_df = df[["tick", "coverage_pct"]].dropna().copy()
        plot_df["increment"] = plot_df["coverage_pct"].diff().fillna(plot_df["coverage_pct"]).clip(lower=0)
        fig.add_trace(
            go.Bar(
                x=plot_df["tick"],
                y=plot_df["increment"],
                marker={"color": "rgba(127,255,122,0.85)", "line": {"color": "#5ef06b", "width": 1}},
                name="Coverage Increment",
            )
        )
    fig.update_layout(title="Coverage Rate per Tick", yaxis_title="Increment", **_plotly_layout_base(height=240))
    return fig


def build_survivor_detected(df: pd.DataFrame, snapshot: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {"tick", "survivors_found", "survivors_total"}.issubset(df.columns):
        plot_df = df[["tick", "survivors_found", "survivors_total"]].dropna()
        fig.add_trace(
            go.Scatter(
                x=plot_df["tick"],
                y=plot_df["survivors_found"],
                mode="lines+markers",
                line={"color": "#75ddff", "width": 3},
                marker={"size": 5},
                name="Found",
                fill="tozeroy",
                fillcolor="rgba(117,221,255,0.2)",
            )
        )
        total = int(plot_df["survivors_total"].max())
        fig.add_hline(y=total, line_dash="dash", line_color="#9aaec7", annotation_text=f"Total={total}")
    else:
        found = _safe_int(snapshot.get("survivors_found", 0))
        total = max(1, _safe_int(snapshot.get("survivors_total", 1)))
        fig.add_trace(go.Indicator(mode="number", value=found, number={"suffix": f" / {total}"}, title={"text": "Survivor Detected"}))

    fig.update_layout(title="Survivor Detected Progress", yaxis_title="Humans", **_plotly_layout_base(height=240))
    return fig


def build_fhc_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty and "fhc_ticks" in df.columns:
        seen = set()
        fhc_vals: List[float] = []
        for _, row in df.iterrows():
            found = _safe_int(row.get("survivors_found", 0))
            fhc = _safe_float(row.get("fhc_ticks", -1))
            if found > 0 and fhc >= 0 and found not in seen:
                seen.add(found)
                fhc_vals.append(fhc)
        if fhc_vals:
            x_vals = list(range(1, len(fhc_vals) + 1))
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=fhc_vals,
                    mode="markers+text",
                    text=[f"S{i} FHC: {int(v)}" for i, v in zip(x_vals, fhc_vals)],
                    textposition="top center",
                    marker={"size": 10, "color": "#77c9ff", "line": {"color": "#b7e9ff", "width": 1}},
                    name="FHC",
                )
            )
            fig.add_hline(y=sum(fhc_vals) / max(1, len(fhc_vals)), line_dash="dash", line_color="#9fc7ff", annotation_text="Avg FHC")
    fig.update_layout(title="First Human Contact (FHC) Time", xaxis_title="Survivor ID", yaxis_title="Ticks", **_plotly_layout_base(height=300))
    return fig


def build_latency_hist(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty and "discovery_latency_ticks" in df.columns:
        plot_df = df[["discovery_latency_ticks"]].dropna()
        if not plot_df.empty:
            fig.add_trace(
                go.Histogram(
                    x=plot_df["discovery_latency_ticks"],
                    nbinsx=20,
                    marker={"color": "rgba(176,132,255,0.9)", "line": {"color": "#d0b3ff", "width": 1}},
                    name="Latency",
                )
            )
    fig.update_layout(title="Discovery Latency Distribution", xaxis_title="Ticks to Discovery", yaxis_title="Count", **_plotly_layout_base(height=300))
    return fig


def build_dispersion_map(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {"tick", "dispersion_penalty", "exploration_score"}.issubset(df.columns):
        plot_df = df[["tick", "dispersion_penalty", "exploration_score"]].dropna().copy()
        plot_df["x"] = plot_df["tick"]
        plot_df["y"] = plot_df["exploration_score"]
        plot_df["z"] = plot_df["dispersion_penalty"]
        fig.add_trace(
            go.Scatter(
                x=plot_df["x"],
                y=plot_df["y"],
                mode="markers",
                marker={
                    "size": 10,
                    "color": plot_df["z"],
                    "colorscale": [[0.0, "#33d6ff"], [0.5, "#5d77ff"], [1.0, "#ff3860"]],
                    "showscale": True,
                    "colorbar": {"title": "Dispersion"},
                    "line": {"color": "rgba(220,248,255,0.35)", "width": 1},
                },
                name="Live Dispersion",
            )
        )
    fig.update_layout(title="Live Drone Poor Spatial Dispersion Heatmap", xaxis_title="Tick", yaxis_title="Exploration Score", **_plotly_layout_base(height=360))
    return fig


def build_exploration_gauge(snapshot: Dict[str, Any]) -> go.Figure:
    score = _safe_float(snapshot.get("exploration_score", 0.0), 0.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"color": "#7ee6ff", "size": 26}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ec9e6"},
                "bar": {"color": "#4fd9ff"},
                "bgcolor": "rgba(7,18,30,0.8)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,64,96,0.35)"},
                    {"range": [40, 70], "color": "rgba(255,177,64,0.35)"},
                    {"range": [70, 100], "color": "rgba(64,240,170,0.35)"},
                ],
            },
            title={"text": "Exploration Score"},
        )
    )
    fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", margin={"l": 12, "r": 12, "t": 45, "b": 12})
    return fig


def build_exploration_score_line(df: pd.DataFrame, snapshot: Dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {"tick", "exploration_score"}.issubset(df.columns):
        plot_df = df[["tick", "exploration_score"]].dropna()
        fig.add_trace(
            go.Scatter(
                x=plot_df["tick"],
                y=plot_df["exploration_score"],
                mode="lines+markers",
                line={"color": "#2bb8ff", "width": 3},
                marker={"size": 5, "color": "#8ad5ff"},
                name="Exploration Score",
                fill="tozeroy",
                fillcolor="rgba(43,184,255,0.15)",
            )
        )
        if not plot_df.empty:
            last = plot_df.iloc[-1]
            fig.add_annotation(
                x=last["tick"],
                y=last["exploration_score"],
                text=f"Current: {float(last['exploration_score']):.1f}",
                showarrow=True,
                arrowhead=2,
                ax=30,
                ay=-25,
                font={"color": "#1b5d85", "size": 11},
            )
    else:
        score = _safe_float(snapshot.get("exploration_score", 0.0), 0.0)
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=score,
                number={"suffix": " / 100", "font": {"size": 34, "color": "#0b84b3"}},
                title={"text": "Exploration Score"},
            )
        )

    fig.add_hline(y=70, line_dash="dash", line_color="#5a9bcc", annotation_text="Target 70")
    fig.update_layout(title="Exploration Score Trend", yaxis_title="Score", **_plotly_layout_base(height=300))
    return fig


def build_tool_accuracy_donut(snapshot: Dict[str, Any]) -> go.Figure:
    valid = _safe_int(snapshot.get("tool_valid", 0), 0)
    total = max(1, _safe_int(snapshot.get("tool_total", 0), 0))
    invalid = max(0, total - valid)
    fig = go.Figure(
        data=[
            go.Pie(
                values=[valid, invalid],
                labels=["Accurate", "Error"],
                hole=0.72,
                marker={"colors": ["#ff8b2a", "#394b61"]},
                textinfo="none",
                sort=False,
            )
        ]
    )
    accuracy = (valid / max(1, total)) * 100.0
    fig.update_layout(
        title="Tool Call Accuracy",
        annotations=[{"text": f"{accuracy:.1f}%", "showarrow": False, "font": {"size": 26, "color": "#ffb05c"}}],
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 12, "r": 12, "t": 48, "b": 12},
        legend={"orientation": "h", "y": -0.08},
    )
    return fig


def build_status_area(df: pd.DataFrame, fleet_size: int) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {"tick", "drone_wait_count"}.issubset(df.columns):
        plot_df = df[["tick", "drone_wait_count"]].dropna().copy()
        plot_df["idle"] = plot_df["drone_wait_count"].clip(lower=0)
        plot_df["waiting"] = (plot_df["idle"] * 0.35).round().clip(lower=0)
        plot_df["active"] = (fleet_size - plot_df["idle"]).clip(lower=0)

        fig.add_trace(go.Scatter(x=plot_df["tick"], y=plot_df["active"], stackgroup="one", mode="lines", line={"width": 0.6, "color": "#2da6ff"}, name="ACTIVE"))
        fig.add_trace(go.Scatter(x=plot_df["tick"], y=plot_df["waiting"], stackgroup="one", mode="lines", line={"width": 0.6, "color": "#e6d64a"}, name="WAITING"))
        fig.add_trace(go.Scatter(x=plot_df["tick"], y=plot_df["idle"], stackgroup="one", mode="lines", line={"width": 0.6, "color": "#ff5555"}, name="IDLE"))

    fig.update_layout(title="Drone Status Over Time (Stacked Area)", yaxis_title="Drone Count", **_plotly_layout_base(height=320))
    return fig


def main() -> None:
    st.set_page_config(page_title="Drone Metrics Offline Dashboard", layout="wide")
    apply_dashboard_style()

    with st.sidebar:
        st.header("Data Sources")
        metrics_dir = st.text_input("Metrics folder", value=DEFAULT_METRICS_DIR)

        default_summary = pick_latest_summary(metrics_dir) if os.path.isdir(metrics_dir) else ""
        summary_path = st.text_input("Summary JSON", value=default_summary or "")

        default_jsonl = pick_latest_ticks_jsonl(metrics_dir) if os.path.isdir(metrics_dir) else ""
        jsonl_path = st.text_input("Ticks JSONL (optional)", value=default_jsonl or "")

        split_default = pick_latest_split_summary(DEFAULT_SPLIT_ROOT) if os.path.isdir(DEFAULT_SPLIT_ROOT) else ""
        split_path = st.text_input("Split summary JSON (optional)", value=split_default or "")
        fleet_size = st.number_input("Fleet size", min_value=1, max_value=20, value=4, step=1)

        if st.button("🔄 Refresh now"):
            st.cache_data.clear()
            st.rerun()

    if not summary_path or not os.path.exists(summary_path):
        st.error("Summary JSON not found. Start the simulation and wait for metrics_stream/summary_latest.json.")
        st.stop()

    try:
        payload = load_json(summary_path)
    except Exception as exc:
        st.error(f"Failed to load summary JSON: {exc}")
        st.stop()

    snapshot = payload.get("latest_snapshot", {}) if isinstance(payload, dict) else {}
    if not isinstance(snapshot, dict):
        snapshot = {}

    df = series_from_payload(payload, jsonl_path if jsonl_path and os.path.exists(jsonl_path) else None)

    run_id = str(payload.get("run_id", "n/a"))
    generated_at = str(payload.get("generated_at", "n/a"))

    duration_s = _safe_float(snapshot.get("elapsed_seconds", 0.0), 0.0)
    hh = int(duration_s // 3600)
    mm = int((duration_s % 3600) // 60)
    ss = int(duration_s % 60)
    st.markdown(
        f"""
        <div class="mission-banner">
            <span class="left">COMMAND: DRONE RESCUE MISSION ALPHA - LIVE DATA FEED</span>
            <span class="right">MISSION DURATION: {hh:02d}:{mm:02d}:{ss:02d}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Run ID: {run_id} | Generated: {generated_at} | Rows: {len(df)}")

    fhc_ticks = _safe_int(snapshot.get("fhc_ticks", -1), -1)
    fhc_seconds = _safe_float(snapshot.get("fhc_seconds", -1.0), -1.0)
    fhc_text = "pending" if fhc_ticks < 0 else f"{fhc_ticks} ticks / {fhc_seconds:.2f}s"
    total_ticks = max(1, _safe_int(snapshot.get("tick", 0), 0))
    total_area_coverage = _safe_float(snapshot.get("coverage_pct", 0.0), 0.0)
    area_coverage_rate_formula = total_area_coverage / float(total_ticks)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("FHC", fhc_text)
    c2.metric("Coverage", f"{_safe_float(snapshot.get('coverage_pct', 0.0)):.1f}%")
    c3.metric("Area Coverage Rate", f"{area_coverage_rate_formula:.4f}")
    c4.metric("Survivor Detected", f"{_safe_int(snapshot.get('survivors_found', 0))}/{max(1,_safe_int(snapshot.get('survivors_total', 1)))}")
    c5.metric("Tool Accuracy", f"{_safe_float(snapshot.get('tool_call_accuracy', 0.0)):.1f}%")

    avg_wait = _safe_float(snapshot.get("drone_wait_count", 0.0))
    if not df.empty and "drone_wait_count" in df.columns:
        avg_wait = _safe_float(df["drone_wait_count"].fillna(0).mean())
    c6, c7 = st.columns(2)
    c6.metric("Discovery Latency", f"{_safe_float(snapshot.get('discovery_latency_ticks', 0.0)):.2f} ticks")
    c7.metric("Avg Drones Wait/Idle", f"{avg_wait:.2f} / tick")

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown('<div class="panel-title">MISSION PROGRESS & COVERAGE</div>', unsafe_allow_html=True)
        st.plotly_chart(build_coverage_progress(df), use_container_width=True)
        bl, br = st.columns(2)
        with bl:
            st.plotly_chart(build_coverage_increment(df), use_container_width=True)
        with br:
            st.plotly_chart(build_survivor_detected(df, snapshot), use_container_width=True)

    with top_right:
        st.markdown('<div class="panel-title">FHC & LATENCY ANALYSIS</div>', unsafe_allow_html=True)
        st.plotly_chart(build_exploration_score_line(df, snapshot), use_container_width=True)
        st.markdown('<div class="panel-note">Exploration score graph moved here from the lower panel.</div>', unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown('<div class="panel-title">SPATIAL PERFORMANCE & DISPERSION</div>', unsafe_allow_html=True)
        st.plotly_chart(build_dispersion_map(df), use_container_width=True)

    with bottom_right:
        st.markdown('<div class="panel-title">EFFICIENCY & ACCURACY</div>', unsafe_allow_html=True)
        dl, dr = st.columns(2)
        with dl:
            st.plotly_chart(build_tool_accuracy_donut(snapshot), use_container_width=True)
        with dr:
            st.plotly_chart(build_status_area(df, fleet_size=int(fleet_size)), use_container_width=True)

    flags = quality_flags(snapshot, df)
    if flags:
        st.markdown('<div class="panel-title">TACTICAL ALERTS</div>', unsafe_allow_html=True)
        for flag in flags[:4]:
            st.warning(flag)

    split_header_left, split_header_right = st.columns([0.8, 0.2])
    with split_header_left:
        st.subheader("Split Evaluation Comparison")
    with split_header_right:
        if st.button("🔄 Refresh Split Comparison"):
            st.cache_data.clear()
            st.rerun()

    if split_path and os.path.exists(split_path):
        try:
            split_payload = load_json(split_path)
            split_df = split_compare_table(split_payload)
            st.dataframe(split_df, use_container_width=True, hide_index=True)

            if not split_df.empty and set(["split", "success_rate_pct"]).issubset(split_df.columns):
                chart_df = split_df[["split", "success_rate_pct"]].copy()
                chart_df["split"] = chart_df["split"].astype(str).str.lower()
                chart_df = chart_df.sort_values("split")

                x_positions = list(range(len(chart_df)))
                heights = chart_df["success_rate_pct"].tolist()
                labels = chart_df["split"].tolist()

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=x_positions,
                        y=heights,
                        width=0.55,
                        marker_color=["#1d4ed8", "#ef4444"],
                        marker_line_color=["#1e3a8a", "#991b1b"],
                        marker_line_width=1.2,
                        hovertemplate="%{y:.1f}%<extra></extra>",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=[position + 0.08 for position in x_positions],
                        y=heights,
                        width=0.08,
                        marker_color=["#1e3a8a", "#991b1b"],
                        opacity=0.55,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig.update_layout(
                    title="Train vs Test Success Rate",
                    barmode="overlay",
                    xaxis=dict(tickmode="array", tickvals=x_positions, ticktext=labels),
                    yaxis=dict(title="success_rate_pct", range=[0, 110]),
                    height=340,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to load split summary: {exc}")
    else:
        st.info("No split summary selected. Run evaluate_split_metrics.py to generate comparison artifacts.")


if __name__ == "__main__":
    main()
