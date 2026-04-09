"""
dashboard.py — Research Intelligence Platform (Multi-User)
-----------------------------------------------------------
McKinsey-quality multi-user dashboard for the LangGraph Content Pipeline.

Multiple users can spawn concurrent research pipelines, monitor all active
sessions in a live Operations Center, and respond to HITL checkpoints.

Architecture
------------
  _SESSIONS : dict[thread_id → SessionRecord]   — global, thread-safe registry
  _QUEUES   : dict[thread_id → queue.Queue]     — per-session resume queues
  _LOCK     : threading.Lock                    — guards both dicts

  Per-browser Streamlit session (st.session_state):
    user_name        : str           — display name entered by the user
    my_thread_ids    : list[str]     — thread_ids this browser started
    viewing_thread_id: str | None    — which session the detail panel shows

Run:  streamlit run dashboard.py  (from content_pipeline/ directory)
"""

import uuid
import time
import threading
import queue
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import Counter
import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command
from graph import graph as _graph

# ── User preferences (persisted to disk so username survives restarts) ────────
_PREFS_FILE = Path(__file__).parent / ".dashboard_prefs.json"

def _load_prefs() -> dict:
    try:
        return json.loads(_PREFS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_prefs(**kwargs) -> None:
    try:
        prefs = _load_prefs()
        prefs.update(kwargs)
        _PREFS_FILE.write_text(json.dumps(prefs), encoding="utf-8")
    except Exception:
        pass

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION PERSISTENCE  (SQLite — zero extra dependencies)
# ══════════════════════════════════════════════════════════════════════════════
_DB_FILE = Path(__file__).parent / "sessions.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id      TEXT PRIMARY KEY,
    user_name      TEXT,
    topic          TEXT,
    status         TEXT,
    created_at     TEXT,
    updated_at     TEXT,
    node_statuses  TEXT,
    node_timings   TEXT,
    hitl_events    TEXT,
    logs           TEXT,
    pipeline_state TEXT,
    run_result     TEXT,
    run_complete   INTEGER DEFAULT 0
)
"""

def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_FILE), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _init_db():
    with _db_conn() as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()

def _dt_serial(obj):
    """Recursively convert datetime → ISO string for JSON storage."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _dt_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dt_serial(i) for i in obj]
    return obj

def _save_session_to_db(thread_id: str, lock, sessions: dict):
    """Snapshot the current session state into SQLite (called from bg thread)."""
    with lock:
        sess = sessions.get(thread_id)
        if not sess:
            return
        # Serialize inside the lock to avoid concurrent mutation
        try:
            status = (
                "hitl"    if sess.get("interrupt") else
                "running" if sess.get("running")   else
                "done"    if sess.get("run_complete") else "pending"
            )
            row = (
                thread_id,
                sess["user_name"],
                sess["topic"],
                status,
                sess["created_at"].isoformat(),
                sess["updated_at"].isoformat(),
                json.dumps(sess["node_statuses"]),
                json.dumps(_dt_serial(dict(sess["node_timings"]))),
                json.dumps(_dt_serial(list(sess["hitl_events"]))),
                json.dumps(_dt_serial(list(sess["logs"]))),
                json.dumps(_dt_serial(dict(sess.get("pipeline_state", {})))),
                json.dumps(_dt_serial(dict(sess.get("run_result", {})))),
                int(bool(sess.get("run_complete"))),
            )
        except Exception as e:
            print(f"[db] serialize error {thread_id}: {e}")
            return

    try:
        with _db_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (thread_id, user_name, topic, status, created_at, updated_at,
                 node_statuses, node_timings, hitl_events, logs,
                 pipeline_state, run_result, run_complete)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, row)
            conn.commit()
    except Exception as e:
        print(f"[db] write error {thread_id}: {e}")

def _parse_dt(s: str | None) -> datetime:
    """Parse an ISO datetime string, falling back to now() for malformed values."""
    if not s:
        return datetime.now()
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        # Legacy rows stored time-only strings (e.g. '17:12:32') — use today
        try:
            t = datetime.strptime(s, "%H:%M:%S")
            return datetime.now().replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=0)
        except Exception:
            return datetime.now()


def _load_history_from_db() -> dict:
    """Load the 200 most-recent sessions from SQLite into memory."""
    if not _DB_FILE.exists():
        return {}
    history: dict = {}
    try:
        with _db_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
    except Exception as e:
        print(f"[db] load error: {e}")
        return history

    for row in rows:
        tid = row["thread_id"]
        try:
            sess: dict = {
                "thread_id":     tid,
                "user_name":     row["user_name"],
                "topic":         row["topic"],
                "active_node":   None,
                "interrupt":     None,
                "running":       False,
                "run_complete":  bool(row["run_complete"]),
                "is_historical": True,
                "created_at":    _parse_dt(row["created_at"]),
                "updated_at":    _parse_dt(row["updated_at"]),
                "node_statuses": json.loads(row["node_statuses"] or "{}"),
                "pipeline_state": json.loads(row["pipeline_state"] or "{}"),
                "run_result":    json.loads(row["run_result"] or "{}"),
                "logs":          [],
                "hitl_events":   [],
                "node_timings":  {},
            }
            # Restore timings (ISO strings → datetime)
            for nid, t in json.loads(row["node_timings"] or "{}").items():
                sess["node_timings"][nid] = {
                    k: _parse_dt(v) for k, v in t.items()
                }
            # Restore HITL events
            for e in json.loads(row["hitl_events"] or "[]"):
                sess["hitl_events"].append({**e, "ts": _parse_dt(e.get("ts"))})
            # Restore logs
            for lg in json.loads(row["logs"] or "[]"):
                sess["logs"].append({**lg, "ts": _parse_dt(lg.get("ts"))})
            # Fix nodes left in active state when app shut down mid-run
            for nid, ns in sess["node_statuses"].items():
                if ns == "active":
                    sess["node_statuses"][nid] = "error"
            history[tid] = sess
        except Exception as e:
            print(f"[db] skipping row {tid}: {e}")
    return history

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Canvas ── */
[data-testid="stAppViewContainer"] { background: #f5f7fa; }
[data-testid="stMain"]             { background: #f5f7fa; }
.main .block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8ecf2;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #4a5568 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1a2535 !important; }

/* ── Platform header ── */
.platform-header {
    background: #ffffff;
    border-bottom: 1px solid #e8ecf2;
    padding: 18px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.platform-logo {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 40%, #0891b2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.platform-sub {
    font-size: 0.7rem;
    font-weight: 500;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 2px;
}
.user-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f0f2ff;
    border: 1px solid #c7d2fe;
    border-radius: 24px;
    padding: 7px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #4338ca;
}
.avatar-circle {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #0891b2);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.6rem;
    font-weight: 800;
    color: #fff;
    flex-shrink: 0;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #a0aec0;
    margin-bottom: 14px;
}

/* ── KPI metric cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 28px; }
.kpi-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e8ecf2;
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    border-radius: 14px 14px 0 0;
}
.kpi-card.kpi-active::after  { background: linear-gradient(90deg, #059669, #0891b2); }
.kpi-card.kpi-hitl::after    { background: linear-gradient(90deg, #d97706, #dc2626); }
.kpi-card.kpi-done::after    { background: linear-gradient(90deg, #4f46e5, #7c3aed); }
.kpi-value {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1a2535;
    line-height: 1;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
}
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #a0aec0;
}
.kpi-sub {
    font-size: 0.76rem;
    color: #059669;
    margin-top: 6px;
}
.kpi-sub.warn { color: #d97706; }
.kpi-sub.neutral { color: #a0aec0; }

/* ── Session cards ── */
.sess-card {
    background: #ffffff;
    border: 1px solid #e8ecf2;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    cursor: default;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.sess-card:hover { border-color: #c7d2fe; box-shadow: 0 6px 24px rgba(79,70,229,0.09); }
.sess-card.is-hitl {
    border-color: #fbbf24;
    background: #fffdf7;
    box-shadow: 0 0 0 3px rgba(251,191,36,0.12);
    animation: cardPulse 2.5s ease-in-out infinite;
}
.sess-card.is-running { border-color: #6ee7b7; background: #f9fffd; }
.sess-card.is-mine    { border-color: #c7d2fe; }
.sess-card.is-done    { opacity: 0.82; }

@keyframes cardPulse {
    0%,100% { box-shadow: 0 0 0 3px rgba(251,191,36,0.10); }
    50%      { box-shadow: 0 0 0 5px rgba(251,191,36,0.22); }
}

.card-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 10px; }
.card-user-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f5f7fa;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 3px 10px 3px 4px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #4a5568;
}
.card-avatar {
    width: 18px; height: 18px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #0891b2);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.55rem;
    font-weight: 800;
    color: #fff;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.pill-running  { background: rgba(5,150,105,0.08);  color: #065f46; border: 1px solid rgba(5,150,105,0.22); }
.pill-hitl     { background: rgba(217,119,6,0.08);  color: #92400e; border: 1px solid rgba(217,119,6,0.22); }
.pill-done     { background: rgba(79,70,229,0.08);  color: #3730a3; border: 1px solid rgba(79,70,229,0.22); }
.pill-error    { background: rgba(220,38,38,0.08);  color: #991b1b; border: 1px solid rgba(220,38,38,0.22); }
.pill-pending  { background: rgba(160,174,192,0.12); color: #718096; border: 1px solid rgba(160,174,192,0.25); }

.card-topic { font-size: 0.92rem; font-weight: 600; color: #1a2535; line-height: 1.4; margin-bottom: 14px; }

.mini-bar { height: 3px; background: #edf0f7; border-radius: 2px; overflow: hidden; margin-bottom: 10px; }
.mini-fill { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
.fill-r { background: linear-gradient(90deg, #059669, #0891b2); }
.fill-h { background: linear-gradient(90deg, #d97706, #dc2626); }
.fill-d { background: linear-gradient(90deg, #4f46e5, #7c3aed); }

.card-meta { font-size: 0.72rem; color: #a0aec0; display: flex; justify-content: space-between; }
.card-meta span { color: #718096; }

/* ── Pipeline graph nodes ── */
.pipe-wrap { padding: 4px 0; }
.pipe-node {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 15px;
    border-radius: 10px;
    border: 1px solid transparent;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.3s;
    position: relative;
}
.pipe-node .node-type {
    margin-left: auto;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.55;
    border: 1px solid currentColor;
    padding: 2px 6px;
    border-radius: 4px;
}
.pn-pending { background: #f5f7fa; border-color: #e2e8f0; color: #a0aec0; }
.pn-active  { background: #ecfdf5; border-color: #6ee7b7; color: #065f46; animation: nodeGlow 1.8s ease-in-out infinite; }
.pn-done    { background: #eef2ff; border-color: #c7d2fe; color: #3730a3; }
.pn-hitl    { background: #fffbeb; border-color: #fde68a; color: #92400e; animation: nodeGlowGold 1.8s ease-in-out infinite; }
.pn-error   { background: #fef2f2; border-color: #fecaca; color: #991b1b; }

@keyframes nodeGlow     { 0%,100%{box-shadow:0 0 0 0 rgba(5,150,105,0.06)} 50%{box-shadow:0 0 10px 3px rgba(5,150,105,0.18)} }
@keyframes nodeGlowGold { 0%,100%{box-shadow:0 0 0 0 rgba(217,119,6,0.06)} 50%{box-shadow:0 0 10px 3px rgba(217,119,6,0.18)} }

.pipe-connector { color: #d4dce8; font-size: 0.7rem; padding-left: 24px; line-height: 1.2; margin: 1px 0; }

/* ── HITL checkpoint banner ── */
.hitl-banner {
    background: linear-gradient(135deg, #fffbeb, #fff8e0);
    border: 1px solid #fde68a;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hitl-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #f59e0b, #ef4444, #f59e0b);
    animation: shimmer 2s linear infinite;
    background-size: 200% 100%;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

.hitl-tag  { font-size: 0.66rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; color: #b45309; margin-bottom: 4px; }
.hitl-step { font-size: 1.05rem; font-weight: 700; color: #92400e; }
.hitl-desc { font-size: 0.78rem; color: #a16207; margin-top: 4px; }

/* ── Running indicator ── */
.running-banner {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1px solid #6ee7b7;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
.running-tag  { font-size: 0.66rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.12em; color: #059669; margin-bottom: 4px; }
.running-step { font-size: 1.05rem; font-weight: 700; color: #065f46; }
.running-sub  { font-size: 0.78rem; color: #6ee7b7; margin-top: 4px; }

/* ── Complete banner ── */
.complete-banner {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
}

/* ── Log entries ── */
.log-container {
    background: #f8f9fb;
    border: 1px solid #e8ecf2;
    border-radius: 10px;
    padding: 12px 14px;
    max-height: 280px;
    overflow-y: auto;
}
.log-line { font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace; font-size: 0.73rem; padding: 2px 0; }
.log-info  { color: #3b82f6; }
.log-hitl  { color: #d97706; }
.log-done  { color: #059669; }
.log-error { color: #dc2626; }
.log-ts    { color: #c0cad8; margin-right: 6px; }

/* ── State inspector ── */
.state-container {
    background: #f8f9fb;
    border: 1px solid #e8ecf2;
    border-radius: 10px;
    padding: 12px 14px;
    max-height: 280px;
    overflow-y: auto;
}
.state-row { display: flex; gap: 10px; padding: 5px 0; border-bottom: 1px solid #edf0f7; font-size: 0.8rem; }
.state-row:last-child { border-bottom: none; }
.skey { color: #7c3aed; font-weight: 600; min-width: 130px; flex-shrink: 0; }
.sval { color: #4a5568; word-break: break-word; }

/* ── Section divider ── */
.section-div {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0 30%, #e2e8f0 70%, transparent);
    margin: 28px 0;
}

/* ── Detail session header ── */
.detail-header { margin-bottom: 20px; }
.detail-topic  { font-size: 1.4rem; font-weight: 700; color: #1a2535; line-height: 1.35; margin-bottom: 6px; letter-spacing: -0.02em; }
.detail-meta   { font-size: 0.76rem; color: #a0aec0; }
.detail-meta span { color: #718096; }

/* ── Streamlit form/input overrides ── */
div[data-testid="stForm"] { background: transparent !important; border: none !important; padding: 0 !important; }
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {
    background: #ffffff !important;
    border: 1px solid #d4dce8 !important;
    color: #1a2535 !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border: none !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338ca, #4f46e5) !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid #d4dce8 !important;
    color: #4a5568 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #4f46e5 !important;
    color: #4f46e5 !important;
    background: #f5f3ff !important;
}
/* Historical (archived) session cards — muted, no border animation */
.sess-card.is-historical {
    opacity: 0.82;
    border-top: 3px solid #cbd5e1 !important;
    animation: none !important;
}
/* Sidebar session list buttons — styled as clean text rows */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 6px !important;
    color: #4a5568 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 6px 10px !important;
    margin-bottom: 2px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: #f5f3ff !important;
    border-left-color: #4f46e5 !important;
    color: #4f46e5 !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: #a0aec0 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4f46e5 !important;
    border-bottom: 2px solid #4f46e5 !important;
}
[data-testid="stTabs"] { border-bottom: 1px solid #e8ecf2; }
.stExpander { border: 1px solid #e8ecf2 !important; border-radius: 10px !important; background: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
PIPELINE_NODES = [
    ("research_agent",  "Research Agent",  "🧠", "agent"),
    ("review_queries",  "Review Queries",  "⏸",  "hitl"),
    ("web_search_node", "Web Search",      "🔍", "tool"),
    ("writer_agent",    "Writer Agent",    "✍",  "agent"),
    ("review_draft",    "Review Draft",    "⏸",  "hitl"),
    ("publisher",       "Publisher",       "🚀", "agent"),
]
NODE_IDS = [n[0] for n in PIPELINE_NODES]

HITL_NODE_MAP = {
    "review_queries":  "review_queries",
    "review_draft":    "review_draft",
    "confirm_publish": "publisher",
}


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL MULTI-USER STATE REGISTRY
#  @st.cache_resource persists across all reruns AND all browser sessions.
#  Plain module-level dicts are re-initialised to {} on every st.rerun(),
#  so they MUST live inside a cached resource that Streamlit only calls once.
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _global_registry():
    """Singleton shared state — survives every st.rerun() and every tab."""
    _init_db()
    reg = {
        "sessions": {},          # thread_id → session record
        "queues":   {},          # thread_id → queue.Queue
        "lock":     threading.Lock(),
    }
    # Pre-populate with persisted history so the dashboard shows past runs
    reg["sessions"].update(_load_history_from_db())
    return reg

_reg      = _global_registry()
_SESSIONS: dict           = _reg["sessions"]
_QUEUES:   dict           = _reg["queues"]
_LOCK:     threading.Lock = _reg["lock"]


def _try_resume(thread_id: str, response: dict) -> bool:
    """
    Send a HITL response only if the session is still waiting for one.
    Clears the interrupt under the lock before enqueuing, so concurrent
    callers (two browsers clicking at the same time) see interrupt=None
    on their second attempt and bail out without double-resuming.
    Returns True if the response was sent, False if the checkpoint was stale.
    """
    with _LOCK:
        sess = _SESSIONS.get(thread_id)
        if not sess or sess.get("interrupt") is None:
            return False
        q = _QUEUES.get(thread_id)
        if q is None:
            return False
        # Optimistically clear so no concurrent caller can sneak in
        _SESSIONS[thread_id]["interrupt"] = None
        q.put(response)
        return True


def _new_session_record(user_name: str, topic: str, thread_id: str) -> dict:
    return {
        "thread_id":      thread_id,
        "user_name":      user_name,
        "topic":          topic,
        "node_statuses":  {n: "pending" for n in NODE_IDS},
        "active_node":    None,
        "interrupt":      None,
        "logs":           [],
        "pipeline_state": {},
        "run_result":     {},
        "running":        True,
        "run_complete":   False,
        "created_at":     datetime.now(),
        "updated_at":     datetime.now(),
        # ── Observability fields ──────────────────────────────────────────────
        # node_timings: {node_id: {"start": datetime, "end": datetime}}
        "node_timings":   {},
        # hitl_events: [{step, action, wait_s, ts}]
        "hitl_events":    [],
    }


# ── Thread-safe session mutators (called from background threads) ─────────────

def _sess_set(tid: str, key: str, value):
    with _LOCK:
        if tid in _SESSIONS:
            _SESSIONS[tid][key] = value
            _SESSIONS[tid]["updated_at"] = datetime.now()


def _sess_log(tid: str, msg: str, level: str = "info"):
    ts = time.strftime("%H:%M:%S")
    with _LOCK:
        if tid in _SESSIONS:
            _SESSIONS[tid]["logs"].append({"ts": ts, "msg": msg, "level": level})
            _SESSIONS[tid]["updated_at"] = datetime.now()


def _sess_set_node(tid: str, name: str, status: str):
    with _LOCK:
        if tid in _SESSIONS:
            now = datetime.now()
            _SESSIONS[tid]["node_statuses"][name] = status
            timings = _SESSIONS[tid]["node_timings"].setdefault(name, {})

            if status == "active":
                # Close out the previously active node when the next one starts
                prev = _SESSIONS[tid].get("active_node")
                if prev and prev != name:
                    prev_t = _SESSIONS[tid]["node_timings"].setdefault(prev, {})
                    if "end" not in prev_t:
                        prev_t["end"] = now
                _SESSIONS[tid]["active_node"] = name
                timings["start"] = now

            elif status == "hitl":
                # Close out previous active agent node; hitl start = interrupt time
                prev = _SESSIONS[tid].get("active_node")
                if prev and prev != name:
                    prev_t = _SESSIONS[tid]["node_timings"].setdefault(prev, {})
                    if "end" not in prev_t:
                        prev_t["end"] = now
                if "start" not in timings:
                    timings["start"] = now

            elif status in ("done", "error"):
                if "end" not in timings:
                    timings["end"] = now
                if "start" not in timings:
                    timings["start"] = now

            _SESSIONS[tid]["updated_at"] = now


def _safe_state(result: dict) -> dict:
    skip = {"__interrupt__", "messages"}
    return {k: v for k, v in result.items() if k not in skip and v is not None}


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND GRAPH RUNNER (never touches st.session_state)
# ══════════════════════════════════════════════════════════════════════════════
def _run_graph(topic: str, thread_id: str):
    config   = {"configurable": {"thread_id": thread_id}}
    resume_q = _QUEUES[thread_id]

    try:
        _sess_log(thread_id, f"Pipeline started: \"{topic}\"")
        _sess_set_node(thread_id, "research_agent", "active")

        result = _graph.invoke(
            {"topic": topic, "status": "started", "human_feedback": ""},
            config=config,
        )

        # ── HITL loop ────────────────────────────────────────────────────────
        while result.get("__interrupt__"):
            payload   = result["__interrupt__"][0].value
            step      = payload.get("step", "unknown")
            hitl_node = HITL_NODE_MAP.get(step, step)

            _sess_set_node(thread_id, hitl_node, "hitl")
            _sess_log(thread_id, f"HITL checkpoint: {step.replace('_', ' ')} — awaiting review", "hitl")
            _sess_set(thread_id, "interrupt", payload)
            _sess_set(thread_id, "pipeline_state", _safe_state(result))
            _save_session_to_db(thread_id, _LOCK, _SESSIONS)   # persist HITL state so recovery survives restarts

            # Block until UI pushes a response — record how long the human takes
            _hitl_t0 = datetime.now()
            human_response = resume_q.get(block=True)
            _hitl_wait_s = (datetime.now() - _hitl_t0).total_seconds()
            with _LOCK:
                if thread_id in _SESSIONS:
                    _SESSIONS[thread_id]["hitl_events"].append({
                        "step":           step,
                        "action":         human_response.get("action", "unknown"),
                        "wait_s":         _hitl_wait_s,
                        "ts":             datetime.now(),
                        "input_payload":  _dt_serial(dict(payload)),
                        "human_response": _dt_serial(dict(human_response)),
                    })

            _sess_set(thread_id, "interrupt", None)
            _sess_set_node(thread_id, hitl_node, "done")
            _sess_log(thread_id, f"Resuming after {step.replace('_', ' ')}")
            _save_session_to_db(thread_id, _LOCK, _SESSIONS)   # persist HITL decision

            idx = NODE_IDS.index(hitl_node)
            if idx + 1 < len(NODE_IDS):
                _sess_set_node(thread_id, NODE_IDS[idx + 1], "active")

            result = _graph.invoke(Command(resume=human_response), config=config)

        # ── Mark remaining nodes complete + finalize their timings ──────────────
        with _LOCK:
            if thread_id in _SESSIONS:
                _batch_now = datetime.now()
                for n in NODE_IDS:
                    if _SESSIONS[thread_id]["node_statuses"][n] in ("pending", "active"):
                        _SESSIONS[thread_id]["node_statuses"][n] = "done"
                        _bt = _SESSIONS[thread_id]["node_timings"].setdefault(n, {})
                        if "end" not in _bt:
                            _bt["end"] = _batch_now
                        if "start" not in _bt:
                            _bt["start"] = _batch_now

        final = result.get("status", "complete")
        _sess_set(thread_id, "pipeline_state", _safe_state(result))
        _sess_set(thread_id, "run_result",     result)
        _sess_set(thread_id, "run_complete",   True)
        _sess_set(thread_id, "running",        False)
        _sess_log(thread_id, f"Pipeline complete — {final}", "done")
        _save_session_to_db(thread_id, _LOCK, _SESSIONS)   # final persist on success

    except Exception as exc:
        _sess_log(thread_id, f"Pipeline error: {exc}", "error")
        with _LOCK:
            if thread_id in _SESSIONS:
                for n in NODE_IDS:
                    if _SESSIONS[thread_id]["node_statuses"].get(n) == "active":
                        _SESSIONS[thread_id]["node_statuses"][n] = "error"
                _SESSIONS[thread_id]["running"]      = False
                _SESSIONS[thread_id]["run_complete"] = True
        _save_session_to_db(thread_id, _LOCK, _SESSIONS)   # final persist on error


def _resume_graph(thread_id: str, resume_q: queue.Queue):
    """
    Resume a pipeline that was paused at a HITL checkpoint when the app restarted.
    The session is already in _SESSIONS with interrupt=payload; this function
    waits for human input then drives the pipeline to completion exactly like
    the inner HITL loop in _run_graph.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # ── First resume (we were already interrupted on startup) ────────────
        with _LOCK:
            sess = _SESSIONS.get(thread_id)
            if not sess:
                return
            payload   = sess.get("interrupt", {})
        step      = payload.get("step", "unknown")
        hitl_node = HITL_NODE_MAP.get(step, step)

        _sess_log(thread_id, f"Session restored — awaiting review at {step.replace('_', ' ')}", "hitl")

        _hitl_t0       = datetime.now()
        human_response = resume_q.get(block=True)
        _hitl_wait_s   = (datetime.now() - _hitl_t0).total_seconds()

        with _LOCK:
            if thread_id in _SESSIONS:
                _SESSIONS[thread_id]["hitl_events"].append({
                    "step":           step,
                    "action":         human_response.get("action", "unknown"),
                    "wait_s":         _hitl_wait_s,
                    "ts":             datetime.now(),
                    "input_payload":  _dt_serial(dict(payload)),
                    "human_response": _dt_serial(dict(human_response)),
                })

        _sess_set(thread_id, "interrupt", None)
        _sess_set_node(thread_id, hitl_node, "done")
        _sess_log(thread_id, f"Resuming after {step.replace('_', ' ')}")
        _save_session_to_db(thread_id, _LOCK, _SESSIONS)

        idx = NODE_IDS.index(hitl_node)
        if idx + 1 < len(NODE_IDS):
            _sess_set_node(thread_id, NODE_IDS[idx + 1], "active")

        result = _graph.invoke(Command(resume=human_response), config=config)

        # ── Subsequent HITL checkpoints (same loop as _run_graph) ────────────
        while result.get("__interrupt__"):
            payload   = result["__interrupt__"][0].value
            step      = payload.get("step", "unknown")
            hitl_node = HITL_NODE_MAP.get(step, step)

            _sess_set_node(thread_id, hitl_node, "hitl")
            _sess_log(thread_id, f"HITL checkpoint: {step.replace('_', ' ')} — awaiting review", "hitl")
            _sess_set(thread_id, "interrupt", payload)
            _sess_set(thread_id, "pipeline_state", _safe_state(result))
            _save_session_to_db(thread_id, _LOCK, _SESSIONS)   # persist HITL state so recovery survives restarts

            _hitl_t0       = datetime.now()
            human_response = resume_q.get(block=True)
            _hitl_wait_s   = (datetime.now() - _hitl_t0).total_seconds()
            with _LOCK:
                if thread_id in _SESSIONS:
                    _SESSIONS[thread_id]["hitl_events"].append({
                        "step":           step,
                        "action":         human_response.get("action", "unknown"),
                        "wait_s":         _hitl_wait_s,
                        "ts":             datetime.now(),
                        "input_payload":  _dt_serial(dict(payload)),
                        "human_response": _dt_serial(dict(human_response)),
                    })

            _sess_set(thread_id, "interrupt", None)
            _sess_set_node(thread_id, hitl_node, "done")
            _sess_log(thread_id, f"Resuming after {step.replace('_', ' ')}")
            _save_session_to_db(thread_id, _LOCK, _SESSIONS)

            idx = NODE_IDS.index(hitl_node)
            if idx + 1 < len(NODE_IDS):
                _sess_set_node(thread_id, NODE_IDS[idx + 1], "active")

            result = _graph.invoke(Command(resume=human_response), config=config)

        # ── Finalise ─────────────────────────────────────────────────────────
        with _LOCK:
            if thread_id in _SESSIONS:
                _batch_now = datetime.now()
                for n in NODE_IDS:
                    if _SESSIONS[thread_id]["node_statuses"][n] in ("pending", "active"):
                        _SESSIONS[thread_id]["node_statuses"][n] = "done"
                        _bt = _SESSIONS[thread_id]["node_timings"].setdefault(n, {})
                        if "end" not in _bt:
                            _bt["end"] = _batch_now
                        if "start" not in _bt:
                            _bt["start"] = _batch_now

        final = result.get("status", "complete")
        _sess_set(thread_id, "pipeline_state", _safe_state(result))
        _sess_set(thread_id, "run_result",     result)
        _sess_set(thread_id, "run_complete",   True)
        _sess_set(thread_id, "running",        False)
        _sess_log(thread_id, f"Pipeline complete — {final}", "done")
        _save_session_to_db(thread_id, _LOCK, _SESSIONS)

    except Exception as exc:
        _sess_log(thread_id, f"Pipeline error: {exc}", "error")
        with _LOCK:
            if thread_id in _SESSIONS:
                for n in NODE_IDS:
                    if _SESSIONS[thread_id]["node_statuses"].get(n) == "active":
                        _SESSIONS[thread_id]["node_statuses"][n] = "error"
                _SESSIONS[thread_id]["running"]      = False
                _SESSIONS[thread_id]["run_complete"] = True
        _save_session_to_db(thread_id, _LOCK, _SESSIONS)


def _recover_hitl_sessions():
    """
    Called once at startup (after _SESSIONS/_QUEUES/_LOCK are ready).
    For every session that was paused at a HITL checkpoint when the process
    died, query LangGraph's SqliteSaver checkpoint to get the interrupt
    payload, restore the session as live, and spin up a _resume_graph thread.
    Safe to call on every Streamlit rerun — already-recovered sessions have
    a queue entry so they are skipped.
    """
    to_recover = []
    with _LOCK:
        for tid, sess in _SESSIONS.items():
            # Skip if already active/managed, already complete, or not a HITL session
            if tid in _QUEUES or sess.get("run_complete") or not sess.get("is_historical"):
                continue
            hitl_nodes = [nid for nid, ns in sess.get("node_statuses", {}).items()
                          if ns == "hitl"]
            if hitl_nodes:
                to_recover.append((tid, sess["topic"], hitl_nodes[0]))

    for tid, topic, hitl_node in to_recover:
        try:
            config   = {"configurable": {"thread_id": tid}}
            snapshot = _graph.get_state(config)

            # Extract interrupt payload from the LangGraph checkpoint
            interrupt_payload = None
            for task in (snapshot.tasks or []):
                for intr in getattr(task, "interrupts", []):
                    interrupt_payload = intr.value
                    break
                if interrupt_payload is not None:
                    break

            if interrupt_payload is None:
                continue   # Checkpoint doesn't match — leave as historical

            with _LOCK:
                _SESSIONS[tid]["is_historical"] = False
                _SESSIONS[tid]["running"]        = True
                _SESSIONS[tid]["interrupt"]      = interrupt_payload
                _SESSIONS[tid]["node_statuses"][hitl_node] = "hitl"
                _QUEUES[tid] = queue.Queue()

            threading.Thread(
                target=_resume_graph,
                args=(tid, _QUEUES[tid]),
                daemon=True,
            ).start()
            print(f"[recovery] Restored session {tid[:8]}… at HITL node '{hitl_node}'")

        except Exception as e:
            print(f"[recovery] Could not recover session {tid[:8]}…: {e}")


def _start_run(user_name: str, topic: str) -> str:
    """Create a session record, queue, and kick off the background thread."""
    thread_id = str(uuid.uuid4())
    with _LOCK:
        _SESSIONS[thread_id] = _new_session_record(user_name, topic, thread_id)
    _QUEUES[thread_id] = queue.Queue()
    _save_session_to_db(thread_id, _LOCK, _SESSIONS)   # initial INSERT
    threading.Thread(target=_run_graph, args=(topic, thread_id), daemon=True).start()
    return thread_id


# Recover sessions that were at HITL checkpoints when the process was killed.
# Safe to run on every Streamlit rerun — sessions already in _QUEUES are skipped.
_recover_hitl_sessions()


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT SESSION STATE BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
_prefs = _load_prefs()
for _key, _default in [
    ("user_name",         _prefs.get("user_name", "")),
    ("llm_provider",      _prefs.get("llm_provider", "gemini")),
    ("ollama_model",      _prefs.get("ollama_model", "llama3.2:latest")),
    ("openrouter_model",  _prefs.get("openrouter_model", "openai/gpt-4o-mini")),
    ("my_thread_ids",     []),
    ("viewing_thread_id", None),
    ("pending_tab",       None),
    ("_launching",        False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Apply saved provider on startup (llm_config persists in sys.modules)
from llm_config import set_provider, get_provider, list_ollama_models, ollama_running, list_openrouter_models
_startup_model = {
    "ollama":      st.session_state.ollama_model,
    "openrouter":  st.session_state.openrouter_model,
}.get(st.session_state.llm_provider)
set_provider(st.session_state.llm_provider, _startup_model)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _initials(name: str) -> str:
    parts = name.strip().split()
    if not parts: return "?"
    return "".join(w[0].upper() for w in parts[:2])


def _elapsed(dt: datetime) -> str:
    secs = int((datetime.now() - dt).total_seconds())
    if secs < 60:   return f"{secs}s"
    if secs < 3600: return f"{secs//60}m {secs%60}s"
    return f"{secs//3600}h {(secs%3600)//60}m"


def _session_status(sess: dict) -> str:
    if sess.get("interrupt"):    return "hitl"
    if sess.get("running"):      return "running"
    if sess.get("run_complete"):
        st = sess.get("run_result", {}).get("status", "")
        if "cancel" in st or "reject" in st: return "cancelled"
        return "done"
    return "pending"


def _progress_pct(sess: dict) -> int:
    statuses = sess.get("node_statuses", {})
    done = sum(1 for s in statuses.values() if s == "done")
    return int(done / max(len(NODE_IDS), 1) * 100)


def _active_label(sess: dict) -> str:
    nid = sess.get("active_node")
    if not nid: return "—"
    for node_id, label, _, _ in PIPELINE_NODES:
        if node_id == nid: return label
    return nid


def _snapshot_all() -> dict:
    with _LOCK:
        return {tid: dict(s) for tid, s in _SESSIONS.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  HTML RENDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _pill(status: str) -> str:
    MAP = {
        "running":   ('<span class="status-pill pill-running">◉ Running</span>'),
        "hitl":      ('<span class="status-pill pill-hitl">⏸ Review Required</span>'),
        "done":      ('<span class="status-pill pill-done">✓ Complete</span>'),
        "cancelled": ('<span class="status-pill pill-error">✗ Cancelled</span>'),
        "pending":   ('<span class="status-pill pill-pending">○ Pending</span>'),
    }
    return MAP.get(status, "")


def _render_session_card(sess: dict, is_mine: bool = False) -> str:
    status      = _session_status(sess)
    pct         = _progress_pct(sess)
    user        = sess.get("user_name") or "Anonymous"
    topic       = sess["topic"]
    topic_s     = topic[:60] + ("…" if len(topic) > 60 else "")
    elapsed     = _elapsed(sess["created_at"])
    initials    = _initials(user)
    step_lbl    = _active_label(sess)
    is_hist     = sess.get("is_historical", False)

    fill_cls  = {"running": "fill-r", "hitl": "fill-h", "done": "fill-d"}.get(status, "fill-r")
    card_cls  = {"hitl": "is-hitl", "running": "is-running", "done": "is-done"}.get(status, "")
    if is_mine: card_cls += " is-mine"
    if is_hist: card_cls += " is-historical"

    arch_badge = (
        '<span style="font-size:0.6rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.08em;color:#a0aec0;background:#f5f5f5;border:1px solid #e0e0e0;'
        'border-radius:4px;padding:2px 6px;margin-left:6px;">ARCHIVED</span>'
        if is_hist else ""
    )
    ts_str = sess["created_at"].strftime("%d %b %Y  %H:%M")

    return f"""
<div class="sess-card {card_cls}">
  <div class="card-header">
    <div class="card-user-tag">
      <span class="card-avatar">{initials}</span>
      {user}{arch_badge}
    </div>
    {_pill(status)}
  </div>
  <div class="card-topic">{topic_s}</div>
  <div class="mini-bar"><div class="mini-fill {fill_cls}" style="width:{pct}%;"></div></div>
  <div class="card-meta">
    <span>{'🕐 ' + ts_str if is_hist else 'Step: ' + step_lbl}</span>
    <span>{elapsed} ago</span>
  </div>
</div>"""


def _render_pipeline_graph(node_statuses: dict) -> str:
    ICONS  = {"pending": "○", "active": "◉", "done": "✓", "hitl": "⏸", "error": "✗"}
    lines  = []
    for i, (nid, lbl, icon, ntype) in enumerate(PIPELINE_NODES):
        st_ = node_statuses.get(nid, "pending")
        ico = ICONS.get(st_, "○")
        if i > 0:
            lines.append('<div class="pipe-connector">│</div>')
        lines.append(
            f'<div class="pipe-node pn-{st_}">'
            f'{icon} {ico} {lbl}'
            f'<span class="node-type">{ntype}</span>'
            f'</div>'
        )
    return '<div class="pipe-wrap">' + "\n".join(lines) + "</div>"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:8px 0 20px;">
      <div style="font-size:1.1rem;font-weight:800;letter-spacing:-0.02em;
          background:linear-gradient(90deg,#4f46e5,#7c3aed,#0891b2);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
          background-clip:text;">◈ Research Intelligence</div>
      <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;
          letter-spacing:0.12em;color:#a0aec0;margin-top:3px;">
          Multi-Agent Content Platform
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Identity ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Your Identity</div>', unsafe_allow_html=True)
    name_val = st.text_input(
        "Name",
        value=st.session_state.user_name,
        placeholder="Enter your name…",
        label_visibility="collapsed",
        key="__name_input__",
    )
    if name_val != st.session_state.user_name:
        st.session_state.user_name = name_val
        _save_prefs(user_name=name_val)   # persist across restarts

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

    # ── LLM Provider ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">LLM Provider</div>', unsafe_allow_html=True)
    _provider_options = ["gemini", "ollama", "openrouter"]
    _provider_labels  = {"gemini": "Gemini API", "ollama": "Ollama (local)", "openrouter": "OpenRouter"}
    _provider_idx     = _provider_options.index(st.session_state.llm_provider) \
                        if st.session_state.llm_provider in _provider_options else 0
    _provider_choice = st.radio(
        "Provider",
        options=_provider_options,
        format_func=lambda x: _provider_labels[x],
        index=_provider_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="__provider_radio__",
    )
    if _provider_choice != st.session_state.llm_provider:
        st.session_state.llm_provider = _provider_choice
        _save_prefs(llm_provider=_provider_choice)
        _new_model = {
            "ollama":     st.session_state.ollama_model,
            "openrouter": st.session_state.openrouter_model,
        }.get(_provider_choice)
        set_provider(_provider_choice, _new_model)

    if _provider_choice == "ollama":
        _ollama_ok = ollama_running()
        if not _ollama_ok:
            st.warning("Ollama not running — start it with `ollama serve`.", icon="⚠")
        else:
            _available_models = list_ollama_models()
            _model_idx = _available_models.index(st.session_state.ollama_model) \
                if st.session_state.ollama_model in _available_models else 0
            _model_choice = st.selectbox(
                "Model",
                options=_available_models,
                index=_model_idx,
                label_visibility="collapsed",
                key="__ollama_model__",
            )
            if _model_choice != st.session_state.ollama_model:
                st.session_state.ollama_model = _model_choice
                _save_prefs(ollama_model=_model_choice)
                set_provider("ollama", _model_choice)
    elif _provider_choice == "openrouter":
        _or_models = list_openrouter_models()
        _or_idx    = _or_models.index(st.session_state.openrouter_model) \
                     if st.session_state.openrouter_model in _or_models else 0
        _or_choice = st.selectbox(
            "Model",
            options=_or_models,
            index=_or_idx,
            label_visibility="collapsed",
            key="__openrouter_model__",
        )
        if _or_choice != st.session_state.openrouter_model:
            st.session_state.openrouter_model = _or_choice
            _save_prefs(openrouter_model=_or_choice)
            set_provider("openrouter", _or_choice)
        st.caption("Routes to 100+ models via openrouter.ai.")
    else:
        st.caption("Using Gemini 2.5 Flash via Google AI Studio.")

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

    # ── Launch form ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">New Research</div>', unsafe_allow_html=True)
    can_launch = bool(st.session_state.user_name.strip())

    with st.form("__launch__"):
        topic_val = st.text_input(
            "Topic",
            placeholder="e.g. AI agents reshaping enterprise software 2025…",
            label_visibility="collapsed",
        )
        launched = st.form_submit_button(
            "▶  Launch Research",
            type="primary",
            disabled=not can_launch,
            use_container_width=True,
        )

    # Reset guard when form is idle so future submissions work normally
    if not launched:
        st.session_state["_launching"] = False

    if launched and topic_val.strip() and can_launch and not st.session_state["_launching"]:
        st.session_state["_launching"] = True
        new_tid = _start_run(st.session_state.user_name.strip(), topic_val.strip())
        st.session_state.my_thread_ids.append(new_tid)
        st.session_state.viewing_thread_id = new_tid
        st.rerun()

    if not can_launch:
        st.caption("⚠ Enter your name above to launch.")

    # ── My sessions ───────────────────────────────────────────────────────────
    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">My Sessions</div>', unsafe_allow_html=True)

    my_tids = [t for t in st.session_state.my_thread_ids if t in _SESSIONS]
    if not my_tids:
        st.markdown('<div style="font-size:0.78rem;color:#a0aec0;padding:4px 0;">No sessions yet.</div>', unsafe_allow_html=True)
    else:
        for tid in reversed(my_tids):
            sess = _SESSIONS.get(tid)
            if not sess:
                continue
            stat   = _session_status(sess)
            s_col  = {"running": "#059669", "hitl": "#d97706", "done": "#4f46e5", "cancelled": "#dc2626"}.get(stat, "#a0aec0")
            s_icon = {"running": "◉", "hitl": "⏸", "done": "✓", "cancelled": "✗"}.get(stat, "○")
            t_short = sess["topic"][:22] + "…" if len(sess["topic"]) > 22 else sess["topic"]
            is_sel  = st.session_state.viewing_thread_id == tid
            prefix  = "▸ " if is_sel else "  "
            lbl     = f"{prefix}{s_icon} {t_short}"
            if st.button(lbl, key=f"sb_{tid}",
                         help=sess["topic"]):
                st.session_state.viewing_thread_id = tid
                st.rerun()

    # Refresh note — inline below session list, no absolute positioning
    st.markdown(
        '<div style="margin-top:16px;font-size:0.65rem;color:#c0cad8;'
        'text-align:center;line-height:1.6;padding-bottom:8px;">'
        'Platform auto-refreshes while agents are running</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — take a single consistent snapshot for this render pass
# ══════════════════════════════════════════════════════════════════════════════
all_sessions = _snapshot_all()

# ── Platform header ───────────────────────────────────────────────────────────
user_display = st.session_state.user_name or "Guest"
initials     = _initials(user_display)
st.markdown(f"""
<div class="platform-header">
  <div>
    <div class="platform-logo">◈ Research Intelligence Platform</div>
    <div class="platform-sub">LangGraph Multi-Agent · HITL Orchestration · TAPO Cycle</div>
  </div>
  <div class="user-chip">
    <span class="avatar-circle">{initials}</span>
    {user_display}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ops, tab_detail, tab_obs = st.tabs([
    "  ◈  Operations Center  ",
    "  ◈  Session Detail  ",
    "  ◈  Observability  ",
])

# Auto-switch to the requested tab (pending_tab is set by "Open in Detail" button)
if st.session_state.pending_tab is not None:
    _tab_idx = st.session_state.pending_tab
    st.session_state.pending_tab = None
    st.components.v1.html(f"""
    <script>
    (function() {{
        var tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
        if (tabs && tabs[{_tab_idx}]) {{
            tabs[{_tab_idx}].click();
        }}
    }})();
    </script>
    """, height=0)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OPERATIONS CENTER
# ══════════════════════════════════════════════════════════════════════════════
with tab_ops:

    # ── KPI metrics ───────────────────────────────────────────────────────────
    total_s   = len(all_sessions)
    active_s  = sum(1 for s in all_sessions.values() if s.get("running") and not s.get("interrupt"))
    hitl_s    = sum(1 for s in all_sessions.values() if s.get("interrupt"))
    done_s    = sum(1 for s in all_sessions.values() if s.get("run_complete"))

    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{total_s}</div>
          <div class="kpi-label">Total Sessions</div>
          <div class="kpi-sub neutral">all time</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        sub2 = f"{active_s} agents processing" if active_s else "all idle"
        st.markdown(f"""
        <div class="kpi-card kpi-active">
          <div class="kpi-value">{active_s}</div>
          <div class="kpi-label">Active Pipelines</div>
          <div class="kpi-sub">{sub2}</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        sub4 = "needs attention" if hitl_s else "none pending"
        cls4 = "kpi-hitl" if hitl_s else ""
        st.markdown(f"""
        <div class="kpi-card {cls4}">
          <div class="kpi-value">{hitl_s}</div>
          <div class="kpi-label">Awaiting Review</div>
          <div class="kpi-sub {'warn' if hitl_s else 'neutral'}">{sub4}</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card kpi-done">
          <div class="kpi-value">{done_s}</div>
          <div class="kpi-label">Completed</div>
          <div class="kpi-sub neutral">published or ended</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Session grid ──────────────────────────────────────────────────────────
    if not all_sessions:
        st.markdown("""
        <div style="text-align:center;padding:70px 0;">
          <div style="font-size:2.5rem;margin-bottom:14px;opacity:0.2;">◈</div>
          <div style="font-size:1rem;font-weight:600;color:#718096;">No active sessions</div>
          <div style="font-size:0.8rem;color:#a0aec0;margin-top:6px;">
              Launch a research session from the sidebar to get started
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Sort: HITL first → running → done/cancelled
        sorted_sessions = sorted(
            all_sessions.items(),
            key=lambda x: (
                0 if x[1].get("interrupt") else
                1 if x[1].get("running") else
                2
            )
        )

        # HITL alert banner
        if hitl_s:
            st.markdown(f"""
            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                border-radius:10px;padding:12px 18px;margin-bottom:18px;font-size:0.82rem;color:#d97706;">
                <strong>⏸ {hitl_s} session{"s" if hitl_s != 1 else ""} awaiting human review.</strong>
                Click "Open in Detail →" on a highlighted session to respond.
            </div>
            """, unsafe_allow_html=True)

        # Render in 3-column grid
        GRID_COLS = 3
        for row_start in range(0, len(sorted_sessions), GRID_COLS):
            row_items = sorted_sessions[row_start:row_start + GRID_COLS]
            cols = st.columns(GRID_COLS, gap="medium")
            for col_idx, (tid, sess) in enumerate(row_items):
                is_mine = tid in st.session_state.my_thread_ids
                with cols[col_idx]:
                    st.markdown(_render_session_card(sess, is_mine=is_mine), unsafe_allow_html=True)
                    btn_label = "⏸ Open for Review →" if sess.get("interrupt") else "Open in Detail →"
                    if st.button(btn_label, key=f"ops_view_{tid}"):
                        st.session_state.viewing_thread_id = tid
                        st.session_state.pending_tab = 1
                        st.rerun()

    # ── Auto-refresh while any pipeline is running (non-HITL) ─────────────────
    has_running = any(
        s.get("running") and not s.get("interrupt")
        for s in all_sessions.values()
    )
    if has_running:
        time.sleep(2.5)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SESSION DETAIL
# ══════════════════════════════════════════════════════════════════════════════
with tab_detail:
    vtid = st.session_state.viewing_thread_id

    if vtid is None or vtid not in all_sessions:
        st.markdown("""
        <div style="text-align:center;padding:80px 0;">
          <div style="font-size:2.5rem;margin-bottom:14px;opacity:0.18;">◈</div>
          <div style="font-size:1rem;font-weight:600;color:#718096;">No session selected</div>
          <div style="font-size:0.8rem;color:#a0aec0;margin-top:6px;">
              Launch a session from the sidebar or open one from the Operations Center
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sess         = all_sessions[vtid]
        status       = _session_status(sess)
        interrupt    = sess.get("interrupt")
        running      = sess.get("running", False)
        run_complete = sess.get("run_complete", False)
        run_result   = sess.get("run_result", {})
        pip_state    = sess.get("pipeline_state", {})
        logs         = sess.get("logs", [])
        node_stats   = sess.get("node_statuses", {})

        # ── Session detail header ─────────────────────────────────────────────
        hcol1, hcol2 = st.columns([3, 1], gap="medium")
        with hcol1:
            is_mine_flag = "  ·  <span style='color:#4f46e5;font-weight:600;'>My session</span>" if vtid in st.session_state.my_thread_ids else ""
            st.markdown(f"""
            <div class="detail-header">
              <div class="detail-topic">{sess['topic']}</div>
              <div class="detail-meta">
                <span style='color:#4a5568;'>{sess['user_name']}</span>
                &ensp;·&ensp; {_elapsed(sess['created_at'])} elapsed
                &ensp;·&ensp; Thread <span>{vtid[:12]}…</span>
                {is_mine_flag}
              </div>
            </div>
            """, unsafe_allow_html=True)
        with hcol2:
            st.markdown(f'<div style="text-align:right;padding-top:20px;">{_pill(status)}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-div" style="margin:16px 0 20px;"></div>', unsafe_allow_html=True)

        # ── Two-column layout: Pipeline graph (left) + Action area (right) ────
        graph_col, action_col = st.columns([1, 2], gap="large")

        with graph_col:
            st.markdown('<div class="section-label">Pipeline Graph</div>', unsafe_allow_html=True)
            st.markdown(_render_pipeline_graph(node_stats), unsafe_allow_html=True)

        with action_col:

            # ════════════════════════════════════════════════════════════
            #  HITL CHECKPOINT FORMS
            # ════════════════════════════════════════════════════════════
            if interrupt:
                step = interrupt.get("step", "")
                tapo = interrupt.get("tapo_phase", "")
                st.markdown(f"""
                <div class="hitl-banner">
                  <div class="hitl-tag">⏸ Human-in-the-Loop Checkpoint</div>
                  <div class="hitl-step">{step.replace("_", " ").title()}</div>
                  <div class="hitl-desc">{tapo}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── HITL #1: Review Queries ─────────────────────────────
                if step == "review_queries":
                    queries = interrupt.get("suggested_queries", [])
                    st.markdown("**Suggested Search Queries**")
                    st.caption(
                        "The Research Agent generated these queries. "
                        "Approve to proceed, edit them, or reject to cancel."
                    )
                    edited = []
                    for i, q in enumerate(queries):
                        edited.append(
                            st.text_input(f"Query {i + 1}", value=q, key=f"q_{vtid}_{i}")
                        )

                    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3, gap="small")
                    with c1:
                        if st.button("✅  Approve All",
                                     type="primary", key=f"aq_{vtid}"):
                            if not _try_resume(vtid, {"action": "approve"}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()
                    with c2:
                        reason = st.text_input(
                            "Reason", key=f"er_{vtid}",
                            label_visibility="collapsed",
                            placeholder="Reason for edits (optional)…"
                        )
                        if st.button("✏️  Use My Edits",
                                     key=f"eq_{vtid}"):
                            if not _try_resume(vtid, {"action": "edit", "queries": edited, "reason": reason}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()
                    with c3:
                        if st.button("❌  Reject", key=f"rq_{vtid}"):
                            if not _try_resume(vtid, {"action": "reject"}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()

                # ── HITL #2: Review Draft ───────────────────────────────
                elif step == "review_draft":
                    draft = interrupt.get("draft", "")
                    st.markdown("**Draft Content**")
                    st.caption(
                        "Review the Writer Agent's draft. "
                        "Approve, edit directly, request a full revision, or reject."
                    )
                    edited_draft = st.text_area(
                        "Draft", value=draft, height=340, key=f"de_{vtid}"
                    )

                    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4, gap="small")
                    with c1:
                        if st.button("✅  Approve",
                                     type="primary", key=f"ad_{vtid}"):
                            if not _try_resume(vtid, {"action": "approve"}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()
                    with c2:
                        if st.button("✏️  My Edits", key=f"ed_{vtid}"):
                            if not _try_resume(vtid, {"action": "edit", "edited_content": edited_draft}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()
                    with c3:
                        notes = st.text_input(
                            "Notes", key=f"rn_{vtid}",
                            label_visibility="collapsed",
                            placeholder="What to change…"
                        )
                        if st.button("🔄  Revise", key=f"rd_{vtid}"):
                            if not _try_resume(vtid, {"action": "revise", "feedback": notes or "Please improve."}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()
                    with c4:
                        if st.button("❌  Reject", key=f"rjd_{vtid}"):
                            if not _try_resume(vtid, {"action": "reject"}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()

                # ── HITL #3: Confirm Publish ────────────────────────────
                elif step == "confirm_publish":
                    wc = interrupt.get("word_count", "?")
                    st.markdown("**Final Publish Confirmation**")
                    st.caption(
                        f"The content is ready — **{wc} words**. "
                        "This action will call the publish API and cannot be undone."
                    )
                    preview = interrupt.get("content_preview", "")
                    if preview:
                        with st.expander("Content Preview", expanded=True):
                            st.markdown(preview)

                    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2, gap="small")
                    with c1:
                        if st.button("🚀  Confirm & Publish",
                                     type="primary", key=f"cp_{vtid}"):
                            if not _try_resume(vtid, {"action": "confirm"}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()
                    with c2:
                        if st.button("🚫  Cancel", key=f"cap_{vtid}"):
                            if not _try_resume(vtid, {"action": "cancel"}):
                                st.warning("Already responded to this checkpoint.")
                            st.rerun()

            # ════════════════════════════════════════════════════════════
            #  RUNNING — show progress banner
            # ════════════════════════════════════════════════════════════
            elif running:
                lbl = _active_label(sess)
                st.markdown(f"""
                <div class="running-banner">
                  <div class="running-tag">◉ Processing</div>
                  <div class="running-step">{lbl}</div>
                  <div class="running-sub">Pipeline is executing — page refreshes automatically</div>
                </div>
                """, unsafe_allow_html=True)

                # Progress bar
                pct = _progress_pct(sess)
                st.progress(pct / 100, text=f"{pct}% complete")

            # ════════════════════════════════════════════════════════════
            #  COMPLETE — show results
            # ════════════════════════════════════════════════════════════
            elif run_complete:
                final_status = run_result.get("status", "complete")

                if final_status == "published":
                    st.success("✅  Content successfully published!")
                elif "cancel" in final_status:
                    st.warning(f"Pipeline ended: **{final_status}**")
                elif "error" in final_status:
                    st.error(f"Pipeline encountered an error: **{final_status}**")
                else:
                    st.info(f"Pipeline finished — **{final_status}**")

                content = run_result.get("final_content") or run_result.get("draft_content")
                if content:
                    with st.expander("📄  Published Content", expanded=True):
                        st.markdown(content)

            # ════════════════════════════════════════════════════════════
            #  IDLE / INITIALISING
            # ════════════════════════════════════════════════════════════
            else:
                st.markdown("""
                <div style="text-align:center;padding:40px 0;color:#a0aec0;">
                  <div style="font-size:1.5rem;margin-bottom:10px;opacity:0.3;">◈</div>
                  <div style="font-size:0.88rem;">Session initialising…</div>
                </div>
                """, unsafe_allow_html=True)

        # ── State Inspector + Execution Log ───────────────────────────────────
        st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

        sc, lc = st.columns(2, gap="medium")

        with sc:
            st.markdown('<div class="section-label">State Inspector</div>', unsafe_allow_html=True)
            if pip_state:
                rows = []
                for k, v in pip_state.items():
                    if isinstance(v, list) and len(v) > 3:
                        disp = f"[{len(v)} items]"
                    elif isinstance(v, str) and len(v) > 110:
                        disp = v[:110] + "…"
                    else:
                        disp = str(v)
                    rows.append(
                        f'<div class="state-row">'
                        f'<span class="skey">{k}</span>'
                        f'<span class="sval">{disp}</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div class="state-container">{"".join(rows)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="state-container" style="color:#a0aec0;font-size:0.8rem;">No state yet.</div>',
                    unsafe_allow_html=True,
                )

        with lc:
            st.markdown('<div class="section-label">Execution Log</div>', unsafe_allow_html=True)
            if logs:
                log_lines = "".join(
                    f'<div class="log-line log-{e["level"]}">'
                    f'<span class="log-ts">[{e["ts"]}]</span>{e["msg"]}'
                    f'</div>'
                    for e in reversed(logs[-35:])
                )
                st.markdown(
                    f'<div class="log-container">{log_lines}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="log-container" style="color:#a0aec0;font-size:0.8rem;">No entries yet.</div>',
                    unsafe_allow_html=True,
                )

        # ── Auto-refresh for running session (but not when HITL is waiting) ───
        if running and not interrupt:
            time.sleep(2)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — OBSERVABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_obs:

    # ── Aggregate metrics across all sessions ─────────────────────────────────
    all_hitl_events: list[dict] = []
    for _s in all_sessions.values():
        all_hitl_events.extend(_s.get("hitl_events", []))

    completed_sessions = {
        tid: s for tid, s in all_sessions.items() if s.get("run_complete")
    }
    completed_durs = [
        (s["updated_at"] - s["created_at"]).total_seconds()
        for s in completed_sessions.values()
    ]
    avg_dur_s   = sum(completed_durs) / len(completed_durs) if completed_durs else 0
    avg_dur_fmt = (f"{int(avg_dur_s//60)}m {int(avg_dur_s%60)}s" if avg_dur_s >= 60
                   else f"{int(avg_dur_s)}s")

    hitl_waits  = [e["wait_s"] for e in all_hitl_events]
    avg_hitl_s  = sum(hitl_waits) / len(hitl_waits) if hitl_waits else 0
    avg_hitl_fmt = f"{int(avg_hitl_s)}s"

    error_count = sum(
        1 for s in all_sessions.values()
        if any(ns == "error" for ns in s.get("node_statuses", {}).values())
    )
    error_rate  = f"{error_count/len(all_sessions)*100:.0f}%" if all_sessions else "0%"

    # KPI row
    ok1, ok2, ok3, ok4 = st.columns(4, gap="medium")
    with ok1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">{len(completed_sessions)}</div>
          <div class="kpi-label">Completed Runs</div>
          <div class="kpi-sub neutral">of {len(all_sessions)} total</div>
        </div>""", unsafe_allow_html=True)
    with ok2:
        st.markdown(f"""
        <div class="kpi-card kpi-active">
          <div class="kpi-value">{avg_dur_fmt if completed_sessions else "—"}</div>
          <div class="kpi-label">Avg Pipeline Duration</div>
          <div class="kpi-sub neutral">end-to-end wall time</div>
        </div>""", unsafe_allow_html=True)
    with ok3:
        st.markdown(f"""
        <div class="kpi-card {'kpi-hitl' if all_hitl_events else ''}">
          <div class="kpi-value">{len(all_hitl_events)}</div>
          <div class="kpi-label">HITL Reviews</div>
          <div class="kpi-sub {'warn' if all_hitl_events else 'neutral'}">
            {f"avg {avg_hitl_fmt} response" if all_hitl_events else "none yet"}
          </div>
        </div>""", unsafe_allow_html=True)
    with ok4:
        st.markdown(f"""
        <div class="kpi-card {'kpi-hitl' if error_count else ''}">
          <div class="kpi-value">{error_rate}</div>
          <div class="kpi-label">Error Rate</div>
          <div class="kpi-sub {'warn' if error_count else 'neutral'}">
            {f"{error_count} session(s) errored" if error_count else "no errors"}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Execution Trace (Gantt) — selected session ────────────────────────────
    vtid_obs   = st.session_state.viewing_thread_id
    sess_obs   = all_sessions.get(vtid_obs) if vtid_obs else None

    if sess_obs:
        topic_short = sess_obs["topic"][:50] + ("…" if len(sess_obs["topic"]) > 50 else "")
        st.markdown(
            f'<div class="section-label">Execution Trace — {topic_short}</div>',
            unsafe_allow_html=True,
        )

        node_timings = sess_obs.get("node_timings", {})
        sess_start   = sess_obs["created_at"]

        # Build Gantt rows for any node that has timing data
        gantt_bars = []
        _color_map = {
            "done":    "#4f7de8",
            "hitl":    "#d97706",
            "active":  "#10b981",
            "error":   "#dc2626",
            "pending": "#cbd5e1",
        }
        for _nid, _lbl, _icon, _ntype in PIPELINE_NODES:
            timing = node_timings.get(_nid, {})
            if "start" not in timing:
                continue
            _start_s = (timing["start"] - sess_start).total_seconds()
            _end_s   = (timing.get("end", datetime.now()) - sess_start).total_seconds()
            _dur_s   = max(_end_s - _start_s, 0.2)
            _status  = sess_obs["node_statuses"].get(_nid, "pending")
            gantt_bars.append({
                "label":    _lbl,
                "type":     _ntype,
                "start":    _start_s,
                "duration": _dur_s,
                "status":   _status,
                "color":    _color_map.get(_status, "#cbd5e1"),
                "text":     f"{_dur_s:.1f}s",
            })

        if gantt_bars and _PLOTLY:
            fig_gantt = go.Figure()
            for bar in gantt_bars:
                fig_gantt.add_trace(go.Bar(
                    name=bar["label"],
                    y=[bar["label"]],
                    x=[bar["duration"]],
                    base=[bar["start"]],
                    orientation="h",
                    marker_color=bar["color"],
                    marker_line_width=0,
                    text=bar["text"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color="#ffffff", size=10, family="Inter, sans-serif"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{bar['label']}</b> [{bar['type']}]<br>"
                        f"Start: {bar['start']:.1f}s<br>"
                        f"Duration: {bar['duration']:.1f}s<br>"
                        f"Status: {bar['status']}<extra></extra>"
                    ),
                ))

            # Overlay HITL wait events on the Gantt
            hitl_step_to_node = {
                "review_queries": "Review Queries",
                "review_draft":   "Review Draft",
                "confirm_publish": "Publisher",
            }
            _evt_colors = {
                "approve":  "#10b981", "confirm": "#10b981",
                "edit":     "#6366f1", "revise":  "#d97706",
                "reject":   "#ef4444", "cancel":  "#ef4444",
            }
            for _evt in sess_obs.get("hitl_events", []):
                _node_lbl = hitl_step_to_node.get(_evt["step"], _evt["step"])
                _evt_start = (_evt["ts"] - sess_start).total_seconds() - _evt["wait_s"]
                fig_gantt.add_annotation(
                    x=_evt_start + _evt["wait_s"] / 2,
                    y=_node_lbl,
                    text=f"⏸ {_evt['action']} ({_evt['wait_s']:.0f}s wait)",
                    showarrow=False,
                    font=dict(
                        color=_evt_colors.get(_evt["action"], "#718096"),
                        size=9, family="Inter, sans-serif",
                    ),
                    yshift=14,
                )

            _max_x = max((b["start"] + b["duration"] for b in gantt_bars), default=1)
            fig_gantt.update_layout(
                barmode="overlay",
                xaxis=dict(
                    title=dict(
                        text="Seconds from session start",
                        font=dict(color="#a0aec0", size=10),
                    ),
                    range=[0, _max_x * 1.05],
                    gridcolor="#e8ecf2",
                    zerolinecolor="#d4dce8",
                    tickfont=dict(color="#a0aec0", size=10),
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(color="#4a5568", size=11),
                    gridcolor="#e8ecf2",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f8f9fb",
                font=dict(family="Inter, sans-serif", color="#718096", size=11),
                height=290,
                margin=dict(l=10, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_gantt)

        elif gantt_bars and not _PLOTLY:
            # Text fallback
            _max_total = max(b["start"] + b["duration"] for b in gantt_bars) or 1
            for bar in gantt_bars:
                _offset_pct = int(bar["start"] / _max_total * 100)
                _width_pct  = max(int(bar["duration"] / _max_total * 100), 2)
                st.markdown(
                    f'<div style="margin:4px 0;">'
                    f'<div style="font-size:0.73rem;color:#4a5568;margin-bottom:2px;">'
                    f'{bar["label"]} — {bar["text"]}</div>'
                    f'<div style="height:10px;background:#e2e8f0;border-radius:4px;overflow:hidden;">'
                    f'<div style="height:100%;margin-left:{_offset_pct}%;width:{_width_pct}%;'
                    f'background:{bar["color"]};border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:0.8rem;color:#a0aec0;padding:16px 0;">'
                'No timing data yet — recorded as the pipeline runs.</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="section-label">Execution Trace</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:0.8rem;color:#a0aec0;padding:16px 0;">'
            'Select a session from the sidebar or Operations Center to view its trace.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Node Performance + HITL Analytics ─────────────────────────────────────
    perf_col, hitl_col = st.columns(2, gap="large")

    with perf_col:
        st.markdown('<div class="section-label">Node Latency — Avg Across All Runs</div>',
                    unsafe_allow_html=True)

        # Collect durations per node from all sessions
        _node_durs: dict[str, list[float]] = {nid: [] for nid in NODE_IDS}
        for _s in all_sessions.values():
            for _nid, _timing in _s.get("node_timings", {}).items():
                if "start" in _timing and "end" in _timing:
                    _dur = (_timing["end"] - _timing["start"]).total_seconds()
                    if _nid in _node_durs and _dur >= 0:
                        _node_durs[_nid].append(_dur)

        _perf_rows = []
        for _nid, _lbl, _icon, _ntype in PIPELINE_NODES:
            _durs = _node_durs.get(_nid, [])
            if _durs:
                _perf_rows.append({
                    "label": _lbl, "type": _ntype,
                    "avg": sum(_durs) / len(_durs),
                    "min": min(_durs), "max": max(_durs),
                    "count": len(_durs),
                })

        if _perf_rows and _PLOTLY:
            _bar_colors = {
                "agent": "#4f7de8", "hitl": "#d97706", "tool": "#10b981",
            }
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(
                y=[r["label"] for r in _perf_rows],
                x=[r["avg"] for r in _perf_rows],
                orientation="h",
                marker=dict(
                    color=[_bar_colors.get(r["type"], "#4f7de8") for r in _perf_rows],
                    line=dict(width=0),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[r["max"] - r["avg"] for r in _perf_rows],
                    arrayminus=[r["avg"] - r["min"] for r in _perf_rows],
                    color="#d4dce8", thickness=1.5, width=5,
                ),
                text=[f"{r['avg']:.1f}s  ({r['count']} runs)" for r in _perf_rows],
                textposition="outside",
                textfont=dict(color="#718096", size=9, family="Inter, sans-serif"),
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Avg: %{x:.1f}s<extra></extra>",
            ))
            _perf_max = max(r["max"] for r in _perf_rows) * 1.25
            fig_perf.update_layout(
                xaxis=dict(
                    title=dict(
                        text="seconds",
                        font=dict(color="#a0aec0", size=9),
                    ),
                    range=[0, _perf_max],
                    gridcolor="#e8ecf2",
                    zerolinecolor="#d4dce8",
                    tickfont=dict(color="#a0aec0", size=9),
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(color="#4a5568", size=10),
                    gridcolor="#e8ecf2",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f8f9fb",
                font=dict(family="Inter, sans-serif", color="#718096", size=10),
                height=260,
                margin=dict(l=10, r=90, t=10, b=30),
            )
            st.plotly_chart(fig_perf)

            # Legend chips
            st.markdown(
                '<div style="display:flex;gap:12px;margin-top:4px;">'
                '<span style="font-size:0.68rem;color:#4f7de8;">■ Agent</span>'
                '<span style="font-size:0.68rem;color:#d97706;">■ HITL (incl. wait)</span>'
                '<span style="font-size:0.68rem;color:#10b981;">■ Tool</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif _perf_rows and not _PLOTLY:
            for r in _perf_rows:
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#4a5568;padding:3px 0;">'
                    f'{r["label"]}: <strong style="color:#6366f1">{r["avg"]:.1f}s</strong>'
                    f' avg ({r["count"]} runs)</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:0.8rem;color:#a0aec0;padding:16px 0;">No completed nodes yet.</div>',
                unsafe_allow_html=True,
            )

    with hitl_col:
        st.markdown('<div class="section-label">HITL Decision Analytics</div>',
                    unsafe_allow_html=True)

        if all_hitl_events:
            # Decision distribution donut
            _action_counts = Counter(e["action"] for e in all_hitl_events)
            _donut_colors  = {
                "approve":  "#10b981", "confirm":  "#10b981",
                "edit":     "#6366f1", "revise":   "#d97706",
                "reject":   "#ef4444", "cancel":   "#ef4444",
                "unknown":  "#a0aec0",
            }
            _labels = list(_action_counts.keys())
            _values = list(_action_counts.values())
            _colors = [_donut_colors.get(l, "#a0aec0") for l in _labels]

            if _PLOTLY:
                fig_donut = go.Figure(go.Pie(
                    labels=_labels,
                    values=_values,
                    hole=0.62,
                    marker=dict(
                        colors=_colors,
                        line=dict(color="#ffffff", width=2),
                    ),
                    textfont=dict(color="#4a5568", size=10, family="Inter, sans-serif"),
                    textposition="outside",
                    textinfo="label+percent",
                    pull=[0.04] * len(_labels),
                ))
                fig_donut.update_layout(
                    annotations=[dict(
                        text=(f"<b style='font-size:20px'>{sum(_values)}</b>"
                              f"<br><span style='font-size:10px;color:#718096'>reviews</span>"),
                        x=0.5, y=0.5,
                        font=dict(color="#4a5568", size=14, family="Inter, sans-serif"),
                        showarrow=False,
                    )],
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#718096", size=10),
                    height=220,
                    margin=dict(l=0, r=0, t=10, b=10),
                    showlegend=True,
                    legend=dict(
                        font=dict(color="#718096", size=9),
                        bgcolor="rgba(0,0,0,0)",
                        orientation="h",
                        x=0.5, xanchor="center", y=-0.05,
                    ),
                )
                st.plotly_chart(fig_donut)
            else:
                for action, count in _action_counts.items():
                    _pct = count / sum(_values) * 100
                    st.markdown(
                        f'<div style="font-size:0.8rem;color:#4a5568;padding:2px 0;">'
                        f'{action}: <strong>{count}</strong> ({_pct:.0f}%)</div>',
                        unsafe_allow_html=True,
                    )

            # Avg wait time per checkpoint
            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
            _step_waits: dict[str, list[float]] = {}
            for _e in all_hitl_events:
                _step_waits.setdefault(_e["step"], []).append(_e["wait_s"])

            _step_label_map = {
                "review_queries":  "Query Review",
                "review_draft":    "Draft Review",
                "confirm_publish": "Publish Confirm",
            }
            _wait_rows = [
                (
                    _step_label_map.get(step, step),
                    sum(waits) / len(waits),
                    len(waits),
                )
                for step, waits in _step_waits.items()
            ]
            if _wait_rows:
                _wait_html = "".join(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:5px 0;border-bottom:1px solid #f0f2f7;font-size:0.76rem;">'
                    f'<span style="color:#4a5568;">{lbl}</span>'
                    f'<span>'
                    f'<strong style="color:#fbbf24;">{avg:.0f}s</strong>'
                    f'<span style="color:#a0aec0;margin-left:6px;">×{cnt}</span>'
                    f'</span></div>'
                    for lbl, avg, cnt in _wait_rows
                )
                st.markdown(
                    f'<div style="margin-top:4px;">'
                    f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.1em;color:#a0aec0;margin-bottom:6px;">Avg Response Time</div>'
                    f'{_wait_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:0.8rem;color:#a0aec0;padding:16px 0;">'
                'No HITL reviews recorded yet.</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Sessions Comparison Table ─────────────────────────────────────────────
    st.markdown('<div class="section-label">Sessions — Comparison View</div>',
                unsafe_allow_html=True)

    if all_sessions:
        _status_color = {
            "running": "#34d399", "hitl": "#fbbf24", "done": "#a5b4fc",
            "cancelled": "#f87171", "pending": "#a0aec0",
        }
        _rows_html = ""
        for _tid, _s in sorted(all_sessions.items(),
                                key=lambda x: x[1]["created_at"], reverse=True):
            _st     = _session_status(_s)
            _dur    = _elapsed(_s["created_at"])
            _evts   = _s.get("hitl_events", [])
            _dec    = " → ".join(e["action"] for e in _evts) or "—"
            _out    = (_s.get("run_result", {}).get("status", "—")
                       if _s.get("run_complete") else "—")
            _done_n = sum(1 for ns in _s["node_statuses"].values() if ns == "done")
            _mine   = "✦ " if _tid in st.session_state.my_thread_ids else ""
            _col    = _status_color.get(_st, "#a0aec0")

            _rows_html += f"""
            <tr style="border-bottom:1px solid #f0f2f7;">
                <td style="padding:9px 14px;color:#4f46e5;font-weight:600;">{_mine}{_s['user_name'] or '—'}</td>
                <td style="padding:9px 14px;color:#1a2535;">{_s['topic'][:40]}{'…' if len(_s['topic']) > 40 else ''}</td>
                <td style="padding:9px 14px;"><span style="color:{_col};font-weight:700;font-size:0.72rem;">{_st.upper()}</span></td>
                <td style="padding:9px 14px;color:#718096;">{_dur}</td>
                <td style="padding:9px 14px;color:#718096;">{_done_n}/{len(NODE_IDS)}</td>
                <td style="padding:9px 14px;color:#d97706;font-size:0.76rem;">{_dec}</td>
                <td style="padding:9px 14px;color:#4a5568;font-size:0.76rem;">{_out}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e8ecf2;border-radius:12px;
                    overflow:hidden;overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
            <thead>
              <tr style="background:#f5f7fa;border-bottom:2px solid #e8ecf2;">
                <th style="text-align:left;padding:10px 14px;color:#a0aec0;font-size:0.65rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">User</th>
                <th style="text-align:left;padding:10px 14px;color:#a0aec0;font-size:0.65rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Topic</th>
                <th style="text-align:left;padding:10px 14px;color:#a0aec0;font-size:0.65rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Status</th>
                <th style="text-align:left;padding:10px 14px;color:#a0aec0;font-size:0.65rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Duration</th>
                <th style="text-align:left;padding:10px 14px;color:#a0aec0;font-size:0.65rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Nodes</th>
                <th style="text-align:left;padding:10px 14px;color:#a0aec0;font-size:0.65rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Decisions</th>
                <th style="text-align:left;padding:10px 14px;color:#a0aec0;font-size:0.65rem;
                    font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Outcome</th>
              </tr>
            </thead>
            <tbody>{_rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:0.68rem;color:#a0aec0;margin-top:6px;">✦ Your session</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.8rem;color:#a0aec0;padding:16px 0;">No sessions yet.</div>',
            unsafe_allow_html=True,
        )

    # ── HITL Event Feed ───────────────────────────────────────────────────────
    if all_hitl_events:
        st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">HITL Event Feed — Most Recent First</div>',
                    unsafe_allow_html=True)

        # Enrich with user/topic/thread_id from parent session; group by session
        _enriched_events = []
        for _s in all_sessions.values():
            for _e in _s.get("hitl_events", []):
                _enriched_events.append({
                    **_e,
                    "user":      _s.get("user_name", "?"),
                    "topic":     _s.get("topic", "?"),
                    "thread_id": _s.get("thread_id", ""),
                })

        # Build session groups ordered by most-recent event per session
        _sess_events: dict[str, list] = {}
        for _e in _enriched_events:
            _sess_events.setdefault(_e["thread_id"], []).append(_e)
        for _evts in _sess_events.values():
            _evts.sort(key=lambda x: x["ts"], reverse=True)
        _sess_order = sorted(
            _sess_events.keys(),
            key=lambda tid: _sess_events[tid][0]["ts"],
            reverse=True,
        )

        for _tid in _sess_order:
            _tevts     = _sess_events[_tid]
            _latest_ts = _tevts[0]["ts"].strftime("%H:%M")
            _user      = _tevts[0]["user"]
            _topic     = _tevts[0]["topic"]
            _tid_short = _tid[:8] if _tid else "?"
            _n         = len(_tevts)
            _outer_label = (
                f"**{_topic}** — "
                f"{_user} · {_n} checkpoint{'s' if _n != 1 else ''} "
                f"· last {_latest_ts} · `{_tid_short}…`"
            )
            with st.expander(_outer_label, expanded=False):
                # ── Tab per checkpoint (avoids nested-expander error) ─────────
                _tab_labels = []
                for _e in _tevts:
                    _act  = _e["action"]
                    _slbl = _e["step"].replace("_", " ").title()
                    _tab_labels.append(f"{_slbl} · {_act.upper()}")

                _tabs = st.tabs(_tab_labels)
                for _tab, _e in zip(_tabs, _tevts):
                    with _tab:
                        _ts     = _e["ts"].strftime("%H:%M:%S")
                        _action = _e["action"]
                        _step   = _e["step"]
                        _inp    = _e.get("input_payload") or {}
                        _resp   = _e.get("human_response") or {}
                        _dc     = ('green' if _action in ('approve','confirm')
                                   else 'violet' if _action == 'edit'
                                   else 'orange' if _action == 'revise'
                                   else 'red')

                        st.caption(f"{_ts} · :{_dc}[**{_action.upper()}**] · {_e['wait_s']:.0f}s wait")

                        # ── Input ─────────────────────────────────────────
                        st.markdown("**Input — what was presented for review**")
                        if _step == "review_queries":
                            _qs = _inp.get("suggested_queries", [])
                            if _qs:
                                for i, q in enumerate(_qs, 1):
                                    st.markdown(f"{i}. {q}")
                            else:
                                st.caption("_(no query data captured)_")

                        elif _step == "review_draft":
                            _draft = _inp.get("draft", "")
                            if _draft:
                                st.text_area("Draft", value=_draft, height=260,
                                             disabled=True, key=f"_fd_{_tid}_{_ts}")
                            else:
                                st.caption("_(no draft data captured)_")

                        elif _step == "confirm_publish":
                            _wc = _inp.get("word_count", "?")
                            st.markdown(f"Word count: **{_wc}**")
                            _preview = _inp.get("content_preview", "")
                            if _preview:
                                st.markdown(_preview)
                            else:
                                st.caption("_(no preview captured)_")

                        else:
                            st.json(_inp) if _inp else st.caption("_(no input data captured)_")

                        st.divider()

                        # ── Human Decision ────────────────────────────────
                        st.markdown(f"**Human Decision — :{_dc}[{_action.upper()}]**")

                        if _action == "edit" and _step == "review_queries":
                            _edited_qs = _resp.get("queries", [])
                            _reason    = _resp.get("reason", "")
                            if _edited_qs:
                                st.markdown("Edited queries:")
                                for i, q in enumerate(_edited_qs, 1):
                                    st.markdown(f"{i}. {q}")
                            if _reason:
                                st.markdown(f"Reason: _{_reason}_")

                        elif _action == "edit" and _step == "review_draft":
                            _edited = _resp.get("edited_content", "")
                            if _edited:
                                st.text_area("Edited draft", value=_edited, height=260,
                                             disabled=True, key=f"_fe_{_tid}_{_ts}")

                        elif _action == "revise":
                            _fb = _resp.get("feedback", "")
                            if _fb:
                                st.markdown(f"Feedback: _{_fb}_")

                        elif _action in ("approve", "confirm"):
                            st.caption("Approved without changes.")

                        elif _action in ("reject", "cancel"):
                            st.caption("Rejected / cancelled.")

                        else:
                            st.json(_resp) if _resp else st.caption("_(no response data captured)_")
