"""tgaer-bridge plugin — data bridge from hermes sessions to TGAER episodes.

Captures tool calls and LLM turns via lifecycle hooks, maps them to
TGAER-compatible Transition/EvalResult structures, and persists episodes
as JSON for offline analysis and GEQ optimization.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from . import config as cfg
from .collector import (
    EpisodeCollector,
    get_collector,
    persist_episode,
    pop_collector,
    register_collector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_session_start(
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    """Initialize episode collector for a new session."""
    config = cfg.load_config()
    if not config.get("enabled", True):
        return
    if not session_id:
        return

    collector = EpisodeCollector(session_id, model, platform, config)
    register_collector(session_id, collector)
    logger.debug("tgaer-bridge: tracking session %s (model=%s)", session_id, model)


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    """Record a tool call as a Transition."""
    collector = get_collector(session_id)
    if collector is None:
        return

    excluded = collector.config.get("exclude_tools", [])
    if tool_name in excluded:
        return

    collector.record_tool_call(tool_name, args, result, task_id, session_id, tool_call_id)


def _on_post_llm_call(
    session_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: Optional[List[Dict]] = None,
    model: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    """Record an LLM turn."""
    collector = get_collector(session_id)
    if collector is None:
        return

    collector.record_llm_turn(
        user_message, assistant_response, model, platform, conversation_history,
    )


def _on_session_finalize(
    session_id: str = "",
    **_: Any,
) -> None:
    """Finalize and persist the episode at session boundary."""
    collector = pop_collector(session_id)
    if collector is None:
        return

    episode = collector.finalize(completed=True, interrupted=False)
    if episode is not None:
        persist_episode(episode, collector.config)


# ---------------------------------------------------------------------------
# Slash command
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
/tgaer-bridge — TGAER data bridge

Subcommands:
  status          Episode count, disk usage, last session score
  list [N]        Show last N episodes (default 10)
  export          Force-flush current session's data now
  config          Show current plugin configuration
"""


def _handle_slash(raw_args: str) -> Optional[str]:
    argv = raw_args.strip().split()
    if not argv or argv[0] in ("help", "-h", "--help"):
        return _HELP_TEXT

    sub = argv[0]
    config = cfg.load_config()

    if sub == "status":
        return _cmd_status(config)
    if sub == "list":
        n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 10
        return _cmd_list(config, n)
    if sub == "export":
        return _cmd_export(config)
    if sub == "config":
        return _cmd_config(config)

    return f"Unknown subcommand: {sub}\n\n{_HELP_TEXT}"


def _cmd_status(config: Dict[str, Any]) -> str:
    out_dir = cfg.output_dir(config)
    ep_dir = cfg.episodes_dir(config)
    index_path = out_dir / "episode_index.jsonl"

    episode_count = 0
    last_score = None
    if index_path.exists():
        lines = index_path.read_text().strip().splitlines()
        episode_count = len(lines)
        if lines:
            try:
                last = json.loads(lines[-1])
                last_score = last.get("score")
            except json.JSONDecodeError:
                pass

    # Disk usage
    total_bytes = sum(f.stat().st_size for f in ep_dir.rglob("*") if f.is_file())
    if index_path.exists():
        total_bytes += index_path.stat().st_size

    def _fmt_size(b: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    parts = [
        f"Episodes: {episode_count}",
        f"Disk usage: {_fmt_size(total_bytes)}",
    ]
    if last_score is not None:
        parts.append(f"Last score: {last_score:.4f}")

    return "[tgaer-bridge] " + " | ".join(parts)


def _cmd_list(config: Dict[str, Any], n: int) -> str:
    index_path = cfg.output_dir(config) / "episode_index.jsonl"
    if not index_path.exists():
        return "[tgaer-bridge] No episodes recorded yet."

    lines = index_path.read_text().strip().splitlines()
    if not lines:
        return "[tgaer-bridge] No episodes recorded yet."

    recent = lines[-n:]
    parts = [f"[tgaer-bridge] Last {min(n, len(recent))} episode(s):\n"]
    for line in reversed(recent):
        try:
            entry = json.loads(line)
            parts.append(
                f"  {entry.get('session_id', '?')[:20]}  "
                f"score={entry.get('score', '?'):.4f}  "
                f"steps={entry.get('steps', '?')}  "
                f"model={entry.get('model', '?')}"
            )
        except json.JSONDecodeError:
            continue

    return "\n".join(parts)


def _cmd_export(config: Dict[str, Any]) -> str:
    from .collector import _collectors, _collectors_lock

    with _collectors_lock:
        sessions = list(_collectors.keys())

    if not sessions:
        return "[tgaer-bridge] No active sessions to export."

    exported = 0
    for sid in sessions:
        collector = pop_collector(sid)
        if collector is None:
            continue
        episode = collector.finalize(completed=True, interrupted=False)
        if episode is not None:
            path = persist_episode(episode, collector.config)
            if path:
                exported += 1

    return f"[tgaer-bridge] Exported {exported} episode(s)."


def _cmd_config(config: Dict[str, Any]) -> str:
    parts = ["[tgaer-bridge] Current configuration:\n"]
    for key, value in sorted(config.items()):
        parts.append(f"  {key}: {value}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_command(
        "tgaer-bridge",
        handler=_handle_slash,
        description="TGAER data bridge: status, list, export, config.",
    )
