"""Thread-safe per-session episode collector.

Accumulates tool calls and LLM turns during a hermes session,
then finalizes into a complete Episode at session end.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config as cfg
from .models import Episode, EvalResult, LLMTurn, Transition, episode_to_dict, episode_to_json
from .scoring import backfill_rewards, compute_eval_result

logger = logging.getLogger(__name__)

# Module-level collector registry
_collectors: Dict[str, "EpisodeCollector"] = {}
_collectors_lock = threading.Lock()


def get_collector(session_id: str) -> Optional["EpisodeCollector"]:
    with _collectors_lock:
        return _collectors.get(session_id)


def register_collector(session_id: str, collector: "EpisodeCollector") -> None:
    with _collectors_lock:
        _collectors[session_id] = collector


def pop_collector(session_id: str) -> Optional["EpisodeCollector"]:
    with _collectors_lock:
        return _collectors.pop(session_id, None)


class EpisodeCollector:
    """Accumulates tool calls and LLM turns for a single hermes session."""

    def __init__(
        self,
        session_id: str,
        model: str,
        platform: str,
        config: Dict[str, Any],
    ) -> None:
        self.session_id = session_id
        self.model = model
        self.platform = platform
        self.config = config
        self.started_at = time.time()
        self._transitions: List[Transition] = []
        self._llm_turns: List[LLMTurn] = []
        self._step_counter = 0
        self._current_turn_index = -1
        self._tool_history: List[Dict[str, Any]] = []
        self._last_user_message = ""
        self._lock = threading.Lock()

    def record_tool_call(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]],
        result: Any,
        task_id: str,
        session_id: str,
        tool_call_id: str,
    ) -> None:
        """Record a single tool call as a Transition."""
        max_chars = self.config.get("max_result_chars", 4096)
        result_str = ""
        if isinstance(result, str):
            result_str = result[:max_chars]
            if len(result) > max_chars:
                result_str += f"... [truncated, {len(result)} total chars]"

        with self._lock:
            step_index = self._step_counter
            self._step_counter += 1

            # Compact state snapshot
            state = {
                "turn_index": self._current_turn_index,
                "step_index": step_index,
                "message_count": len(self._llm_turns),
                "tool_history_summary": list(self._tool_history[-10:]),
                "last_user_intent": self._last_user_message[:200],
            }

            transition = Transition(
                state=state,
                action={"tool_name": tool_name, "args": args or {}},
                reward=0.0,
                done=False,
                info={
                    "tool_result": result_str,
                    "tool_call_id": tool_call_id,
                    "task_id": task_id,
                    "timestamp": time.time(),
                    "turn_index": self._current_turn_index,
                },
            )
            self._transitions.append(transition)

            # Track tool history for state snapshots
            is_error = result_str.lower().startswith("error:") or (
                result_str.startswith("{")
                and '"error"' in result_str[:200]
            )
            self._tool_history.append({
                "tool": tool_name,
                "success": not is_error,
            })

    def record_llm_turn(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        platform: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> None:
        """Record an LLM turn."""
        with self._lock:
            self._current_turn_index += 1
            self._last_user_message = user_message or ""

            # Count tool calls that belong to this turn
            tool_calls_in_turn = sum(
                1 for t in self._transitions
                if t.info.get("turn_index") == self._current_turn_index
            )

            turn = LLMTurn(
                user_message=user_message[:500] if user_message else "",
                assistant_response=assistant_response[:500] if assistant_response else "",
                model=model,
                timestamp=time.time(),
                tool_calls_in_turn=tool_calls_in_turn,
            )
            self._llm_turns.append(turn)

    def finalize(self, completed: bool, interrupted: bool) -> Optional[Episode]:
        """Finalize the episode: backfill rewards, compute EvalResult.

        Returns None if the episode doesn't meet min_steps_to_export.
        """
        with self._lock:
            min_steps = self.config.get("min_steps_to_export", 1)
            if len(self._transitions) < min_steps:
                logger.debug(
                    "tgaer-bridge: session %s has %d steps (min=%d), skipping export",
                    self.session_id, len(self._transitions), min_steps,
                )
                return None

            eval_result = compute_eval_result(
                self._transitions,
                self._llm_turns,
                completed,
                interrupted,
                self.config,
            )

            strategy = self.config.get("reward_strategy", "terminal_only")
            backfill_rewards(self._transitions, eval_result.score, strategy)

            return Episode(
                session_id=self.session_id,
                model=self.model,
                platform=self.platform,
                started_at=self.started_at,
                ended_at=time.time(),
                transitions=list(self._transitions),
                llm_turns=list(self._llm_turns),
                eval_result=eval_result,
                metadata={
                    "tool_call_count": len(self._transitions),
                    "llm_turn_count": len(self._llm_turns),
                    "reward_strategy": strategy,
                },
            )


def persist_episode(episode: Episode, config: Dict[str, Any]) -> Optional[Path]:
    """Write episode JSON + append to index. Returns path or None on error."""
    try:
        ep_dir = cfg.episodes_dir(config)
        timestamp = int(episode.ended_at)
        filename = f"{episode.session_id}_{timestamp}.json"
        ep_path = ep_dir / filename

        ep_path.write_text(episode_to_json(episode))

        # Append to index
        index_path = cfg.output_dir(config) / "episode_index.jsonl"
        index_entry = json.dumps({
            "session_id": episode.session_id,
            "model": episode.model,
            "platform": episode.platform,
            "score": episode.eval_result.score,
            "steps": len(episode.transitions),
            "llm_turns": len(episode.llm_turns),
            "started_at": episode.started_at,
            "ended_at": episode.ended_at,
            "file": f"episodes/{filename}",
        }, default=str)

        with open(index_path, "a") as f:
            f.write(index_entry + "\n")

        logger.info(
            "tgaer-bridge: persisted episode %s (score=%.4f, steps=%d) to %s",
            episode.session_id, episode.eval_result.score,
            len(episode.transitions), ep_path,
        )
        return ep_path

    except Exception as exc:
        logger.error("tgaer-bridge: failed to persist episode: %s", exc)
        return None
