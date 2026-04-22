"""TGAER-compatible data structures for episode capture.

Mirrors tgaer.core.env_base.Transition and tgaer.evaluation.metrics.EvalResult
without importing them, so the plugin has no runtime dependency on TGAER.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Transition:
    """Single agent step — mirrors tgaer.core.env_base.Transition."""

    state: Any
    action: Any
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class EvalResult:
    """Episode evaluation outcome — mirrors tgaer.evaluation.metrics.EvalResult."""

    score: float
    details: Dict[str, Any]


@dataclass
class LLMTurn:
    """One user-assistant exchange."""

    user_message: str
    assistant_response: str
    model: str
    timestamp: float
    tool_calls_in_turn: int


@dataclass
class Episode:
    """Complete hermes session mapped as a TGAER episode."""

    session_id: str
    model: str
    platform: str
    started_at: float
    ended_at: float
    transitions: List[Transition]
    llm_turns: List[LLMTurn]
    eval_result: EvalResult
    metadata: Dict[str, Any] = field(default_factory=dict)


def episode_to_dict(episode: Episode) -> Dict[str, Any]:
    """Serialize an Episode to a JSON-compatible dict."""
    return {
        "schema_version": "1.0.0",
        "episode": {
            "session_id": episode.session_id,
            "model": episode.model,
            "platform": episode.platform,
            "started_at": episode.started_at,
            "ended_at": episode.ended_at,
            "metadata": episode.metadata,
        },
        "transitions": [dataclasses.asdict(t) for t in episode.transitions],
        "llm_turns": [dataclasses.asdict(t) for t in episode.llm_turns],
        "eval_result": dataclasses.asdict(episode.eval_result),
    }


def episode_to_json(episode: Episode, indent: int = 2) -> str:
    """Serialize an Episode to a JSON string."""
    return json.dumps(episode_to_dict(episode), indent=indent, default=str)
