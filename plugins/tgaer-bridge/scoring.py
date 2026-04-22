"""Session-end scoring logic for tgaer-bridge.

Pure functions — no side effects, fully unit-testable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .models import EvalResult, LLMTurn, Transition


def _is_error_result(tool_result: str) -> bool:
    """Heuristic check for tool call errors."""
    if not isinstance(tool_result, str):
        return False
    # Check for JSON error payloads
    if tool_result.startswith("{"):
        try:
            parsed = json.loads(tool_result)
            if isinstance(parsed, dict) and "error" in parsed:
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    # Check for common error prefixes
    lower = tool_result.lower()
    return lower.startswith("error:") or lower.startswith("traceback")


def _per_tool_metrics(transitions: List[Transition]) -> Dict[str, Any]:
    """Compute per-tool-name success rate and count."""
    tools: Dict[str, Dict[str, int]] = {}
    for t in transitions:
        name = t.action.get("tool_name", "unknown") if isinstance(t.action, dict) else "unknown"
        bucket = tools.setdefault(name, {"count": 0, "successes": 0})
        bucket["count"] += 1
        if not _is_error_result(t.info.get("tool_result", "")):
            bucket["successes"] += 1

    return {
        name: {
            "count": data["count"],
            "success_rate": data["successes"] / data["count"] if data["count"] else 0.0,
        }
        for name, data in tools.items()
    }


def compute_eval_result(
    transitions: List[Transition],
    llm_turns: List[LLMTurn],
    completed: bool,
    interrupted: bool,
    config: Dict[str, Any],
) -> EvalResult:
    """Compute a composite EvalResult for a completed session."""
    total_steps = len(transitions)
    if total_steps == 0:
        return EvalResult(score=0.0, details={"reason": "no_tool_calls"})

    # Tool success rate
    success_count = sum(
        1 for t in transitions
        if not _is_error_result(t.info.get("tool_result", ""))
    )
    tool_success_rate = success_count / total_steps

    # Completion score
    if completed and not interrupted:
        completion_score = 1.0
    elif completed:
        completion_score = 0.5
    else:
        completion_score = 0.0

    # Efficiency: fewer tool calls per LLM turn is better
    turns = max(len(llm_turns), 1)
    efficiency = min(1.0, turns / total_steps) if total_steps > 0 else 1.0

    # Weighted composite
    weights = config.get("score_weights", {
        "completion": 0.5,
        "tool_success": 0.3,
        "efficiency": 0.2,
    })
    score = (
        weights.get("completion", 0.5) * completion_score
        + weights.get("tool_success", 0.3) * tool_success_rate
        + weights.get("efficiency", 0.2) * efficiency
    )

    details = {
        "completed": completed,
        "interrupted": interrupted,
        "total_steps": total_steps,
        "total_llm_turns": len(llm_turns),
        "tool_success_rate": tool_success_rate,
        "completion_score": completion_score,
        "efficiency": efficiency,
        "per_tool_metrics": _per_tool_metrics(transitions),
    }
    return EvalResult(score=round(score, 4), details=details)


def backfill_rewards(
    transitions: List[Transition],
    score: float,
    strategy: str = "terminal_only",
) -> None:
    """Backfill reward values on transitions in-place.

    Strategies:
        terminal_only: Final step gets the score, rest stay 0.0.
        uniform: score / num_steps to each.
        success_weighted: Proportional share to successful steps only.
    """
    if not transitions:
        return

    # Mark last transition as done
    transitions[-1].done = True

    if strategy == "terminal_only":
        transitions[-1].reward = score

    elif strategy == "uniform":
        per_step = score / len(transitions)
        for t in transitions:
            t.reward = round(per_step, 6)

    elif strategy == "success_weighted":
        successful = [
            t for t in transitions
            if not _is_error_result(t.info.get("tool_result", ""))
        ]
        if successful:
            per_step = score / len(successful)
            for t in successful:
                t.reward = round(per_step, 6)
        else:
            # All failed — give terminal step the score
            transitions[-1].reward = score

    else:
        # Fallback to terminal_only
        transitions[-1].reward = score
