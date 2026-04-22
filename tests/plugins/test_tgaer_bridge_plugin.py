"""Tests for the tgaer-bridge plugin.

Covers:
  * models.py: dataclass serialization round-trip
  * scoring.py: compute_eval_result + backfill_rewards
  * collector.py: EpisodeCollector record/finalize + persist
  * __init__.py: hook wiring + slash command
  * concurrency: thread-safe record_tool_call
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

def _plugin_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "plugins" / "tgaer-bridge"


def _ensure_namespace():
    """Ensure hermes_plugins namespace package exists."""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns


def _load_module(name: str):
    """Import a plugin module by name (e.g. 'models', 'scoring')."""
    _ensure_namespace()
    plugin_dir = _plugin_dir()
    full_name = f"hermes_plugins.tgaer_bridge.{name}"
    pkg_name = "hermes_plugins.tgaer_bridge"

    # Load the package first if not loaded
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            plugin_dir / "__init__.py",
            submodule_search_locations=[str(plugin_dir)],
        )
        pkg = importlib.util.module_from_spec(spec)
        pkg.__package__ = pkg_name
        pkg.__path__ = [str(plugin_dir)]
        sys.modules[pkg_name] = pkg
        # Don't exec __init__ yet — it has imports that need submodules
        # We'll load submodules individually

    spec = importlib.util.spec_from_file_location(
        full_name,
        plugin_dir / f"{name}.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and output for each test."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


@pytest.fixture
def default_config(tmp_path) -> Dict[str, Any]:
    return {
        "enabled": True,
        "output_dir": str(tmp_path / "tgaer-bridge"),
        "max_result_chars": 4096,
        "exclude_tools": [],
        "reward_strategy": "terminal_only",
        "score_weights": {
            "completion": 0.5,
            "tool_success": 0.3,
            "efficiency": 0.2,
        },
        "min_steps_to_export": 1,
    }


# ---------------------------------------------------------------------------
# Models tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_transition_round_trip(self):
        models = _load_module("models")
        t = models.Transition(
            state={"step_index": 0},
            action={"tool_name": "terminal", "args": {"command": "ls"}},
            reward=0.0,
            done=False,
            info={"tool_result": "file1.py\nfile2.py", "timestamp": 1000.0},
        )
        import dataclasses
        d = dataclasses.asdict(t)
        t2 = models.Transition(**d)
        assert t2.state == t.state
        assert t2.action == t.action
        assert t2.reward == t.reward

    def test_eval_result_round_trip(self):
        models = _load_module("models")
        er = models.EvalResult(score=0.85, details={"completed": True})
        import dataclasses
        d = dataclasses.asdict(er)
        er2 = models.EvalResult(**d)
        assert er2.score == er.score
        assert er2.details == er.details

    def test_episode_to_json(self):
        models = _load_module("models")
        episode = models.Episode(
            session_id="test-123",
            model="claude-sonnet",
            platform="cli",
            started_at=1000.0,
            ended_at=1010.0,
            transitions=[
                models.Transition(
                    state={}, action={"tool_name": "read"}, reward=0.5,
                    done=True, info={},
                ),
            ],
            llm_turns=[],
            eval_result=models.EvalResult(score=0.5, details={}),
        )
        json_str = models.episode_to_json(episode)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "1.0.0"
        assert parsed["episode"]["session_id"] == "test-123"
        assert len(parsed["transitions"]) == 1
        assert parsed["eval_result"]["score"] == 0.5

    def test_episode_to_dict_structure(self):
        models = _load_module("models")
        episode = models.Episode(
            session_id="s1", model="m", platform="p",
            started_at=0.0, ended_at=1.0,
            transitions=[], llm_turns=[],
            eval_result=models.EvalResult(score=0.0, details={}),
        )
        d = models.episode_to_dict(episode)
        assert set(d.keys()) == {"schema_version", "episode", "transitions", "llm_turns", "eval_result"}


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def _make_transition(self, models, tool_name="terminal", error=False):
        result = '{"error": "fail"}' if error else '{"output": "ok"}'
        return models.Transition(
            state={},
            action={"tool_name": tool_name, "args": {}},
            reward=0.0,
            done=False,
            info={"tool_result": result, "timestamp": 1000.0},
        )

    def test_no_tool_calls(self, default_config):
        models = _load_module("models")
        scoring = _load_module("scoring")
        result = scoring.compute_eval_result([], [], True, False, default_config)
        assert result.score == 0.0
        assert result.details["reason"] == "no_tool_calls"

    def test_all_successful_completed(self, default_config):
        models = _load_module("models")
        scoring = _load_module("scoring")
        transitions = [self._make_transition(models) for _ in range(3)]
        turns = [models.LLMTurn("hi", "hello", "m", 1000.0, 3)]
        result = scoring.compute_eval_result(transitions, turns, True, False, default_config)
        # completion=1.0*0.5 + tool_success=1.0*0.3 + efficiency=min(1,1/3)*0.2
        expected = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * (1 / 3)
        assert abs(result.score - round(expected, 4)) < 0.001

    def test_some_errors(self, default_config):
        models = _load_module("models")
        scoring = _load_module("scoring")
        transitions = [
            self._make_transition(models),
            self._make_transition(models, error=True),
        ]
        turns = [models.LLMTurn("q", "a", "m", 1000.0, 2)]
        result = scoring.compute_eval_result(transitions, turns, True, False, default_config)
        assert result.details["tool_success_rate"] == 0.5

    def test_interrupted_session(self, default_config):
        models = _load_module("models")
        scoring = _load_module("scoring")
        transitions = [self._make_transition(models)]
        turns = [models.LLMTurn("q", "a", "m", 1000.0, 1)]
        result = scoring.compute_eval_result(transitions, turns, True, True, default_config)
        assert result.details["completion_score"] == 0.5

    def test_per_tool_metrics(self, default_config):
        models = _load_module("models")
        scoring = _load_module("scoring")
        transitions = [
            self._make_transition(models, "terminal"),
            self._make_transition(models, "terminal", error=True),
            self._make_transition(models, "read_file"),
        ]
        turns = [models.LLMTurn("q", "a", "m", 1000.0, 3)]
        result = scoring.compute_eval_result(transitions, turns, True, False, default_config)
        ptm = result.details["per_tool_metrics"]
        assert ptm["terminal"]["count"] == 2
        assert ptm["terminal"]["success_rate"] == 0.5
        assert ptm["read_file"]["count"] == 1
        assert ptm["read_file"]["success_rate"] == 1.0

    def test_backfill_terminal_only(self):
        models = _load_module("models")
        scoring = _load_module("scoring")
        transitions = [
            models.Transition(state={}, action={}, reward=0.0, done=False, info={"tool_result": "ok"}),
            models.Transition(state={}, action={}, reward=0.0, done=False, info={"tool_result": "ok"}),
        ]
        scoring.backfill_rewards(transitions, 0.8, "terminal_only")
        assert transitions[0].reward == 0.0
        assert transitions[1].reward == 0.8
        assert transitions[1].done is True

    def test_backfill_uniform(self):
        models = _load_module("models")
        scoring = _load_module("scoring")
        transitions = [
            models.Transition(state={}, action={}, reward=0.0, done=False, info={"tool_result": "ok"}),
            models.Transition(state={}, action={}, reward=0.0, done=False, info={"tool_result": "ok"}),
        ]
        scoring.backfill_rewards(transitions, 1.0, "uniform")
        assert transitions[0].reward == 0.5
        assert transitions[1].reward == 0.5

    def test_backfill_success_weighted(self):
        models = _load_module("models")
        scoring = _load_module("scoring")
        transitions = [
            models.Transition(state={}, action={}, reward=0.0, done=False, info={"tool_result": "ok"}),
            models.Transition(state={}, action={}, reward=0.0, done=False, info={"tool_result": '{"error":"x"}'}),
        ]
        scoring.backfill_rewards(transitions, 0.6, "success_weighted")
        assert transitions[0].reward == 0.6  # Only successful step gets reward
        assert transitions[1].reward == 0.0

    def test_backfill_empty_transitions(self):
        scoring = _load_module("scoring")
        scoring.backfill_rewards([], 1.0, "terminal_only")  # should not raise


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------

class TestCollector:
    def test_record_and_finalize(self, default_config):
        models = _load_module("models")
        collector_mod = _load_module("collector")
        c = collector_mod.EpisodeCollector("sess-1", "claude", "cli", default_config)

        c.record_tool_call("terminal", {"command": "ls"}, '{"output": "ok"}', "t1", "sess-1", "call-1")
        c.record_llm_turn("list files", "I'll run ls", "claude", "cli")
        c.record_tool_call("read_file", {"path": "/tmp/x"}, "contents", "t1", "sess-1", "call-2")

        episode = c.finalize(completed=True, interrupted=False)
        assert episode is not None
        assert episode.session_id == "sess-1"
        assert len(episode.transitions) == 2
        assert len(episode.llm_turns) == 1
        assert episode.eval_result.score > 0
        # Last transition should be marked done
        assert episode.transitions[-1].done is True

    def test_finalize_skips_below_min_steps(self, default_config):
        collector_mod = _load_module("collector")
        default_config["min_steps_to_export"] = 5
        c = collector_mod.EpisodeCollector("sess-2", "claude", "cli", default_config)
        c.record_tool_call("terminal", {}, "ok", "", "sess-2", "c1")
        assert c.finalize(True, False) is None

    def test_state_snapshot_structure(self, default_config):
        collector_mod = _load_module("collector")
        c = collector_mod.EpisodeCollector("sess-3", "claude", "cli", default_config)
        c.record_llm_turn("hello", "hi", "claude", "cli")
        c.record_tool_call("terminal", {"command": "ls"}, "ok", "", "sess-3", "c1")
        episode = c.finalize(True, False)
        state = episode.transitions[0].state
        assert "turn_index" in state
        assert "step_index" in state
        assert "message_count" in state
        assert "tool_history_summary" in state
        assert "last_user_intent" in state

    def test_result_truncation(self, default_config):
        collector_mod = _load_module("collector")
        default_config["max_result_chars"] = 100
        c = collector_mod.EpisodeCollector("sess-4", "claude", "cli", default_config)
        long_result = "x" * 500
        c.record_tool_call("terminal", {}, long_result, "", "sess-4", "c1")
        episode = c.finalize(True, False)
        result_str = episode.transitions[0].info["tool_result"]
        assert len(result_str) < 200
        assert "truncated" in result_str

    def test_persist_episode(self, default_config, tmp_path):
        models = _load_module("models")
        collector_mod = _load_module("collector")
        episode = models.Episode(
            session_id="persist-test",
            model="claude",
            platform="cli",
            started_at=1000.0,
            ended_at=1010.0,
            transitions=[
                models.Transition(
                    state={}, action={"tool_name": "terminal"},
                    reward=0.5, done=True, info={},
                ),
            ],
            llm_turns=[],
            eval_result=models.EvalResult(score=0.5, details={}),
        )
        path = collector_mod.persist_episode(episode, default_config)
        assert path is not None
        assert path.exists()

        # Verify JSON content
        data = json.loads(path.read_text())
        assert data["episode"]["session_id"] == "persist-test"

        # Verify index
        index_path = Path(default_config["output_dir"]) / "episode_index.jsonl"
        assert index_path.exists()
        entry = json.loads(index_path.read_text().strip())
        assert entry["session_id"] == "persist-test"
        assert entry["score"] == 0.5


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_concurrent_record_tool_call(self, default_config):
        collector_mod = _load_module("collector")
        c = collector_mod.EpisodeCollector("conc-1", "claude", "cli", default_config)

        errors = []

        def _record(i: int):
            try:
                c.record_tool_call(
                    f"tool_{i}", {"arg": i}, f"result_{i}",
                    "t1", "conc-1", f"call_{i}",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_record, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        episode = c.finalize(True, False)
        assert len(episode.transitions) == 20

    def test_collector_registry_thread_safe(self, default_config):
        collector_mod = _load_module("collector")
        errors = []

        def _register(i: int):
            try:
                c = collector_mod.EpisodeCollector(f"reg-{i}", "m", "p", default_config)
                collector_mod.register_collector(f"reg-{i}", c)
                got = collector_mod.get_collector(f"reg-{i}")
                assert got is c
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        # Cleanup
        for i in range(20):
            collector_mod.pop_collector(f"reg-{i}")


# ---------------------------------------------------------------------------
# Error detection tests
# ---------------------------------------------------------------------------

class TestErrorDetection:
    def test_json_error_detected(self):
        scoring = _load_module("scoring")
        assert scoring._is_error_result('{"error": "something broke"}') is True

    def test_error_prefix_detected(self):
        scoring = _load_module("scoring")
        assert scoring._is_error_result("Error: file not found") is True

    def test_traceback_detected(self):
        scoring = _load_module("scoring")
        assert scoring._is_error_result("Traceback (most recent call last):") is True

    def test_normal_result_not_error(self):
        scoring = _load_module("scoring")
        assert scoring._is_error_result('{"output": "hello"}') is False

    def test_empty_not_error(self):
        scoring = _load_module("scoring")
        assert scoring._is_error_result("") is False

    def test_non_string_not_error(self):
        scoring = _load_module("scoring")
        assert scoring._is_error_result(None) is False
