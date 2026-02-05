"""
Tests for SubagentLoopController.

Tests the subagent loop system that enables agents to deploy
iterative execution loops for fix-test-fix cycles, code quality
improvements, and build repair.
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch

from web.services.subagent_loop import (
    SubagentLoopController,
    LoopStatus,
    LoopRequest,
    LoopResult,
    LoopIteration,
    get_subagent_loop_controller,
)


@pytest.fixture
def controller():
    """Fresh SubagentLoopController for each test."""
    ctrl = SubagentLoopController()
    yield ctrl
    ctrl.shutdown()


@pytest.fixture
def sample_task_small():
    """Sample small-scope task."""
    return {
        "id": "task-1",
        "title": "Fix typo in README",
        "description": "Change 'teh' to 'the'",
        "category": "todo",
        "scope": "small",
        "repo": "shadow-bridge"
    }


@pytest.fixture
def sample_task_medium_multi():
    """Sample medium-scope task with multiple items."""
    return {
        "id": "task-2",
        "title": "Fix all TODO comments in auth module",
        "description": "Find and implement each TODO in the authentication module",
        "category": "todo",
        "scope": "medium",
        "repo": "shadow-android"
    }


@pytest.fixture
def sample_task_refactor():
    """Sample refactoring task."""
    return {
        "id": "task-3",
        "title": "Refactor database manager",
        "description": "Clean up and optimize the database manager code",
        "category": "smell",
        "scope": "large",
        "repo": "shadow-bridge"
    }


class TestLoopMarkerDetection:
    """Tests for parsing loop protocol markers from agent output."""

    def test_detect_loop_start(self, controller):
        """Should detect LOOP_START marker with valid JSON."""
        line = '<<<SUBAGENT_LOOP_START {"iterations": 5, "purpose": "fix lint errors"}>>>'
        result = controller.detect_loop_trigger(line)
        
        assert result is not None
        assert result["type"] == "LOOP_START"
        assert result["iterations"] == 5
        assert result["purpose"] == "fix lint errors"

    def test_detect_loop_start_defaults(self, controller):
        """Should provide defaults for missing fields."""
        line = '<<<SUBAGENT_LOOP_START {}>>>'
        result = controller.detect_loop_trigger(line)
        
        assert result is not None
        assert result["type"] == "LOOP_START"
        assert result["iterations"] == 5  # default
        assert "purpose" in result

    def test_detect_loop_iteration(self, controller):
        """Should detect LOOP_ITERATION marker."""
        line = '<<<SUBAGENT_LOOP_ITERATION {"num": 3, "result": "fixed 7 of 10 errors"}>>>'
        result = controller.detect_loop_trigger(line)
        
        assert result is not None
        assert result["type"] == "LOOP_ITERATION"
        assert result["num"] == 3
        assert result["result"] == "fixed 7 of 10 errors"

    def test_detect_loop_end_success(self, controller):
        """Should detect LOOP_END marker with success status."""
        line = '<<<SUBAGENT_LOOP_END {"status": "success", "final_result": "all errors fixed"}>>>'
        result = controller.detect_loop_trigger(line)
        
        assert result is not None
        assert result["type"] == "LOOP_END"
        assert result["status"] == "success"
        assert result["final_result"] == "all errors fixed"

    def test_detect_loop_end_fail(self, controller):
        """Should detect LOOP_END marker with fail status."""
        line = '<<<SUBAGENT_LOOP_END {"status": "fail", "final_result": "could not fix all errors"}>>>'
        result = controller.detect_loop_trigger(line)
        
        assert result is not None
        assert result["type"] == "LOOP_END"
        assert result["status"] == "fail"

    def test_no_marker_returns_none(self, controller):
        """Should return None for lines without loop markers."""
        lines = [
            "Hello world",
            "Processing files...",
            "<<<TASK_STARTED {}>>>>>",  # Different marker
            "SUBAGENT_LOOP_START",  # Missing delimiters
        ]
        for line in lines:
            assert controller.detect_loop_trigger(line) is None

    def test_invalid_json_handles_gracefully(self, controller):
        """Should handle invalid JSON in markers gracefully."""
        line = '<<<SUBAGENT_LOOP_START {invalid json}>>>'
        result = controller.detect_loop_trigger(line)
        
        # Should still detect marker type with parse_error
        assert result is not None
        assert result["type"] == "LOOP_START"
        assert "parse_error" in result


class TestShouldDeployLoop:
    """Tests for loop deployment heuristics."""

    def test_small_scope_never_loops(self, controller, sample_task_small):
        """Small scope tasks should never use loops."""
        should_loop, reason = controller.should_deploy_loop(sample_task_small)
        assert should_loop is False
        assert "small" in reason.lower()

    def test_fix_all_triggers_loop(self, controller, sample_task_medium_multi):
        """Tasks with 'fix all' should trigger loops."""
        should_loop, reason = controller.should_deploy_loop(sample_task_medium_multi)
        assert should_loop is True

    def test_refactor_triggers_loop(self, controller, sample_task_refactor):
        """Refactoring tasks should trigger loops."""
        should_loop, reason = controller.should_deploy_loop(sample_task_refactor)
        assert should_loop is True
        # Reason may be from 'refactor' or 'clean up' indicator

    def test_smell_category_medium_scope(self, controller):
        """Category 'smell' with medium scope should trigger loops."""
        task = {
            "title": "Remove code smells",
            "description": "Clean up code smells in the module",
            "category": "smell",
            "scope": "medium"
        }
        should_loop, reason = controller.should_deploy_loop(task)
        assert should_loop is True

    def test_no_indicators_no_loop(self, controller):
        """Tasks without loop indicators should not loop."""
        task = {
            "title": "Update version number",
            "description": "Bump version to 1.2.3",
            "category": "other",
            "scope": "large"
        }
        should_loop, reason = controller.should_deploy_loop(task)
        assert should_loop is False


class TestLoopLifecycle:
    """Tests for full loop lifecycle: start -> iterations -> end."""

    def test_start_creates_active_loop(self, controller):
        """Starting a loop should create an active loop entry."""
        agent_id = "agent-1"
        event = {"type": "LOOP_START", "iterations": 5, "purpose": "testing"}
        
        controller.handle_loop_event(agent_id, event, task_id="task-1")
        
        loops = controller.get_active_loops()
        assert len(loops) == 1
        assert loops[0]["agent_id"] == agent_id
        assert loops[0]["max_iterations"] == 5

    def test_iteration_updates_progress(self, controller):
        """Iterations should update loop progress."""
        agent_id = "agent-1"
        
        # Start loop
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_START", "iterations": 5, "purpose": "testing"
        })
        
        # Send iteration
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_ITERATION", "num": 1, "result": "first pass done"
        })
        
        loop = controller.get_loop_for_agent(agent_id)
        assert loop is not None
        assert loop["current_iteration"] == 1

    def test_end_completes_loop(self, controller):
        """Ending a loop should return LoopResult and cleanup."""
        agent_id = "agent-1"
        
        # Start loop
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_START", "iterations": 5, "purpose": "testing"
        })
        
        # End loop
        result = controller.handle_loop_event(agent_id, {
            "type": "LOOP_END", "status": "success", "final_result": "done"
        })
        
        assert result is not None
        assert isinstance(result, LoopResult)
        assert result.status == LoopStatus.COMPLETED
        assert result.final_result == "done"
        
        # Should be no active loops
        assert len(controller.get_active_loops()) == 0

    def test_full_lifecycle(self, controller):
        """Full loop lifecycle with multiple iterations."""
        agent_id = "agent-1"
        
        # Start
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_START", "iterations": 3, "purpose": "fix errors"
        })
        
        # Iterations
        for i in range(1, 4):
            controller.handle_loop_event(agent_id, {
                "type": "LOOP_ITERATION", "num": i, "result": f"pass {i}"
            })
        
        # End
        result = controller.handle_loop_event(agent_id, {
            "type": "LOOP_END", "status": "success", "final_result": "all done"
        })
        
        assert result is not None
        assert result.iterations_completed == 3
        assert len(result.iteration_history) == 3


class TestSafetyLimits:
    """Tests for loop safety controls."""

    def test_max_iterations_clamped(self, controller):
        """Iterations should be clamped to MAX_ITERATIONS."""
        agent_id = "agent-1"
        
        # Request more than max
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_START", "iterations": 100, "purpose": "too many"
        })
        
        loop = controller.get_loop_for_agent(agent_id)
        assert loop["max_iterations"] == controller.MAX_ITERATIONS

    def test_concurrent_loop_limit(self, controller):
        """Should respect MAX_CONCURRENT_LOOPS limit."""
        # Fill up to max
        for i in range(controller.MAX_CONCURRENT_LOOPS):
            controller.handle_loop_event(f"agent-{i}", {
                "type": "LOOP_START", "iterations": 5, "purpose": "testing"
            })
        
        # Try one more
        controller.handle_loop_event("agent-extra", {
            "type": "LOOP_START", "iterations": 5, "purpose": "overflow"
        })
        
        # Should not create new loop
        assert len(controller.get_active_loops()) == controller.MAX_CONCURRENT_LOOPS
        assert controller.get_loop_for_agent("agent-extra") is None

    def test_duplicate_loop_prevented(self, controller):
        """Agent with active loop should not start another."""
        agent_id = "agent-1"
        
        # Start first loop
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_START", "iterations": 5, "purpose": "first"
        })
        
        # Try to start second
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_START", "iterations": 3, "purpose": "second"
        })
        
        # Should still be first loop
        loop = controller.get_loop_for_agent(agent_id)
        assert loop["max_iterations"] == 5

    def test_cancel_loop(self, controller):
        """Should be able to cancel an active loop."""
        agent_id = "agent-1"
        
        controller.handle_loop_event(agent_id, {
            "type": "LOOP_START", "iterations": 5, "purpose": "will cancel"
        })
        
        result = controller.cancel_loop(agent_id, "user cancelled")
        
        assert result is not None
        assert result.status == LoopStatus.CANCELLED
        assert controller.get_loop_for_agent(agent_id) is None


class TestEventCallbacks:
    """Tests for loop event broadcasting."""

    def test_callback_on_start(self, controller):
        """Should broadcast event on loop start."""
        events = []
        controller.register_callback(lambda t, d: events.append((t, d)))
        
        controller.handle_loop_event("agent-1", {
            "type": "LOOP_START", "iterations": 5, "purpose": "testing"
        })
        
        assert len(events) == 1
        assert events[0][0] == "subagent_loop_started"

    def test_callback_on_iteration(self, controller):
        """Should broadcast event on iteration."""
        events = []
        controller.register_callback(lambda t, d: events.append((t, d)))
        
        controller.handle_loop_event("agent-1", {
            "type": "LOOP_START", "iterations": 5, "purpose": "testing"
        })
        controller.handle_loop_event("agent-1", {
            "type": "LOOP_ITERATION", "num": 1, "result": "done"
        })
        
        assert len(events) == 2
        assert events[1][0] == "subagent_loop_iteration"

    def test_callback_on_complete(self, controller):
        """Should broadcast event on completion."""
        events = []
        controller.register_callback(lambda t, d: events.append((t, d)))
        
        controller.handle_loop_event("agent-1", {
            "type": "LOOP_START", "iterations": 5, "purpose": "testing"
        })
        controller.handle_loop_event("agent-1", {
            "type": "LOOP_END", "status": "success", "final_result": "done"
        })
        
        assert len(events) == 2
        assert events[1][0] == "subagent_loop_completed"


class TestSingleton:
    """Tests for global singleton instance."""

    def test_get_subagent_loop_controller_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        ctrl1 = get_subagent_loop_controller()
        ctrl2 = get_subagent_loop_controller()
        assert ctrl1 is ctrl2
