"""
Tests for agent-related API endpoints in shadow-bridge.
"""

import pytest
import json
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_agents():
    """Sample agent data for testing."""
    return [
        {
            "id": "agent-1",
            "name": "Lead Architect",
            "role": "Lead-Architect",
            "status": "idle",
            "project_id": "project-1"
        },
        {
            "id": "agent-2",
            "name": "UI Designer",
            "role": "UI-Designer",
            "status": "active",
            "current_task": "Refactor CSS",
            "project_id": "project-1"
        }
    ]


class TestAgentEndpoints:
    """Tests for agent API endpoints."""

    def test_get_agents(self, flask_client, sample_agents):
        """GET /api/agents should return all agents."""
        with patch("web.routes.api.get_agents", return_value=sample_agents):
            response = flask_client.get("/api/agents")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data) == 2
            assert data[0]["name"] == "Lead Architect"

    def test_get_agent_by_id(self, flask_client, sample_agents):
        """GET /api/agents/<id> should return specific agent."""
        with patch("web.routes.api.get_agent", return_value=sample_agents[0]):
            response = flask_client.get("/api/agents/agent-1")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["id"] == "agent-1"
            assert data["name"] == "Lead Architect"

    def test_create_agent(self, flask_client):
        """POST /api/agents should create a new agent."""
        new_agent_data = {
            "name": "New Agent",
            "role": "QA-Specialist",
            "project_id": "project-1"
        }
        with patch("web.routes.api.add_agent", return_value={"success": True, "id": "agent-new"}):
            response = flask_client.post("/api/agents", 
                                      data=json.dumps(new_agent_data),
                                      content_type="application/json")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert data["id"] == "agent-new"

    def test_pause_agents(self, flask_client):
        """POST /api/agents/pause should send pause command."""
        with patch("web.routes.api.send_command_to_device") as mock_send:
            response = flask_client.post("/api/agents/pause")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            mock_send.assert_called_once_with("agent_command", {"action": "pause_all"})

    def test_kill_all_agents(self, flask_client):
        """POST /api/agents/kill-all should send kill command."""
        with patch("web.routes.api.send_command_to_device") as mock_send:
            response = flask_client.post("/api/agents/kill-all")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            mock_send.assert_called_once_with("agent_command", {"action": "kill_all"})

    def test_delete_agent(self, flask_client):
        """DELETE /api/agents/<id> should remove agent."""
        with patch("web.routes.api.delete_agent", return_value={"success": True}):
            response = flask_client.delete("/api/agents/agent-1")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
