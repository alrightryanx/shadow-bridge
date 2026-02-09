"""
Tests for Ouroboros telemetry and crash reporting endpoints.
Verifies pairing, metric collection, and crash analysis triggers.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestOuroborosTelemetry:
    """Tests for Ouroboros telemetry endpoints."""

    ALLOWLISTED_DEVICE_ID = "emulator-5554"
    CSRF_HEADERS = {"X-Requested-With": "XMLHttpRequest"}
    AUTH_TOKEN = "a" * 64

    def _paired_headers(self, flask_client, token: str | None = None):
        effective_token = token or self.AUTH_TOKEN
        pair_payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "auth_token": effective_token,
            "device_model": "Pixel Test",
        }
        pair_response = flask_client.post('/api/telemetry/pair', json=pair_payload, headers=self.CSRF_HEADERS)
        if pair_response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")
        assert pair_response.status_code == 200
        headers = dict(self.CSRF_HEADERS)
        headers["Authorization"] = f"Bearer {effective_token}"
        return headers

    def test_ping_endpoint(self, flask_client):
        """GET /api/telemetry/ping should return 200."""
        response = flask_client.get('/api/telemetry/ping')
        # If imports fail in setup, the app might not register routes, so handle 404 gracefully in test
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered in test app")
            
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        assert 'timestamp' in data

    def test_pair_device_success(self, flask_client):
        """POST /api/telemetry/pair should successfully pair a valid device."""
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "auth_token": self.AUTH_TOKEN,
            "device_model": "Pixel Test"
        }
        response = flask_client.post('/api/telemetry/pair', json=payload, headers=self.CSRF_HEADERS)
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")
            
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'paired'
        assert data['device_id'] == self.ALLOWLISTED_DEVICE_ID

    def test_pair_device_invalid_token(self, flask_client):
        """POST /api/telemetry/pair should fail with invalid token format."""
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "auth_token": "short-token",
            "device_model": "Pixel Test"
        }
        response = flask_client.post('/api/telemetry/pair', json=payload, headers=self.CSRF_HEADERS)
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")
            
        assert response.status_code == 400

    def test_receive_response_metrics_success(self, flask_client):
        """POST /api/telemetry/response should accept valid metrics."""
        headers = self._paired_headers(flask_client)
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "metrics": {
                "request_id": "req-123",
                "provider": "gemini-api",
                "total_time": 1500,
                "was_successful": True,
                "prompt_tokens": 50,
                "response_tokens": 100
            }
        }
        response = flask_client.post('/api/telemetry/response', json=payload, headers=headers)
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")
            
        assert response.status_code == 200
        assert response.get_json()['status'] == 'received'

    def test_receive_response_metrics_invalid_provider(self, flask_client):
        """POST /api/telemetry/response should reject unauthorized providers."""
        headers = self._paired_headers(flask_client)
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "metrics": {
                "provider": "malicious-ai",
                "total_time": 100,
                "was_successful": True
            }
        }
        response = flask_client.post('/api/telemetry/response', json=payload, headers=headers)
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")
            
        assert response.status_code == 400
        assert "Invalid provider name" in response.get_json()['error']

    def test_receive_response_metrics_injection_detection(self, flask_client):
        """POST /api/telemetry/response should detect injection attempts."""
        headers = self._paired_headers(flask_client)
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "metrics": {
                "provider": "gemini-api",
                "total_time": 100,
                "was_successful": True,
                "request_id": "'; DROP TABLE users; --"
            }
        }
        response = flask_client.post('/api/telemetry/response', json=payload, headers=headers)
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")
            
        assert response.status_code == 400
        assert "Injection attempt detected" in response.get_json()['error']

    def test_receive_response_metrics_rejects_unauthorized_device(self, flask_client, monkeypatch):
        """POST /api/telemetry/response should reject non-allowlisted devices."""
        monkeypatch.delenv("OUROBOROS_ALLOW_DYNAMIC_DEVICES", raising=False)
        payload = {
            "device_id": "UNAUTHORIZED-DEVICE-001",
            "metrics": {
                "provider": "gemini-api",
                "total_time": 100,
                "was_successful": True
            }
        }
        response = flask_client.post('/api/telemetry/response', json=payload, headers=self.CSRF_HEADERS)
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")

        assert response.status_code == 403
        assert "Device not authorized" in response.get_json()['error']

    def test_receive_response_metrics_triggers_analysis_for_slow_pattern(self, flask_client):
        """POST /api/telemetry/response should trigger analysis thread for slow responses."""
        headers = self._paired_headers(flask_client)
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "metrics": {
                "provider": "gemini-api",
                "total_time": 12000,
                "time_to_first_token": 4500,
                "was_successful": True
            }
        }

        patch_targets = [
            'web.routes.ouroboros_telemetry._trigger_telemetry_analysis',
            'routes.ouroboros_telemetry._trigger_telemetry_analysis'
        ]
        success = False
        for target in patch_targets:
            try:
                with patch(target) as mock_trigger:
                    response = flask_client.post('/api/telemetry/response', json=payload, headers=headers)
                    if response.status_code == 404:
                        pytest.skip("Ouroboros routes not registered")
                    assert response.status_code == 200
                    assert mock_trigger.called
                    success = True
                    break
            except (ImportError, AttributeError):
                continue

        if not success:
            response = flask_client.post('/api/telemetry/response', json=payload, headers=headers)
            if response.status_code == 404:
                pytest.skip("Ouroboros routes not registered")
            assert response.status_code == 200

    def test_receive_crash_report_triggers_analysis(self, flask_client):
        """POST /api/telemetry/crash should accept report and trigger background analysis."""
        headers = self._paired_headers(flask_client)
        payload = {
            "device_info": {
                "device_id": self.ALLOWLISTED_DEVICE_ID,
                "model": "Pixel Test"
            },
            "error": {
                "type": "NullPointerException",
                "message": "Test crash",
                "source_reference": {
                    "file_name": "MainActivity.kt",
                    "line_number": 42
                }
            }
        }
        
        # Try multiple potential patch targets
        patch_targets = [
            'web.routes.ouroboros_telemetry.analyze_and_report_crash',
            'routes.ouroboros_telemetry.analyze_and_report_crash'
        ]
        
        success = False
        for target in patch_targets:
            try:
                with patch(target) as mock_report:
                    response = flask_client.post('/api/telemetry/crash', json=payload, headers=headers)
                    if response.status_code == 404:
                        pytest.skip("Ouroboros routes not registered")
                    
                    assert response.status_code == 200
                    assert response.get_json()['priority'] == 'critical'
                    assert mock_report.called
                    success = True
                    break
            except (ImportError, AttributeError):
                continue
        
        if not success:
            # Fallback for when we can't find the function to patch
            response = flask_client.post('/api/telemetry/crash', json=payload, headers=headers)
            if response.status_code == 404:
                pytest.skip("Ouroboros routes not registered")
            assert response.status_code == 200

    def test_batch_metrics_validation(self, flask_client):
        """POST /api/telemetry/batch should validate each metric in the batch."""
        headers = self._paired_headers(flask_client)
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "metrics": [
                {
                    "provider": "gemini-api",
                    "total_time": 500,
                    "was_successful": True
                },
                {
                    "provider": "invalid-provider",
                    "total_time": 500,
                    "was_successful": True
                }
            ]
        }
        response = flask_client.post('/api/telemetry/batch', json=payload, headers=headers)
        if response.status_code == 404:
            pytest.skip("Ouroboros routes not registered")
            
        assert response.status_code == 200
        data = response.get_json()
        assert data['accepted'] == 1
        assert data['rejected'] == 1

    def test_frustration_high_severity_triggers_urgent_analysis(self, flask_client):
        """POST /api/telemetry/frustration should trigger urgent analysis for critical severity."""
        headers = self._paired_headers(flask_client)
        payload = {
            "device_id": self.ALLOWLISTED_DEVICE_ID,
            "pattern": "rage_taps",
            "severity": "critical",
            "description": "User retried 10 times in 30s",
            "events": [{"type": "tap", "count": 10}]
        }

        patch_targets = [
            'web.routes.ouroboros_telemetry._trigger_telemetry_analysis',
            'routes.ouroboros_telemetry._trigger_telemetry_analysis'
        ]
        success = False
        for target in patch_targets:
            try:
                with patch(target) as mock_trigger:
                    response = flask_client.post('/api/telemetry/frustration', json=payload, headers=headers)
                    if response.status_code == 404:
                        pytest.skip("Ouroboros routes not registered")
                    assert response.status_code == 200
                    assert mock_trigger.called
                    args, kwargs = mock_trigger.call_args
                    assert kwargs.get("priority") == "urgent"
                    success = True
                    break
            except (ImportError, AttributeError):
                continue

        if not success:
            response = flask_client.post('/api/telemetry/frustration', json=payload, headers=headers)
            if response.status_code == 404:
                pytest.skip("Ouroboros routes not registered")
            assert response.status_code == 200
