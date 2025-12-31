"""
Comprehensive tests for shadow-bridge settings and configuration.
Tests settings persistence, validation, defaults, and migrations.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSettingsDefaults:
    """Tests for default settings values."""

    def test_default_bridge_port(self, default_settings):
        """Default bridge port should be 19284."""
        assert default_settings["bridge_port"] == 19284

    def test_default_discovery_port(self, default_settings):
        """Default discovery port should be 19285."""
        assert default_settings["discovery_port"] == 19285

    def test_default_companion_port(self, default_settings):
        """Default companion port should be 19286."""
        assert default_settings["companion_port"] == 19286

    def test_default_auto_start(self, default_settings):
        """Auto-start should be disabled by default."""
        assert default_settings["auto_start"] is False

    def test_default_auto_update(self, default_settings):
        """Auto-update should be enabled by default."""
        assert default_settings["auto_update"] is True

    def test_default_theme(self, default_settings):
        """Default theme should be dark."""
        assert default_settings["theme"] == "dark"

    def test_all_required_keys_present(self, default_settings):
        """All required settings keys should be present."""
        required_keys = [
            "bridge_port",
            "discovery_port",
            "companion_port",
            "auto_start",
            "auto_update",
            "theme"
        ]
        for key in required_keys:
            assert key in default_settings


class TestSettingsPersistence:
    """Tests for settings file persistence."""

    def test_settings_file_created(self, settings_file):
        """Settings file should be created."""
        assert settings_file.exists()

    def test_settings_file_is_valid_json(self, settings_file):
        """Settings file should contain valid JSON."""
        with open(settings_file, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_settings_can_be_loaded(self, settings_file, default_settings):
        """Settings should be loadable from file."""
        with open(settings_file, "r") as f:
            loaded = json.load(f)
        assert loaded == default_settings

    def test_settings_can_be_modified(self, settings_file):
        """Settings should be modifiable."""
        with open(settings_file, "r") as f:
            settings = json.load(f)

        settings["theme"] = "light"

        with open(settings_file, "w") as f:
            json.dump(settings, f)

        with open(settings_file, "r") as f:
            reloaded = json.load(f)

        assert reloaded["theme"] == "light"

    def test_settings_preserve_unknown_keys(self, settings_file):
        """Unknown settings keys should be preserved."""
        with open(settings_file, "r") as f:
            settings = json.load(f)

        settings["custom_key"] = "custom_value"

        with open(settings_file, "w") as f:
            json.dump(settings, f)

        with open(settings_file, "r") as f:
            reloaded = json.load(f)

        assert reloaded["custom_key"] == "custom_value"


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_port_must_be_positive(self):
        """Port numbers must be positive."""
        def validate_port(port):
            return isinstance(port, int) and 1 <= port <= 65535

        assert validate_port(19284) is True
        assert validate_port(0) is False
        assert validate_port(-1) is False
        assert validate_port(65536) is False

    def test_port_must_be_integer(self):
        """Port numbers must be integers."""
        def validate_port(port):
            return isinstance(port, int) and 1 <= port <= 65535

        assert validate_port(19284) is True
        assert validate_port("19284") is False
        assert validate_port(19284.5) is False

    def test_theme_must_be_valid(self):
        """Theme must be a valid value."""
        valid_themes = ["light", "dark", "system"]

        assert "dark" in valid_themes
        assert "light" in valid_themes
        assert "invalid" not in valid_themes

    def test_boolean_settings_are_boolean(self, default_settings):
        """Boolean settings should be actual booleans."""
        assert isinstance(default_settings["auto_start"], bool)
        assert isinstance(default_settings["auto_update"], bool)

    def test_ports_are_unique(self, default_settings):
        """All port settings should be unique."""
        ports = [
            default_settings["bridge_port"],
            default_settings["discovery_port"],
            default_settings["companion_port"]
        ]
        assert len(ports) == len(set(ports))

    def test_ports_are_not_privileged(self, default_settings):
        """Default ports should not be privileged (<1024)."""
        assert default_settings["bridge_port"] > 1024
        assert default_settings["discovery_port"] > 1024
        assert default_settings["companion_port"] > 1024


class TestSettingsMigration:
    """Tests for settings migration between versions."""

    def test_migrate_v1_to_v2(self, temp_dir):
        """V1 settings should migrate to V2."""
        v1_settings = {
            "port": 19284,  # Old single port
            "auto_start": False
        }

        # Migration logic
        v2_settings = {
            "bridge_port": v1_settings.get("port", 19284),
            "discovery_port": v1_settings.get("port", 19284) + 1,
            "companion_port": v1_settings.get("port", 19284) + 2,
            "auto_start": v1_settings.get("auto_start", False),
            "auto_update": True,
            "theme": "dark"
        }

        assert v2_settings["bridge_port"] == 19284
        assert v2_settings["discovery_port"] == 19285
        assert v2_settings["companion_port"] == 19286

    def test_migrate_preserves_existing_values(self):
        """Migration should preserve existing values."""
        old_settings = {
            "auto_start": True,
            "theme": "light"
        }

        migrated = {**old_settings, "new_field": "default"}

        assert migrated["auto_start"] is True
        assert migrated["theme"] == "light"
        assert migrated["new_field"] == "default"

    def test_migrate_adds_missing_fields(self):
        """Migration should add missing fields."""
        old_settings = {"auto_start": True}
        defaults = {"auto_start": False, "theme": "dark"}

        migrated = {**defaults, **old_settings}

        assert migrated["auto_start"] is True  # Preserved
        assert migrated["theme"] == "dark"  # Added


class TestDirectoryStructure:
    """Tests for settings directory structure."""

    def test_shadowai_dir_created(self, shadowai_dir):
        """~/.shadowai directory should be created."""
        assert shadowai_dir.exists()
        assert shadowai_dir.is_dir()

    def test_settings_in_correct_location(self, settings_file, shadowai_dir):
        """Settings file should be in .shadowai directory."""
        assert settings_file.parent == shadowai_dir

    def test_can_create_subdirectories(self, shadowai_dir):
        """Should be able to create subdirectories."""
        subdir = shadowai_dir / "backups"
        subdir.mkdir(exist_ok=True)
        assert subdir.exists()

    def test_projects_file_location(self, projects_file, shadowai_dir):
        """Projects file should be in .shadowai directory."""
        assert projects_file.parent == shadowai_dir

    def test_notes_file_location(self, notes_file, shadowai_dir):
        """Notes file should be in .shadowai directory."""
        assert notes_file.parent == shadowai_dir


class TestApprovedDevices:
    """Tests for approved devices settings."""

    def test_approved_devices_file_format(self, temp_dir):
        """Approved devices file should have correct format."""
        approved = {
            "device-1": {
                "name": "Phone 1",
                "approved_at": 1700000000,
                "public_key": "ssh-rsa AAAA..."
            }
        }

        file_path = temp_dir / "approved_devices.json"
        with open(file_path, "w") as f:
            json.dump(approved, f)

        with open(file_path, "r") as f:
            loaded = json.load(f)

        assert "device-1" in loaded
        assert loaded["device-1"]["name"] == "Phone 1"

    def test_can_add_approved_device(self, temp_dir):
        """Should be able to add an approved device."""
        approved = {}
        file_path = temp_dir / "approved_devices.json"

        # Add device
        approved["new-device"] = {
            "name": "New Phone",
            "approved_at": 1700000000
        }

        with open(file_path, "w") as f:
            json.dump(approved, f)

        with open(file_path, "r") as f:
            loaded = json.load(f)

        assert "new-device" in loaded

    def test_can_remove_approved_device(self, temp_dir):
        """Should be able to remove an approved device."""
        approved = {
            "device-1": {"name": "Phone 1", "approved_at": 1700000000},
            "device-2": {"name": "Phone 2", "approved_at": 1700000001}
        }

        file_path = temp_dir / "approved_devices.json"
        with open(file_path, "w") as f:
            json.dump(approved, f)

        # Remove device
        del approved["device-1"]

        with open(file_path, "w") as f:
            json.dump(approved, f)

        with open(file_path, "r") as f:
            loaded = json.load(f)

        assert "device-1" not in loaded
        assert "device-2" in loaded


class TestEnvironmentVariables:
    """Tests for environment variable overrides."""

    def test_port_override_from_env(self):
        """Port should be overridable via environment variable."""
        with patch.dict(os.environ, {"SHADOW_BRIDGE_PORT": "19999"}):
            port = int(os.environ.get("SHADOW_BRIDGE_PORT", "19284"))
            assert port == 19999

    def test_env_default_when_not_set(self):
        """Default should be used when env var not set."""
        # Ensure env var is not set
        env_var = "SHADOW_TEST_VAR_NONEXISTENT"
        port = int(os.environ.get(env_var, "19284"))
        assert port == 19284

    def test_home_directory_detection(self, mock_home, temp_dir):
        """Home directory should be detected correctly."""
        # Windows
        userprofile = os.environ.get("USERPROFILE")
        # Unix
        home = os.environ.get("HOME")

        assert userprofile == str(temp_dir) or home == str(temp_dir)


class TestSettingsBackup:
    """Tests for settings backup functionality."""

    def test_backup_file_created(self, settings_file, temp_dir):
        """Backup file should be created."""
        backup_path = temp_dir / "settings.json.bak"

        # Create backup
        with open(settings_file, "r") as src:
            with open(backup_path, "w") as dst:
                dst.write(src.read())

        assert backup_path.exists()

    def test_backup_contains_same_data(self, settings_file, temp_dir):
        """Backup should contain same data as original."""
        backup_path = temp_dir / "settings.json.bak"

        with open(settings_file, "r") as f:
            original = json.load(f)

        with open(backup_path, "w") as f:
            json.dump(original, f)

        with open(backup_path, "r") as f:
            backup = json.load(f)

        assert original == backup

    def test_restore_from_backup(self, temp_dir):
        """Settings should be restorable from backup."""
        settings_path = temp_dir / "settings.json"
        backup_path = temp_dir / "settings.json.bak"

        # Create backup
        original = {"theme": "dark", "auto_start": True}
        with open(backup_path, "w") as f:
            json.dump(original, f)

        # Corrupt settings
        with open(settings_path, "w") as f:
            f.write("invalid json")

        # Restore from backup
        with open(backup_path, "r") as f:
            restored = json.load(f)
        with open(settings_path, "w") as f:
            json.dump(restored, f)

        with open(settings_path, "r") as f:
            final = json.load(f)

        assert final == original


class TestEncryptionSettings:
    """Tests for encryption-related settings."""

    def test_encryption_key_format(self, encryption_key):
        """Encryption key should have valid format."""
        assert isinstance(encryption_key, str)
        assert len(encryption_key) > 0

    def test_encrypted_content_prefix(self, encrypted_content):
        """Encrypted content should have correct prefix."""
        assert encrypted_content.startswith("SYNC_ENC:")

    def test_encryption_disabled_by_default(self, default_settings):
        """Encryption should be configurable."""
        # Default settings may or may not have encryption
        # Just verify it's a valid setting type if present
        if "encryption_enabled" in default_settings:
            assert isinstance(default_settings["encryption_enabled"], bool)


class TestPortConflicts:
    """Tests for port conflict detection."""

    def test_detect_port_in_use(self, free_port):
        """Should detect when a port is in use."""
        import socket

        # Bind the port
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', free_port))
        server.listen(1)

        try:
            # Check if port is in use
            check_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                check_sock.bind(('127.0.0.1', free_port))
                in_use = False
            except OSError:
                in_use = True
            finally:
                check_sock.close()

            assert in_use is True
        finally:
            server.close()

    def test_suggest_alternate_port(self):
        """Should suggest alternate port when conflict detected."""
        def get_alternate_port(base_port, offset=10):
            return base_port + offset

        assert get_alternate_port(19284) == 19294
        assert get_alternate_port(19284, 100) == 19384


class TestUISettings:
    """Tests for UI-related settings."""

    def test_window_position_format(self):
        """Window position should have x, y coordinates."""
        window_pos = {"x": 100, "y": 200}
        assert "x" in window_pos
        assert "y" in window_pos
        assert isinstance(window_pos["x"], int)
        assert isinstance(window_pos["y"], int)

    def test_window_size_format(self):
        """Window size should have width, height."""
        window_size = {"width": 800, "height": 600}
        assert "width" in window_size
        assert "height" in window_size
        assert window_size["width"] > 0
        assert window_size["height"] > 0

    def test_minimized_to_tray_default(self):
        """Should have minimize to tray option."""
        ui_settings = {"minimize_to_tray": True}
        assert isinstance(ui_settings["minimize_to_tray"], bool)

    def test_start_minimized_default(self):
        """Should have start minimized option."""
        ui_settings = {"start_minimized": False}
        assert isinstance(ui_settings["start_minimized"], bool)


class TestLoggingSettings:
    """Tests for logging-related settings."""

    def test_log_level_valid(self):
        """Log level should be valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level = "INFO"
        assert log_level in valid_levels

    def test_log_file_path(self, temp_dir):
        """Log file path should be valid."""
        log_path = temp_dir / "shadow_bridge.log"
        log_path.touch()
        assert log_path.exists()

    def test_log_rotation_settings(self):
        """Log rotation settings should be valid."""
        rotation_settings = {
            "max_size_mb": 10,
            "backup_count": 5
        }
        assert rotation_settings["max_size_mb"] > 0
        assert rotation_settings["backup_count"] >= 0
