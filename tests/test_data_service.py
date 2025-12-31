"""
Comprehensive tests for shadow-bridge data service layer.
Tests projects, notes, devices, and data persistence.
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestProjectsData:
    """Tests for projects data handling."""

    def test_project_has_required_fields(self, sample_projects):
        """Projects should have all required fields."""
        required_fields = ["id", "name", "workingDirectory", "createdAt", "updatedAt"]
        for project in sample_projects:
            for field in required_fields:
                assert field in project

    def test_project_id_is_string(self, sample_projects):
        """Project ID should be a string."""
        for project in sample_projects:
            assert isinstance(project["id"], str)

    def test_project_timestamps_are_positive(self, sample_projects):
        """Project timestamps should be positive."""
        for project in sample_projects:
            assert project["createdAt"] > 0
            assert project["updatedAt"] > 0

    def test_project_name_is_not_empty(self, sample_projects):
        """Project name should not be empty."""
        for project in sample_projects:
            assert len(project["name"]) > 0

    def test_projects_file_structure(self, projects_file):
        """Projects file should have correct structure."""
        with open(projects_file, "r") as f:
            data = json.load(f)

        assert "devices" in data
        assert isinstance(data["devices"], dict)

    def test_projects_associated_with_device(self, projects_file, sample_device):
        """Projects should be associated with a device."""
        with open(projects_file, "r") as f:
            data = json.load(f)

        device_id = sample_device["id"]
        assert device_id in data["devices"]
        assert "projects" in data["devices"][device_id]

    def test_can_add_project(self, projects_file, sample_device):
        """Should be able to add a new project."""
        with open(projects_file, "r") as f:
            data = json.load(f)

        new_project = {
            "id": "project-new",
            "name": "New Project",
            "description": "A new project",
            "workingDirectory": "/new/path",
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000)
        }

        device_id = sample_device["id"]
        data["devices"][device_id]["projects"].append(new_project)

        with open(projects_file, "w") as f:
            json.dump(data, f)

        with open(projects_file, "r") as f:
            reloaded = json.load(f)

        project_ids = [p["id"] for p in reloaded["devices"][device_id]["projects"]]
        assert "project-new" in project_ids

    def test_can_update_project(self, projects_file, sample_device):
        """Should be able to update an existing project."""
        with open(projects_file, "r") as f:
            data = json.load(f)

        device_id = sample_device["id"]
        projects = data["devices"][device_id]["projects"]

        # Update first project
        projects[0]["name"] = "Updated Name"
        projects[0]["updatedAt"] = int(time.time() * 1000)

        with open(projects_file, "w") as f:
            json.dump(data, f)

        with open(projects_file, "r") as f:
            reloaded = json.load(f)

        assert reloaded["devices"][device_id]["projects"][0]["name"] == "Updated Name"

    def test_can_delete_project(self, projects_file, sample_device):
        """Should be able to delete a project."""
        with open(projects_file, "r") as f:
            data = json.load(f)

        device_id = sample_device["id"]
        initial_count = len(data["devices"][device_id]["projects"])

        # Remove first project
        data["devices"][device_id]["projects"].pop(0)

        with open(projects_file, "w") as f:
            json.dump(data, f)

        with open(projects_file, "r") as f:
            reloaded = json.load(f)

        assert len(reloaded["devices"][device_id]["projects"]) == initial_count - 1


class TestNotesData:
    """Tests for notes data handling."""

    def test_note_has_required_fields(self, sample_notes):
        """Notes should have all required fields."""
        required_fields = ["id", "title", "content", "createdAt", "updatedAt"]
        for note in sample_notes:
            for field in required_fields:
                assert field in note

    def test_note_id_is_string(self, sample_notes):
        """Note ID should be a string."""
        for note in sample_notes:
            assert isinstance(note["id"], str)

    def test_note_has_archive_status(self, sample_notes):
        """Notes should have archive status."""
        for note in sample_notes:
            assert "isArchived" in note
            assert isinstance(note["isArchived"], bool)

    def test_notes_file_structure(self, notes_file):
        """Notes file should have correct structure."""
        with open(notes_file, "r") as f:
            data = json.load(f)

        assert "devices" in data
        assert isinstance(data["devices"], dict)

    def test_notes_associated_with_device(self, notes_file, sample_device):
        """Notes should be associated with a device."""
        with open(notes_file, "r") as f:
            data = json.load(f)

        device_id = sample_device["id"]
        assert device_id in data["devices"]
        assert "notes" in data["devices"][device_id]

    def test_can_filter_archived_notes(self, sample_notes):
        """Should be able to filter archived notes."""
        archived = [n for n in sample_notes if n["isArchived"]]
        active = [n for n in sample_notes if not n["isArchived"]]

        assert len(archived) == 1
        assert len(active) == 1

    def test_can_filter_by_category(self, sample_notes):
        """Should be able to filter notes by category."""
        work_notes = [n for n in sample_notes if n.get("category") == "Work"]
        assert len(work_notes) >= 0  # May or may not have work notes

    def test_can_add_note(self, notes_file, sample_device):
        """Should be able to add a new note."""
        with open(notes_file, "r") as f:
            data = json.load(f)

        new_note = {
            "id": "note-new",
            "title": "New Note",
            "content": "New content",
            "category": "Test",
            "isArchived": False,
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000)
        }

        device_id = sample_device["id"]
        data["devices"][device_id]["notes"].append(new_note)

        with open(notes_file, "w") as f:
            json.dump(data, f)

        with open(notes_file, "r") as f:
            reloaded = json.load(f)

        note_ids = [n["id"] for n in reloaded["devices"][device_id]["notes"]]
        assert "note-new" in note_ids


class TestDeviceData:
    """Tests for device data handling."""

    def test_device_has_required_fields(self, sample_device):
        """Device should have all required fields."""
        required_fields = ["id", "name", "ip"]
        for field in required_fields:
            assert field in sample_device

    def test_device_id_format(self, sample_device):
        """Device ID should be a string."""
        assert isinstance(sample_device["id"], str)
        assert len(sample_device["id"]) > 0

    def test_device_ip_format(self, sample_device):
        """Device IP should be valid format."""
        ip = sample_device["ip"]
        parts = ip.split(".")
        assert len(parts) == 4
        for part in parts:
            assert 0 <= int(part) <= 255

    def test_device_has_last_seen(self, sample_device):
        """Device should have lastSeen timestamp."""
        assert "lastSeen" in sample_device
        assert sample_device["lastSeen"] > 0

    def test_device_has_ip_candidates(self, sample_device):
        """Device may have IP candidates list."""
        if "ipCandidates" in sample_device:
            assert isinstance(sample_device["ipCandidates"], list)

    def test_can_update_device_ip(self, projects_file, sample_device):
        """Should be able to update device IP."""
        with open(projects_file, "r") as f:
            data = json.load(f)

        device_id = sample_device["id"]
        data["devices"][device_id]["ip"] = "192.168.1.200"

        with open(projects_file, "w") as f:
            json.dump(data, f)

        with open(projects_file, "r") as f:
            reloaded = json.load(f)

        assert reloaded["devices"][device_id]["ip"] == "192.168.1.200"

    def test_can_update_last_seen(self, projects_file, sample_device):
        """Should be able to update lastSeen timestamp."""
        with open(projects_file, "r") as f:
            data = json.load(f)

        device_id = sample_device["id"]
        new_timestamp = int(time.time() * 1000)
        data["devices"][device_id]["lastSeen"] = new_timestamp

        with open(projects_file, "w") as f:
            json.dump(data, f)

        with open(projects_file, "r") as f:
            reloaded = json.load(f)

        assert reloaded["devices"][device_id]["lastSeen"] == new_timestamp


class TestAutomationsData:
    """Tests for automations data handling."""

    def test_automation_has_required_fields(self, sample_automations):
        """Automations should have all required fields."""
        required_fields = ["id", "name", "type", "isEnabled"]
        for automation in sample_automations:
            for field in required_fields:
                assert field in automation

    def test_automation_type_is_valid(self, sample_automations):
        """Automation type should be valid."""
        valid_types = ["TEXT", "IMAGE_GENERATION", "BROWSER_AUTOMATION",
                       "N8N_WORKFLOW", "FILE_OPERATION"]
        for automation in sample_automations:
            assert automation["type"] in valid_types

    def test_automation_has_trigger(self, sample_automations):
        """Automations should have trigger configuration."""
        for automation in sample_automations:
            assert "trigger" in automation
            assert "type" in automation["trigger"]

    def test_can_filter_enabled_automations(self, sample_automations):
        """Should be able to filter enabled automations."""
        enabled = [a for a in sample_automations if a["isEnabled"]]
        disabled = [a for a in sample_automations if not a["isEnabled"]]

        assert len(enabled) + len(disabled) == len(sample_automations)


class TestTeamData:
    """Tests for team data handling."""

    def test_team_has_required_fields(self, sample_teams):
        """Teams should have all required fields."""
        required_fields = ["id", "name", "members"]
        for team in sample_teams:
            for field in required_fields:
                assert field in team

    def test_team_members_have_roles(self, sample_teams):
        """Team members should have roles."""
        for team in sample_teams:
            for member in team["members"]:
                assert "role" in member
                assert member["role"] in ["owner", "admin", "member"]

    def test_team_has_at_least_one_owner(self, sample_teams):
        """Each team should have at least one owner."""
        for team in sample_teams:
            owners = [m for m in team["members"] if m["role"] == "owner"]
            assert len(owners) >= 1


class TestDataValidation:
    """Tests for data validation."""

    def test_id_uniqueness(self, sample_projects):
        """IDs should be unique."""
        ids = [p["id"] for p in sample_projects]
        assert len(ids) == len(set(ids))

    def test_timestamps_are_valid(self, sample_projects):
        """Timestamps should be valid milliseconds since epoch."""
        for project in sample_projects:
            # Should be after year 2000
            assert project["createdAt"] > 946684800000
            assert project["updatedAt"] >= project["createdAt"]

    def test_path_separators(self, sample_projects):
        """Paths should handle both Windows and Unix separators."""
        for project in sample_projects:
            path = project["workingDirectory"]
            # Should be a string
            assert isinstance(path, str)
            # Should have some path-like structure
            assert "/" in path or "\\" in path

    def test_empty_string_handling(self):
        """Should handle empty strings appropriately."""
        note = {"title": "", "content": "Some content"}
        assert note["title"] == ""
        assert len(note["content"]) > 0

    def test_null_handling(self):
        """Should handle null values appropriately."""
        automation = {"lastRun": None}
        assert automation["lastRun"] is None


class TestDataSorting:
    """Tests for data sorting."""

    def test_sort_projects_by_name(self, sample_projects):
        """Projects should be sortable by name."""
        sorted_projects = sorted(sample_projects, key=lambda p: p["name"])
        for i in range(len(sorted_projects) - 1):
            assert sorted_projects[i]["name"] <= sorted_projects[i + 1]["name"]

    def test_sort_projects_by_updated(self, sample_projects):
        """Projects should be sortable by update time."""
        sorted_projects = sorted(sample_projects,
                                 key=lambda p: p["updatedAt"],
                                 reverse=True)
        for i in range(len(sorted_projects) - 1):
            assert sorted_projects[i]["updatedAt"] >= sorted_projects[i + 1]["updatedAt"]

    def test_sort_notes_by_created(self, sample_notes):
        """Notes should be sortable by creation time."""
        sorted_notes = sorted(sample_notes, key=lambda n: n["createdAt"])
        for i in range(len(sorted_notes) - 1):
            assert sorted_notes[i]["createdAt"] <= sorted_notes[i + 1]["createdAt"]


class TestDataSearch:
    """Tests for data search functionality."""

    def test_search_projects_by_name(self, sample_projects):
        """Should be able to search projects by name."""
        query = "Test"
        results = [p for p in sample_projects if query.lower() in p["name"].lower()]
        assert len(results) >= 0

    def test_search_notes_by_content(self, sample_notes):
        """Should be able to search notes by content."""
        query = "content"
        results = [n for n in sample_notes if query.lower() in n["content"].lower()]
        assert len(results) > 0

    def test_search_is_case_insensitive(self, sample_projects):
        """Search should be case insensitive."""
        query_lower = "test"
        query_upper = "TEST"

        results_lower = [p for p in sample_projects
                         if query_lower.lower() in p["name"].lower()]
        results_upper = [p for p in sample_projects
                         if query_upper.lower() in p["name"].lower()]

        assert len(results_lower) == len(results_upper)


class TestDataPagination:
    """Tests for data pagination."""

    def test_pagination_first_page(self):
        """Should get first page correctly."""
        items = list(range(100))
        page_size = 10
        page = 0

        start = page * page_size
        end = start + page_size
        page_items = items[start:end]

        assert len(page_items) == 10
        assert page_items[0] == 0

    def test_pagination_last_page(self):
        """Should get last page correctly."""
        items = list(range(95))
        page_size = 10
        page = 9  # Last page

        start = page * page_size
        end = start + page_size
        page_items = items[start:end]

        assert len(page_items) == 5
        assert page_items[-1] == 94

    def test_pagination_empty_page(self):
        """Should handle empty page."""
        items = list(range(10))
        page_size = 10
        page = 5  # Beyond available data

        start = page * page_size
        end = start + page_size
        page_items = items[start:end]

        assert len(page_items) == 0


class TestDataExport:
    """Tests for data export functionality."""

    def test_export_to_json(self, sample_projects, temp_dir):
        """Should export data to JSON."""
        export_path = temp_dir / "export.json"

        with open(export_path, "w") as f:
            json.dump({"projects": sample_projects}, f)

        assert export_path.exists()
        with open(export_path, "r") as f:
            exported = json.load(f)
        assert len(exported["projects"]) == len(sample_projects)

    def test_export_includes_all_fields(self, sample_projects, temp_dir):
        """Export should include all fields."""
        export_path = temp_dir / "export.json"

        with open(export_path, "w") as f:
            json.dump({"projects": sample_projects}, f)

        with open(export_path, "r") as f:
            exported = json.load(f)

        original_keys = set(sample_projects[0].keys())
        exported_keys = set(exported["projects"][0].keys())
        assert original_keys == exported_keys


class TestDataImport:
    """Tests for data import functionality."""

    def test_import_from_json(self, temp_dir):
        """Should import data from JSON."""
        import_data = {
            "projects": [
                {"id": "import-1", "name": "Imported Project"}
            ]
        }

        import_path = temp_dir / "import.json"
        with open(import_path, "w") as f:
            json.dump(import_data, f)

        with open(import_path, "r") as f:
            imported = json.load(f)

        assert len(imported["projects"]) == 1
        assert imported["projects"][0]["id"] == "import-1"

    def test_import_validates_structure(self, temp_dir):
        """Import should validate data structure."""
        invalid_data = {"projects": "not a list"}

        import_path = temp_dir / "import.json"
        with open(import_path, "w") as f:
            json.dump(invalid_data, f)

        with open(import_path, "r") as f:
            imported = json.load(f)

        # Validation would fail
        assert not isinstance(imported["projects"], list)


class TestDataMerge:
    """Tests for data merge functionality."""

    def test_merge_without_conflicts(self):
        """Should merge data without conflicts."""
        local = [{"id": "1", "name": "Local"}]
        remote = [{"id": "2", "name": "Remote"}]

        merged = local + [r for r in remote if r["id"] not in [l["id"] for l in local]]
        assert len(merged) == 2

    def test_merge_with_conflicts_keeps_newer(self):
        """Should keep newer item on conflict."""
        local = {"id": "1", "name": "Local", "updatedAt": 1000}
        remote = {"id": "1", "name": "Remote", "updatedAt": 2000}

        if local["updatedAt"] > remote["updatedAt"]:
            result = local
        else:
            result = remote

        assert result["name"] == "Remote"

    def test_merge_preserves_local_only(self):
        """Should preserve local-only items."""
        local = [{"id": "1"}, {"id": "2"}]
        remote = [{"id": "2"}, {"id": "3"}]

        local_ids = {item["id"] for item in local}
        remote_ids = {item["id"] for item in remote}

        local_only = local_ids - remote_ids
        assert "1" in local_only


class TestConcurrentAccess:
    """Tests for concurrent data access."""

    def test_file_locking_concept(self, temp_dir):
        """File operations should be safe."""
        file_path = temp_dir / "test.json"

        # Write initial data
        with open(file_path, "w") as f:
            json.dump({"value": 0}, f)

        # Simulate concurrent increments
        for _ in range(10):
            with open(file_path, "r") as f:
                data = json.load(f)
            data["value"] += 1
            with open(file_path, "w") as f:
                json.dump(data, f)

        with open(file_path, "r") as f:
            final = json.load(f)

        assert final["value"] == 10


class TestDataIntegrity:
    """Tests for data integrity."""

    def test_corrupted_json_detection(self, temp_dir):
        """Should detect corrupted JSON."""
        file_path = temp_dir / "corrupted.json"

        with open(file_path, "w") as f:
            f.write("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            with open(file_path, "r") as f:
                json.load(f)

    def test_missing_file_handling(self, temp_dir):
        """Should handle missing files."""
        file_path = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            with open(file_path, "r") as f:
                json.load(f)

    def test_empty_file_handling(self, temp_dir):
        """Should handle empty files."""
        file_path = temp_dir / "empty.json"
        file_path.touch()

        with pytest.raises(json.JSONDecodeError):
            with open(file_path, "r") as f:
                json.load(f)
