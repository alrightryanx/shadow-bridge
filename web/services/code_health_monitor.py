"""Code health monitor for proactive Ouroboros integration.

Analyzes project health (dependency staleness, build health, doc freshness)
and tracks issues/fixes for the self-healing pipeline.
"""

import json
import logging
import os
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.path.join(str(Path.home()), ".shadowai")
HEALTH_STORE_FILE = "code_health_store.json"


class CodeHealthMonitor:
    """Monitors project health and tracks issues for Ouroboros."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.file_path = os.path.join(self.data_dir, HEALTH_STORE_FILE)
        self._lock = threading.Lock()
        self.issues: Dict[str, dict] = {}
        self.project_health: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load state from JSON file."""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.issues = data.get("issues", {})
                self.project_health = data.get("project_health", {})
                log.info(f"CodeHealthMonitor loaded: {len(self.issues)} issues, "
                         f"{len(self.project_health)} projects")
        except Exception as e:
            log.warning(f"Failed to load code health store: {e}")

    def _save(self):
        """Persist state to JSON file."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            payload = {
                "issues": self.issues,
                "project_health": self.project_health,
                "updated": time.time(),
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save code health store: {e}")

    # ---- Issue Tracking ----

    def get_issues(self, status: str = "open", project_id: Optional[str] = None,
                   limit: int = 20) -> List[dict]:
        """Get issues filtered by status and optional project."""
        with self._lock:
            issues = list(self.issues.values())
        if status:
            issues = [i for i in issues if i.get("status") == status]
        if project_id:
            issues = [i for i in issues if i.get("project_id") == project_id]
        issues.sort(key=lambda i: i.get("severity_score", 0), reverse=True)
        return issues[:limit]

    def add_issue(self, project_id: str, category: str, title: str,
                  description: str, severity_score: float = 0.5) -> dict:
        """Add a new health issue."""
        import uuid
        issue_id = str(uuid.uuid4())
        issue = {
            "id": issue_id,
            "project_id": project_id,
            "category": category,
            "title": title,
            "description": description,
            "severity_score": severity_score,
            "status": "open",
            "created_at": time.time(),
            "fixed_at": None,
            "fix_description": None,
            "commit_hash": None,
        }
        with self._lock:
            self.issues[issue_id] = issue
            self._save()
        return issue

    def record_fix(self, issue_id: str, fix_description: str,
                   commit_hash: Optional[str] = None) -> dict:
        """Record that an issue has been fixed."""
        with self._lock:
            if issue_id not in self.issues:
                return {"error": "Issue not found"}
            issue = self.issues[issue_id]
            issue["status"] = "fixed"
            issue["fixed_at"] = time.time()
            issue["fix_description"] = fix_description
            if commit_hash:
                issue["commit_hash"] = commit_hash
            self._save()
            return issue

    # ---- Health Analysis ----

    def analyze_project_health(self, project_dir: str) -> dict:
        """Analyze a project directory for health indicators.

        Checks: dependency staleness, build health, doc freshness,
        git activity, and file complexity heuristics.
        """
        result = {
            "project_dir": project_dir,
            "analyzed_at": time.time(),
            "health_score": 1.0,
            "checks": {},
        }

        if not os.path.isdir(project_dir):
            result["health_score"] = 0.0
            result["error"] = "Directory not found"
            return result

        # Check git status
        git_health = self._check_git_health(project_dir)
        result["checks"]["git"] = git_health

        # Check for build files and their age
        build_health = self._check_build_health(project_dir)
        result["checks"]["build"] = build_health

        # Check documentation freshness
        doc_health = self._check_doc_health(project_dir)
        result["checks"]["documentation"] = doc_health

        # Check for dependency files
        dep_health = self._check_dependency_health(project_dir)
        result["checks"]["dependencies"] = dep_health

        # Calculate overall score
        scores = [
            git_health.get("score", 0.5),
            build_health.get("score", 0.5),
            doc_health.get("score", 0.5),
            dep_health.get("score", 0.5),
        ]
        result["health_score"] = round(sum(scores) / len(scores), 3)

        return result

    def _check_git_health(self, project_dir: str) -> dict:
        """Check git repository health."""
        git_dir = os.path.join(project_dir, ".git")
        if not os.path.isdir(git_dir):
            return {"score": 0.5, "status": "not_a_repo"}

        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_dir, capture_output=True, text=True, timeout=10,
            )
            uncommitted = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

            # Check last commit age
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ct"],
                cwd=project_dir, capture_output=True, text=True, timeout=10,
            )
            last_commit_age = 0
            if result.stdout.strip():
                last_commit_ts = int(result.stdout.strip())
                last_commit_age = time.time() - last_commit_ts

            days_since_commit = last_commit_age / 86400
            score = 1.0
            if uncommitted > 10:
                score -= 0.2
            if days_since_commit > 30:
                score -= 0.3
            elif days_since_commit > 7:
                score -= 0.1

            return {
                "score": max(0.0, score),
                "uncommitted_files": uncommitted,
                "days_since_last_commit": round(days_since_commit, 1),
            }
        except Exception as e:
            return {"score": 0.5, "error": str(e)}

    def _check_build_health(self, project_dir: str) -> dict:
        """Check build system health indicators."""
        build_files = {
            "build.gradle.kts": "gradle",
            "build.gradle": "gradle",
            "pom.xml": "maven",
            "package.json": "npm",
            "Cargo.toml": "cargo",
            "setup.py": "python",
            "pyproject.toml": "python",
            "requirements.txt": "python",
        }

        found_system = None
        build_file_path = None

        for filename, system in build_files.items():
            path = os.path.join(project_dir, filename)
            if os.path.exists(path):
                found_system = system
                build_file_path = path
                break

        if not found_system:
            return {"score": 0.5, "status": "no_build_system_detected"}

        # Check build file age
        try:
            mtime = os.path.getmtime(build_file_path)
            age_days = (time.time() - mtime) / 86400
            score = 1.0 if age_days < 30 else (0.7 if age_days < 90 else 0.4)
            return {
                "score": score,
                "build_system": found_system,
                "build_file_age_days": round(age_days, 1),
            }
        except Exception as e:
            return {"score": 0.5, "build_system": found_system, "error": str(e)}

    def _check_doc_health(self, project_dir: str) -> dict:
        """Check documentation freshness."""
        doc_files = []
        doc_dirs = ["docs", "doc", "documentation"]

        # Check README
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            path = os.path.join(project_dir, name)
            if os.path.exists(path):
                doc_files.append(path)

        # Check doc directories
        for dirname in doc_dirs:
            dirpath = os.path.join(project_dir, dirname)
            if os.path.isdir(dirpath):
                for f in os.listdir(dirpath):
                    fpath = os.path.join(dirpath, f)
                    if os.path.isfile(fpath) and f.endswith((".md", ".rst", ".txt")):
                        doc_files.append(fpath)

        if not doc_files:
            return {"score": 0.3, "status": "no_documentation", "doc_count": 0}

        # Check freshness of docs
        ages = []
        for fpath in doc_files:
            try:
                mtime = os.path.getmtime(fpath)
                ages.append((time.time() - mtime) / 86400)
            except Exception:
                pass

        if not ages:
            return {"score": 0.5, "doc_count": len(doc_files)}

        avg_age = sum(ages) / len(ages)
        score = 1.0 if avg_age < 14 else (0.7 if avg_age < 60 else 0.4)

        return {
            "score": score,
            "doc_count": len(doc_files),
            "avg_doc_age_days": round(avg_age, 1),
            "newest_doc_age_days": round(min(ages), 1),
        }

    def _check_dependency_health(self, project_dir: str) -> dict:
        """Check dependency file health (lock file freshness)."""
        lock_files = {
            "gradle.lockfile": "gradle",
            "package-lock.json": "npm",
            "yarn.lock": "yarn",
            "Cargo.lock": "cargo",
            "poetry.lock": "poetry",
            "Pipfile.lock": "pipenv",
        }

        for filename, system in lock_files.items():
            path = os.path.join(project_dir, filename)
            if os.path.exists(path):
                try:
                    mtime = os.path.getmtime(path)
                    age_days = (time.time() - mtime) / 86400
                    score = 1.0 if age_days < 30 else (0.6 if age_days < 90 else 0.3)
                    return {
                        "score": score,
                        "lock_system": system,
                        "lock_age_days": round(age_days, 1),
                    }
                except Exception as e:
                    return {"score": 0.5, "error": str(e)}

        return {"score": 0.7, "status": "no_lock_file"}

    def generate_health_tasks(self, project_id: str,
                              results: dict) -> List[dict]:
        """Convert health analysis results into actionable task suggestions.

        NOTE: These are suggestions only -- never modify code directly.
        """
        tasks = []
        checks = results.get("checks", {})

        git = checks.get("git", {})
        if git.get("uncommitted_files", 0) > 10:
            tasks.append({
                "title": "Clean up uncommitted files",
                "description": f"{git['uncommitted_files']} uncommitted files detected. "
                               f"Review and commit or gitignore stale files.",
                "priority": "MEDIUM",
                "category": "git_hygiene",
                "source": "code_health",
            })

        if git.get("days_since_last_commit", 0) > 30:
            tasks.append({
                "title": "Stale repository check",
                "description": f"No commits in {git['days_since_last_commit']:.0f} days. "
                               f"Verify project is still active or archive it.",
                "priority": "LOW",
                "category": "project_status",
                "source": "code_health",
            })

        doc = checks.get("documentation", {})
        if doc.get("status") == "no_documentation":
            tasks.append({
                "title": "Add project documentation",
                "description": "No README or docs found. Add basic project documentation.",
                "priority": "MEDIUM",
                "category": "documentation",
                "source": "code_health",
            })
        elif doc.get("avg_doc_age_days", 0) > 60:
            tasks.append({
                "title": "Update project documentation",
                "description": f"Documentation hasn't been updated in "
                               f"{doc['avg_doc_age_days']:.0f} days.",
                "priority": "LOW",
                "category": "documentation",
                "source": "code_health",
            })

        deps = checks.get("dependencies", {})
        if deps.get("lock_age_days", 0) > 90:
            tasks.append({
                "title": "Update project dependencies",
                "description": f"Dependency lock file is {deps['lock_age_days']:.0f} days old. "
                               f"Review for security updates.",
                "priority": "HIGH",
                "category": "dependencies",
                "source": "code_health",
            })

        # Store issues for any critical findings
        for task in tasks:
            if task["priority"] == "HIGH":
                self.add_issue(
                    project_id=project_id,
                    category=task["category"],
                    title=task["title"],
                    description=task["description"],
                    severity_score=0.8,
                )

        return tasks

    # ---- Aggregate Queries ----

    def get_health_context(self) -> dict:
        """Get aggregate health context for the Ouroboros refiner."""
        with self._lock:
            open_issues = [i for i in self.issues.values()
                           if i.get("status") == "open"]
            fixed_issues = [i for i in self.issues.values()
                            if i.get("status") == "fixed"]

        # Compute crash patterns from open issues
        categories = {}
        for issue in open_issues:
            cat = issue.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        # Overall health from cached project analyses
        with self._lock:
            project_scores = [
                p.get("health_score", 0.5)
                for p in self.project_health.values()
            ]

        avg_health = (sum(project_scores) / len(project_scores)
                      if project_scores else 0.5)

        return {
            "health_score": round(avg_health, 3),
            "open_issues": len(open_issues),
            "fixed_issues": len(fixed_issues),
            "crash_patterns": [
                {"category": cat, "count": count}
                for cat, count in sorted(categories.items(),
                                         key=lambda x: x[1], reverse=True)
            ],
            "projects_analyzed": len(self.project_health),
        }

    def get_project_health(self, project_id: str) -> dict:
        """Get cached health data for a project, or empty result."""
        with self._lock:
            cached = self.project_health.get(project_id)
        if cached:
            return cached
        return {
            "project_id": project_id,
            "health_score": 0.0,
            "status": "not_analyzed",
            "message": "Project has not been analyzed yet. "
                       "Health scans run periodically.",
        }

    def cache_project_health(self, project_id: str, health_data: dict):
        """Cache a project health analysis result."""
        with self._lock:
            self.project_health[project_id] = {
                **health_data,
                "project_id": project_id,
            }
            self._save()


# ---- Singleton ----

_monitor: Optional[CodeHealthMonitor] = None


def get_code_health_monitor() -> CodeHealthMonitor:
    """Return the singleton CodeHealthMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = CodeHealthMonitor()
    return _monitor
