"""
Autonomous Code Scanner

Scans shadow-android and shadow-bridge repos for actionable improvements.
Generates prioritized task queue for autonomous agents.
"""

import os
import re
import uuid
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Persistence
TASKS_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "autonomous_tasks.json")
COMPLETED_HASHES_FILE = os.path.join(os.path.expanduser("~"), ".shadowai", "completed_task_hashes.json")

# Scanner config
MAX_METHOD_LINES = 80
MAX_NESTING_DEPTH = 4

# File extensions to scan per repo
KOTLIN_EXTS = {".kt", ".kts"}
PYTHON_EXTS = {".py"}
JAVA_EXTS = {".java"}

# Directories to skip
SKIP_DIRS = {
    "build", ".gradle", ".idea", "node_modules", "__pycache__",
    ".git", "venv", "env", ".aidev", "locks", ".opencode",
    "llama.cpp", "whisper.cpp",  # git submodules
}

# Important files get higher priority scores
IMPORTANT_FILE_PATTERNS = [
    "MainActivity", "ViewModel", "Repository", "Service",
    "Manager", "Helper", "Utils", "Database", "Migration",
    "shadow_bridge_gui", "data_service", "agent_orchestrator",
    "api.py", "websocket.py",
]


class AutonomousScanner:
    """Scans codebases for actionable improvements."""

    def __init__(self, repos: Optional[List[str]] = None):
        self.repos = repos or [
            "C:/shadow/shadow-android",
            "C:/shadow/shadow-bridge",
        ]
        self._completed_hashes = self._load_completed_hashes()

    def scan_all(self) -> List[Dict]:
        """Run all scanners across all repos, deduplicate, and prioritize."""
        all_tasks = []

        for repo in self.repos:
            if not os.path.isdir(repo):
                logger.warning(f"Repo not found: {repo}")
                continue

            repo_name = os.path.basename(repo)
            logger.info(f"Scanning {repo_name}...")

            all_tasks.extend(self.scan_todos(repo))
            all_tasks.extend(self.scan_code_smells(repo))
            all_tasks.extend(self.scan_error_handling(repo))
            all_tasks.extend(self.scan_performance(repo))
            all_tasks.extend(self.scan_test_gaps(repo))

        # Deduplicate
        deduped = self._deduplicate(all_tasks)

        # Filter already completed
        filtered = [t for t in deduped if self._task_hash(t) not in self._completed_hashes]

        # Prioritize
        prioritized = self.prioritize(filtered)

        # Persist
        self._save_tasks(prioritized)

        logger.info(f"Scanner found {len(prioritized)} actionable tasks")
        return prioritized

    def scan_todos(self, repo: str) -> List[Dict]:
        """Find TODO, FIXME, HACK, XXX comments with context."""
        tasks = []
        repo_name = os.path.basename(repo)
        pattern = re.compile(r"(TODO|FIXME|HACK|XXX)\s*:?\s*(.*)", re.IGNORECASE)

        for filepath in self._walk_source_files(repo):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, 1):
                        match = pattern.search(line)
                        if match:
                            marker = match.group(1).upper()
                            comment = match.group(2).strip()
                            rel_path = os.path.relpath(filepath, repo)

                            # Skip production checklist markers
                            if "REMOVE FOR PRODUCTION" in comment:
                                continue

                            priority = 2 if marker in ("FIXME", "HACK") else 3
                            tasks.append({
                                "id": str(uuid.uuid4()),
                                "title": f"Resolve {marker} in {os.path.basename(filepath)}:{line_num}",
                                "description": f"{marker}: {comment}\n\nFile: {rel_path}\nLine: {line_num}",
                                "file_path": rel_path,
                                "line_number": line_num,
                                "category": "todo",
                                "priority": priority,
                                "repo": repo_name,
                                "scope": "small",
                                "assigned_to": None,
                                "status": "pending",
                                "created_at": datetime.utcnow().isoformat(),
                            })
            except (OSError, UnicodeDecodeError):
                continue

        return tasks

    def scan_code_smells(self, repo: str) -> List[Dict]:
        """Find long methods, deep nesting, and unused imports."""
        tasks = []
        repo_name = os.path.basename(repo)

        for filepath in self._walk_source_files(repo):
            ext = os.path.splitext(filepath)[1]
            rel_path = os.path.relpath(filepath, repo)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except (OSError, UnicodeDecodeError):
                continue

            # Check for long methods
            if ext in KOTLIN_EXTS | JAVA_EXTS:
                tasks.extend(self._find_long_methods_kt(lines, rel_path, repo_name))
            elif ext in PYTHON_EXTS:
                tasks.extend(self._find_long_methods_py(lines, rel_path, repo_name))

            # Check for deep nesting
            tasks.extend(self._find_deep_nesting(lines, rel_path, repo_name, ext))

        return tasks

    def scan_error_handling(self, repo: str) -> List[Dict]:
        """Find bare except/catch, empty catch blocks, swallowed exceptions."""
        tasks = []
        repo_name = os.path.basename(repo)

        for filepath in self._walk_source_files(repo):
            ext = os.path.splitext(filepath)[1]
            rel_path = os.path.relpath(filepath, repo)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    lines = content.split("\n")
            except (OSError, UnicodeDecodeError):
                continue

            if ext in KOTLIN_EXTS | JAVA_EXTS:
                # Bare catch (catches everything)
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if re.match(r"}\s*catch\s*\(\s*e\s*:\s*Exception\s*\)", stripped):
                        # Check if catch block is empty or just logs
                        if i < len(lines):
                            next_lines = "".join(lines[i:i+3])
                            if re.search(r"}\s*$", next_lines.strip()):
                                tasks.append(self._make_task(
                                    f"Empty catch block in {os.path.basename(filepath)}:{i}",
                                    f"Empty or trivial catch block that swallows exceptions.\n\n"
                                    f"File: {rel_path}\nLine: {i}\n\n"
                                    f"Replace with specific exception types and proper handling.",
                                    rel_path, i, "error_handling", 2, repo_name, "small"
                                ))

                    # Catch Throwable (too broad)
                    if re.match(r"}\s*catch\s*\(\s*\w+\s*:\s*Throwable\s*\)", stripped):
                        tasks.append(self._make_task(
                            f"Catch Throwable in {os.path.basename(filepath)}:{i}",
                            f"Catching Throwable is too broad - catches OutOfMemoryError etc.\n\n"
                            f"File: {rel_path}\nLine: {i}\n\n"
                            f"Narrow to specific exception types.",
                            rel_path, i, "error_handling", 2, repo_name, "small"
                        ))

            elif ext in PYTHON_EXTS:
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    # Bare except
                    if stripped == "except:" or stripped == "except Exception:":
                        tasks.append(self._make_task(
                            f"Bare except in {os.path.basename(filepath)}:{i}",
                            f"Bare except clause catches all exceptions including KeyboardInterrupt.\n\n"
                            f"File: {rel_path}\nLine: {i}\n\n"
                            f"Use specific exception types.",
                            rel_path, i, "error_handling", 3, repo_name, "small"
                        ))
                    # pass in except
                    if stripped == "pass" and i >= 2:
                        prev = lines[i-2].strip() if i >= 2 else ""
                        if "except" in prev:
                            tasks.append(self._make_task(
                                f"Swallowed exception in {os.path.basename(filepath)}:{i}",
                                f"Exception caught and silently ignored with pass.\n\n"
                                f"File: {rel_path}\nLine: {i}\n\n"
                                f"Add logging or proper error handling.",
                                rel_path, i, "error_handling", 3, repo_name, "small"
                            ))

        return tasks

    def scan_performance(self, repo: str) -> List[Dict]:
        """Find performance issues: blocking calls, unnecessary allocations."""
        tasks = []
        repo_name = os.path.basename(repo)

        for filepath in self._walk_source_files(repo):
            ext = os.path.splitext(filepath)[1]
            rel_path = os.path.relpath(filepath, repo)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except (OSError, UnicodeDecodeError):
                continue

            if ext in KOTLIN_EXTS:
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()

                    # Thread.sleep on main thread indicators
                    if "Thread.sleep" in stripped and "runBlocking" not in stripped:
                        tasks.append(self._make_task(
                            f"Thread.sleep in {os.path.basename(filepath)}:{i}",
                            f"Thread.sleep blocks the current thread.\n\n"
                            f"File: {rel_path}\nLine: {i}\n\n"
                            f"Use delay() in coroutines or Handler.postDelayed() instead.",
                            rel_path, i, "performance", 3, repo_name, "small"
                        ))

                    # String concatenation in loops
                    if "+=" in stripped and "String" in "".join(lines[max(0,i-5):i]):
                        if any(kw in "".join(lines[max(0,i-10):i]) for kw in ["for ", "while "]):
                            tasks.append(self._make_task(
                                f"String concatenation in loop at {os.path.basename(filepath)}:{i}",
                                f"String concatenation in a loop creates unnecessary objects.\n\n"
                                f"File: {rel_path}\nLine: {i}\n\n"
                                f"Use StringBuilder instead.",
                                rel_path, i, "performance", 4, repo_name, "small"
                            ))

            elif ext in PYTHON_EXTS:
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()

                    # time.sleep in async context
                    if "time.sleep" in stripped:
                        # Check if we're in an async function
                        for j in range(max(0, i-20), i):
                            if "async def" in lines[j]:
                                tasks.append(self._make_task(
                                    f"Blocking sleep in async function at {os.path.basename(filepath)}:{i}",
                                    f"time.sleep() blocks the event loop in async code.\n\n"
                                    f"File: {rel_path}\nLine: {i}\n\n"
                                    f"Use await asyncio.sleep() instead.",
                                    rel_path, i, "performance", 2, repo_name, "small"
                                ))
                                break

        return tasks

    def scan_test_gaps(self, repo: str) -> List[Dict]:
        """Find classes/modules with no corresponding test file."""
        tasks = []
        repo_name = os.path.basename(repo)

        source_files = set()
        test_files = set()

        for filepath in self._walk_source_files(repo):
            basename = os.path.basename(filepath)
            name_no_ext = os.path.splitext(basename)[0]

            if "test" in filepath.lower() or "Test" in basename:
                test_files.add(name_no_ext.replace("Test", "").replace("test_", ""))
            else:
                # Only track important-looking files
                if any(p.lower() in name_no_ext.lower() for p in
                       ["ViewModel", "Repository", "Service", "Manager", "UseCase"]):
                    source_files.add(name_no_ext)

        untested = source_files - test_files
        for name in sorted(untested):
            tasks.append(self._make_task(
                f"No test file for {name}",
                f"Class {name} has no corresponding test file.\n\n"
                f"Consider adding unit tests for core logic.",
                name, 0, "test_gap", 4, repo_name, "medium"
            ))

        return tasks

    def prioritize(self, tasks: List[Dict]) -> List[Dict]:
        """Score and sort tasks by priority and importance."""
        for task in tasks:
            score = self._calculate_score(task)
            task["score"] = score

        return sorted(tasks, key=lambda t: t.get("score", 0), reverse=True)

    def mark_completed(self, task: Dict):
        """Mark a task hash as completed so it won't be regenerated."""
        h = self._task_hash(task)
        self._completed_hashes.add(h)
        self._save_completed_hashes()

    # ---- Internal helpers ----

    def _make_task(self, title, description, file_path, line_number,
                   category, priority, repo, scope) -> Dict:
        return {
            "id": str(uuid.uuid4()),
            "title": title,
            "description": description,
            "file_path": file_path,
            "line_number": line_number,
            "category": category,
            "priority": priority,
            "repo": repo,
            "scope": scope,
            "assigned_to": None,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }

    def _find_long_methods_kt(self, lines, rel_path, repo_name) -> List[Dict]:
        """Find Kotlin/Java methods longer than MAX_METHOD_LINES."""
        tasks = []
        method_pattern = re.compile(r"^\s*(private|public|protected|internal|override)?\s*(fun|suspend fun)\s+(\w+)")
        brace_depth = 0
        in_method = False
        method_name = ""
        method_start = 0

        for i, line in enumerate(lines, 1):
            match = method_pattern.match(line)
            if match and not in_method:
                method_name = match.group(3)
                method_start = i
                in_method = True
                brace_depth = 0

            if in_method:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0 and i > method_start:
                    length = i - method_start
                    if length > MAX_METHOD_LINES:
                        tasks.append(self._make_task(
                            f"Long method {method_name}() ({length} lines) in {os.path.basename(rel_path)}",
                            f"Method {method_name}() is {length} lines (max recommended: {MAX_METHOD_LINES}).\n\n"
                            f"File: {rel_path}\nLine: {method_start}\n\n"
                            f"Consider extracting helper methods.",
                            rel_path, method_start, "smell", 4, repo_name, "medium"
                        ))
                    in_method = False

        return tasks

    def _find_long_methods_py(self, lines, rel_path, repo_name) -> List[Dict]:
        """Find Python functions longer than MAX_METHOD_LINES."""
        tasks = []
        func_pattern = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)")
        functions = []

        for i, line in enumerate(lines, 1):
            match = func_pattern.match(line)
            if match:
                indent = len(match.group(1))
                name = match.group(3)
                functions.append((name, i, indent))

        for idx, (name, start, indent) in enumerate(functions):
            # Find end: next function at same or lower indent, or end of file
            if idx + 1 < len(functions):
                end = functions[idx + 1][1] - 1
            else:
                end = len(lines)

            length = end - start
            if length > MAX_METHOD_LINES:
                tasks.append(self._make_task(
                    f"Long function {name}() ({length} lines) in {os.path.basename(rel_path)}",
                    f"Function {name}() is {length} lines (max recommended: {MAX_METHOD_LINES}).\n\n"
                    f"File: {rel_path}\nLine: {start}\n\n"
                    f"Consider breaking into smaller functions.",
                    rel_path, start, "smell", 4, repo_name, "medium"
                ))

        return tasks

    def _find_deep_nesting(self, lines, rel_path, repo_name, ext) -> List[Dict]:
        """Find deeply nested code blocks."""
        tasks = []
        if ext in KOTLIN_EXTS | JAVA_EXTS:
            brace_depth = 0
            for i, line in enumerate(lines, 1):
                brace_depth += line.count("{") - line.count("}")
                if brace_depth > MAX_NESTING_DEPTH + 2:  # +2 for class + method
                    stripped = line.strip()
                    if stripped and not stripped.startswith("//"):
                        tasks.append(self._make_task(
                            f"Deep nesting (depth {brace_depth}) in {os.path.basename(rel_path)}:{i}",
                            f"Code nesting depth of {brace_depth} exceeds max of {MAX_NESTING_DEPTH + 2}.\n\n"
                            f"File: {rel_path}\nLine: {i}\n\n"
                            f"Extract into helper methods or use early returns.",
                            rel_path, i, "smell", 4, repo_name, "medium"
                        ))
                        break  # One per file to avoid noise

        elif ext in PYTHON_EXTS:
            for i, line in enumerate(lines, 1):
                if line.strip() and not line.strip().startswith("#"):
                    indent = len(line) - len(line.lstrip())
                    spaces_per_level = 4
                    depth = indent // spaces_per_level
                    if depth > MAX_NESTING_DEPTH:
                        tasks.append(self._make_task(
                            f"Deep nesting (depth {depth}) in {os.path.basename(rel_path)}:{i}",
                            f"Indentation depth of {depth} exceeds max of {MAX_NESTING_DEPTH}.\n\n"
                            f"File: {rel_path}\nLine: {i}\n\n"
                            f"Consider early returns or extracting functions.",
                            rel_path, i, "smell", 4, repo_name, "medium"
                        ))
                        break  # One per file

        return tasks

    def _walk_source_files(self, repo: str):
        """Walk source files, skipping build dirs and non-source files."""
        valid_exts = KOTLIN_EXTS | PYTHON_EXTS | JAVA_EXTS
        for root, dirs, files in os.walk(repo):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in files:
                ext = os.path.splitext(fname)[1]
                if ext in valid_exts:
                    yield os.path.join(root, fname)

    def _calculate_score(self, task: Dict) -> float:
        """Calculate priority score for a task."""
        # Base score from priority (1=critical=5pts, 5=nice-to-have=1pt)
        severity = 6 - task.get("priority", 3)

        # File importance multiplier
        importance = 1.0
        file_path = task.get("file_path", "")
        for pattern in IMPORTANT_FILE_PATTERNS:
            if pattern.lower() in file_path.lower():
                importance = 1.5
                break

        # Scope penalty (smaller = better for autonomous)
        scope_mult = {"small": 1.0, "medium": 0.7, "large": 0.3}
        scope = scope_mult.get(task.get("scope", "small"), 0.5)

        return severity * importance * scope

    def _task_hash(self, task: Dict) -> str:
        """Generate dedup hash from file_path + line_number + category."""
        key = f"{task.get('file_path')}:{task.get('line_number')}:{task.get('category')}"
        return hashlib.md5(key.encode()).hexdigest()

    def _deduplicate(self, tasks: List[Dict]) -> List[Dict]:
        """Remove duplicate tasks by hash."""
        seen = set()
        unique = []
        for task in tasks:
            h = self._task_hash(task)
            if h not in seen:
                seen.add(h)
                unique.append(task)
        return unique

    def _load_completed_hashes(self) -> set:
        """Load set of completed task hashes."""
        try:
            if os.path.exists(COMPLETED_HASHES_FILE):
                with open(COMPLETED_HASHES_FILE, "r") as f:
                    return set(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
        return set()

    def _save_completed_hashes(self):
        """Save completed task hashes to disk."""
        os.makedirs(os.path.dirname(COMPLETED_HASHES_FILE), exist_ok=True)
        with open(COMPLETED_HASHES_FILE, "w") as f:
            json.dump(list(self._completed_hashes), f)

    def _save_tasks(self, tasks: List[Dict]):
        """Save task list to disk."""
        os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=2)

    def _load_tasks(self) -> List[Dict]:
        """Load task list from disk."""
        try:
            if os.path.exists(TASKS_FILE):
                with open(TASKS_FILE, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        return []
