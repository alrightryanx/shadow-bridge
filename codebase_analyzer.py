"""
Codebase Analyzer for Shadow AI autonomous task generation.
Analyzes git repos to find TODOs, test gaps, hot files, and generate task suggestions.
"""
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
import json
import logging

log = logging.getLogger("codebase_analyzer")

@dataclass
class TodoItem:
    file: str
    line: int
    text: str
    priority: str = "MEDIUM"  # HIGH if FIXME, MEDIUM if TODO, LOW if HACK/NOTE

@dataclass
class HotFile:
    file: str
    change_count: int
    last_changed: str

@dataclass
class ComplexityInfo:
    file: str
    lines: int
    functions: int = 0

@dataclass
class TestCoverage:
    test_files: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    untested_files: List[str] = field(default_factory=list)
    estimated_percent: float = 0.0

@dataclass
class TaskSuggestion:
    title: str
    description: str
    task_type: str  # CODE, DOCS, DEVOPS, GENERAL
    priority: str   # HIGH, MEDIUM, LOW
    source: str     # todo, coverage, complexity, churn

@dataclass
class CodebaseReport:
    tech_stack: List[str] = field(default_factory=list)
    todos: List[TodoItem] = field(default_factory=list)
    test_coverage: Optional[TestCoverage] = None
    hot_files: List[HotFile] = field(default_factory=list)
    complexity: List[ComplexityInfo] = field(default_factory=list)
    suggested_tasks: List[TaskSuggestion] = field(default_factory=list)
    analyzed_at: str = ""

    def to_dict(self):
        return asdict(self)


class CodebaseAnalyzer:
    """Analyzes a codebase to generate autonomous task suggestions."""

    # File extensions to language mapping
    LANG_MAP = {
        '.py': 'Python', '.kt': 'Kotlin', '.java': 'Java', '.js': 'JavaScript',
        '.ts': 'TypeScript', '.tsx': 'TypeScript/React', '.jsx': 'JavaScript/React',
        '.go': 'Go', '.rs': 'Rust', '.cpp': 'C++', '.c': 'C', '.rb': 'Ruby',
        '.swift': 'Swift', '.dart': 'Dart', '.cs': 'C#', '.php': 'PHP',
    }

    FRAMEWORK_MARKERS = {
        'requirements.txt': 'Python/pip', 'setup.py': 'Python/setuptools',
        'package.json': 'Node.js', 'build.gradle': 'Gradle', 'build.gradle.kts': 'Gradle/Kotlin',
        'Cargo.toml': 'Rust/Cargo', 'go.mod': 'Go Modules', 'Gemfile': 'Ruby/Bundler',
        'Dockerfile': 'Docker', 'docker-compose.yml': 'Docker Compose',
        'Makefile': 'Make', '.github/workflows': 'GitHub Actions',
    }

    def analyze(self, repo_path: str) -> CodebaseReport:
        """Run full analysis on a repository."""
        if not os.path.isdir(repo_path):
            log.error(f"Not a directory: {repo_path}")
            return CodebaseReport(analyzed_at=datetime.now().isoformat())

        report = CodebaseReport(analyzed_at=datetime.now().isoformat())
        report.tech_stack = self.detect_stack(repo_path)
        report.todos = self.extract_todos(repo_path)
        report.test_coverage = self.estimate_coverage(repo_path)
        report.hot_files = self.git_churn_analysis(repo_path)
        report.complexity = self.find_complex_files(repo_path)
        report.suggested_tasks = self.generate_suggestions(report)
        return report

    def detect_stack(self, repo_path: str) -> List[str]:
        """Detect languages and frameworks from file extensions and markers."""
        stack = set()
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden dirs, node_modules, build dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'build', 'dist', '__pycache__', '.gradle')]
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext in self.LANG_MAP:
                    stack.add(self.LANG_MAP[ext])
                if f in self.FRAMEWORK_MARKERS:
                    stack.add(self.FRAMEWORK_MARKERS[f])
        return sorted(stack)

    def extract_todos(self, repo_path: str, max_results: int = 50) -> List[TodoItem]:
        """Find TODO, FIXME, HACK, XXX comments in source files."""
        todos = []
        pattern = re.compile(r'(TODO|FIXME|HACK|XXX|NOTE)[\s:]+(.+)', re.IGNORECASE)
        priority_map = {'FIXME': 'HIGH', 'TODO': 'MEDIUM', 'HACK': 'MEDIUM', 'XXX': 'HIGH', 'NOTE': 'LOW'}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'build', 'dist', '__pycache__', '.gradle')]
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext not in self.LANG_MAP:
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                        for i, line in enumerate(fh, 1):
                            m = pattern.search(line)
                            if m:
                                tag = m.group(1).upper()
                                text = m.group(2).strip()
                                rel = os.path.relpath(filepath, repo_path)
                                todos.append(TodoItem(
                                    file=rel, line=i, text=text[:200],
                                    priority=priority_map.get(tag, 'MEDIUM')
                                ))
                                if len(todos) >= max_results:
                                    return todos
                except (IOError, OSError):
                    continue
        return todos

    def estimate_coverage(self, repo_path: str) -> TestCoverage:
        """Estimate test coverage by matching test files to source files."""
        test_files = []
        source_files = []

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'build', 'dist', '__pycache__', '.gradle')]
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext not in self.LANG_MAP:
                    continue
                rel = os.path.relpath(os.path.join(root, f), repo_path)
                name_lower = f.lower()
                if ('test' in name_lower or 'spec' in name_lower or
                    '/test/' in rel.replace('\\', '/') or '/tests/' in rel.replace('\\', '/')):
                    test_files.append(rel)
                else:
                    source_files.append(rel)

        # Find source files without corresponding tests
        test_basenames = {os.path.splitext(os.path.basename(t))[0].replace('test_', '').replace('_test', '').replace('Test', '').lower() for t in test_files}
        untested = [s for s in source_files if os.path.splitext(os.path.basename(s))[0].lower() not in test_basenames]

        total = len(source_files)
        tested = total - len(untested)
        pct = (tested / total * 100) if total > 0 else 0

        return TestCoverage(
            test_files=test_files[:20],
            source_files=source_files[:20],
            untested_files=untested[:20],
            estimated_percent=round(pct, 1)
        )

    def git_churn_analysis(self, repo_path: str, days: int = 30) -> List[HotFile]:
        """Find most-changed files from git log."""
        try:
            result = subprocess.run(
                ['git', 'log', f'--since={days} days ago', '--pretty=format:', '--name-only'],
                cwd=repo_path, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return []

            counts = {}
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('.'):
                    counts[line] = counts.get(line, 0) + 1

            sorted_files = sorted(counts.items(), key=lambda x: -x[1])[:15]

            hot = []
            for f, count in sorted_files:
                # Get last change date
                try:
                    date_result = subprocess.run(
                        ['git', 'log', '-1', '--format=%ci', '--', f],
                        cwd=repo_path, capture_output=True, text=True, timeout=10
                    )
                    last = date_result.stdout.strip()[:10] if date_result.returncode == 0 else ''
                except Exception:
                    last = ''
                hot.append(HotFile(file=f, change_count=count, last_changed=last))
            return hot
        except Exception as e:
            log.warning(f"Git churn analysis failed: {e}")
            return []

    def find_complex_files(self, repo_path: str, min_lines: int = 300) -> List[ComplexityInfo]:
        """Find files with high line counts."""
        complex_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'build', 'dist', '__pycache__', '.gradle')]
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext not in self.LANG_MAP:
                    continue
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                        lines = sum(1 for _ in fh)
                    if lines >= min_lines:
                        rel = os.path.relpath(filepath, repo_path)
                        complex_files.append(ComplexityInfo(file=rel, lines=lines))
                except (IOError, OSError):
                    continue
        return sorted(complex_files, key=lambda x: -x.lines)[:15]

    def generate_suggestions(self, report: CodebaseReport) -> List[TaskSuggestion]:
        """Generate actionable task suggestions from analysis report."""
        tasks = []

        # From TODOs
        for todo in report.todos[:10]:
            prio = todo.priority
            tasks.append(TaskSuggestion(
                title=f"Address {todo.priority} {todo.text[:60]}",
                description=f"Found in {todo.file}:{todo.line}: {todo.text}",
                task_type="CODE",
                priority=prio,
                source="todo"
            ))

        # From test coverage gaps
        if report.test_coverage and report.test_coverage.untested_files:
            for f in report.test_coverage.untested_files[:5]:
                tasks.append(TaskSuggestion(
                    title=f"Add tests for {os.path.basename(f)}",
                    description=f"File {f} has no corresponding test file. Add unit tests to improve coverage.",
                    task_type="CODE",
                    priority="MEDIUM",
                    source="coverage"
                ))

        # From complex files
        for cf in report.complexity[:5]:
            if cf.lines > 500:
                tasks.append(TaskSuggestion(
                    title=f"Refactor {os.path.basename(cf.file)} ({cf.lines} lines)",
                    description=f"File {cf.file} has {cf.lines} lines. Consider splitting into smaller modules.",
                    task_type="CODE",
                    priority="LOW",
                    source="complexity"
                ))

        # From hot files (frequently changed = potential instability)
        for hf in report.hot_files[:3]:
            if hf.change_count > 10:
                tasks.append(TaskSuggestion(
                    title=f"Stabilize {os.path.basename(hf.file)} ({hf.change_count} changes)",
                    description=f"File {hf.file} changed {hf.change_count} times in the last 30 days. May need refactoring or better tests.",
                    task_type="CODE",
                    priority="MEDIUM",
                    source="churn"
                ))

        return tasks
