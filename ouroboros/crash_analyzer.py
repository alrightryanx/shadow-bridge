"""
The Architect - Analyzes crashes and generates fixes.

This is the brain of Ouroboros. When a crash is received:
1. Path Mapper: Locates the source file on disk
2. Context Builder: Reads code around the crash line
3. AI Analyst: Sends context to local LLM for analysis
4. Reporter: Creates GitHub issue with the fix

"""

import os
import glob
import re
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PathMapper:
    """Maps stack trace references to actual file paths on disk."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self._file_cache = {}  # Cache for faster lookups

    def find_file(self, file_name: str) -> Optional[Path]:
        """
        Finds a source file in the project directory.

        Args:
            file_name: Name of file (e.g., "MainActivity.kt")

        Returns:
            Full path to file, or None if not found
        """
        # Check cache first
        if file_name in self._file_cache:
            return self._file_cache[file_name]

        # Search for file recursively
        pattern = f"**/{file_name}"
        matches = list(self.project_root.glob(pattern))

        if not matches:
            logger.warning(f"File not found: {file_name}")
            return None

        if len(matches) > 1:
            # Multiple matches - prefer app/src over test directories
            for match in matches:
                if "test" not in str(match).lower():
                    self._file_cache[file_name] = match
                    return match

        # Use first match
        file_path = matches[0]
        self._file_cache[file_name] = file_path
        logger.info(f"Mapped {file_name} -> {file_path}")
        return file_path


class ContextBuilder:
    """Reads code context around crash location."""

    @staticmethod
    def read_context(file_path: Path, line_number: int, context_lines: int = 20) -> Optional[Dict]:
        """
        Reads code context around specified line.

        Args:
            file_path: Path to source file
            line_number: Line number where crash occurred
            context_lines: Number of lines before/after to include

        Returns:
            Dict with code snippet and metadata, or None if file can't be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_line = max(0, line_number - context_lines - 1)  # -1 for 0-indexing
            end_line = min(total_lines, line_number + context_lines)

            code_snippet = "".join(lines[start_line:end_line])

            return {
                "file_path": str(file_path),
                "crash_line": line_number,
                "start_line": start_line + 1,  # 1-indexed for display
                "end_line": end_line,
                "code_snippet": code_snippet,
                "total_lines": total_lines
            }

        except Exception as e:
            logger.error(f"Failed to read context from {file_path}: {e}")
            return None


class CrashAnalyzer:
    """Main crash analysis coordinator."""

    def __init__(self, project_root: str, db_path: str = None):
        self.project_root = project_root
        self.path_mapper = PathMapper(project_root)
        self.context_builder = ContextBuilder()
        self.pattern_memory = None

        # Initialize pattern memory if db_path provided
        if db_path:
            try:
                from .pattern_memory import PatternMemory
                self.pattern_memory = PatternMemory(db_path)
                logger.info("Pattern memory initialized")
            except Exception as e:
                logger.warning(f"Pattern memory init failed: {e}")

    def analyze_crash(self, crash_payload: Dict) -> Optional[Dict]:
        """
        Analyzes a crash and generates a fix suggestion.

        Args:
            crash_payload: Crash data from Android app

        Returns:
            Analysis result with fix suggestion, or None if analysis fails
        """
        try:
            # Extract crash details
            error = crash_payload.get("error", {})
            source_ref = error.get("source_reference", {})
            file_name = source_ref.get("file_name")
            line_number = source_ref.get("line_number")
            error_type = error.get("type")
            error_message = error.get("message")
            stack_trace = error.get("stack_trace_raw")
            flight_recorder = crash_payload.get("flight_recorder", [])

            if not file_name or not line_number:
                logger.error("Missing source reference in crash payload")
                return None

            # Step 1: Locate file on disk
            file_path = self.path_mapper.find_file(file_name)
            if not file_path:
                return {
                    "success": False,
                    "error": f"Could not locate file: {file_name}",
                    "file_name": file_name
                }

            # Step 2: Read code context
            context = self.context_builder.read_context(file_path, line_number)
            if not context:
                return {
                    "success": False,
                    "error": f"Could not read context from: {file_path}",
                    "file_name": file_name
                }

            # Step 2.5: Check pattern memory for prior occurrences
            pattern_info = None
            pattern_context = ""
            if self.pattern_memory:
                pattern_info = self.pattern_memory.record_crash(crash_payload)
                if pattern_info.get('is_known'):
                    sig = pattern_info.get('signature', '')
                    pattern_context = self.pattern_memory.get_pattern_context(sig) or ''
                    if pattern_info.get('is_regression'):
                        logger.warning(f"REGRESSION: {error_type} in {file_name} seen {pattern_info['occurrence_count']} times")

            # Step 3: Build AI prompt (with pattern memory context)
            prompt = self._build_analysis_prompt(
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                code_context=context,
                flight_recorder=flight_recorder,
                pattern_context=pattern_context
            )

            # Step 4: Get AI analysis (using multi-backend system)
            analysis = self._call_ai_analyst(prompt)

            if not analysis:
                return {
                    "success": False,
                    "error": "AI analysis failed",
                    "file_name": file_name
                }

            # Step 4.5: Record fix attempt in pattern memory
            if self.pattern_memory and pattern_info and analysis:
                sig = pattern_info.get('signature', '')
                fix_desc = analysis.get('fix', {}).get('description', 'AI-generated fix')
                self.pattern_memory.record_fix_attempt(sig, fix_desc, 'multi-backend')

            # Step 5: Return complete analysis
            return {
                "success": True,
                "file_path": str(file_path),
                "file_name": file_name,
                "line_number": line_number,
                "error_type": error_type,
                "error_message": error_message,
                "code_context": context,
                "ai_analysis": analysis,
                "flight_recorder": flight_recorder,
                "crash_payload": crash_payload,
                "pattern_info": pattern_info
            }

        except Exception as e:
            logger.error(f"Crash analysis failed: {e}")
            return None

    def _build_analysis_prompt(
        self,
        error_type: str,
        error_message: str,
        stack_trace: str,
        code_context: Dict,
        flight_recorder: List[Dict],
        pattern_context: str = ""
    ) -> str:
        """Builds comprehensive AI prompt for crash analysis."""

        # Format flight recorder events
        user_actions = "\n".join([
            f"  {i+1}. [{event.get('timestamp_formatted', '')}] "
            f"{event.get('event', '')} - {event.get('target', '')} @ {event.get('screen', '')}"
            for i, event in enumerate(flight_recorder)
        ])

        # Include pattern memory context if available
        pattern_section = ""
        if pattern_context:
            pattern_section = f"\n{pattern_context}\n"

        prompt = f"""
# Crash Analysis Task

You are analyzing a crash in the Shadow Android app to identify the root cause and provide a fix.

## Crash Details

**Error Type:** `{error_type}`
**Error Message:** `{error_message}`

**Location:** `{code_context['file_path']}:{code_context['crash_line']}`

## User Actions Leading to Crash (Last 20 interactions)

{user_actions if user_actions else "  (No flight recorder data available)"}

## Code Context

File: {code_context['file_path']}
Lines {code_context['start_line']}-{code_context['end_line']} (crash at line {code_context['crash_line']}):

```kotlin
{code_context['code_snippet']}
```

## Stack Trace (First 30 lines)

```
{chr(10).join(stack_trace.split(chr(10))[:30])}
```

{pattern_section}
## Analysis Required

Please provide:

1. **Root Cause:** What caused this crash?
2. **Reproduction Steps:** How to reproduce based on user actions
3. **Fix:** Specific code change to prevent this crash
4. **Risk Assessment:** Is this fix low/medium/high risk?

## Output Format

Return your analysis as JSON:

```json
{{
  "root_cause": "Brief explanation of what went wrong",
  "reproduction_steps": [
    "Step 1",
    "Step 2",
    "..."
  ],
  "fix": {{
    "description": "What the fix does",
    "code_changes": [
      {{
        "file": "path/to/file.kt",
        "line_number": 123,
        "old_code": "// Code that caused crash",
        "new_code": "// Fixed code"
      }}
    ]
  }},
  "risk_level": "low|medium|high",
  "confidence": 0.9
}}
```

Analyze the crash and provide your fix:
""".strip()

        return prompt

    def _call_ai_analyst(self, prompt: str) -> Optional[Dict]:
        """
        Calls AI backend to analyze crash.
        Uses multi-backend system from ai_analyzer.py
        """
        try:
            # Import the multi-backend AI analyzer
            from .ai_analyzer import AIAnalyzer

            analyzer = AIAnalyzer(self.project_root)

            # Define a custom task for crash analysis that AIAnalyzer can handle
            # We're reusing the _query_backend logic by constructing a pseudo-metric
            # or by adding a dedicated method to AIAnalyzer. 
            # For now, let's add a direct analyze_prompt method to AIAnalyzer to handle raw prompts.
            
            # Since we can't easily modify AIAnalyzer in this step without a separate call,
            # let's assume we'll update AIAnalyzer to have an `analyze_prompt` method.
            # If not, we can use the existing analyze_telemetry and trick it, but that's messy.
            # Better approach: We will implement analyze_prompt in AIAnalyzer in the next step.
            
            return analyzer.analyze_prompt(prompt)

        except Exception as e:
            logger.error(f"AI analyst call failed: {e}")
            return None




class GitHubReporter:
    """Creates GitHub issues for crashes with AI analysis."""

    # Dedup: 24-hour cooldown, persisted to JSON file
    DEDUP_COOLDOWN_SECS = 24 * 60 * 60
    DEDUP_FILE = "github_issue_dedup.json"

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._dedup_path = Path(repo_path).parent / "shadow-bridge" / "telemetry_data" / self.DEDUP_FILE

    def _compute_fingerprint(self, title: str) -> str:
        """Normalize title to a dedup fingerprint, stripping variable numbers."""
        fp = title
        fp = re.sub(r'\d+x\d+', 'NxN', fp)
        fp = re.sub(r'\d+dp', 'Ndp', fp)
        fp = re.sub(r'\d+ms', 'Nms', fp)
        fp = re.sub(r'\d+\.\d+', 'N.N', fp)
        fp = re.sub(r'\d+', 'N', fp)
        return fp.strip().lower()

    def _load_dedup_cache(self) -> dict:
        try:
            if self._dedup_path.exists():
                with open(self._dedup_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_dedup_cache(self, cache: dict):
        try:
            self._dedup_path.parent.mkdir(parents=True, exist_ok=True)
            # Prune expired entries
            now = datetime.now().timestamp()
            pruned = {k: v for k, v in cache.items()
                      if (now - v) < self.DEDUP_COOLDOWN_SECS}
            with open(self._dedup_path, 'w') as f:
                json.dump(pruned, f)
        except Exception as e:
            logger.warning(f"Failed to save dedup cache: {e}")

    def _is_duplicate(self, fingerprint: str) -> bool:
        cache = self._load_dedup_cache()
        last_filed = cache.get(fingerprint, 0)
        if last_filed == 0:
            return False
        return (datetime.now().timestamp() - last_filed) < self.DEDUP_COOLDOWN_SECS

    def _record_issue(self, fingerprint: str):
        cache = self._load_dedup_cache()
        cache[fingerprint] = datetime.now().timestamp()
        self._save_dedup_cache(cache)

    def create_crash_issue(self, analysis: Dict) -> bool:
        """
        Creates a GitHub issue for the crash with AI-generated fix.

        Args:
            analysis: Complete analysis from CrashAnalyzer

        Returns:
            True if issue created successfully
        """
        try:
            # Build issue title
            error_type = analysis.get("error_type", "Unknown")
            file_name = analysis.get("file_name", "Unknown")
            line_number = analysis.get("line_number", "?")

            title = f"[AUTO-FIX] {error_type} in {file_name}:{line_number}"

            # Dedup: skip if a similar issue was recently filed
            fingerprint = self._compute_fingerprint(title)
            if self._is_duplicate(fingerprint):
                logger.info(f"Skipping duplicate crash issue: {title}")
                return False

            # Build issue body
            body = self._build_issue_body(analysis)

            # Create issue using gh CLI
            result = subprocess.run(
                [
                    "gh", "issue", "create",
                    "--title", title,
                    "--body", body,
                    "--label", "bug",
                    "--label", "auto-generated",
                    "--label", "ouroboros",
                    "--assignee", "@me"
                ],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )

            if result.returncode == 0:
                issue_url = result.stdout.strip()
                logger.info(f"Created GitHub issue: {issue_url}")
                self._record_issue(fingerprint)
                return True
            else:
                logger.error(f"Failed to create issue: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"GitHub issue creation failed: {e}")
            return False

    def _build_issue_body(self, analysis: Dict) -> str:
        """Builds markdown body for GitHub issue."""

        ai_analysis = analysis.get("ai_analysis", {})
        root_cause = ai_analysis.get("root_cause", "Unknown")
        fix_desc = ai_analysis.get("fix", {}).get("description", "No fix provided")
        code_changes = ai_analysis.get("fix", {}).get("code_changes", [])
        risk_level = ai_analysis.get("risk_level", "unknown")
        flight_recorder = analysis.get("flight_recorder", [])

        # Format code changes
        changes_md = ""
        for change in code_changes:
            changes_md += f"""
**File:** `{change.get('file', 'unknown')}:{change.get('line_number', '?')}`

```kotlin
// OLD:
{change.get('old_code', '')}

// NEW:
{change.get('new_code', '')}
```

"""

        # Format user actions
        actions_md = "\n".join([
            f"{i+1}. `{event.get('event', '')}` - {event.get('target', '')} @ {event.get('screen', '')}"
            for i, event in enumerate(flight_recorder[:10])  # First 10 actions
        ])

        body = f"""
## Auto-Detected Crash

**Error:** `{analysis.get('error_type', 'Unknown')}`
**Message:** `{analysis.get('error_message', 'No message')}`
**Location:** `{analysis.get('file_path', 'Unknown')}:{analysis.get('line_number', '?')}`

### User Actions Before Crash

{actions_md if actions_md else "*(No flight recorder data)*"}

### Root Cause

{root_cause}

### Proposed Fix

{fix_desc}

**Risk Level:** `{risk_level.upper()}`

### Code Changes

{changes_md if changes_md else "*(No code changes provided)*"}

---

🤖 Generated by Ouroboros Auto-Fix System
📅 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""".strip()

        return body
