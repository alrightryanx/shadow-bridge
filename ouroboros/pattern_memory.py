"""
Ouroboros V2 - Pattern Memory

Hashes crashes by (error_type, file_name, method_name, line_range) to detect
recurring patterns. Stores occurrence count, fix attempts, root cause category,
and resolution status.

Detects regressions (resolved pattern recurring) and provides prior fix context
to the AI analyzer for better suggestions.
"""

import sqlite3
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PatternMemory:
    """
    Persistent memory for crash patterns.

    Each unique crash signature is stored with:
    - Occurrence count
    - First/last seen timestamps
    - Fix attempts history
    - Root cause category
    - Resolution status
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def compute_signature(error_type: str, file_name: str,
                          method_name: str = '', line_number: int = 0) -> str:
        """
        Create a stable hash for a crash pattern.
        Uses a line range (±5 lines) to account for minor code shifts.
        """
        line_range = f"{max(0, line_number - 5)}-{line_number + 5}"
        payload = f"{error_type}:{file_name}:{method_name}:{line_range}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def record_crash(self, crash_payload: Dict) -> Dict:
        """
        Record a crash occurrence. Returns pattern info including whether
        this is a known pattern and any prior fix context.
        """
        error = crash_payload.get('error', {})
        source_ref = error.get('source_reference', {})

        error_type = error.get('type', 'Unknown')
        file_name = source_ref.get('file_name', 'Unknown')
        method_name = source_ref.get('method_name', '')
        line_number = source_ref.get('line_number', 0)

        signature = self.compute_signature(error_type, file_name, method_name, line_number)
        line_range = f"{max(0, line_number - 5)}-{line_number + 5}"

        try:
            conn = self._get_connection()

            # Check if pattern exists
            cursor = conn.execute(
                "SELECT * FROM crash_patterns WHERE id = ?",
                (signature,)
            )
            row = cursor.fetchone()

            if row:
                columns = [desc[0] for desc in cursor.description]
                existing = dict(zip(columns, row))

                # Increment occurrence count
                new_count = existing['occurrence_count'] + 1
                conn.execute(
                    "UPDATE crash_patterns SET occurrence_count = ?, last_seen = ? WHERE id = ?",
                    (new_count, datetime.now().isoformat(), signature)
                )
                conn.commit()

                # Check for regression
                is_regression = existing.get('resolved', 0) == 1
                if is_regression:
                    # Mark as unresolved again
                    conn.execute(
                        "UPDATE crash_patterns SET resolved = 0 WHERE id = ?",
                        (signature,)
                    )
                    conn.commit()
                    logger.warning(
                        f"[PatternMemory] REGRESSION detected: {error_type} in {file_name} "
                        f"(previously resolved, occurred {new_count} times)"
                    )

                conn.close()

                # Parse fix attempts
                fix_attempts = []
                try:
                    fix_attempts = json.loads(existing.get('fix_attempts', '[]'))
                except (json.JSONDecodeError, TypeError):
                    pass

                return {
                    'is_known': True,
                    'is_regression': is_regression,
                    'occurrence_count': new_count,
                    'first_seen': existing.get('first_seen'),
                    'fix_attempts': fix_attempts,
                    'root_cause_category': existing.get('root_cause_category'),
                    'resolution_summary': existing.get('resolution_summary'),
                    'signature': signature
                }

            else:
                # New pattern
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT INTO crash_patterns
                    (id, error_type, file_name, method_name, line_range,
                     occurrence_count, first_seen, last_seen, fix_attempts,
                     root_cause_category, resolved, resolution_summary)
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?, '[]', NULL, 0, NULL)
                """, (signature, error_type, file_name, method_name, line_range, now, now))
                conn.commit()
                conn.close()

                logger.info(f"[PatternMemory] New pattern: {error_type} in {file_name}")

                return {
                    'is_known': False,
                    'is_regression': False,
                    'occurrence_count': 1,
                    'first_seen': now,
                    'fix_attempts': [],
                    'root_cause_category': None,
                    'resolution_summary': None,
                    'signature': signature
                }

        except Exception as e:
            logger.error(f"[PatternMemory] Failed to record crash: {e}")
            return {
                'is_known': False,
                'is_regression': False,
                'occurrence_count': 0,
                'error': str(e),
                'signature': signature
            }

    def record_fix_attempt(self, signature: str, fix_description: str,
                           ai_model: str = 'unknown', success: bool = False) -> bool:
        """Record a fix attempt for a crash pattern."""
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT fix_attempts FROM crash_patterns WHERE id = ?",
                (signature,)
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False

            attempts = []
            try:
                attempts = json.loads(row[0] or '[]')
            except (json.JSONDecodeError, TypeError):
                pass

            attempts.append({
                'timestamp': datetime.now().isoformat(),
                'description': fix_description,
                'ai_model': ai_model,
                'success': success
            })

            conn.execute(
                "UPDATE crash_patterns SET fix_attempts = ? WHERE id = ?",
                (json.dumps(attempts), signature)
            )
            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"[PatternMemory] Failed to record fix attempt: {e}")
            return False

    def resolve_pattern(self, signature: str, root_cause: str,
                        resolution_summary: str) -> bool:
        """Mark a crash pattern as resolved."""
        try:
            conn = self._get_connection()
            conn.execute("""
                UPDATE crash_patterns
                SET resolved = 1, root_cause_category = ?, resolution_summary = ?
                WHERE id = ?
            """, (root_cause, resolution_summary, signature))
            conn.commit()
            conn.close()
            logger.info(f"[PatternMemory] Pattern {signature[:12]}... resolved: {root_cause}")
            return True
        except Exception as e:
            logger.error(f"[PatternMemory] Failed to resolve pattern: {e}")
            return False

    def get_top_patterns(self, limit: int = 10) -> List[Dict]:
        """Get the most frequent unresolved crash patterns."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT * FROM crash_patterns
                WHERE resolved = 0
                ORDER BY occurrence_count DESC
                LIMIT ?
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            patterns = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()

            # Parse fix_attempts JSON
            for p in patterns:
                try:
                    p['fix_attempts'] = json.loads(p.get('fix_attempts', '[]'))
                except (json.JSONDecodeError, TypeError):
                    p['fix_attempts'] = []

            return patterns
        except Exception as e:
            logger.error(f"[PatternMemory] Failed to get top patterns: {e}")
            return []

    def get_pattern_context(self, signature: str) -> Optional[str]:
        """
        Build a context string for the AI analyzer with prior fix history.
        Used to enrich crash analysis prompts with pattern memory.
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM crash_patterns WHERE id = ?",
                (signature,)
            )
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            columns = [desc[0] for desc in cursor.description]
            pattern = dict(zip(columns, row))

            fix_attempts = []
            try:
                fix_attempts = json.loads(pattern.get('fix_attempts', '[]'))
            except (json.JSONDecodeError, TypeError):
                pass

            context = f"""
## Pattern Memory Context

This crash has been seen **{pattern['occurrence_count']} times** since {pattern['first_seen']}.
Error: {pattern['error_type']} in {pattern['file_name']}:{pattern['line_range']}
"""

            if pattern.get('root_cause_category'):
                context += f"\nPreviously categorized as: **{pattern['root_cause_category']}**\n"

            if pattern.get('resolution_summary'):
                context += f"\nPrevious resolution: {pattern['resolution_summary']}\n"
                context += "\n**WARNING: This is a REGRESSION** - the previous fix did not hold.\n"

            if fix_attempts:
                context += f"\n### Prior Fix Attempts ({len(fix_attempts)} total):\n"
                for i, attempt in enumerate(fix_attempts[-3:], 1):  # Last 3 attempts
                    status = "SUCCESS" if attempt.get('success') else "FAILED"
                    context += (
                        f"{i}. [{status}] {attempt.get('description', 'No description')} "
                        f"(via {attempt.get('ai_model', 'unknown')}, "
                        f"{attempt.get('timestamp', 'unknown')})\n"
                    )
                context += "\nDo NOT repeat the same fix approaches that failed before.\n"

            return context.strip()

        except Exception as e:
            logger.error(f"[PatternMemory] Failed to build context: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get pattern memory statistics."""
        try:
            conn = self._get_connection()

            total = conn.execute("SELECT COUNT(*) FROM crash_patterns").fetchone()[0]
            unresolved = conn.execute(
                "SELECT COUNT(*) FROM crash_patterns WHERE resolved = 0"
            ).fetchone()[0]
            resolved = conn.execute(
                "SELECT COUNT(*) FROM crash_patterns WHERE resolved = 1"
            ).fetchone()[0]
            total_occurrences = conn.execute(
                "SELECT COALESCE(SUM(occurrence_count), 0) FROM crash_patterns"
            ).fetchone()[0]

            conn.close()

            return {
                'total_patterns': total,
                'unresolved': unresolved,
                'resolved': resolved,
                'total_occurrences': total_occurrences
            }
        except Exception as e:
            logger.error(f"[PatternMemory] Failed to get stats: {e}")
            return {}
