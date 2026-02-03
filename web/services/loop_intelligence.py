"""
Loop Intelligence - Multi-factor scoring for subagent loop deployment decisions.

Decides WHEN to deploy an iterative subagent loop based on:
- Keyword matching (iterative task indicators)
- Task complexity estimation
- Historical loop success rates
- Scope breadth of the task
- Cost efficiency projection

Usage:
    intel = LoopIntelligence(db_path)
    should_deploy, score, factors = intel.should_deploy_loop(task_description, context)
    if should_deploy:
        iterations = intel.recommend_iterations(score, task_description)
"""

import os
import re
import json
import math
import logging
import sqlite3
from typing import Tuple, Dict, Optional, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Configurable thresholds via environment
DEPLOY_THRESHOLD = float(os.environ.get("LOOP_DEPLOY_THRESHOLD", "0.55"))
MAX_LOOP_ITERATIONS = int(os.environ.get("MAX_LOOP_ITERATIONS", "10"))

# Scoring weights
WEIGHTS = {
    "keyword_match": 0.20,
    "task_complexity": 0.25,
    "historical_success": 0.20,
    "scope_breadth": 0.15,
    "cost_efficiency": 0.20,
}

# Iterative keyword categories with sub-scores
KEYWORD_GROUPS = {
    "refinement": {
        "weight": 1.0,
        "keywords": ["refine", "improve", "enhance", "polish", "iterate", "optimize", "tweak"],
    },
    "bulk_operations": {
        "weight": 0.9,
        "keywords": ["scan each", "test all", "check every", "for each file", "all files",
                      "every module", "across all", "batch"],
    },
    "quality_loops": {
        "weight": 0.95,
        "keywords": ["until passing", "until clean", "until stable", "until fixed",
                      "keep trying", "retry until", "fix until"],
    },
    "exploration": {
        "weight": 0.7,
        "keywords": ["try different", "explore options", "compare approaches",
                      "experiment with", "evaluate alternatives"],
    },
}

# Complexity indicators
COMPLEXITY_INDICATORS = {
    "high": {
        "weight": 1.0,
        "patterns": [
            r"refactor\s+(?:entire|whole|all|complete)",
            r"migrat(?:e|ion)",
            r"redesign",
            r"architect",
            r"rewrite",
            r"integration\s+test",
            r"end.to.end",
            r"cross.module",
            r"backward.compat",
        ],
    },
    "medium": {
        "weight": 0.6,
        "patterns": [
            r"add\s+(?:error|exception)\s+handling",
            r"implement\s+(?:caching|retry|fallback)",
            r"update\s+(?:multiple|several)",
            r"fix\s+(?:flaky|intermittent)",
            r"performance",
            r"security\s+(?:audit|scan|review)",
        ],
    },
    "low": {
        "weight": 0.3,
        "patterns": [
            r"update\s+(?:comment|doc|readme)",
            r"rename",
            r"format",
            r"lint",
            r"typo",
        ],
    },
}


class LoopIntelligence:
    """Multi-factor scoring system for subagent loop deployment decisions."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            "C:", os.sep, "shadow", "backend", "data", "shadow_ai.db"
        )
        self.deploy_threshold = DEPLOY_THRESHOLD
        self.max_iterations = MAX_LOOP_ITERATIONS

    def should_deploy_loop(
        self, task_description: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Evaluate whether a subagent loop should be deployed for a given task.

        Returns:
            (should_deploy, composite_score, factor_scores)
        """
        context = context or {}
        task_lower = task_description.lower()

        factors = {
            "keyword_match": self._score_keywords(task_lower),
            "task_complexity": self._score_complexity(task_lower),
            "historical_success": self._score_history(task_lower),
            "scope_breadth": self._score_scope(task_lower, context),
            "cost_efficiency": self._score_cost_efficiency(task_lower),
        }

        score = sum(factors[k] * WEIGHTS[k] for k in factors)

        logger.info(
            f"Loop deployment score: {score:.3f} (threshold: {self.deploy_threshold}) "
            f"factors: {json.dumps({k: round(v, 3) for k, v in factors.items()})}"
        )

        return score >= self.deploy_threshold, score, factors

    def recommend_iterations(self, score: float, task_description: str) -> int:
        """Recommend number of iterations based on score and task characteristics."""
        if score >= 0.85:
            base = 10
        elif score >= 0.70:
            base = 5
        elif score >= 0.55:
            base = 3
        else:
            base = 2

        # Cap at configured max
        return min(base, self.max_iterations)

    def record_loop_result(
        self, task_description: str, success: bool, iterations_used: int, score: float
    ):
        """Record loop outcome for future historical scoring."""
        try:
            conn = self._get_db()
            if conn is None:
                return
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO loop_history
                   (parent_goal_id, loop_prompt, iteration_count, max_iterations, status, deploy_score, started_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))""",
                [
                    None,
                    task_description[:500],
                    iterations_used,
                    self.max_iterations,
                    "completed" if success else "failed",
                    score,
                ],
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to record loop result: {e}")

    # ---- Scoring Functions ----

    def _score_keywords(self, task_lower: str) -> float:
        """Score based on presence of iterative keywords."""
        max_group_score = 0.0
        total_matches = 0

        for group_name, group in KEYWORD_GROUPS.items():
            for keyword in group["keywords"]:
                if keyword in task_lower:
                    group_score = group["weight"]
                    max_group_score = max(max_group_score, group_score)
                    total_matches += 1

        if total_matches == 0:
            return 0.0

        # Base score from best matching group, boosted by number of matches
        boost = min(total_matches * 0.1, 0.3)
        return min(max_group_score + boost, 1.0)

    def _score_complexity(self, task_lower: str) -> float:
        """Score based on estimated task complexity."""
        best_level_score = 0.0

        for level, config in COMPLEXITY_INDICATORS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, task_lower):
                    best_level_score = max(best_level_score, config["weight"])

        # Also factor in prompt length as a rough complexity proxy
        word_count = len(task_lower.split())
        length_score = min(word_count / 100.0, 0.5)  # 100 words = 0.5

        # Multi-step indicators
        step_patterns = [r"\d+\.", r"first.*then", r"step\s+\d", r"after that"]
        step_bonus = 0.0
        for pat in step_patterns:
            if re.search(pat, task_lower):
                step_bonus = 0.2
                break

        return min(best_level_score + length_score + step_bonus, 1.0)

    def _score_history(self, task_lower: str) -> float:
        """Score based on historical loop success for similar tasks."""
        try:
            conn = self._get_db()
            if conn is None:
                return 0.5  # Neutral if no DB access

            cursor = conn.cursor()

            # Get recent loop history
            cursor.execute(
                "SELECT status, deploy_score, loop_prompt FROM loop_history ORDER BY started_at DESC LIMIT 50"
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return 0.5  # Neutral score for no history

            # Calculate overall success rate
            completed = sum(1 for r in rows if r[0] == "completed")
            total = len(rows)
            success_rate = completed / total if total > 0 else 0.5

            # Check for similar task descriptions (simple keyword overlap)
            task_words = set(task_lower.split())
            similar_results = []
            for row in rows:
                prompt_words = set((row[2] or "").lower().split())
                overlap = len(task_words & prompt_words)
                if overlap >= 3:  # At least 3 words in common
                    similar_results.append(row[0] == "completed")

            if similar_results:
                similar_rate = sum(similar_results) / len(similar_results)
                # Weight similar task history more heavily
                return similar_rate * 0.7 + success_rate * 0.3

            return success_rate

        except Exception as e:
            logger.warning(f"Failed to score history: {e}")
            return 0.5

    def _score_scope(self, task_lower: str, context: Dict[str, Any]) -> float:
        """Score based on the breadth of the task scope."""
        score = 0.0

        # File/target counting patterns
        file_patterns = [
            r"all\s+(?:files|modules|components|services|tests)",
            r"every\s+(?:file|module|component|service|test)",
            r"each\s+(?:file|module|component|service|test)",
            r"across\s+(?:the\s+)?(?:codebase|project|repo)",
            r"entire\s+(?:codebase|project|directory)",
        ]
        for pat in file_patterns:
            if re.search(pat, task_lower):
                score = max(score, 0.8)

        # Check context for explicit file lists
        files = context.get("files", [])
        if isinstance(files, list):
            if len(files) > 10:
                score = max(score, 0.9)
            elif len(files) > 5:
                score = max(score, 0.7)
            elif len(files) > 2:
                score = max(score, 0.5)

        # Multi-repo indicators
        if re.search(r"(?:both|all)\s+repos?", task_lower):
            score = max(score, 0.85)

        # If no scope indicators found, estimate from description length
        if score == 0.0:
            word_count = len(task_lower.split())
            score = min(word_count / 150.0, 0.4)

        return min(score, 1.0)

    def _score_cost_efficiency(self, task_lower: str) -> float:
        """Score based on estimated cost efficiency of iterative approach vs single-shot."""
        # Tasks where iteration is clearly cost-effective
        high_efficiency_patterns = [
            r"fix.*test.*(?:fail|error)",   # Fix-test cycles
            r"build.*(?:error|fail)",        # Build repair
            r"lint.*(?:fix|clean)",           # Lint cleanup
            r"type.*(?:error|check)",         # Type checking
            r"compil(?:e|ation).*(?:error|fix)",  # Compilation fixes
        ]

        for pat in high_efficiency_patterns:
            if re.search(pat, task_lower):
                return 0.9

        # Tasks where iteration has moderate efficiency
        medium_efficiency_patterns = [
            r"refactor",
            r"performance.*(?:improve|optimize)",
            r"security.*(?:fix|patch)",
            r"test.*coverage",
        ]

        for pat in medium_efficiency_patterns:
            if re.search(pat, task_lower):
                return 0.7

        # Tasks where single-shot is usually sufficient
        low_efficiency_patterns = [
            r"add.*(?:feature|endpoint|page)",
            r"create.*(?:new|component|service)",
            r"document",
            r"update.*readme",
        ]

        for pat in low_efficiency_patterns:
            if re.search(pat, task_lower):
                return 0.3

        return 0.5  # Neutral default

    def _get_db(self) -> Optional[sqlite3.Connection]:
        """Get a database connection, or None if unavailable."""
        try:
            if os.path.exists(self.db_path):
                return sqlite3.connect(self.db_path)
            return None
        except Exception as e:
            logger.warning(f"Failed to connect to DB: {e}")
            return None


# Module-level singleton
_instance: Optional[LoopIntelligence] = None


def get_loop_intelligence(db_path: Optional[str] = None) -> LoopIntelligence:
    """Get or create the singleton LoopIntelligence instance."""
    global _instance
    if _instance is None:
        _instance = LoopIntelligence(db_path)
    return _instance
