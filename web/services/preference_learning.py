"""
User Preference Learning Service - AGI-Readiness

Learns user preferences from behavior patterns:
1. Interaction patterns (what/when/how they use features)
2. Content preferences (types of projects, note styles)
3. Response preferences (verbosity, format, tone)
4. Workflow patterns (common sequences of actions)

AGI-Readiness: When AI becomes more capable, it should already
know how the user prefers to interact.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Storage location
PREFERENCES_FILE = Path.home() / ".shadowai" / "user_preferences.json"

# Learning thresholds
MIN_SAMPLES_FOR_PREFERENCE = 3  # Need at least N samples to learn a pattern
PREFERENCE_DECAY_DAYS = 30  # Preferences decay if not reinforced


@dataclass
class PreferenceScore:
    """A single preference with confidence."""
    value: str
    score: float  # 0.0 to 1.0
    sample_count: int
    last_updated: datetime

    def to_dict(self) -> Dict:
        return {
            'value': self.value,
            'score': self.score,
            'sample_count': self.sample_count,
            'last_updated': self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PreferenceScore':
        return cls(
            value=data['value'],
            score=data['score'],
            sample_count=data['sample_count'],
            last_updated=datetime.fromisoformat(data['last_updated'])
        )


@dataclass
class UserPreferences:
    """Collection of learned user preferences."""

    # Interaction preferences
    preferred_search_mode: Optional[str] = None  # keyword, semantic, hybrid
    preferred_view_mode: Optional[str] = None  # list, grid, compact
    preferred_theme: Optional[str] = None  # light, dark, auto

    # Content preferences
    favorite_categories: List[str] = field(default_factory=list)
    common_tags: List[str] = field(default_factory=list)
    active_hours: List[int] = field(default_factory=list)  # Hours of day (0-23)

    # Response preferences
    verbosity_preference: Optional[str] = None  # concise, balanced, detailed
    code_style: Optional[str] = None  # language/framework preferences
    format_preference: Optional[str] = None  # markdown, plain, structured

    # Workflow patterns
    common_sequences: List[List[str]] = field(default_factory=list)
    frequent_actions: Dict[str, int] = field(default_factory=dict)

    # Detailed scores for each preference
    scores: Dict[str, List[PreferenceScore]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert PreferenceScore lists to dicts
        result['scores'] = {
            k: [s.to_dict() for s in v] if v else []
            for k, v in self.scores.items()
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserPreferences':
        # Convert scores back to PreferenceScore objects
        scores = {}
        if 'scores' in data:
            for k, v in data['scores'].items():
                scores[k] = [PreferenceScore.from_dict(s) for s in v] if v else []
            data['scores'] = scores
        return cls(**data)


class PreferenceLearningService:
    """
    Learns and applies user preferences.

    Observes user behavior and builds a preference model that can be
    used to personalize AI interactions and UI.
    """

    def __init__(self):
        self.preferences = UserPreferences()
        self._action_history: List[Tuple[datetime, str, str]] = []  # (time, action, resource)
        self._load_preferences()

    def _load_preferences(self) -> None:
        """Load preferences from disk."""
        try:
            if PREFERENCES_FILE.exists():
                with open(PREFERENCES_FILE, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    self.preferences = UserPreferences.from_dict(data)
                logger.info("Loaded user preferences")
        except Exception as e:
            logger.warning(f"Failed to load preferences: {e}")
            self.preferences = UserPreferences()

    def _save_preferences(self) -> None:
        """Save preferences to disk."""
        try:
            PREFERENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PREFERENCES_FILE, 'w', encoding="utf-8") as f:
                json.dump(self.preferences.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

    def observe_action(
        self,
        action: str,
        resource_type: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Observe a user action to learn preferences.

        Args:
            action: Type of action (view, edit, search, create, etc.)
            resource_type: Type of resource acted on
            metadata: Additional context
        """
        now = datetime.now()
        metadata = metadata or {}

        # Track action frequency
        action_key = f"{action}:{resource_type}"
        self.preferences.frequent_actions[action_key] = \
            self.preferences.frequent_actions.get(action_key, 0) + 1

        # Track active hours
        hour = now.hour
        if hour not in self.preferences.active_hours:
            self.preferences.active_hours.append(hour)
            self.preferences.active_hours = sorted(self.preferences.active_hours)

        # Add to action history for sequence learning
        self._action_history.append((now, action, resource_type))

        # Keep only last 100 actions
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-100:]

        # Learn specific preferences from metadata
        if 'search_mode' in metadata:
            self._update_preference('search_mode', metadata['search_mode'])

        if 'category' in metadata:
            if metadata['category'] not in self.preferences.favorite_categories:
                self.preferences.favorite_categories.append(metadata['category'])

        if 'tags' in metadata:
            for tag in metadata['tags']:
                if tag not in self.preferences.common_tags:
                    self.preferences.common_tags.append(tag)

        # Learn workflow patterns every 10 actions
        if len(self._action_history) % 10 == 0:
            self._learn_sequences()

        # Save periodically
        if len(self._action_history) % 5 == 0:
            self._save_preferences()

        logger.debug(f"Observed action: {action_key}")

    def _update_preference(
        self,
        category: str,
        value: str,
        weight: float = 1.0
    ) -> None:
        """Update a preference score."""
        if category not in self.preferences.scores:
            self.preferences.scores[category] = []

        scores = self.preferences.scores[category]

        # Find existing score for this value
        existing = None
        for s in scores:
            if s.value == value:
                existing = s
                break

        if existing:
            # Update existing score
            existing.sample_count += 1
            # Weighted moving average
            existing.score = (existing.score * 0.7) + (weight * 0.3)
            existing.last_updated = datetime.now()
        else:
            # New preference
            scores.append(PreferenceScore(
                value=value,
                score=weight * 0.5,  # Start at half confidence
                sample_count=1,
                last_updated=datetime.now()
            ))

        # Apply the top preference if confident enough
        top_pref = self._get_top_preference(category)
        if top_pref:
            if category == 'search_mode':
                self.preferences.preferred_search_mode = top_pref
            elif category == 'theme':
                self.preferences.preferred_theme = top_pref
            elif category == 'verbosity':
                self.preferences.verbosity_preference = top_pref

    def _get_top_preference(self, category: str) -> Optional[str]:
        """Get the top preference for a category if confident enough."""
        if category not in self.preferences.scores:
            return None

        scores = self.preferences.scores[category]
        if not scores:
            return None

        # Decay old scores
        now = datetime.now()
        valid_scores = []
        for s in scores:
            age_days = (now - s.last_updated).days
            if age_days < PREFERENCE_DECAY_DAYS:
                decay = 1.0 - (age_days / PREFERENCE_DECAY_DAYS)
                valid_scores.append((s.value, s.score * decay, s.sample_count))

        if not valid_scores:
            return None

        # Need minimum samples
        valid_scores = [s for s in valid_scores if s[2] >= MIN_SAMPLES_FOR_PREFERENCE]
        if not valid_scores:
            return None

        # Return highest scoring
        valid_scores.sort(key=lambda x: x[1], reverse=True)
        return valid_scores[0][0] if valid_scores[0][1] > 0.3 else None

    def _learn_sequences(self) -> None:
        """Learn common action sequences."""
        if len(self._action_history) < 5:
            return

        # Look for repeated 2-3 action sequences
        sequence_counts: Dict[str, int] = defaultdict(int)

        for i in range(len(self._action_history) - 2):
            # 2-action sequence
            seq2 = f"{self._action_history[i][1]}:{self._action_history[i][2]}" + \
                   f" -> {self._action_history[i+1][1]}:{self._action_history[i+1][2]}"
            sequence_counts[seq2] += 1

            # 3-action sequence
            if i < len(self._action_history) - 3:
                seq3 = seq2 + f" -> {self._action_history[i+2][1]}:{self._action_history[i+2][2]}"
                sequence_counts[seq3] += 1

        # Keep top sequences
        top_sequences = sorted(
            sequence_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        self.preferences.common_sequences = [
            seq.split(' -> ') for seq, count in top_sequences if count >= 2
        ]

    def learn_from_feedback(
        self,
        category: str,
        value: str,
        is_positive: bool
    ) -> None:
        """
        Learn from explicit user feedback.

        Args:
            category: Preference category
            value: The preference value
            is_positive: Whether user liked it (True) or not (False)
        """
        weight = 1.0 if is_positive else -0.5
        self._update_preference(category, value, weight)
        self._save_preferences()
        logger.info(f"Learned from feedback: {category}={value} ({'positive' if is_positive else 'negative'})")

    def predict_next_action(
        self,
        current_action: str,
        current_resource: str
    ) -> Optional[Tuple[str, str, float]]:
        """
        Predict the next likely action based on learned sequences.

        Returns:
            Tuple of (action, resource_type, confidence) or None
        """
        current_key = f"{current_action}:{current_resource}"

        # Simple 1-step lookahead from learned sequences
        # common_sequences is a list of lists: [["view:project", "edit:note"], ...]
        candidates = []
        for seq in self.preferences.common_sequences:
            for i, step in enumerate(seq):
                if step == current_key and i < len(seq) - 1:
                    candidates.append(seq[i + 1])

        if not candidates:
            return None

        # Find most frequent next step
        from collections import Counter
        counts = Counter(candidates)
        most_common, count = counts.most_common(1)[0]
        
        confidence = count / len(candidates) if candidates else 0.0
        
        # Parse "action:resource" back to components
        if ":" in most_common:
            action, resource = most_common.split(":", 1)
            return (action, resource, confidence)
            
        return None

    def get_preferences(self) -> UserPreferences:
        """Get current user preferences."""
        return self.preferences

    def get_preference(self, category: str) -> Optional[str]:
        """Get a specific preference by category."""
        if category == 'search_mode':
            return self.preferences.preferred_search_mode
        elif category == 'theme':
            return self.preferences.preferred_theme
        elif category == 'verbosity':
            return self.preferences.verbosity_preference
        elif category == 'view_mode':
            return self.preferences.preferred_view_mode
        return None

    def get_context_for_ai(self) -> Dict[str, Any]:
        """
        Get preferences formatted for AI context injection.

        Returns a dict that can be added to AI prompts to personalize responses.
        """
        prefs = self.preferences

        context = {
            'user_preferences': {}
        }

        if prefs.verbosity_preference:
            context['user_preferences']['verbosity'] = prefs.verbosity_preference

        if prefs.code_style:
            context['user_preferences']['code_style'] = prefs.code_style

        if prefs.format_preference:
            context['user_preferences']['format'] = prefs.format_preference

        if prefs.active_hours:
            # Determine if user is morning/afternoon/evening person
            morning = sum(1 for h in prefs.active_hours if 5 <= h < 12)
            afternoon = sum(1 for h in prefs.active_hours if 12 <= h < 18)
            evening = sum(1 for h in prefs.active_hours if 18 <= h < 24 or 0 <= h < 5)

            if morning > afternoon and morning > evening:
                context['user_preferences']['active_time'] = 'morning'
            elif afternoon > evening:
                context['user_preferences']['active_time'] = 'afternoon'
            else:
                context['user_preferences']['active_time'] = 'evening'

        if prefs.favorite_categories:
            context['user_preferences']['interests'] = prefs.favorite_categories[:5]

        if prefs.common_tags:
            context['user_preferences']['common_topics'] = prefs.common_tags[:10]

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        prefs = self.preferences

        return {
            'total_actions_observed': sum(prefs.frequent_actions.values()),
            'action_types': len(prefs.frequent_actions),
            'active_hours': prefs.active_hours,
            'favorite_categories': prefs.favorite_categories[:5],
            'common_tags_count': len(prefs.common_tags),
            'learned_sequences': len(prefs.common_sequences),
            'preference_categories': list(prefs.scores.keys()),
            'top_preferences': {
                'search_mode': prefs.preferred_search_mode,
                'theme': prefs.preferred_theme,
                'verbosity': prefs.verbosity_preference
            }
        }

    def reset(self) -> None:
        """Reset all learned preferences."""
        self.preferences = UserPreferences()
        self._action_history.clear()
        self._save_preferences()
        logger.info("User preferences reset")


# Global instance
_preference_service: Optional[PreferenceLearningService] = None


def get_preference_service() -> PreferenceLearningService:
    """Get or create the global preference learning service."""
    global _preference_service

    if _preference_service is None:
        _preference_service = PreferenceLearningService()

    return _preference_service


def observe(action: str, resource_type: str, metadata: Optional[Dict] = None) -> None:
    """Convenience function to observe an action."""
    get_preference_service().observe_action(action, resource_type, metadata)


def get_preferences() -> UserPreferences:
    """Convenience function to get preferences."""
    return get_preference_service().get_preferences()
