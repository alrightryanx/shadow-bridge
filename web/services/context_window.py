"""
Context Window Service - Intelligent Context for AI Queries

Builds dynamic context windows based on:
1. Current activity (what project/note is open)
2. Recent interactions (what was recently accessed)
3. Semantic relevance (what's related to the query)
4. User preferences (learned patterns)

AGI-Readiness: This enables AI to have relevant context without
needing the user to explicitly provide it.
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field, asdict
import json

logger = logging.getLogger(__name__)

# Maximum items in each context category
MAX_RECENT_ITEMS = 10
MAX_CONTEXT_TOKENS = 4000  # Approximate token budget for context

# Activity tracking window
ACTIVITY_WINDOW_MINUTES = 30


@dataclass
class ActivityEvent:
    """Represents a user activity event."""
    timestamp: datetime
    event_type: str  # view, edit, search, create, delete
    resource_type: str  # project, note, automation, agent
    resource_id: str
    resource_title: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ContextWindow:
    """A compiled context window for AI queries."""
    active_context: Optional[Dict] = None  # Currently focused item
    recent_items: List[Dict] = field(default_factory=list)
    related_items: List[Dict] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_summary: str = ""
    token_estimate: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """Convert to a prompt-ready context string."""
        parts = []

        if self.active_context:
            parts.append(f"Currently working on: {self.active_context.get('title', 'Unknown')}")
            if self.active_context.get('content'):
                parts.append(f"Content preview: {self.active_context['content'][:500]}")

        if self.recent_items:
            recent_titles = [item.get('title', 'Unknown') for item in self.recent_items[:5]]
            parts.append(f"Recently accessed: {', '.join(recent_titles)}")

        if self.related_items:
            related_titles = [item.get('title', 'Unknown') for item in self.related_items[:3]]
            parts.append(f"Related context: {', '.join(related_titles)}")

        if self.session_summary:
            parts.append(f"Session summary: {self.session_summary}")

        return "\n".join(parts) if parts else ""


class ContextWindowService:
    """
    Manages context windows for AI interactions.

    Tracks user activity and builds intelligent context based on:
    - What the user is currently looking at
    - What they've recently accessed
    - What's semantically related to their query
    """

    def __init__(self, data_service=None, vector_store=None):
        self.data_service = data_service
        self.vector_store = vector_store

        # Resource details cache (LRU)
        self._resource_cache: Dict[str, Dict] = {}
        self._cache_order: deque = deque(maxlen=50)

        # Activity tracking
        self._activity_history: deque = deque(maxlen=100)
        self._current_focus: Optional[ActivityEvent] = None

        # Session state
        self._session_start: datetime = datetime.now()
        self._query_count: int = 0
        self._topics_discussed: List[str] = []

    def preload_context(self, resource_type: str, resource_id: str) -> None:
        """Pre-load resource details into cache."""
        # This triggers _get_resource_details which handles caching
        self._get_resource_details(resource_type, resource_id)
        logger.debug(f"Preloaded context for {resource_type}:{resource_id}")

    def track_activity(
        self,
        event_type: str,
        resource_type: str,
        resource_id: str,
        resource_title: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Track a user activity event.

        Args:
            event_type: Type of activity (view, edit, search, etc.)
            resource_type: Type of resource (project, note, etc.)
            resource_id: ID of the resource
            resource_title: Human-readable title
            metadata: Additional context
        """
        event = ActivityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_title=resource_title,
            metadata=metadata or {}
        )

        self._activity_history.append(event)

        # Update focus if this is a view or edit event
        if event_type in ('view', 'edit', 'open'):
            self._current_focus = event

        # Broadcast via WebSocket
        try:
            from ..routes import websocket
            websocket.broadcast_activity(
                event_type, 
                resource_type, 
                resource_id, 
                resource_title, 
                metadata
            )
        except Exception as e:
            logger.debug(f"Failed to broadcast activity: {e}")

        logger.debug(f"Tracked activity: {event_type} {resource_type}:{resource_id}")

    def get_recent_activity(
        self,
        minutes: int = ACTIVITY_WINDOW_MINUTES,
        event_types: Optional[List[str]] = None,
        resource_types: Optional[List[str]] = None,
        limit: int = MAX_RECENT_ITEMS
    ) -> List[ActivityEvent]:
        """
        Get recent activity events.

        Args:
            minutes: How far back to look
            event_types: Filter by event types
            resource_types: Filter by resource types
            limit: Maximum events to return
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)

        events = []
        for event in reversed(self._activity_history):
            if event.timestamp < cutoff:
                break

            if event_types and event.event_type not in event_types:
                continue

            if resource_types and event.resource_type not in resource_types:
                continue

            events.append(event)

            if len(events) >= limit:
                break

        return events

    def build_context_window(
        self,
        query: Optional[str] = None,
        include_semantic: bool = True,
        max_tokens: int = MAX_CONTEXT_TOKENS
    ) -> ContextWindow:
        """
        Build a context window for an AI query.

        Args:
            query: The user's query (for semantic relevance)
            include_semantic: Whether to include semantically related items
            max_tokens: Approximate token budget

        Returns:
            ContextWindow with relevant context
        """
        context = ContextWindow()
        token_count = 0

        # 1. Add current focus (highest priority)
        if self._current_focus:
            focus_data = self._get_resource_details(
                self._current_focus.resource_type,
                self._current_focus.resource_id
            )
            if focus_data:
                context.active_context = focus_data
                token_count += self._estimate_tokens(json.dumps(focus_data))

        # 2. Add recent activity
        recent_events = self.get_recent_activity(limit=5)
        seen_ids = set()

        if self._current_focus:
            seen_ids.add(f"{self._current_focus.resource_type}:{self._current_focus.resource_id}")

        for event in recent_events:
            key = f"{event.resource_type}:{event.resource_id}"
            if key in seen_ids:
                continue
            seen_ids.add(key)

            item_data = self._get_resource_details(event.resource_type, event.resource_id)
            if item_data:
                item_tokens = self._estimate_tokens(json.dumps(item_data))
                if token_count + item_tokens < max_tokens * 0.6:  # Reserve space for semantic
                    context.recent_items.append(item_data)
                    token_count += item_tokens

        # 3. Add semantically related items (if query provided and vector store available)
        if query and include_semantic and self.vector_store:
            try:
                from . import vector_store as vs
                if vs.is_available():
                    semantic_results = vs.semantic_search(query, limit=5)
                    for result in semantic_results:
                        key = f"{result.get('source_type')}:{result.get('source_id')}"
                        if key in seen_ids:
                            continue
                        seen_ids.add(key)

                        item_tokens = self._estimate_tokens(json.dumps(result))
                        if token_count + item_tokens < max_tokens * 0.9:
                            context.related_items.append({
                                'type': result.get('source_type'),
                                'id': result.get('source_id'),
                                'title': result.get('title'),
                                'score': result.get('score'),
                                'preview': result.get('content', '')[:200]
                            })
                            token_count += item_tokens
            except Exception as e:
                logger.warning(f"Semantic search failed in context window: {e}")

        # 4. Build session summary
        context.session_summary = self._build_session_summary()

        # 5. Add user preferences (placeholder for Phase 2.4)
        context.user_preferences = self._get_user_preferences()

        context.token_estimate = token_count

        return context

    def _get_resource_details(self, resource_type: str, resource_id: str) -> Optional[Dict]:
        """Fetch details for a resource (cached)."""
        cache_key = f"{resource_type}:{resource_id}"
        
        # Check cache
        if cache_key in self._resource_cache:
            # Refresh LRU position
            if cache_key in self._cache_order:
                self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._resource_cache[cache_key]

        if not self.data_service:
            return None

        result = None
        try:
            if resource_type == 'project':
                projects = self.data_service.get_projects() or []
                for p in projects:
                    if str(p.get('id', '')) == str(resource_id) or p.get('path', '') == resource_id:
                        result = {
                            'type': 'project',
                            'id': p.get('id', p.get('path', '')),
                            'title': p.get('name', 'Unknown Project'),
                            'content': p.get('description', '')
                        }
                        break

            elif resource_type == 'note':
                # Try getting cached content first for speed
                cached_note = self.data_service.get_cached_note_content(resource_id)
                if cached_note:
                     result = {
                        'type': 'note',
                        'id': cached_note.get('id', ''),
                        'title': cached_note.get('title', 'Untitled Note'),
                        'content': cached_note.get('content', '')[:500]
                    }
                else:
                    notes = self.data_service.get_notes() or []
                    for n in notes:
                        if str(n.get('id', '')) == str(resource_id):
                            result = {
                                'type': 'note',
                                'id': n.get('id', ''),
                                'title': n.get('title', 'Untitled Note'),
                                'content': n.get('content', n.get('preview', ''))[:500]
                            }
                            break

            elif resource_type == 'automation':
                automations = self.data_service.get_automations() or []
                for a in automations:
                    if str(a.get('id', '')) == str(resource_id):
                        result = {
                            'type': 'automation',
                            'id': a.get('id', ''),
                            'title': a.get('name', 'Unknown Automation'),
                            'content': a.get('description', '') or a.get('prompt', '')
                        }
                        break

            elif resource_type == 'agent':
                agents = self.data_service.get_agents() or []
                for ag in agents:
                    if str(ag.get('id', '')) == str(resource_id):
                        result = {
                            'type': 'agent',
                            'id': ag.get('id', ''),
                            'title': ag.get('name', 'Unknown Agent'),
                            'content': ag.get('specialty', '') or ag.get('description', '')
                        }
                        break
        except Exception as e:
            logger.warning(f"Failed to get resource details: {e}")

        # Update cache if found
        if result:
            if len(self._cache_order) >= 50:
                oldest = self._cache_order.popleft()
                del self._resource_cache[oldest]
            
            self._resource_cache[cache_key] = result
            self._cache_order.append(cache_key)

        return result

    def _build_session_summary(self) -> str:
        """Build a summary of the current session."""
        duration = datetime.now() - self._session_start
        minutes = int(duration.total_seconds() / 60)

        # Count activity by type
        activity_counts = {}
        for event in self._activity_history:
            key = event.resource_type
            activity_counts[key] = activity_counts.get(key, 0) + 1

        parts = [f"Session duration: {minutes} minutes"]

        if activity_counts:
            activity_str = ", ".join(f"{count} {rtype}s" for rtype, count in activity_counts.items())
            parts.append(f"Activity: {activity_str}")

        if self._topics_discussed:
            parts.append(f"Topics: {', '.join(self._topics_discussed[-5:])}")

        return "; ".join(parts)

    def _get_user_preferences(self) -> Dict:
        """Get user preferences for context. (Placeholder for Phase 2.4)"""
        return {
            'preferred_verbosity': 'concise',
            'code_style': 'kotlin_android',
            'timezone': 'local'
        }

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token on average)."""
        return len(text) // 4

    def record_topic(self, topic: str) -> None:
        """Record a topic discussed in this session."""
        if topic not in self._topics_discussed:
            self._topics_discussed.append(topic)

    def get_stats(self) -> Dict:
        """Get context window service statistics."""
        return {
            'session_duration_minutes': int((datetime.now() - self._session_start).total_seconds() / 60),
            'activity_count': len(self._activity_history),
            'current_focus': self._current_focus.to_dict() if self._current_focus else None,
            'query_count': self._query_count,
            'topics_discussed': self._topics_discussed[-10:],
            'has_vector_store': self.vector_store is not None
        }

    def clear_session(self) -> None:
        """Clear session state (e.g., on logout or session end)."""
        self._activity_history.clear()
        self._current_focus = None
        self._session_start = datetime.now()
        self._query_count = 0
        self._topics_discussed.clear()
        logger.info("Context window session cleared")


# Global instance
_context_service: Optional[ContextWindowService] = None


def get_context_service(data_service=None, vector_store=None) -> ContextWindowService:
    """Get or create the global context window service."""
    global _context_service

    if _context_service is None:
        _context_service = ContextWindowService(
            data_service=data_service,
            vector_store=vector_store
        )

    return _context_service


def track_activity(
    event_type: str,
    resource_type: str,
    resource_id: str,
    resource_title: str,
    metadata: Optional[Dict] = None
) -> None:
    """Convenience function to track activity."""
    service = get_context_service()
    service.track_activity(event_type, resource_type, resource_id, resource_title, metadata)


def build_context(query: Optional[str] = None) -> ContextWindow:
    """Convenience function to build context window."""
    service = get_context_service()
    return service.build_context_window(query)
