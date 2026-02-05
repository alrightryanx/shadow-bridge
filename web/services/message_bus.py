"""
Agent Message Bus - AGI-Readiness Infrastructure

Enables agent-to-agent and orchestrator-to-agent communication:
1. Message routing (point-to-point and broadcast)
2. Message persistence for audit trail
3. Pub/sub patterns for event-driven coordination
4. Priority queues for urgent messages
5. Dead letter handling for failed deliveries
6. Distributed SQLite MQ for cross-machine messaging

This is the nervous system of the multi-agent system.
"""
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock, Thread
from queue import PriorityQueue, Empty
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from uuid import uuid4
import os

from .agent_protocol import (
    AgentId, AgentMessage, MessageType, AgentState, AgentStateType
)

logger = logging.getLogger(__name__)

# Persistence
MESSAGE_LOG_FILE = Path.home() / ".shadowai" / "message_log.jsonl"
MAX_MESSAGE_LOG_SIZE = 10000  # Max messages to keep in log

# Queue settings
MAX_QUEUE_SIZE = 1000
MESSAGE_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5


@dataclass(order=True)
class PrioritizedMessage:
    """Message wrapper with priority for queue ordering."""
    priority: int
    timestamp: float = field(compare=False)
    message: AgentMessage = field(compare=False)


class MessageHandler:
    """Wrapper for message handler functions."""

    def __init__(
        self,
        handler: Callable[[AgentMessage], Optional[AgentMessage]],
        message_types: Optional[Set[MessageType]] = None
    ):
        self.handler = handler
        self.message_types = message_types  # None = handle all types
        self.call_count = 0
        self.last_called = None

    def can_handle(self, message_type: MessageType) -> bool:
        if self.message_types is None:
            return True
        return message_type in self.message_types

    def __call__(self, message: AgentMessage) -> Optional[AgentMessage]:
        self.call_count += 1
        self.last_called = datetime.now()
        return self.handler(message)


class MessageBus:
    """
    Central message bus for agent communication.

    Provides:
    - Point-to-point messaging between agents
    - Broadcast messaging to all agents
    - Topic-based pub/sub
    - Message persistence for audit
    - Priority-based delivery
    - Dead letter queue for failed messages
    """

    def __init__(self):
        self._lock = Lock()

        # Message queues per agent
        self._queues: Dict[str, PriorityQueue] = defaultdict(PriorityQueue)

        # Broadcast queue
        self._broadcast_queue: deque = deque(maxlen=100)

        # Topic subscriptions: topic -> set of agent IDs
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # Message handlers: agent_id -> list of handlers
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)

        # Global handlers (receive all messages)
        self._global_handlers: List[MessageHandler] = []

        # Message log for persistence
        self._message_log: deque = deque(maxlen=MAX_MESSAGE_LOG_SIZE)

        # Dead letter queue
        self._dead_letters: deque = deque(maxlen=500)

        # Metrics
        self._metrics = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'broadcasts': 0
        }

        # Load message history
        self._load_message_log()

    def _load_message_log(self) -> None:
        """Load recent messages from disk."""
        try:
            if MESSAGE_LOG_FILE.exists():
                with open(MESSAGE_LOG_FILE, 'r', encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            msg = AgentMessage.from_dict(data)
                            self._message_log.append(msg)
                        except (json.JSONDecodeError, KeyError, ValueError, AttributeError, TypeError) as e:
                            logger.debug(f"Skipping malformed message: {e}")
                            continue
                logger.info(f"Loaded {len(self._message_log)} messages from log")
        except Exception as e:
            logger.warning(f"Failed to load message log: {e}")

    def _persist_message(self, message: AgentMessage) -> None:
        """Persist a message to disk."""
        try:
            MESSAGE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(MESSAGE_LOG_FILE, 'a', encoding="utf-8") as f:
                f.write(json.dumps(message.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to persist message: {e}")

    def send(
        self,
        message: AgentMessage,
        persist: bool = True
    ) -> bool:
        """
        Send a message to a specific agent.

        Args:
            message: The message to send
            persist: Whether to persist for audit

        Returns:
            True if message was queued successfully
        """
        with self._lock:
            if not message.to_agent:
                logger.error("Cannot send point-to-point message without recipient")
                return False

            recipient_key = str(message.to_agent)

            # Check queue size
            if self._queues[recipient_key].qsize() >= MAX_QUEUE_SIZE:
                logger.warning(f"Queue full for agent {recipient_key}")
                self._dead_letters.append(message)
                self._metrics['messages_failed'] += 1
                return False

            # Queue the message
            prioritized = PrioritizedMessage(
                priority=message.priority,
                timestamp=time.time(),
                message=message
            )
            self._queues[recipient_key].put(prioritized)

            # Log and persist
            self._message_log.append(message)
            if persist:
                self._persist_message(message)

            self._metrics['messages_sent'] += 1
            logger.debug(f"Queued message {message.id} for {recipient_key}")

            # Trigger handlers
            self._invoke_handlers(message)

            return True

    def broadcast(
        self,
        message: AgentMessage,
        persist: bool = True
    ) -> int:
        """
        Broadcast a message to all agents.

        Args:
            message: The message to broadcast (to_agent should be None)
            persist: Whether to persist for audit

        Returns:
            Number of agents message was queued for
        """
        with self._lock:
            # Add to broadcast queue for late joiners
            self._broadcast_queue.append(message)

            # Queue for all known agents
            delivered = 0
            for agent_key in list(self._queues.keys()):
                prioritized = PrioritizedMessage(
                    priority=message.priority,
                    timestamp=time.time(),
                    message=message
                )
                self._queues[agent_key].put(prioritized)
                delivered += 1

            # Log and persist
            self._message_log.append(message)
            if persist:
                self._persist_message(message)

            self._metrics['broadcasts'] += 1
            self._metrics['messages_sent'] += delivered

            # Trigger global handlers
            self._invoke_handlers(message, broadcast=True)

            return delivered

    def publish(
        self,
        topic: str,
        message: AgentMessage,
        persist: bool = True
    ) -> int:
        """
        Publish a message to a topic.

        Args:
            topic: Topic name
            message: The message to publish
            persist: Whether to persist

        Returns:
            Number of subscribers message was sent to
        """
        with self._lock:
            subscribers = self._subscriptions.get(topic, set())

            delivered = 0
            for agent_key in subscribers:
                prioritized = PrioritizedMessage(
                    priority=message.priority,
                    timestamp=time.time(),
                    message=message
                )
                self._queues[agent_key].put(prioritized)
                delivered += 1

            if persist:
                self._persist_message(message)

            self._metrics['messages_sent'] += delivered
            return delivered

    def subscribe(self, agent_id: AgentId, topic: str) -> None:
        """Subscribe an agent to a topic."""
        with self._lock:
            self._subscriptions[topic].add(str(agent_id))
            logger.debug(f"Agent {agent_id} subscribed to {topic}")

    def unsubscribe(self, agent_id: AgentId, topic: str) -> None:
        """Unsubscribe an agent from a topic."""
        with self._lock:
            self._subscriptions[topic].discard(str(agent_id))

    def receive(
        self,
        agent_id: AgentId,
        timeout: float = 0.0,
        message_types: Optional[Set[MessageType]] = None
    ) -> Optional[AgentMessage]:
        """
        Receive a message for an agent.

        Args:
            agent_id: The receiving agent
            timeout: How long to wait (0 = non-blocking)
            message_types: Optional filter by message types

        Returns:
            Next message or None if queue empty/timeout
        """
        agent_key = str(agent_id)

        try:
            if timeout > 0:
                prioritized = self._queues[agent_key].get(timeout=timeout)
            else:
                prioritized = self._queues[agent_key].get_nowait()

            message = prioritized.message

            # Filter by type if specified
            if message_types and message.message_type not in message_types:
                # Put it back (at original priority)
                self._queues[agent_key].put(prioritized)
                return None

            # Check if expired
            if message.is_expired():
                self._dead_letters.append(message)
                self._metrics['messages_failed'] += 1
                return None

            self._metrics['messages_delivered'] += 1
            return message

        except Empty:
            return None

    def receive_all(
        self,
        agent_id: AgentId,
        max_messages: int = 100
    ) -> List[AgentMessage]:
        """Receive all pending messages for an agent."""
        messages = []
        while len(messages) < max_messages:
            msg = self.receive(agent_id)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def register_handler(
        self,
        agent_id: AgentId,
        handler: Callable[[AgentMessage], Optional[AgentMessage]],
        message_types: Optional[Set[MessageType]] = None
    ) -> None:
        """
        Register a message handler for an agent.

        Args:
            agent_id: The agent to handle messages for
            handler: Function that processes messages
            message_types: Optional filter - only handle these types
        """
        with self._lock:
            agent_key = str(agent_id)
            wrapped = MessageHandler(handler, message_types)
            self._handlers[agent_key].append(wrapped)

    def register_global_handler(
        self,
        handler: Callable[[AgentMessage], Optional[AgentMessage]],
        message_types: Optional[Set[MessageType]] = None
    ) -> None:
        """Register a handler that receives ALL messages."""
        with self._lock:
            wrapped = MessageHandler(handler, message_types)
            self._global_handlers.append(wrapped)

    def _invoke_handlers(
        self,
        message: AgentMessage,
        broadcast: bool = False
    ) -> None:
        """Invoke registered handlers for a message."""
        # Global handlers
        for handler in self._global_handlers:
            if handler.can_handle(message.message_type):
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Global handler error: {e}")

        # Agent-specific handlers
        if message.to_agent:
            agent_key = str(message.to_agent)
            for handler in self._handlers.get(agent_key, []):
                if handler.can_handle(message.message_type):
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for {agent_key}: {e}")

    def get_queue_size(self, agent_id: AgentId) -> int:
        """Get the number of pending messages for an agent."""
        return self._queues[str(agent_id)].qsize()

    def get_dead_letters(self, limit: int = 50) -> List[AgentMessage]:
        """Get recent dead letters."""
        return list(self._dead_letters)[-limit:]

    def get_message_history(
        self,
        agent_id: Optional[AgentId] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get message history with optional filters."""
        messages = list(self._message_log)

        if agent_id:
            agent_key = str(agent_id)
            messages = [
                m for m in messages
                if str(m.from_agent) == agent_key or
                   (m.to_agent and str(m.to_agent) == agent_key)
            ]

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        return messages[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        with self._lock:
            queue_sizes = {k: q.qsize() for k, q in self._queues.items()}
            total_queued = sum(queue_sizes.values())

            return {
                **self._metrics,
                'total_queued': total_queued,
                'queue_sizes': queue_sizes,
                'dead_letter_count': len(self._dead_letters),
                'message_log_size': len(self._message_log),
                'subscriptions': {
                    topic: len(subs)
                    for topic, subs in self._subscriptions.items()
                },
                'handlers_registered': sum(
                    len(h) for h in self._handlers.values()
                ) + len(self._global_handlers)
            }

    def clear_queue(self, agent_id: AgentId) -> int:
        """Clear all pending messages for an agent."""
        with self._lock:
            agent_key = str(agent_id)
            count = self._queues[agent_key].qsize()

            # Create new empty queue
            self._queues[agent_key] = PriorityQueue()

            return count

    def create_reply(
        self,
        original: AgentMessage,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> AgentMessage:
        """Create a reply to a message."""
        return AgentMessage.create(
            message_type=message_type,
            from_agent=original.to_agent,  # Swap sender/receiver
            to_agent=original.from_agent,
            payload=payload,
            reply_to=original.id
        )


# ---- Distributed Message Bus (SQLite MQ Layer) ----

DISTRIBUTED_MODE = os.environ.get("MESSAGE_BUS_DISTRIBUTED", "0") == "1"
RELAY_POLL_INTERVAL = float(os.environ.get("RELAY_POLL_INTERVAL", "2.0"))


class DistributedMessageBus:
    """
    Extends MessageBus with SQLite-backed distributed messaging.

    When DISTRIBUTED_MODE is enabled:
    - Messages to remote agents are persisted to SQLite message_queue
    - A relay thread polls SQLite and forwards messages to remote workers
    - Workers poll their own channels via the coordinator API
    - Broadcasts are written to a '__broadcast__' channel for all workers

    This is a polling-based MQ (not real-time push), suitable for
    100-500 agents across a few machines. For 500+ agents, replace
    with Redis Pub/Sub + Streams.
    """

    def __init__(self, local_bus: 'MessageBus'):
        self._local_bus = local_bus
        self._lock = Lock()
        self._running = False
        self._relay_thread: Optional[Thread] = None
        self._remote_channels: Set[str] = set()  # agent IDs known to be remote
        self._local_machine_id = os.environ.get("MACHINE_ID", "local")

    def start(self):
        """Start the relay polling thread."""
        if self._running:
            return
        self._running = True
        self._relay_thread = Thread(
            target=self._relay_loop, daemon=True, name="mq-relay"
        )
        self._relay_thread.start()
        logger.info("DistributedMessageBus relay started")

    def stop(self):
        """Stop the relay thread."""
        self._running = False
        if self._relay_thread:
            self._relay_thread.join(timeout=5)

    def register_remote_agent(self, agent_id: str):
        """Mark an agent ID as being on a remote worker."""
        with self._lock:
            self._remote_channels.add(agent_id)

    def unregister_remote_agent(self, agent_id: str):
        """Remove agent from remote tracking."""
        with self._lock:
            self._remote_channels.discard(agent_id)

    def is_remote(self, agent_id: str) -> bool:
        """Check if an agent is on a remote worker."""
        with self._lock:
            return agent_id in self._remote_channels

    def send_distributed(self, channel: str, message_type: str,
                         sender: str, payload: dict, priority: int = 3,
                         ttl_seconds: int = 300) -> bool:
        """
        Send a message via the distributed SQLite MQ.

        Used for cross-machine messaging when the target agent
        is not on the local machine.
        """
        try:
            from web.services.state_store import get_state_store
            store = get_state_store()
            message_id = uuid4().hex
            store.enqueue_message(
                message_id=message_id,
                channel=channel,
                message_type=message_type,
                sender=sender,
                payload=payload,
                priority=priority,
                ttl_seconds=ttl_seconds,
            )
            logger.debug(f"Distributed message {message_id} queued for {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue distributed message: {e}")
            return False

    def broadcast_distributed(self, sender: str, payload: dict,
                              message_type: str = "broadcast",
                              priority: int = 3) -> bool:
        """Broadcast a message to all remote workers via __broadcast__ channel."""
        return self.send_distributed(
            channel="__broadcast__",
            message_type=message_type,
            sender=sender,
            payload=payload,
            priority=priority,
            ttl_seconds=60,
        )

    def poll_messages(self, channel: str, limit: int = 50) -> List[Dict]:
        """
        Poll for pending messages on a channel.

        Called by remote workers to fetch their messages.
        """
        try:
            from web.services.state_store import get_state_store
            store = get_state_store()
            return store.dequeue_messages(channel, limit=limit)
        except Exception as e:
            logger.error(f"Failed to poll messages for {channel}: {e}")
            return []

    def get_pending_count(self, channel: str = None) -> int:
        """Get count of pending messages."""
        try:
            from web.services.state_store import get_state_store
            store = get_state_store()
            return store.get_pending_message_count(channel)
        except Exception:
            return 0

    def get_stats(self) -> Dict:
        """Get distributed MQ statistics."""
        try:
            from web.services.state_store import get_state_store
            store = get_state_store()
            return {
                "distributed_mode": DISTRIBUTED_MODE,
                "relay_running": self._running,
                "remote_agents": len(self._remote_channels),
                "pending_messages": store.get_pending_message_count(),
                "local_machine_id": self._local_machine_id,
            }
        except Exception:
            return {
                "distributed_mode": DISTRIBUTED_MODE,
                "relay_running": self._running,
                "remote_agents": len(self._remote_channels),
                "pending_messages": 0,
                "local_machine_id": self._local_machine_id,
            }

    def _relay_loop(self):
        """
        Relay thread: polls SQLite for messages addressed to local agents
        and injects them into the in-memory MessageBus.
        """
        cleanup_counter = 0
        while self._running:
            try:
                self._relay_local_messages()

                # Periodic cleanup of old delivered/expired messages
                cleanup_counter += 1
                if cleanup_counter >= 1800:  # ~every hour at 2s interval
                    self._cleanup()
                    cleanup_counter = 0

            except Exception as e:
                logger.error(f"Relay loop error: {e}")

            time.sleep(RELAY_POLL_INTERVAL)

    def _relay_local_messages(self):
        """
        Check for messages addressed to local agents and deliver them
        via the in-memory bus.
        """
        from web.services.state_store import get_state_store
        store = get_state_store()

        # Poll messages for the local machine channel
        messages = store.dequeue_messages(f"machine:{self._local_machine_id}", limit=100)

        # Also poll broadcast channel
        broadcasts = store.dequeue_messages("__broadcast__", limit=50)

        for msg in messages + broadcasts:
            try:
                target = msg.get("channel", "")
                payload = msg.get("payload", {})
                msg_type = msg.get("message_type", "data")

                # Try to deliver to local in-memory bus
                if target.startswith("machine:"):
                    # Machine-level message, deliver to all local agents
                    for agent_key in list(self._local_bus._queues.keys()):
                        agent_msg = AgentMessage.create(
                            message_type=MessageType.DATA,
                            from_agent=msg.get("sender", "coordinator"),
                            to_agent=agent_key,
                            payload=payload,
                        )
                        self._local_bus.send(agent_msg, persist=False)
                else:
                    # Agent-specific message
                    agent_msg = AgentMessage.create(
                        message_type=MessageType.DATA,
                        from_agent=msg.get("sender", "coordinator"),
                        to_agent=target,
                        payload=payload,
                    )
                    self._local_bus.send(agent_msg, persist=False)

            except Exception as e:
                logger.error(f"Failed to relay message: {e}")

    def _cleanup(self):
        """Clean up old delivered/expired messages."""
        try:
            from web.services.state_store import get_state_store
            store = get_state_store()
            removed = store.cleanup_delivered_messages(older_than_hours=24)
            if removed > 0:
                logger.info(f"Cleaned up {removed} old distributed messages")
        except Exception as e:
            logger.warning(f"MQ cleanup failed: {e}")


# Global instances
_message_bus: Optional[MessageBus] = None
_distributed_bus: Optional[DistributedMessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create the global message bus."""
    global _message_bus

    if _message_bus is None:
        _message_bus = MessageBus()

    return _message_bus


def get_distributed_bus() -> DistributedMessageBus:
    """Get or create the distributed message bus layer."""
    global _distributed_bus

    if _distributed_bus is None:
        bus = get_message_bus()
        _distributed_bus = DistributedMessageBus(bus)
        if DISTRIBUTED_MODE:
            _distributed_bus.start()

    return _distributed_bus


# Convenience functions

def send_message(
    from_agent: AgentId,
    to_agent: AgentId,
    message_type: MessageType,
    payload: Dict[str, Any],
    priority: int = 5
) -> bool:
    """Send a message between agents."""
    message = AgentMessage.create(
        message_type=message_type,
        from_agent=from_agent,
        to_agent=to_agent,
        payload=payload,
        priority=priority
    )
    return get_message_bus().send(message)


def broadcast_message(
    from_agent: AgentId,
    message_type: MessageType,
    payload: Dict[str, Any]
) -> int:
    """Broadcast a message to all agents."""
    message = AgentMessage.create(
        message_type=message_type,
        from_agent=from_agent,
        to_agent=None,
        payload=payload
    )
    return get_message_bus().broadcast(message)
