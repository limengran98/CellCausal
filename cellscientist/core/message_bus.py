# -*- coding: utf-8 -*-
"""Async message bus for Multi-Agent Orchestration.

Provides :class:`SimpleMessageBus`, a lightweight publish/subscribe bus
built on :mod:`asyncio` queues.  Agents use this bus to exchange messages
without direct coupling to one another.

Supported topics
----------------
``biology_insights``, ``code_execution``, ``evaluation``, ``modeling``,
``orchestration``
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical set of supported topics.
SUPPORTED_TOPICS: List[str] = [
    "biology_insights",
    "code_execution",
    "evaluation",
    "modeling",
    "orchestration",
]


class SimpleMessageBus:
    """Async publish/subscribe message bus backed by :class:`asyncio.Queue`.

    Each topic has one queue; multiple subscribers on the same topic each
    receive their own copy of every published message.

    Example usage::

        bus = SimpleMessageBus()
        queue = bus.subscribe("biology_insights")
        await bus.publish("biology_insights", {"hello": "world"})
        msg = await queue.get()

    Attributes:
        _queues: Mapping from topic → list of subscriber queues.
    """

    def __init__(self) -> None:
        self._queues: Dict[str, List[asyncio.Queue]] = {
            topic: [] for topic in SUPPORTED_TOPICS
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str) -> asyncio.Queue:
        """Register a new subscriber for *topic*.

        Args:
            topic: One of the supported topic names.

        Returns:
            A dedicated :class:`asyncio.Queue` that will receive every
            message published to *topic* after this call.

        Raises:
            ValueError: If *topic* is not a supported topic name.
        """
        self._validate_topic(topic)
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[topic].append(queue)
        logger.debug("[MessageBus] New subscriber on topic '%s'.", topic)
        return queue

    async def publish(self, topic: str, message: Any) -> None:
        """Publish *message* to all subscribers of *topic*.

        Args:
            topic: Destination topic.  Must be one of the supported topics.
            message: Arbitrary payload (typically a :class:`dict`).

        Raises:
            ValueError: If *topic* is not a supported topic name.
        """
        self._validate_topic(topic)
        subscribers = self._queues[topic]
        if not subscribers:
            logger.debug(
                "[MessageBus] No subscribers for topic '%s'; message dropped.", topic
            )
            return
        for queue in subscribers:
            await queue.put(message)
        logger.debug(
            "[MessageBus] Published to topic '%s' (%d subscriber(s)).",
            topic,
            len(subscribers),
        )

    def subscriber_count(self, topic: str) -> int:
        """Return the number of active subscribers for *topic*.

        Args:
            topic: A supported topic name.

        Returns:
            Number of subscribers currently registered.
        """
        self._validate_topic(topic)
        return len(self._queues[topic])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_topic(self, topic: str) -> None:
        if topic not in self._queues:
            raise ValueError(
                f"Unknown topic '{topic}'. Supported topics: {SUPPORTED_TOPICS}"
            )
