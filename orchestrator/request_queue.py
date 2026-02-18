"""
Priority Request Queue for inference requests.

Supports:
- Priority levels (higher = more important)
- FIFO within same priority
- Concurrent execution limit
- Request cancellation
- Status tracking
"""

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List
from enum import Enum


class RequestStatus(Enum):
    """Request lifecycle states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class QueuedRequest:
    """
    A queued inference request with priority ordering.

    Priority is negated for max-heap behavior (higher priority = processed first).
    Timestamp ensures FIFO ordering within same priority.
    """
    priority: int  # Negated: -10 (high) processes before -1 (low)
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    instance_id: str = field(compare=False)
    request_data: Dict[str, Any] = field(compare=False)
    callback: Optional[Callable] = field(compare=False, default=None)
    status: RequestStatus = field(compare=False, default=RequestStatus.PENDING)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    future: Optional[asyncio.Future] = field(compare=False, default=None)


class RequestQueue:
    """
    Priority queue for inference requests with:
    - Priority levels (0 = default, higher = more important)
    - FIFO within same priority
    - Concurrent execution limit
    - Cancellation support
    - Status tracking
    """

    # Auto-cleanup thresholds
    _COMPLETED_CLEANUP_THRESHOLD = 100  # Cleanup when this many completed
    _COMPLETED_MAX_AGE_SECONDS = 300  # Remove completed requests older than 5 min

    def __init__(self, max_concurrent: int = 4):
        """
        Initialize request queue.

        Args:
            max_concurrent: Maximum concurrent processing requests
        """
        self.max_concurrent = max_concurrent
        self._queue: List[QueuedRequest] = []
        self._processing: Dict[str, QueuedRequest] = {}
        self._completed: Dict[str, QueuedRequest] = {}
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._capacity_available = asyncio.Event()
        self._capacity_available.set()
        self._shutdown = False
        self._processor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the queue processor."""
        self._shutdown = False
        self._processor_task = asyncio.create_task(self._processor_loop())

    async def stop(self):
        """Stop the queue processor gracefully."""
        self._shutdown = True
        self._not_empty.set()
        self._capacity_available.set()

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def enqueue(
        self,
        request_id: str,
        instance_id: str,
        request_data: Dict[str, Any],
        priority: int = 0,
        callback: Optional[Callable] = None,
    ) -> asyncio.Future:
        """
        Add request to queue.

        Args:
            request_id: Unique request identifier
            instance_id: Model instance to use
            request_data: Inference parameters
            priority: Higher = more important (default 0)
            callback: Optional callback for completion

        Returns:
            Future that resolves with results
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        item = QueuedRequest(
            priority=-priority,  # Negate for max-heap behavior
            timestamp=time.time(),
            request_id=request_id,
            instance_id=instance_id,
            request_data=request_data,
            callback=callback,
            future=future,
        )

        async with self._lock:
            heapq.heappush(self._queue, item)
            self._not_empty.set()

        return future

    async def dequeue(self) -> Optional[QueuedRequest]:
        """
        Get next request if under concurrent limit.

        Returns:
            QueuedRequest or None if shutdown
        """
        while not self._shutdown:
            wait_for_capacity = False
            wait_for_items = False

            async with self._lock:
                # Check capacity
                if len(self._processing) >= self.max_concurrent:
                    self._capacity_available.clear()
                    wait_for_capacity = True
                elif self._queue:
                    item = heapq.heappop(self._queue)
                    item.status = RequestStatus.PROCESSING
                    self._processing[item.request_id] = item

                    if not self._queue:
                        self._not_empty.clear()

                    return item
                else:
                    self._not_empty.clear()
                    wait_for_items = True

            # Wait for the specific condition we need (not both)
            if wait_for_capacity:
                await self._capacity_available.wait()
            elif wait_for_items:
                await self._not_empty.wait()

        return None

    async def complete(
        self,
        request_id: str,
        result: Any = None,
        error: Optional[str] = None
    ):
        """
        Mark request as complete.

        Args:
            request_id: Request to complete
            result: Success result (list of InferenceResponse)
            error: Error message if failed
        """
        async with self._lock:
            if request_id in self._processing:
                item = self._processing.pop(request_id)
                item.status = RequestStatus.COMPLETED if not error else RequestStatus.FAILED
                item.result = result
                item.error = error
                self._completed[request_id] = item

                # Resolve future
                if item.future and not item.future.done():
                    if error:
                        item.future.set_exception(Exception(error))
                    else:
                        item.future.set_result(result)

                # Call callback
                if item.callback:
                    try:
                        item.callback(item)
                    except Exception:
                        pass

                # Signal capacity available
                self._capacity_available.set()

                # Auto-cleanup old completed requests to prevent memory growth
                if len(self._completed) >= self._COMPLETED_CLEANUP_THRESHOLD:
                    self._cleanup_completed_unlocked()

    async def cancel(self, request_id: str) -> bool:
        """
        Cancel a pending request.

        Args:
            request_id: Request to cancel

        Returns:
            True if cancelled, False if not found or already processing
        """
        async with self._lock:
            # Check if in queue (pending)
            for i, item in enumerate(self._queue):
                if item.request_id == request_id:
                    item.status = RequestStatus.CANCELLED
                    self._queue.pop(i)
                    heapq.heapify(self._queue)

                    # Resolve future with cancellation
                    if item.future and not item.future.done():
                        item.future.cancel()

                    return True
        return False

    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get status of a request."""
        if request_id in self._processing:
            return self._processing[request_id].status
        if request_id in self._completed:
            return self._completed[request_id].status
        for item in self._queue:
            if item.request_id == request_id:
                return item.status
        return None

    def get_queue_length(self) -> int:
        """Get number of pending requests."""
        return len(self._queue)

    def get_processing_count(self) -> int:
        """Get number of currently processing requests."""
        return len(self._processing)

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get info about pending requests."""
        return [
            {
                "request_id": item.request_id,
                "instance_id": item.instance_id,
                "priority": -item.priority,
                "queued_at": item.timestamp,
            }
            for item in self._queue
        ]

    def clear_completed(self, older_than_seconds: float = 3600):
        """
        Clear completed requests older than specified age.

        Args:
            older_than_seconds: Clear requests older than this
        """
        cutoff = time.time() - older_than_seconds
        to_remove = [
            rid for rid, item in self._completed.items()
            if item.timestamp < cutoff
        ]
        for rid in to_remove:
            del self._completed[rid]

    def _cleanup_completed_unlocked(self):
        """
        Auto-cleanup old completed requests. Called from complete() while lock is held.
        """
        cutoff = time.time() - self._COMPLETED_MAX_AGE_SECONDS
        to_remove = [
            rid for rid, item in self._completed.items()
            if item.timestamp < cutoff
        ]
        for rid in to_remove:
            del self._completed[rid]

    async def _processor_loop(self):
        """Background processor loop (to be connected to orchestrator)."""
        # This is a placeholder - actual processing happens in orchestrator
        while not self._shutdown:
            await asyncio.sleep(0.1)
