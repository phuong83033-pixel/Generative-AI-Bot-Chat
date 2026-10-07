import asyncio
import concurrent.futures
import threading
import logging
from collections import defaultdict
from typing import Callable, Any, Dict, List, Optional

class EventBus:
    """Pub/sub between pipeline stages.

    The bus is deliberately NOT bound to an event loop at construction time.
    Capturing a loop in __init__ binds whatever loop happens to exist when the
    object is built - which, for anything constructed before asyncio.run(), is a
    loop that never runs. publish_sync() would then post callbacks into a dead
    loop and every cross-thread event would vanish silently.

    The loop is bound instead on first async use, or explicitly via bind_loop()
    from inside the running loop.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> asyncio.AbstractEventLoop:
        """Bind the bus to the running loop. Call from inside the event loop."""
        self._loop = loop or asyncio.get_running_loop()
        return self._loop

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def subscribe(self, event_name: str, callback: Callable) -> None:
        with self._lock:
            self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        with self._lock:
            if callback in self._subscribers[event_name]:
                self._subscribers[event_name].remove(callback)

    async def publish(self, event_name: str, data: Any = None) -> None:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        with self._lock:
            callbacks = list(self._subscribers[event_name])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logging.exception(f"Error in subscriber for event {event_name}: {e}")

    def publish_sync(self, event_name: str, data: Any = None) -> None:
        """Publish from a non-async context or another thread.

        Requires the bus to be bound. Raising here is deliberate: a dropped
        event is far harder to diagnose than a loud failure at startup.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError(
                f"EventBus is not bound to a running loop, so '{event_name}' cannot be "
                "delivered. Call bind_loop() from inside the event loop during startup."
            )

        future = asyncio.run_coroutine_threadsafe(self.publish(event_name, data), loop)

        def _report(f: Any) -> None:
            try:
                f.result()
            except concurrent.futures.CancelledError:
                # The loop shut down before this publish ran. Normal during
                # teardown - the last few StateChanged events are simply dropped.
                pass
            except Exception as e:
                logging.exception(f"Error publishing event {event_name}: {e}")

        future.add_done_callback(_report)

__all__ = ["EventBus"]
