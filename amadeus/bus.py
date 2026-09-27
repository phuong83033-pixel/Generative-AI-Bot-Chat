import asyncio
import threading
import logging
from collections import defaultdict
from typing import Callable, Any, Dict, List

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

    def subscribe(self, event_name: str, callback: Callable) -> None:
        with self._lock:
            self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        with self._lock:
            if callback in self._subscribers[event_name]:
                self._subscribers[event_name].remove(callback)

    async def publish(self, event_name: str, data: Any = None) -> None:
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
        def _publish_wrapper():
            asyncio.ensure_future(self.publish(event_name, data))
        
        self._loop.call_soon_threadsafe(_publish_wrapper)
