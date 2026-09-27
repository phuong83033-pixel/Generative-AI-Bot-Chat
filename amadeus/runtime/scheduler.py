import asyncio
from typing import Callable, Awaitable
import logging
from ..bus import EventBus

class Scheduler:
    def __init__(self, bus: EventBus):
        self._bus = bus
        self._stop_event = asyncio.Event()

    def create_stop(self) -> asyncio.Event:
        self._stop_event = asyncio.Event()
        return self._stop_event

    def cancel_current(self) -> None:
        self._stop_event.set()

    async def run_turn(self, 
                       listen_fn: Callable[[asyncio.Event], Awaitable[None]], 
                       think_fn: Callable[[asyncio.Event], Awaitable[None]], 
                       speak_fn: Callable[[asyncio.Event], Awaitable[None]]) -> None:
        
        stop_event = self.create_stop()
        
        try:
            await listen_fn(stop_event)
            if stop_event.is_set():
                return
                
            await think_fn(stop_event)
            if stop_event.is_set():
                return
                
            await speak_fn(stop_event)
            
        except asyncio.CancelledError:
            self.cancel_current()
            raise
        except Exception as e:
            logging.exception(f"Error in turn orchestration: {e}")
