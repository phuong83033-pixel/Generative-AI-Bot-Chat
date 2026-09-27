from enum import Enum, auto
import threading
from .bus import EventBus
import logging

try:
    from amadeus.trace import log_transition
except ImportError:
    def log_transition(old, new):
        logging.info(f"State transition: {old} -> {new}")

class SystemState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()

class StateManager:
    def __init__(self, bus: EventBus):
        self._bus = bus
        self._current = SystemState.IDLE
        self._lock = threading.Lock()
        
        self._valid_transitions = {
            SystemState.IDLE: {SystemState.LISTENING},
            SystemState.LISTENING: {SystemState.THINKING, SystemState.IDLE},
            SystemState.THINKING: {SystemState.SPEAKING, SystemState.LISTENING},
            SystemState.SPEAKING: {SystemState.IDLE, SystemState.LISTENING},
        }

    @property
    def current(self) -> SystemState:
        with self._lock:
            return self._current

    def transition(self, new_state: SystemState) -> None:
        with self._lock:
            if new_state != SystemState.IDLE and new_state not in self._valid_transitions[self._current]:
                raise ValueError(f"Invalid state transition from {self._current.name} to {new_state.name}")
            
            old_state = self._current
            self._current = new_state
            
            log_transition(old_state.name, new_state.name)
            
        self._bus.publish_sync("StateChanged", {"old": old_state, "new": new_state})
