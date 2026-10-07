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
        
        # Mirrors the state machine in README section 6.
        #
        # IDLE -> THINKING is the edge that separates this from a voice
        # assistant: it is the only transition not caused by the user. It fires
        # when drive.py crosses its threshold (README section 19), and is also
        # the path taken by --text mode, which has no LISTENING phase.
        #
        # THINKING -> IDLE exists so an empty transcript or a failed turn can
        # unwind without forcing a fake SPEAKING state.
        self._valid_transitions = {
            SystemState.IDLE: {SystemState.LISTENING, SystemState.THINKING},
            SystemState.LISTENING: {SystemState.THINKING, SystemState.IDLE},
            SystemState.THINKING: {
                SystemState.SPEAKING,
                SystemState.LISTENING,
                SystemState.IDLE,
            },
            SystemState.SPEAKING: {SystemState.IDLE, SystemState.LISTENING},
        }

    @property
    def current(self) -> SystemState:
        with self._lock:
            return self._current

    def transition(self, new_state: SystemState) -> None:
        """Move to a new state, enforcing the transition table.

        The previous version exempted any transition into IDLE from validation,
        which meant the table understated what was actually legal and could not
        be trusted as documentation. IDLE is now reachable only from states that
        declare it, and every edge is in the table.
        """
        with self._lock:
            old_state = self._current

            if new_state is old_state:
                return  # no-op; callers may transition defensively

            if new_state not in self._valid_transitions[old_state]:
                legal = ", ".join(sorted(s.name for s in self._valid_transitions[old_state]))
                raise ValueError(
                    f"Invalid state transition {old_state.name} -> {new_state.name}. "
                    f"Legal from {old_state.name}: {legal}"
                )

            self._current = new_state
            log_transition(old_state.name, new_state.name)

        self._bus.publish_sync("StateChanged", {"old": old_state, "new": new_state})
