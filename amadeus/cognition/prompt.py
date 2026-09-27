from dataclasses import dataclass, field
from amadeus.trace import get_tracer

@dataclass
class DialogueTurn:
    role: str  # "user" or "assistant"
    content: str

class PromptBuilder:
    """Builds chat messages for the LLM.
    
    Follows the stable prefix + volatile tail architecture:
    - STABLE PREFIX: system prompt (persona, instructions, user portrait)
    - VOLATILE TAIL: recent dialogue turns
    """
    
    def __init__(self, system_prompt: str, max_history_turns: int = 10):
        self._system_prompt = system_prompt
        self._max_history_turns = max_history_turns
        self._history: list[DialogueTurn] = []
        self._tracer = get_tracer("prompt_builder")
    
    def add_user_message(self, text: str) -> None:
        """Add a user message to history."""
        self._history.append(DialogueTurn(role="user", content=text))
        self._trim_history()
    
    def add_assistant_message(self, text: str) -> None:
        """Add an assistant response to history."""
        self._history.append(DialogueTurn(role="assistant", content=text))
        self._trim_history()
    
    def build(self, user_input: str | None = None) -> list[dict]:
        """Build the message list for the LLM.
        
        If user_input is provided, it's added as the latest user message.
        
        Returns:
            List of message dicts ready for chat completion.
        """
        messages = [{"role": "system", "content": self._system_prompt}]
        
        for turn in self._history:
            messages.append({"role": turn.role, "content": turn.content})
            
        if user_input is not None:
            messages.append({"role": "user", "content": user_input})
            
        self._tracer.info(f"Built prompt with {len(messages)} total messages")
        return messages
    
    def clear_history(self) -> None:
        """Clear dialogue history."""
        self._history.clear()
        self._tracer.info("Dialogue history cleared")
        
    def _trim_history(self) -> None:
        """Ensure history does not exceed the maximum number of turns."""
        if len(self._history) > self._max_history_turns * 2:
            # Drop older turns but keep pairs intact if possible
            excess = len(self._history) - self._max_history_turns * 2
            self._history = self._history[excess:]
