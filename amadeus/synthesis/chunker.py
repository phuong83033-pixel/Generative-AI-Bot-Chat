import asyncio
from typing import AsyncIterator
from amadeus.trace import get_tracer
from amadeus.bus import EventBus

class ClauseChunker:
    """Splits streaming LLM tokens into speech-sized clauses.
    
    Instead of waiting for the full response, splits on natural
    boundaries (punctuation, clause endings) so TTS can start early.
    """
    
    # Clause-ending punctuation
    CLAUSE_ENDS = {'.', '!', '?', ';', ':', '\n'}
    SOFT_BREAKS = {',', '—', '–', '...'}
    MIN_CLAUSE_LENGTH = 10  # minimum chars before allowing a break
    MAX_CLAUSE_LENGTH = 150  # force break at this length
    
    def __init__(self, bus: EventBus | None = None):
        self._bus = bus
        self._tracer = get_tracer("chunker")
    
    async def process(self, token_stream: AsyncIterator[str],
                      stop: asyncio.Event | None = None) -> AsyncIterator[str]:
        """Process streaming tokens into clauses.
        
        Args:
            token_stream: Async iterator of tokens from LLM
            stop: Cancellation event
            
        Yields:
            Complete clauses ready for TTS
        """
        buffer = ""
        clause_count = 0
        
        async for token in token_stream:
            if stop and stop.is_set():
                break
            buffer += token
            
            # Check if we should emit a clause
            if self._should_break(buffer):
                clause = buffer.strip()
                if clause:
                    clause_count += 1
                    self._tracer.info(f"Clause {clause_count}: {clause[:50]}..." if len(clause) > 50 else f"Clause {clause_count}: {clause}")
                    if self._bus:
                        await self._bus.publish("LLMClause", {"clause": clause, "index": clause_count})
                    yield clause
                buffer = ""
        
        # Emit remaining buffer
        remaining = buffer.strip()
        if remaining:
            clause_count += 1
            self._tracer.info(f"Final clause {clause_count}: {remaining[:50]}..." if len(remaining) > 50 else f"Final clause {clause_count}: {remaining}")
            if self._bus:
                await self._bus.publish("LLMClause", {"clause": remaining, "index": clause_count})
            yield remaining
    
    def _should_break(self, buffer: str) -> bool:
        """Determine if the buffer should be emitted as a clause."""
        stripped = buffer.strip()
        if not stripped:
            return False
        
        # Force break at max length
        if len(stripped) >= self.MAX_CLAUSE_LENGTH:
            return True
        
        # Don't break too-short clauses
        if len(stripped) < self.MIN_CLAUSE_LENGTH:
            return False
        
        # Check for clause-ending punctuation
        if stripped[-1] in self.CLAUSE_ENDS:
            return True
        
        # Check for soft breaks (only if buffer is reasonably long)
        if len(stripped) >= 30 and stripped[-1] in self.SOFT_BREAKS:
            return True
        
        return False
