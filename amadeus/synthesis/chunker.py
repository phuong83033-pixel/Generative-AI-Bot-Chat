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

    # Words a clause should not END on. When the first-clause cap lands right after
    # one, the cut moves back before it, so the seam falls at a phrase boundary
    # ("... a beloved stimulant | that energizes ...") rather than splitting
    # "that" from what it introduces. Nouns and adjectives are deliberately not
    # listed: no word list can tell "personal | preferences" from "the brain".
    FUNCTION_WORDS = frozenset({
        "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her",
        "its", "our", "their", "of", "to", "in", "on", "at", "for", "from", "by", "with",
        "into", "about", "than", "as", "and", "or", "but", "so", "if", "because", "while",
        "which", "who", "whose", "when", "where", "is", "are", "was", "were", "be", "been",
        "i", "i'm", "i'd", "i'll", "i've", "you", "we", "they", "it", "it's", "don't",
        "can't", "won't", "not", "very", "quite",
    })
    
    def __init__(self, bus: EventBus | None = None,
                 first_clause_min: int | None = None,
                 first_clause_max: int | None = None):
        """
        Args:
            first_clause_min: When set, the FIRST clause of a reply breaks at the
                first sentence end or soft break once it holds at least this many
                characters, replacing the usual length rules for that clause only.

                Why only the first: time to first sound is dominated by how much
                text must be synthesised before anything plays, so a short opener
                pays off there. Later clauses are synthesised while earlier ones
                play, so their length costs nothing and longer is smoother.

                Why not as short as possible: clause 2 must be ready when clause 1
                finishes playing, i.e. clause 1 must last about as long as clause 2
                takes to synthesise. At ~15 characters of speech per second, a
                6-character "Hello!" lasts 0.8 s but the next clause needs ~1.3 s to
                synthesise, so the listener hears a gap. See scripts/bench_latency.py,
                which reports first-sound latency and the longest gap together.

                None keeps the original behaviour for every clause.

            first_clause_max: When set, the first clause ends at the next word
                boundary once it reaches this many characters, even with no
                punctuation. OFF by default because it puts a prosodic seam
                mid-sentence: each clause is synthesised as its own utterance, so
                the voice falls and resets at the join. It exists because a reply
                whose opening sentence runs 60-110 characters with no comma costs
                ~1.5-2.5 s of synthesis before anything is heard (about 440 ms
                fixed + 19 ms per character on this CPU). Keep it >= first_clause_min;
                the cap wins if they conflict.
        """
        self._bus = bus
        self._tracer = get_tracer("chunker")
        self.first_clause_min = first_clause_min
        self.first_clause_max = first_clause_max
    
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

            # First-clause cap: a token that starts with a space means the previous
            # word is finished, so this is a safe place to cut if the clause has
            # already reached the cap without any punctuation.
            if (clause_count == 0 and self.first_clause_max is not None
                    and token[:1].isspace()
                    and len(buffer.strip()) >= self.first_clause_max):
                head, tail = self._split_at_cap(buffer)
                clause_count += 1
                await self._announce(head, clause_count)
                yield head
                buffer = tail

            buffer += token
            
            # Check if we should emit a clause
            if self._should_break(buffer, first=(clause_count == 0)):
                clause = buffer.strip()
                if clause:
                    clause_count += 1
                    await self._announce(clause, clause_count)
                    yield clause
                buffer = ""
        
        # Emit remaining buffer
        remaining = buffer.strip()
        if remaining:
            clause_count += 1
            await self._announce(remaining, clause_count, final=True)
            yield remaining

    def _split_at_cap(self, buffer: str) -> tuple[str, str]:
        """Split `buffer` for a cap cut: (clause to speak, text to carry forward).

        Trailing function words move to the carry, so a clause does not end on
        "and" or "that". Never backs up below half the cap (or the first-clause
        minimum), because a too-short opener would leave a gap before clause 2.
        """
        words = buffer.strip().split(" ")
        floor = max(self.first_clause_min or 0, (self.first_clause_max or 0) // 2, 8)
        carried: list[str] = []
        while len(words) > 1:
            last = words[-1].lower().strip(".,;:!?\"()")
            head_len = len(" ".join(words[:-1]))
            if last in self.FUNCTION_WORDS and head_len >= floor:
                carried.insert(0, words.pop())
            else:
                break
        head = " ".join(words).strip()
        tail = (" " + " ".join(carried)) if carried else ""
        return head, tail

    async def _announce(self, clause: str, index: int, final: bool = False) -> None:
        """Log an emitted clause and publish it on the bus."""
        label = "Final clause" if final else "Clause"
        shown = clause[:50] + "..." if len(clause) > 50 else clause
        self._tracer.info(f"{label} {index}: {shown}")
        if self._bus:
            await self._bus.publish("LLMClause", {"clause": clause, "index": index})
    
    def _should_break(self, buffer: str, first: bool = False) -> bool:
        """Determine if the buffer should be emitted as a clause.

        `first` marks the first clause of a reply; see first_clause_min.
        """
        stripped = buffer.strip()
        if not stripped:
            return False

        # Force break at max length
        if len(stripped) >= self.MAX_CLAUSE_LENGTH:
            return True

        # First clause: one threshold decides, for hard and soft breaks alike.
        if first and self.first_clause_min is not None:
            return (len(stripped) >= self.first_clause_min
                    and stripped[-1] in (self.CLAUSE_ENDS | self.SOFT_BREAKS))

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
