import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator

# MUST run before `import llama_cpp`. The prebuilt CUDA wheel needs cudart64_12.dll
# and cublas64_12.dll on the DLL search path; this machine has no CUDA Toolkit, so
# they are borrowed from torch's bundled cu12 runtime. See runtime/cuda_dlls.py.
from amadeus.runtime.cuda_dlls import ensure_cuda_dlls

ensure_cuda_dlls()

from llama_cpp import Llama  # noqa: E402  - import order is load-bearing
from amadeus.trace import get_tracer
from amadeus.bus import EventBus
from amadeus.runtime.vram import vram_guard, log_vram

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    async def stream(self, messages: list[dict], 
                     stop: asyncio.Event | None = None) -> AsyncIterator[str]:
        """Stream tokens from the LLM.
        
        Args:
            messages: Chat messages [{"role": "system"|"user"|"assistant", "content": str}]
            stop: Cancellation event
            
        Yields:
            Individual tokens as strings
        """
        ...

class QwenProvider(LLMProvider):
    """Qwen2.5-3B-Instruct via llama-cpp-python."""
    
    def __init__(self, model_path: str, 
                 n_ctx: int = 2048,
                 n_gpu_layers: int = -1,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 top_p: float = 0.9,
                 repeat_penalty: float = 1.1,
                 bus: EventBus | None = None):
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._repeat_penalty = repeat_penalty
        self._bus = bus
        self._tracer = get_tracer("qwen_provider")
        
        with vram_guard("Qwen model load"):
            self._model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
        self._tracer.info(f"Loaded Qwen model from {model_path}")
    
    async def stream(self, messages: list[dict],
                     stop: asyncio.Event | None = None) -> AsyncIterator[str]:
        """Stream tokens from Qwen."""
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        def _generate():
            try:
                response = self._model.create_chat_completion(
                    messages=messages,
                    stream=True,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    top_p=self._top_p,
                    repeat_penalty=self._repeat_penalty,
                )
                for chunk in response:
                    if stop and stop.is_set():
                        break
                    
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]
                            loop.call_soon_threadsafe(queue.put_nowait, token)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
        
        fut = loop.run_in_executor(None, _generate)
        
        token_count = 0
        first_token_time = None
        start_time = time.perf_counter()
        
        with self._tracer.span("llm_stream"):
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                if stop and stop.is_set():
                    break
                    
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    ttft = first_token_time - start_time
                    self._tracer.info(f"Time to first token: {ttft:.3f}s")
                
                token_count += 1
                if self._bus:
                    await self._bus.publish("LLMToken", {"token": item})
                yield item
        
        await fut
        
        total_time = time.perf_counter() - start_time
        tps = token_count / total_time if total_time > 0 else 0
        self._tracer.info(f"Stream complete: {token_count} tokens at {tps:.1f} tokens/s")
        if self._bus:
            await self._bus.publish("LLMDone", {
                "token_count": token_count, 
                "tokens_per_sec": tps, 
                "total_time": total_time
            })
