import logging
from typing import Dict, Any, Optional

from my_mem.configs.base import MemoryConfig
from my_mem.memory.main import Memory
from my_mem.rag.rag_pipeline import RAGPipeline
from typing import Generator
logger = logging.getLogger(__name__)

class MemoryClient:
    """
    High-level interface for interacting with memory (STM + LTM) and RAG querying.
    """
    def __init__(self, config: Optional[MemoryConfig] = None, top_k: int = 5, ltm_threshold: float = 0.75):
        self.memory = Memory(config or MemoryConfig())
        self.rag    = RAGPipeline(self.memory, top_k=top_k, ltm_threshold=ltm_threshold)


    def add_message(self, text: str, *, user_id: str, infer: bool = True) -> Dict[str, Any]:
        """
        Adds message to STM (always) and LTM (if infer=True).
        """
        return self.memory.add(text, user_id=user_id, infer=infer)

    def retrieve(self, query: str, *, user_id: str, limit: int = 5) -> Dict[str, Any]:
        """
        Retrieves ranked memories from STM + LTM.
        """
        return self.memory.search(query, user_id=user_id, limit=limit)

    def query_rag(self, prompt: str, *, user_id: str) -> Dict[str, Any]:
        """
        Adds prompt to memory and performs a RAG query over the memory context.
        """
        self.memory.add(prompt, user_id=user_id, infer=True)
        return self.rag.query(prompt, user_id=user_id)
    
    def stream_rag(self, prompt: str, *, user_id: str) -> Generator[str, None, None]:
        """
        Streaming version of RAG query — yields the response token by token.
        """
        self.memory.add(prompt, user_id=user_id, infer=True)
        yield from self.rag.stream_query(prompt, user_id=user_id)


    def reset_memory(self) -> None:
        """Resets FAISS + SQLite + STM buffers (if reset() supported)."""
        self.memory.reset()

from my_mem.memory.main import AsyncMemory
from my_mem.rag.rag_pipeline import AsyncRAGPipeline 

from typing import Dict, Any, Optional, AsyncGenerator



class AsyncMemoryClient:
    """
    High-level async interface for interacting with memory (STM + LTM) and RAG querying.
    Mirrors the sync MemoryClient interface.
    """
    def __init__(self, config: Optional[MemoryConfig] = None, top_k: int = 5, ltm_threshold: float = 0.75):
        self.memory = AsyncMemory(config)
        self.rag    = AsyncRAGPipeline(self.memory, top_k=top_k, ltm_threshold=ltm_threshold)

    async def add_message(self, text: str, *, user_id: str, infer: bool = True) -> Dict[str, Any]:
        return await self.memory.add(text, user_id=user_id, infer=infer)

    async def retrieve(self, query: str, *, user_id: str, limit: int = 5) -> Dict[str, Any]:
        return await self.memory.search(query, user_id=user_id, limit=limit)

    async def query_rag(self, prompt: str, *, user_id: str) -> Dict[str, Any]:
        await self.memory.add(prompt, user_id=user_id, infer=True)
        return await self.rag.query(prompt, user_id=user_id)

    async def stream_rag(self, prompt: str, *, user_id: str):
        await self.memory.add(prompt, user_id=user_id, infer=True)
        async for token in self.rag.stream_query(prompt, user_id=user_id):
            yield token

    async def reset_memory(self) -> None:
        await self.memory.reset()