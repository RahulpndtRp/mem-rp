"""
RAG (Retrieval-Augmented Generation) pipeline for my_mem.

• Retrieves top-k memories (STM + FAISS) via `Memory.search`.
• Builds a numbered context block.
• Calls the LLM with a citation-aware prompt.
• Returns answer + traceable sources.

Author: chatGPT (2025-05-09)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from my_mem.memory.main       import Memory
from my_mem.utils.factory     import LlmFactory
from my_mem.configs.base      import MemoryConfig

logger = logging.getLogger(__name__)


_CITATION_SYSTEM_PROMPT = """
You are an assistant that answers using the provided context.
Cite supporting facts with bracketed numbers, e.g. “[2]”.
If the answer cannot be derived, say you don’t have enough information.
"""


def _build_context(results: List[Dict[str, Any]]) -> str:
    """
    Turns Memory.search() results → numbered context block.
    Returns the block **and** a light “sources” list for downstream use.
    """
    lines   = []
    sources = []
    for idx, hit in enumerate(results, start=1):
        label = f"[{idx}]"
        text  = hit["memory"]
        lines.append(f"{label} {text}")
        sources.append({"id": hit["id"], "text": text})
    return "\n".join(lines), sources


class RAGPipeline:
    """Small wrapper so you can do:  rag = RAGPipeline(mem); rag.query("…")"""

    def __init__(self, memory: Memory, top_k: int = 5):
        self.memory = memory
        self.top_k  = top_k

        # Re-use same LLM that Memory already built
        self.llm = memory.llm

    # public --------------------------------------------------------------
    def query(self, question: str, *, user_id: str) -> Dict[str, Any]:
        """Run RAG & return { answer, sources }."""
        retrieved = self.memory.search(question, user_id=user_id, limit=self.top_k)["results"]
        logger.debug(f"RAG retrieved {len(retrieved)} memories")

        context_block, sources = _build_context(retrieved)
        answer = self._ask_llm(question, context_block)

        return {"answer": answer, "sources": sources}

    # internal ------------------------------------------------------------
    def _ask_llm(self, question: str, context: str) -> str:
        messages = [
            {"role": "system",    "content": _CITATION_SYSTEM_PROMPT},
            {"role": "system",    "content": f"Context:\n{context}"},
            {"role": "user",      "content": question},
        ]
        resp = self.llm.generate_response(messages=messages)
        return resp.strip()


# ----------------------------------------------------------------------- #
#  Simple convenience factory (optional)                                  #
# ----------------------------------------------------------------------- #
def get_default_rag(top_k: int = 5) -> RAGPipeline:
    """
    Utility for rapid prototyping:

        rag = get_default_rag()
        print(rag.query("What food do I like?", user_id="u1"))
    """
    mem   = Memory(MemoryConfig())        # uses default FAISS cfg etc.
    return RAGPipeline(mem, top_k=top_k)
