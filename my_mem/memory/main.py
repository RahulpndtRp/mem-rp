"""
Slimmed-down but feature-complete Memory implementation.

• Long-term / short-term logic identical to memrp (semantic & episodic stored
  in the vector DB; procedural summaries optional).
• Graph memory disabled by default (enable later).
"""
import asyncio
import concurrent.futures, hashlib, json, logging, os, uuid
from copy     import deepcopy
from datetime import datetime
from typing   import Any, Dict, List, Optional

import pytz

from my_mem.configs.base import MemoryConfig, MemoryItem
from my_mem.utils.telemetry import capture_event          # <- no-op stub
from my_mem.utils.prompts   import (
    FACT_RETRIEVAL_PROMPT,
    get_update_memory_messages,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
)
from my_mem.utils.utils     import (
    parse_messages,
    remove_code_blocks,
)

from my_mem.vector_stores.base   import BaseVectorStore
from my_mem.utils.factory        import EmbedderFactory, LlmFactory, VectorStoreFactory
from my_mem.memory.storage_sqlite import SQLiteManager
from my_mem.memory.short_memory import ShortTermMemory

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Memory (sync only for v1)                                                  #
# --------------------------------------------------------------------------- #
class Memory:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.cfg  = config

        # Build provider instances
        self.embedder     = EmbedderFactory.create(
            config.embedder.provider, config.embedder.config, config.vector_store.config
        )
        self.vector_store: BaseVectorStore = VectorStoreFactory.create(
            config.vector_store.provider, config.vector_store.config
        )
        self.llm          = LlmFactory.create(config.llm.provider, config.llm.config)
        self.db           = SQLiteManager(config.history_db_path)
        self.short_term = ShortTermMemory(max_items=32)

        capture_event("memory.init", self, {"sync_type": "sync"})

    # ────────────────────────────────────────────────────────────────────────────
    #  ADD  (store → STM ▸ optionally infer & promote to LTM)
    # ────────────────────────────────────────────────────────────────────────────
    def add(self, message: str, *, user_id: str, infer: bool = True) -> Dict:
        """
        • Always caches the raw `message` in short-term memory (STM).
        • If `infer` is True → extract facts and reconcile them with long-term
        memory (vector store).  Otherwise we only keep STM + a single LTM row.
        """
        # -------------------------------------------------------- short-term ----
        msg_vec = self.embedder.embed(message, "add")
        self.short_term.add(user_id, message, msg_vec)

        # --------------------------------------------------------- long-term ----
        metadata = {"user_id": user_id}
        filters  = {"user_id": user_id}

        # ❶  Simple path – no inference, just store the raw text in FAISS
        if not infer:
            mem_id = self._create_memory(message, msg_vec, metadata)
            return {"results": [{"id": mem_id, "memory": message, "event": "ADD"}]}

        # ❷  Fact-extraction ----------------------------------------------------
        system_prompt, user_prompt = FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"
        resp = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        try:
            facts = json.loads(remove_code_blocks(resp))["facts"]
        except Exception:
            facts = []

        if not facts:          # nothing promotable → done
            return {"results": []}

        # ❸  Check overlap with existing long-term memory -----------------------
        existing = {}
        for fact in facts:
            vec   = self.embedder.embed(fact, "add")
            hits  = self.vector_store.search(query=fact, vectors=vec, limit=5, filters=filters)
            for h in hits:
                existing[h.id] = h.payload["data"]

        # ❹  Ask LLM which ADD/UPDATE/DELETE/NONE ops we should perform ---------
        old_mem_list = [{"id": k, "text": v} for k, v in existing.items()]
        up_prompt    = get_update_memory_messages(old_mem_list, facts, self.cfg.custom_update_memory_prompt)
        resp = self.llm.generate_response(
            messages=[{"role": "user", "content": up_prompt}],
            response_format={"type": "json_object"},
        )
        try:
            actions = json.loads(remove_code_blocks(resp))["memory"]
        except Exception:
            logger.error("LLM update-memory JSON parse failure – falling back to ADD")
            actions = [{"id": str(uuid.uuid4()), "text": f, "event": "ADD"} for f in facts]

        # ❺  Apply the actions ---------------------------------------------------
        results = []
        for act in actions:
            ev = act["event"]
            if ev == "ADD":
                vec = self.embedder.embed(act["text"], "add")
                mid = self._create_memory(act["text"], vec, deepcopy(metadata))
                results.append({"id": mid, "memory": act["text"], "event": "ADD"})
            elif ev == "UPDATE":
                vec = self.embedder.embed(act["text"], "update")
                self._update_memory(act["id"], act["text"], vec, deepcopy(metadata))
                results.append(
                    {"id": act["id"], "memory": act["text"], "event": "UPDATE",
                    "previous_memory": act.get("old_memory")}
                )
            elif ev == "DELETE":
                self._delete_memory(act["id"])
                results.append({"id": act["id"], "memory": act["text"], "event": "DELETE"})
            else:  # NONE
                results.append({"id": act["id"], "memory": act["text"], "event": "NONE"})

        capture_event("memory.add", self, {"facts": len(facts)})
        return {"results": results}


    # ────────────────────────────────────────────────────────────────────────────
    #  SEARCH  (blend STM + FAISS and rank by score)
    # ────────────────────────────────────────────────────────────────────────────
    def search(self, query: str, *, user_id: str, limit: int = 5, ltm_threshold: float = 0.75) -> Dict:
        import numpy as np

        filters = {"user_id": user_id}
        qvec    = self.embedder.embed(query, "search")

        # ---- LTM: vector store hits over threshold ----
        lt_hits = self.vector_store.search(query=query, vectors=qvec, limit=10, filters=filters)
        lt_items = [
            MemoryItem(
                id=h.id,
                memory=h.payload["data"],
                hash=h.payload.get("hash"),
                created_at=h.payload.get("created_at"),
                updated_at=h.payload.get("updated_at"),
                score=h.score,
            ).model_dump()
            for h in lt_hits if h.score >= ltm_threshold
        ][:3]  # Take top 3 above threshold

        # ---- STM: last 5 turns only ----
        st_buf = self.short_term.recent(user_id, limit=32)
        st_items = [
            MemoryItem(
                id=it["id"],
                memory=it["text"],
                hash=None,
                created_at=it["created"],
                updated_at=None,
                score=0.99,  # Artificially high score for ordering
            ).model_dump()
            for it in st_buf[-5:]
        ]

        # ---- Merge and return ----
        merged = sorted(lt_items + st_items, key=lambda x: x["score"], reverse=True)
        return {"results": merged[:limit]}


    # -------------------------------------------------------------- helpers --
    def _create_memory(self, data: str, vec, meta: Dict) -> str:
        mid = str(uuid.uuid4())
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update({"data": data, "hash": hashlib.md5(data.encode()).hexdigest(),
                     "created_at": now, "__vector": vec})
        self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[meta])
        self.db.add_history(mid, None, data, "ADD", created_at=now)
        return mid

    def _update_memory(self, mid, new_data, vec, meta):
        existing = self.vector_store.search(query="", vectors=vec, limit=1, filters={})  # hack
        created  = existing[0].payload.get("created_at") if existing else None
        now      = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update({"data": new_data, "hash": hashlib.md5(new_data.encode()).hexdigest(),
                     "created_at": created, "updated_at": now, "__vector": vec})
        self.vector_store.update(vector_id=mid, vector=vec, payload=meta)
        self.db.add_history(mid, existing[0].payload["data"] if existing else None,
                            new_data, "UPDATE", created_at=created, updated_at=now)

    def _delete_memory(self, mid):
        self.vector_store.delete(vector_id=mid)
        self.db.add_history(mid, None, None, "DELETE", is_deleted=1)

    def reset(self):
        """Reset the memory by wiping vector DB and history if needed."""
        if hasattr(self.vector_store, "reset"):
            self.vector_store.reset()
        else:
            raise NotImplementedError("The current vector store does not support reset().")



class AsyncMemory:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.cfg = config
        self.embedder = EmbedderFactory.create(config.embedder.provider, config.embedder.config, config.vector_store.config)
        self.vector_store: BaseVectorStore = VectorStoreFactory.create(config.vector_store.provider, config.vector_store.config)
        self.llm = LlmFactory.create(config.llm.provider, config.llm.config)
        self.db = SQLiteManager(config.history_db_path)
        self.short_term = ShortTermMemory(max_items=32)

        capture_event("memory.init", self, {"sync_type": "async"})

    async def add(self, message: str, *, user_id: str, infer: bool = True) -> Dict:
        msg_vec = await asyncio.to_thread(self.embedder.embed, message, "add")
        self.short_term.add(user_id, message, msg_vec)

        metadata = {"user_id": user_id}
        filters = {"user_id": user_id}

        if not infer:
            mem_id = await self._create_memory(message, msg_vec, metadata)
            return {"results": [{"id": mem_id, "memory": message, "event": "ADD"}]}

        # Fact extraction
        system_prompt, user_prompt = FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"
        resp = await asyncio.to_thread(self.llm.generate_response_async, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], {"type": "json_object"})
        try:
            facts = json.loads(remove_code_blocks(resp))["facts"]
        except Exception:
            facts = []

        if not facts:
            return {"results": []}

        # Search for overlaps
        existing = {}
        for fact in facts:
            vec = await asyncio.to_thread(self.embedder.embed, fact, "add")
            hits = await asyncio.to_thread(self.vector_store.search, fact, vec, 5, filters)
            for h in hits:
                existing[h.id] = h.payload["data"]

        old_mem_list = [{"id": k, "text": v} for k, v in existing.items()]
        prompt = get_update_memory_messages(old_mem_list, facts, self.cfg.custom_update_memory_prompt)
        resp = await asyncio.to_thread(self.llm.generate_response, [{"role": "user", "content": prompt}], {"type": "json_object"})

        try:
            actions = json.loads(remove_code_blocks(resp))["memory"]
        except Exception:
            logger.error("LLM update-memory JSON parse failure – storing all as ADD")
            actions = [{"id": str(uuid.uuid4()), "text": f, "event": "ADD"} for f in facts]

        results = []
        for act in actions:
            ev = act["event"]
            if ev == "ADD":
                vec = await asyncio.to_thread(self.embedder.embed, act["text"], "add")
                mid = await self._create_memory(act["text"], vec, deepcopy(metadata))
                results.append({"id": mid, "memory": act["text"], "event": "ADD"})
            elif ev == "UPDATE":
                vec = await asyncio.to_thread(self.embedder.embed, act["text"], "update")
                await self._update_memory(act["id"], act["text"], vec, deepcopy(metadata))
                results.append({"id": act["id"], "memory": act["text"], "event": "UPDATE", "previous_memory": act.get("old_memory")})
            elif ev == "DELETE":
                await self._delete_memory(act["id"])
                results.append({"id": act["id"], "memory": act["text"], "event": "DELETE"})
            else:
                results.append({"id": act["id"], "memory": act["text"], "event": "NONE"})

        capture_event("memory.add", self, {"facts": len(facts)})
        return {"results": results}

    async def search(self, query: str, *, user_id: str, limit: int = 5, ltm_threshold: float = 0.75) -> Dict:
        import numpy as np

        filters = {"user_id": user_id}
        qvec = await asyncio.to_thread(self.embedder.embed, query, "search")

        lt_hits = await asyncio.to_thread(self.vector_store.search, query, qvec, 10, filters)
        lt_items = [
            MemoryItem(
                id=h.id,
                memory=h.payload["data"],
                hash=h.payload.get("hash"),
                created_at=h.payload.get("created_at"),
                updated_at=h.payload.get("updated_at"),
                score=h.score,
            ).model_dump()
            for h in lt_hits if h.score >= ltm_threshold
        ][:3]

        st_buf = self.short_term.recent(user_id, limit=32)
        st_items = [
            MemoryItem(
                id=it["id"],
                memory=it["text"],
                hash=None,
                created_at=it["created"],
                updated_at=None,
                score=0.99,
            ).model_dump()
            for it in st_buf[-5:]
        ]

        merged = sorted(lt_items + st_items, key=lambda x: x["score"], reverse=True)
        return {"results": merged[:limit]}

    async def _create_memory(self, data: str, vec, meta: Dict) -> str:
        mid = str(uuid.uuid4())
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update({"data": data, "hash": hashlib.md5(data.encode()).hexdigest(), "created_at": now, "__vector": vec})
        await asyncio.to_thread(self.vector_store.insert, [vec], [mid], [meta])
        await asyncio.to_thread(self.db.add_history, mid, None, data, "ADD", created_at=now)
        return mid

    async def _update_memory(self, mid, new_data, vec, meta):
        existing = await asyncio.to_thread(self.vector_store.search, "", vec, 1, {})
        created = existing[0].payload.get("created_at") if existing else None
        now = datetime.now(pytz.timezone("UTC")).isoformat()
        meta.update({"data": new_data, "hash": hashlib.md5(new_data.encode()).hexdigest(),
                     "created_at": created, "updated_at": now, "__vector": vec})
        await asyncio.to_thread(self.vector_store.update, mid, vec, meta)
        await asyncio.to_thread(self.db.add_history, mid, existing[0].payload["data"] if existing else None,
                                new_data, "UPDATE", created_at=created, updated_at=now)

    async def _delete_memory(self, mid):
        await asyncio.to_thread(self.vector_store.delete, mid)
        await asyncio.to_thread(self.db.add_history, mid, None, None, "DELETE", is_deleted=1)

    async def reset(self):
        if hasattr(self.vector_store, "reset"):
            await asyncio.to_thread(self.vector_store.reset)
        else:
            raise NotImplementedError("The current vector store does not support reset().")
