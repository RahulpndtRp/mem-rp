# my_mem/vector_stores/faiss.py
import os, uuid, json
from typing import Dict, Any, List, Optional, Tuple

import faiss
import numpy as np

from my_mem.vector_stores.base import BaseVectorStore


# --------------------------------------------------------------------------- #
#  Result container                                                           #
# --------------------------------------------------------------------------- #
class _Res:
    def __init__(self, _id: str, vec: np.ndarray | None, payload: Dict[str, Any], score: float = 0.0):
        self.id      = _id
        self.vector  = vec
        self.payload = payload or {}
        self.score   = score


# --------------------------------------------------------------------------- #
#  FAISS implementation with persistence                                      #
# --------------------------------------------------------------------------- #
class FAISS(BaseVectorStore):
    """
    Lightweight, local-only FAISS backend.

    • Saves the index to `<path>/<collection>.index`
    • Saves payloads   to `<path>/<collection>.payload.json`

    After every insert / update / delete the index and payload file are flushed
    so memories survive interpreter restarts.
    """

    # ----------------------------- init / load ----------------------------- #
    def __init__(
        self,
        path: str,
        collection_name: str,
        embedding_model_dims: int,
        metric_type: str = "IP",
        **_,
    ):
        os.makedirs(path, exist_ok=True)

        self.idx_path      = os.path.join(path, f"{collection_name}.index")
        self.payload_path  = os.path.join(path, f"{collection_name}.payload.json")
        self.collection    = collection_name
        self.dim           = int(embedding_model_dims)
        self.metric_type   = metric_type.upper()

        self._payloads: Dict[str, Dict[str, Any]] = {}

        # create empty index of correct metric
        self._index = (
            faiss.IndexFlatIP(self.dim) if self.metric_type == "IP"
            else faiss.IndexFlatL2(self.dim) if self.metric_type == "L2"
            else None
        )
        if self._index is None:
            raise ValueError(f"Unsupported metric_type '{metric_type}'")

        # Attempt to load saved state
        self._try_load()

    # ----------------------------- persistence ----------------------------- #
    def _jsonable(self, obj):
        """Convert non-serialisable objects to JSON-friendly ones."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode()            # or base64 - adjust if needed
        if isinstance(obj, set):
            return list(obj)
        # fallback – let json default handle or str()
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    def _save(self):
        faiss.write_index(self._index, self.idx_path)

        # Walk the payload dict and sanitise values
        serialisable = {
            _id: {k: self._jsonable(v) for k, v in pl.items()}
            for _id, pl in self._payloads.items()
        }
        with open(self.payload_path, "w") as f:
            json.dump(serialisable, f)

    def _try_load(self):
        try:
            if os.path.exists(self.idx_path) and os.path.exists(self.payload_path):
                self._index = faiss.read_index(self.idx_path)
                with open(self.payload_path, "r") as f:
                    self._payloads = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load vectorstore index or payloads: {e}")
            self._index = faiss.IndexFlatIP(self.dim if self.metric_type == "IP" else faiss.IndexFlatL2(self.dim))
            self._payloads = {}



    # ----------------------------- CRUD ops -------------------------------- #
    def insert(
        self,
        vectors: List[np.ndarray],
        payloads: Optional[List[Dict]] = None,
        ids:      Optional[List[str]]  = None,
    ):
        vecs = np.vstack(vectors).astype("float32")
        self._index.add(vecs)

        ids      = ids or [str(uuid.uuid4()) for _ in vectors]
        payloads = payloads or [{} for _ in vectors]
        for _id, pl in zip(ids, payloads):
            self._payloads[_id] = pl

        self._save()
        return ids

    def search(
        self,
        query: str,                     # not used (we already have `vectors`)
        vectors: np.ndarray,
        limit: int = 5,
        filters: Optional[Dict] = None,
    ):
        qvec = np.asarray(vectors, dtype="float32").reshape(1, -1)
        D, I = self._index.search(qvec, limit)
        keys = list(self._payloads.keys())
        hits: List[_Res] = []

        for dist, idx in zip(D[0], I[0]):
            if idx == -1 or idx >= len(keys):
                continue
            _id   = keys[idx]
            pload = self._payloads[_id]
            if filters and any(pload.get(k) != v for k, v in filters.items()):
                continue
            hits.append(_Res(_id, None, pload, float(dist)))
        return hits

    def update(self, vector_id: str, vector: Optional[np.ndarray] = None, payload: Optional[Dict] = None):
        self.delete(vector_id)
        self.insert([vector], [payload or {}], [vector_id])   # _save already called

    def delete(self, vector_id: str):
        if vector_id not in self._payloads:
            return
        idx = list(self._payloads.keys()).index(vector_id)
        self._index.remove_ids(np.array([idx], dtype="int64"))
        del self._payloads[vector_id]
        self._save()

    def get(self, vector_id: str):
        pl = self._payloads.get(vector_id)
        return _Res(vector_id, None, pl) if pl else None

    # ------------------------ listing / collection ------------------------- #
    def list(self, filters: Optional[Dict] = None, limit: Optional[int] = None):
        results = []
        for _id, pl in self._payloads.items():
            if filters and any(pl.get(k) != v for k, v in filters.items()):
                continue
            results.append(_Res(_id, None, pl))
            if limit and len(results) >= limit:
                break
        return results, None   # matches expectation

    def create_col(self, name, vector_size, distance):
        # Single in-memory collection already initialised
        pass

    def list_cols(self):
        return [self.collection]

    def delete_col(self):
        self._index.reset()
        self._payloads.clear()
        if os.path.exists(self.idx_path):
            os.remove(self.idx_path)
        if os.path.exists(self.payload_path):
            os.remove(self.payload_path)

    def col_info(self):
        return {
            "name": self.collection,
            "vector_dim": self.dim,
            "metric": self.metric_type,
            "size": self._index.ntotal,
        }

    def reset(self):
        """Delete everything and recreate an empty, persisted collection."""
        self.delete_col()
        # rebuild empty index
        self._index = faiss.IndexFlatIP(self.dim) if self.metric_type == "IP" else faiss.IndexFlatL2(self.dim)
        self._save()




