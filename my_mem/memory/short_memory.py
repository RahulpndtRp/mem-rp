"""
Very-lightweight short-term memory.

• Per-user ring buffer (FIFO) that stores the raw utterance _and_ its
  embedding so we can rank it together with the long-term FAISS hits.
• No persistence – it lives only for the current process.
"""

from collections import deque
from datetime    import datetime
from typing      import Dict, List, Optional, Tuple

import numpy as np


class ShortTermMemory:
    def __init__(self, max_items: int = 32):
        """
        :param max_items: how many items to keep _per user_ before overwriting
        """
        self.max_items      = max_items
        self._store: Dict[str, deque] = {}          # user_id → deque[Item]

    # --------------------------------------------------------- public API ---
    def add(self, user_id: str, text: str, embedding: np.ndarray):
        dq = self._store.setdefault(user_id, deque(maxlen=self.max_items))
        dq.append(
            {
                "id":        f"stm-{len(dq)}-{datetime.utcnow().isoformat(timespec='seconds')}",
                "text":      text,
                "embedding": embedding,
                "created":   datetime.utcnow().isoformat(),
            }
        )

    def recent(self, user_id: str, limit: int = 10) -> List[Dict]:
        return list(self._store.get(user_id, []))[-limit:]

    def clear(self, user_id: Optional[str] = None):
        if user_id is None:
            self._store.clear()
        else:
            self._store.pop(user_id, None)
