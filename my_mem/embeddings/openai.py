import os
import numpy as np
from openai import OpenAI
from my_mem.embeddings.base import BaseEmbedderConfig

class OpenAIEmbedding:
    def __init__(self, cfg: BaseEmbedderConfig):
        self.cfg    = cfg
        self.client = OpenAI(
            api_key = cfg.api_key or os.getenv("OPENAI_API_KEY"),
            base_url= cfg.openai_base_url or "https://api.openai.com/v1",
        )

    def embed(self, text: str, _action="add"):
        resp = self.client.embeddings.create(
            model       = self.cfg.model,
            input       = text,
            encoding_format="float",
        )
        return np.array(resp.data[0].embedding, dtype="float32")
