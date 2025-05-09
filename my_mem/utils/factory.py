"""
Maps provider strings → concrete classes and instantiates them.

If you add a new backend, just extend the *_to_class dict.
"""

from importlib import import_module
from typing    import Dict, Union, Optional, Type
from pydantic import BaseModel  # <-- added for Pylance

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _load(path: str):
    if not path:
        raise ValueError("Factory received None for implementation path. Check provider_to_class mapping.")
    mod, cls = path.rsplit(".", 1)
    return getattr(import_module(mod), cls)


def _ensure(obj: Union[Dict, "BaseModel"], cfg_cls: Type):
    """
    Accept either an already-built config object or a plain dict and
    return an instance of cfg_cls.
    """
    if isinstance(obj, cfg_cls):
        return obj
    if isinstance(obj, dict):
        return cfg_cls(**obj)
    raise TypeError(f"Config must be dict or {cfg_cls.__name__}, got {type(obj)}")


# --------------------------------------------------------------------------- #
#  LLM factory                                                                #
# --------------------------------------------------------------------------- #
from my_mem.llms.base        import BaseLlmConfig
from my_mem.embeddings.base  import BaseEmbedderConfig
from my_mem.vector_stores.base       import BaseVectorStore   # just for typing

class LlmFactory:
    provider_to_class = {
        "openai": "my_mem.llms.openai.OpenAILLM",
        "openai_async": "my_mem.llms.openai.AsyncOpenAILLM",  # ✅ Add this line
    }

    @classmethod
    def create(cls, provider: str, cfg: Union[Dict, BaseLlmConfig]):
        impl = _load(cls.provider_to_class.get(provider))
        cfg_obj = _ensure(cfg, BaseLlmConfig)
        return impl(cfg_obj)


# --------------------------------------------------------------------------- #
#  Embedder factory                                                           #
# --------------------------------------------------------------------------- #
class EmbedderFactory:
    provider_to_class = {
        "openai": "my_mem.embeddings.openai.OpenAIEmbedding",
    }

    @classmethod
    def create(
        cls,
        provider: str,
        cfg: Union[Dict, BaseEmbedderConfig],
        _vector_cfg: Optional[Dict] = None,   # kept for compatibility
    ):
        impl_path = cls.provider_to_class.get(provider)
        if not impl_path:
            raise ValueError(f"Unknown LLM provider: {provider}")
        impl = _load(cls.provider_to_class.get(provider))
        cfg_obj = _ensure(cfg, BaseEmbedderConfig)
        return impl(cfg_obj)


# --------------------------------------------------------------------------- #
#  Vector-store factory                                                       #
# --------------------------------------------------------------------------- #
class VectorStoreFactory:
    provider_to_class = {
        "faiss": "my_mem.vector_stores.faiss.FAISS",
    }

    @classmethod
    def create(cls, provider: str, cfg: Dict) -> BaseVectorStore:
        impl = _load(cls.provider_to_class.get(provider))
        # vector-store constructors are **kwargs only
        return impl(**cfg)

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance
