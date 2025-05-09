"""
Core Pydantic config models that every other module imports.

If you change a field name here, touch:
  • embeddings.base.BaseEmbedderConfig
  • llms.base.BaseLlmConfig
  • vector_stores.configs (per-backend)
  • memory.main.Memory / AsyncMemory
"""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
home_dir   = os.path.expanduser("~")
memrp_dir   = os.environ.get("MEMRP_DIR") or os.path.join(home_dir, ".memrp")
os.makedirs(memrp_dir, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Generic item returned by Memory.{add,get,search,…}                         #
# --------------------------------------------------------------------------- #
class MemoryItem(BaseModel):
    id:         str                    = Field(...,  description="Vector-store UUID")
    memory:     str                    = Field(...,  description="Human-readable memory text")
    hash:       Optional[str]          = Field(None, description="MD5 of memory string")
    metadata:   Optional[Dict[str,Any]]= Field(None, description="Arbitrary extra info")
    score:      Optional[float]        = Field(None, description="Similarity score (search only)")
    created_at: Optional[str]          = Field(None)
    updated_at: Optional[str]          = Field(None)


# --------------------------------------------------------------------------- #
#  Minimal configs for LLM / Embedding / VectorStore                          #
# --------------------------------------------------------------------------- #
class LlmConfig(BaseModel):
    provider: str  = Field("openai")
    config:   Dict = Field(default_factory=dict)

class EmbedderConfig(BaseModel):
    provider: str  = Field("openai")
    config:   Dict = Field(default_factory=dict)

class VectorStoreConfig(BaseModel):
    provider: str  = Field("faiss")
    config:   Dict = Field(
        default_factory=lambda: {
            "path": ".faiss",
            "collection_name": "memrp",
            "embedding_model_dims": 1536,
        }
    )

class GraphStoreConfig(BaseModel):
    provider: Optional[str] = None   # we’ll wire later
    config:   Dict           = Field(default_factory=dict)
    llm:      Optional[LlmConfig] = None


# --------------------------------------------------------------------------- #
#  Composite “everything” config consumed by Memory                           #
# --------------------------------------------------------------------------- #
class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm:           LlmConfig        = Field(default_factory=LlmConfig)
    embedder:      EmbedderConfig   = Field(default_factory=EmbedderConfig)
    graph_store:   GraphStoreConfig = Field(default_factory=GraphStoreConfig)

    history_db_path: str = Field(
        default=os.path.join(memrp_dir, "history.db"),
        description="SQLite file for event history",
    )
    version: str = Field(default="v1.1")

    # Optional custom prompts
    custom_fact_extraction_prompt: Optional[str] = None
    custom_update_memory_prompt:   Optional[str] = None


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
