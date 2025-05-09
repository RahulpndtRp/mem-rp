from abc import ABC
from typing import Dict, Optional, Union
import httpx
from my_mem.configs.base import AzureConfig

class BaseEmbedderConfig(ABC):
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        embedding_dims: int = 1536,
        openai_base_url: Optional[str] = None,
        http_client_proxies: Optional[Union[Dict, str]] = None,
        **extra,
    ):
        self.model          = model
        self.api_key        = api_key
        self.embedding_dims = embedding_dims
        self.openai_base_url= openai_base_url
        self.http_client    = httpx.Client(proxies=http_client_proxies) if http_client_proxies else None
        self.azure_kwargs   = AzureConfig(**extra.get("azure_kwargs", {}))
