from abc import ABC
from typing import Dict, Optional, Union
import httpx
from my_mem.configs.base import AzureConfig

class BaseLlmConfig(ABC):
    def __init__(
        self,
        model: Optional[Union[str, Dict]] = "gpt-4o-mini",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 2000,
        top_p: float = 0.1,
        enable_vision: bool = False,
        openai_base_url: Optional[str] = None,
        http_client_proxies: Optional[Union[Dict, str]] = None,
        **extra,
    ):
        self.model        = model
        self.temperature  = temperature
        self.api_key      = api_key
        self.max_tokens   = max_tokens
        self.top_p        = top_p
        self.enable_vision= enable_vision
        self.openai_base_url = openai_base_url
        self.http_client  = httpx.Client(proxies=http_client_proxies) if http_client_proxies else None
        self.azure_kwargs = AzureConfig(**extra.get("azure_kwargs", {}))
