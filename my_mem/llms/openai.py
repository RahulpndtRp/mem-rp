from openai import OpenAI
from my_mem.llms.base import BaseLlmConfig
import os
class OpenAILLM:
    def __init__(self, cfg: BaseLlmConfig):
        self.cfg    = cfg
        self.client = OpenAI(
            api_key = cfg.api_key or os.getenv("OPENAI_API_KEY"),
            base_url= cfg.openai_base_url or "https://api.openai.com/v1",
        )

    # simple “generate_response” expected by Memory
    def generate_response(self, messages, response_format=None, tools=None):
        chat = self.client.chat.completions.create(
            model       = self.cfg.model,
            temperature = self.cfg.temperature,
            max_tokens  = self.cfg.max_tokens,
            top_p       = self.cfg.top_p,
            messages    = messages,
            tools       = tools,
            response_format = response_format,
        )
        return chat.choices[0].message.content or chat.choices[0].message.tool_calls[0].to_dict()
