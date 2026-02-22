"""vLLM engine adapter.

Uses the OpenAI-compatible /v1/completions endpoint. The logprobs response
uses the "content" array format (same as Lemonade), not the legacy
token_ids/token_logprobs lists from older vLLM versions.
"""
import aiohttp
import os
import time
from typing import List, Optional
from .base import EngineAdapter, GenerationResult, PositionLogprobs, TokenLogprob
from .token_utils import TokenMapper
from transformers import AutoTokenizer


def _parse_content_logprobs(content: list, token_mapper: Optional[TokenMapper]) -> tuple:
    """Parse the 'content' array format returned by vLLM and Lemonade.

    Returns (token_ids, per_position_logprobs).
    """
    token_ids = []
    per_position_logprobs = []

    for i, item in enumerate(content):
        token_str = item["token"]
        logprob = item["logprob"]
        raw_id = item["id"]

        mapped_id = raw_id
        if token_mapper is not None:
            mid = token_mapper.token_str_to_id(token_str)
            if mid != -1:
                mapped_id = mid

        top_list = []
        for top in item.get("top_logprobs", []):
            top_str = top["token"]
            top_id = top["id"]
            if token_mapper is not None:
                mid = token_mapper.token_str_to_id(top_str)
                if mid != -1:
                    top_id = mid
            top_list.append(TokenLogprob(token_id=top_id, token_str=top_str, logprob=top["logprob"]))

        token_ids.append(mapped_id)
        per_position_logprobs.append(PositionLogprobs(
            generated_token=TokenLogprob(mapped_id, token_str, logprob),
            top_logprobs=top_list,
            position=i,
        ))

    return token_ids, per_position_logprobs


class VLLMAdapter(EngineAdapter):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8001/v1",
        tokenizer_name: Optional[str] = None,
    ):
        super().__init__("vllm", model_name)
        self.base_url = base_url.rstrip("/")
        self.session = None
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        self.token_mapper = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def _ensure_tokenizer(self):
        if self.tokenizer is None and self.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.token_mapper = TokenMapper(tokenizer=self.tokenizer)

    async def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 50,
        temperature: float = 0.0,
        **kwargs,
    ) -> GenerationResult:
        await self._ensure_session()
        await self._ensure_tokenizer()
        start_time = time.time()

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "logprobs": top_logprobs,
            "temperature": temperature,
            **kwargs,
        }

        url = f"{self.base_url}/completions"
        async with self.session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        choice = data["choices"][0]
        generated_text = choice["text"]

        content = choice.get("logprobs", {}).get("content", [])
        token_ids, per_position_logprobs = _parse_content_logprobs(content, self.token_mapper)

        # Prompt logprobs: attempt, but silently skip if unsupported.
        try:
            prompt_logprobs = await self.get_prompt_logprobs(prompt, top_logprobs)
        except Exception:
            prompt_logprobs = []

        return self._create_generation_result(
            prompt=prompt,
            generated_text=generated_text,
            generated_token_ids=token_ids,
            per_position_logprobs=per_position_logprobs,
            prompt_logprobs=prompt_logprobs,
            start_time=start_time,
        )

    async def get_prompt_logprobs(self, text: str, top_logprobs: int = 50) -> List[PositionLogprobs]:
        """Attempt to retrieve per-prompt-token logprobs.

        vLLM with llama.cpp backend may not correctly support echo=True for
        all prompt tokens; returns [] if the response is empty or malformed.
        """
        await self._ensure_session()
        await self._ensure_tokenizer()

        payload = {
            "model": self.model_name,
            "prompt": text,
            "max_tokens": 0,
            "logprobs": top_logprobs,
            "echo": True,
        }

        url = f"{self.base_url}/completions"
        async with self.session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        content = data["choices"][0].get("logprobs", {}).get("content", [])
        if not content:
            return []
        _, result = _parse_content_logprobs(content, self.token_mapper)
        return result

    async def health_check(self) -> bool:
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.base_url}/models") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def close(self):
        if self.session:
            await self.session.close()
