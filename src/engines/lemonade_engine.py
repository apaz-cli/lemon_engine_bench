import aiohttp
import time
import json
import asyncio
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer
from .base import EngineAdapter, GenerationResult, PositionLogprobs, TokenLogprob
from .token_utils import TokenMapper

class LemonadeAdapter(EngineAdapter):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/api/v1", tokenizer_name: Optional[str] = None, model_id: Optional[str] = None):
        super().__init__("lemonade", model_name)
        self.base_url = base_url.rstrip("/")
        self.session = None
        self.tokenizer = None
        self.tokenizer_name = tokenizer_name
        self._logprobs_supported = None
        self.model_id = model_id or model_name
        self.token_mapper = None
    
    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def _request_with_retry(self, method: str, url: str, max_retries: int = 3, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                await self._ensure_session()
                async with self.session.request(method, url, **kwargs) as resp:
                    resp.raise_for_status()
                    return resp
            except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # exponential backoff
                print(f"Lemonade request failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                # Recreate session if closed
                if self.session and self.session.closed:
                    self.session = None
        raise RuntimeError("Should not reach here")
    
    async def _post_json_with_retry(self, url: str, max_retries: int = 3, **kwargs) -> dict:
        """POST request with JSON payload and retry, returns parsed JSON."""
        for attempt in range(max_retries):
            try:
                await self._ensure_session()
                async with self.session.post(url, **kwargs) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data
            except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError, aiohttp.ClientPayloadError) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"Lemonade POST failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                if self.session and self.session.closed:
                    self.session = None
        raise RuntimeError("Should not reach here")
    
    async def _ensure_tokenizer(self):
        if self.tokenizer is None and self.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.token_mapper = TokenMapper(tokenizer=self.tokenizer)
    
    async def _probe_logprobs(self) -> bool:
        if self._logprobs_supported is not None:
            return self._logprobs_supported
        await self._ensure_session()
        payload = {
            "model": self.model_id,
            "prompt": "test",
            "max_tokens": 1,
            "logprobs": 5,
            "temperature": 0.0,
            "stream": False,
        }
        url = f"{self.base_url}/completions"
        try:
            data = await self._post_json_with_retry(url, json=payload, timeout=5)
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "logprobs" in choice:
                    self._logprobs_supported = True
                    return True
        except Exception:
            pass
        self._logprobs_supported = False
        return False
    
    async def _apply_chat_template(self, prompt: str) -> str:
        # Disable chat template for completions endpoint
        return prompt
    
    def _parse_lemonade_logprobs(self, logprobs_dict: Dict[str, Any]) -> tuple[List[int], List[PositionLogprobs]]:
        """Parse Lemonade's logprobs format (content array)."""
        token_ids = []
        per_position_logprobs = []
        
        if not logprobs_dict or "content" not in logprobs_dict:
            return token_ids, per_position_logprobs
        
        for i, item in enumerate(logprobs_dict["content"]):
            token_id = item["id"]
            logprob = item["logprob"]
            token_str = item["token"]
            
            # Map token ID using token_mapper if available
            mapped_id = token_id
            if self.token_mapper is not None:
                mapped_id = self.token_mapper.token_str_to_id(token_str)
                # If mapping fails, fall back to original ID
                if mapped_id == -1:
                    mapped_id = token_id
            
            top_list = []
            if "top_logprobs" in item and item["top_logprobs"]:
                for top_item in item["top_logprobs"]:
                    top_token_str = top_item["token"]
                    top_token_id = top_item["id"]
                    if self.token_mapper is not None:
                        mapped_top_id = self.token_mapper.token_str_to_id(top_token_str)
                        if mapped_top_id != -1:
                            top_token_id = mapped_top_id
                    top_list.append(TokenLogprob(
                        token_id=top_token_id,
                        token_str=top_token_str,
                        logprob=top_item["logprob"]
                    ))
            
            token_ids.append(mapped_id)
            per_position_logprobs.append(PositionLogprobs(
                generated_token=TokenLogprob(mapped_id, token_str, logprob),
                top_logprobs=top_list,
                position=i
            ))
        
        return token_ids, per_position_logprobs
    
    async def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 50,
        temperature: float = 0.0,
        **kwargs
    ) -> GenerationResult:
        await self._ensure_session()
        start_time = time.time()
        
        # Apply chat template if needed
        processed_prompt = await self._apply_chat_template(prompt)
        
        payload = {
            "model": self.model_id,
            "prompt": processed_prompt,
            "max_tokens": max_tokens,
            "logprobs": top_logprobs,
            "temperature": temperature,
            "stream": False,
            **kwargs,
        }
        
        url = f"{self.base_url}/completions"
        data = await self._post_json_with_retry(url, json=payload)
        
        choice = data["choices"][0]
        generated_text = choice["text"]
        
        per_position_logprobs = []
        token_ids = []
        if "logprobs" in choice and choice["logprobs"]:
            await self._ensure_tokenizer()
            token_ids, per_position_logprobs = self._parse_lemonade_logprobs(choice["logprobs"])
        else:
            # Fallback: no logprobs, just tokenize generated text
            await self._ensure_tokenizer()
            if self.tokenizer is None:
                # Cannot tokenize, return empty token IDs and placeholder logprobs
                print(f"WARNING: No logprobs and no tokenizer for model {self.model_id}")
                # Create placeholder token IDs and logprobs based on generated text length? Skip.
                # We'll assign token IDs as -1 and logprob -999
                # This is a fallback for emergency; should not happen in normal operation
                token_ids = [-1] * len(generated_text.split())  # crude approximation
                for i, token_id in enumerate(token_ids):
                    per_position_logprobs.append(PositionLogprobs(
                        generated_token=TokenLogprob(token_id, f"token_{i}", -999.0),
                        top_logprobs=[],
                        position=i
                    ))
            else:
                token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
                for i, token_id in enumerate(token_ids):
                    token_str = self.tokenizer.decode([token_id])
                    per_position_logprobs.append(PositionLogprobs(
                        generated_token=TokenLogprob(token_id, token_str, -999.0),
                        top_logprobs=[],
                        position=i
                    ))
        
        prompt_logprobs = await self.get_prompt_logprobs(prompt, top_logprobs)
        
        return self._create_generation_result(
            prompt=prompt,
            generated_text=generated_text,
            generated_token_ids=token_ids,
            per_position_logprobs=per_position_logprobs,
            prompt_logprobs=prompt_logprobs,
            start_time=start_time,
            logprobs_supported=self._logprobs_supported,
        )
    
    async def get_prompt_logprobs(self, text: str, top_logprobs: int = 50) -> List[PositionLogprobs]:
        
        await self._ensure_session()
        processed_prompt = await self._apply_chat_template(text)
        
        payload = {
            "model": self.model_id,
            "prompt": processed_prompt,
            "max_tokens": 0,
            "logprobs": top_logprobs,
            "echo": True,
            "stream": False,
        }
        
        url = f"{self.base_url}/completions"
        data = await self._post_json_with_retry(url, json=payload)
        
        choice = data["choices"][0]
        if "logprobs" not in choice or not choice["logprobs"]:
            return []
        await self._ensure_tokenizer()
        token_ids, result = self._parse_lemonade_logprobs(choice["logprobs"])
        return result
    
    async def health_check(self) -> bool:
        try:
            resp = await self._request_with_retry("GET", f"{self.base_url}/models", max_retries=1)
            return resp.status == 200
        except Exception:
            return False
    
    async def close(self):
        if self.session:
            await self.session.close()