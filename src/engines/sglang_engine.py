import aiohttp
import time
from typing import List, Optional
from .base import EngineAdapter, GenerationResult, PositionLogprobs, TokenLogprob

class SGLangAdapter(EngineAdapter):
    def __init__(self, model_name: str, base_url: str = "http://localhost:8002"):
        super().__init__("sglang", model_name)
        self.base_url = base_url.rstrip("/")
        self.session = None
    
    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
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
        
        payload = {
            "text": prompt,
            "max_tokens": max_tokens,
            "return_logprob": True,
            "top_logprobs": top_logprobs,
            "temperature": temperature,
            **kwargs,
        }
        
        url = f"{self.base_url}/generate"
        async with self.session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
        
        generated_text = data["text"]
        token_ids = data.get("token_ids", [])
        token_logprobs = data.get("logprobs", [])
        top_logprobs_list = data.get("top_logprobs", [])
        
        per_position_logprobs = []
        for i, (token_id, logprob) in enumerate(zip(token_ids, token_logprobs)):
            token_str = data["tokens"][i]
            top_list = []
            if i < len(top_logprobs_list) and top_logprobs_list[i]:
                for top_item in top_logprobs_list[i]:
                    top_list.append(TokenLogprob(
                        token_id=top_item["token_id"],
                        token_str=top_item["token_str"],
                        logprob=top_item["logprob"]
                    ))
            
            per_position_logprobs.append(PositionLogprobs(
                generated_token=TokenLogprob(token_id, token_str, logprob),
                top_logprobs=top_list,
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
        )
    
    async def get_prompt_logprobs(self, text: str, top_logprobs: int = 50) -> List[PositionLogprobs]:
        await self._ensure_session()
        
        payload = {
            "text": text,
            "max_tokens": 0,
            "return_logprob": True,
            "top_logprobs": top_logprobs,
            "logprob_start_len": 0,
        }
        
        url = f"{self.base_url}/generate"
        async with self.session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
        
        token_ids = data.get("token_ids", [])
        token_logprobs = data.get("logprobs", [])
        top_logprobs_list = data.get("top_logprobs", [])
        
        result = []
        for i, (token_id, logprob) in enumerate(zip(token_ids, token_logprobs)):
            token_str = data["tokens"][i]
            top_list = []
            if i < len(top_logprobs_list) and top_logprobs_list[i]:
                for top_item in top_logprobs_list[i]:
                    top_list.append(TokenLogprob(
                        token_id=top_item["token_id"],
                        token_str=top_item["token_str"],
                        logprob=top_item["logprob"]
                    ))
            
            result.append(PositionLogprobs(
                generated_token=TokenLogprob(token_id, token_str, logprob),
                top_logprobs=top_list,
                position=i
            ))
        return result
    
    async def health_check(self) -> bool:
        try:
            await self._ensure_session()
            url = f"{self.base_url}/health"
            async with self.session.get(url) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    async def close(self):
        if self.session:
            await self.session.close()