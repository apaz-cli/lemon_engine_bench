from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

@dataclass
class TokenLogprob:
    token_id: int
    token_str: str
    logprob: float

@dataclass
class PositionLogprobs:
    generated_token: TokenLogprob
    top_logprobs: List[TokenLogprob]
    position: int

@dataclass
class GenerationResult:
    engine_name: str
    model_name: str
    prompt: str
    generated_text: str
    generated_token_ids: List[int]
    per_position_logprobs: List[PositionLogprobs]
    prompt_logprobs: Optional[List[PositionLogprobs]]
    wall_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class EngineAdapter(ABC):
    def __init__(self, engine_name: str, model_name: str, **kwargs):
        self.engine_name = engine_name
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    async def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 50,
        temperature: float = 0.0,
        **kwargs
    ) -> GenerationResult:
        pass
    
    @abstractmethod
    async def get_prompt_logprobs(
        self,
        text: str,
        top_logprobs: int = 50
    ) -> List[PositionLogprobs]:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass
    
    def _create_generation_result(
        self,
        prompt: str,
        generated_text: str,
        generated_token_ids: List[int],
        per_position_logprobs: List[PositionLogprobs],
        prompt_logprobs: Optional[List[PositionLogprobs]] = None,
        start_time: Optional[float] = None,
        **metadata
    ) -> GenerationResult:
        wall_time = time.time() - start_time if start_time else 0.0
        return GenerationResult(
            engine_name=self.engine_name,
            model_name=self.model_name,
            prompt=prompt,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            per_position_logprobs=per_position_logprobs,
            prompt_logprobs=prompt_logprobs,
            wall_time_seconds=wall_time,
            metadata=metadata,
        )