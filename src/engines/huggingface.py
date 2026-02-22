import time
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import EngineAdapter, GenerationResult, PositionLogprobs, TokenLogprob

class HuggingFaceAdapter(EngineAdapter):
    def __init__(self, model_name: str, model_path: Optional[str] = None, dtype=torch.float16, device=None):
        super().__init__("huggingface", model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.model_path = model_path or model_name
        self.tokenizer = None
        self.model = None
    
    async def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.device == "cpu":
            # For CPU, don't use device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            self.model = self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
        self.model.eval()
    
    async def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 50,
        temperature: float = 0.0,
        **kwargs
    ) -> GenerationResult:
        if self.model is None:
            await self.load()
        
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                output_logits=True,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                **kwargs
            )
        
        generated_token_ids = outputs.sequences[0, input_len:].tolist()
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        per_position_logprobs = []
        for i, logits in enumerate(outputs.logits):
            logits = logits[0].float()
            if temperature > 0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            logprobs = torch.log(probs + 1e-12)
            topk_values, topk_indices = torch.topk(logprobs, k=top_logprobs)
            
            generated_token_id = generated_token_ids[i]
            generated_token_str = self.tokenizer.decode([generated_token_id])
            generated_logprob = logprobs[generated_token_id].item()
            
            top_logprobs_list = []
            for value, idx in zip(topk_values.tolist(), topk_indices.tolist()):
                token_str = self.tokenizer.decode([idx])
                top_logprobs_list.append(TokenLogprob(idx, token_str, value))
            
            per_position_logprobs.append(PositionLogprobs(
                generated_token=TokenLogprob(generated_token_id, generated_token_str, generated_logprob),
                top_logprobs=top_logprobs_list,
                position=i
            ))
        
        prompt_logprobs = await self.get_prompt_logprobs(prompt, top_logprobs)
        
        return self._create_generation_result(
            prompt=prompt,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            per_position_logprobs=per_position_logprobs,
            prompt_logprobs=prompt_logprobs,
            start_time=start_time,
        )
    
    async def get_prompt_logprobs(self, text: str, top_logprobs: int = 50) -> List[PositionLogprobs]:
        if self.model is None:
            await self.load()
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False, output_attentions=False)
            logits = outputs.logits[0]
        
        logprobs = torch.log_softmax(logits, dim=-1)
        token_ids = inputs.input_ids[0].tolist()
        result = []
        for i, token_id in enumerate(token_ids):
            token_str = self.tokenizer.decode([token_id])
            topk_values, topk_indices = torch.topk(logprobs[i], k=top_logprobs)
            top_list = []
            for val, idx in zip(topk_values.tolist(), topk_indices.tolist()):
                top_list.append(TokenLogprob(idx, self.tokenizer.decode([idx]), val))
            result.append(PositionLogprobs(
                generated_token=TokenLogprob(token_id, token_str, logprobs[i, token_id].item()),
                top_logprobs=top_list,
                position=i
            ))
        return result
    
    async def health_check(self) -> bool:
        try:
            if self.model is None:
                await self.load()
            return True
        except Exception:
            return False
    
    async def close(self):
        pass

