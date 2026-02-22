import time
import llama_cpp
from typing import List, Optional, Dict
from transformers import AutoTokenizer
from .base import EngineAdapter, GenerationResult, PositionLogprobs, TokenLogprob

class LlamaCppAdapter(EngineAdapter):
    def __init__(self, model_name: str, model_path: Optional[str] = None, n_gpu_layers: int = -1, tokenizer_name: Optional[str] = None):
        super().__init__("llama.cpp", model_name)
        self.model_path = model_path or model_name
        self.n_gpu_layers = n_gpu_layers
        self.tokenizer_name = tokenizer_name
        self.llm = None
        self.tokenizer = None
        self._token_str_to_id_cache: Dict[str, int] = {}
    
    async def _ensure_loaded(self):
        if self.llm is None:
            self.llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                logits_all=True,  # needed for prompt logprobs
                n_ctx=512,
            )
    
    async def _ensure_tokenizer(self):
        if self.tokenizer is None and self.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _normalize_token_str(self, token_str: str) -> str:
        if token_str == ' ':
            return '▁'
        if token_str.startswith(' '):
            stripped = token_str.lstrip()
            if stripped == '':
                return token_str
            return '▁' + stripped
        if token_str.startswith('▁'):
            return token_str
        return token_str
    
    def _token_str_to_id(self, token_str: str) -> int:
        if token_str in self._token_str_to_id_cache:
            return self._token_str_to_id_cache[token_str]
        normalized = self._normalize_token_str(token_str)
        if self.tokenizer is not None:
            vocab = self.tokenizer.get_vocab()
            if normalized in vocab:
                token_id = vocab[normalized]
                self._token_str_to_id_cache[token_str] = token_id
                # DEBUG
                # print(f"DEBUG _token_str_to_id: '{token_str}' -> '{normalized}' -> {token_id}")
                return token_id
        # Fallback: try to tokenize with llama.cpp (should not happen)
        if self.llm and hasattr(self.llm, 'tokenize'):
            try:
                token_bytes = token_str.encode('utf-8')
                token_ids = self.llm.tokenize(token_bytes, add_bos=False)
                if token_ids:
                    token_id = token_ids[0]
                    self._token_str_to_id_cache[token_str] = token_id
                    return token_id
            except:
                pass
        # Ultimate fallback: -1
        return -1
    
    async def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 1,
        top_logprobs: int = 50,
        temperature: float = 0.0,
        **kwargs
    ) -> GenerationResult:
        await self._ensure_loaded()
        await self._ensure_tokenizer()
        start_time = time.time()
        
        result = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=top_logprobs if top_logprobs > 0 else None,
            echo=False,
            **kwargs,
        )
        
        choice = result["choices"][0]

        generated_text = choice["text"]
        
        # Parse logprobs if available
        per_position_logprobs = []
        generated_token_ids = []
        
        if "logprobs" in choice and choice["logprobs"]:
            logprobs_data = choice["logprobs"]
            tokens = logprobs_data.get("tokens", [])
            token_logprobs = logprobs_data.get("token_logprobs", [])
            top_logprobs_list = logprobs_data.get("top_logprobs", [])
            
            # Debug
            # print(f"llama.cpp debug: tokens={len(tokens)}, token_logprobs={len(token_logprobs)}, top_logprobs_list={len(top_logprobs_list)}")
            
            # Handle case where token_logprobs length doesn't match tokens
            if len(token_logprobs) != len(tokens):
                # If token_logprobs empty but tokens present, create placeholders
                if tokens and not token_logprobs:
                    token_logprobs = [None] * len(tokens)
                else:
                    # Truncate to min length
                    min_len = min(len(tokens), len(token_logprobs))
                    tokens = tokens[:min_len]
                    token_logprobs = token_logprobs[:min_len]
            

            for i, token_str in enumerate(tokens):
                logprob_val = token_logprobs[i] if i < len(token_logprobs) else None
                if logprob_val is None:
                    logprob = -1000.0
                else:
                    logprob = float(logprob_val)
                
                token_id = self._token_str_to_id(token_str)
                
                top_list = []
                if i < len(top_logprobs_list) and top_logprobs_list[i]:
                    for top_token_str, top_logprob_val in top_logprobs_list[i].items():
                        if top_logprob_val is None:
                            continue
                        top_logprob = float(top_logprob_val)
                        top_token_id = self._token_str_to_id(top_token_str)
                        top_list.append(TokenLogprob(
                            token_id=top_token_id,
                            token_str=top_token_str,
                            logprob=top_logprob
                        ))
                
                per_position_logprobs.append(PositionLogprobs(
                    generated_token=TokenLogprob(token_id, token_str, logprob),
                    top_logprobs=top_list,
                    position=i
                ))
                generated_token_ids.append(token_id)
        
        # print(f"DEBUG after loop: generated_token_ids={generated_token_ids}, len={len(generated_token_ids)}")
        # If no logprobs or empty tokens, fallback to tokenizing generated_text

        if not generated_token_ids and generated_text:
            # Try to tokenize using reference tokenizer first (matches HF token IDs)
            if self.tokenizer is not None:
                try:
                    token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
                    for i, tid in enumerate(token_ids):
                        token_str = self.tokenizer.decode([tid])
                        per_position_logprobs.append(PositionLogprobs(
                            generated_token=TokenLogprob(tid, token_str, -999.0),
                            top_logprobs=[],
                            position=i
                        ))
                        generated_token_ids.append(tid)
                except Exception as e:
                    # Fallback to llama.cpp tokenize
                    pass
            # If tokenizer not available or failed, use llama.cpp tokenize
            if not generated_token_ids and hasattr(self.llm, 'tokenize'):
                try:
                    token_ids = self.llm.tokenize(generated_text.encode('utf-8'), add_bos=False)
                    # Decode each token ID to string
                    for i, tid in enumerate(token_ids):
                        # Try to detokenize
                        token_str = ""
                        if hasattr(self.llm, 'detokenize'):
                            try:
                                token_bytes = self.llm.detokenize([tid])
                                token_str = token_bytes.decode('utf-8', errors='replace')
                            except:
                                token_str = f"token_{tid}"
                        else:
                            token_str = f"token_{tid}"
                        per_position_logprobs.append(PositionLogprobs(
                            generated_token=TokenLogprob(tid, token_str, -999.0),
                            top_logprobs=[],
                            position=i
                        ))
                        generated_token_ids.append(tid)
                except Exception as e:
                    # print(f"Warning: Failed to tokenize generated text: {e}")
                    # Last resort: assign placeholder IDs
                    generated_token_ids = [-1] * len(generated_text.split())
                    for i in range(len(generated_token_ids)):
                        per_position_logprobs.append(PositionLogprobs(
                            generated_token=TokenLogprob(-1, f"token_{i}", -999.0),
                            top_logprobs=[],
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
        await self._ensure_loaded()
        await self._ensure_tokenizer()
        
        # Use echo=True to get prompt logprobs
        result = self.llm(
            prompt=text,
            max_tokens=0,
            logprobs=top_logprobs if top_logprobs > 0 else None,
            echo=True,
        )
        
        choice = result["choices"][0]
        if "logprobs" not in choice or not choice["logprobs"]:
            return []
        
        logprobs_data = choice["logprobs"]
        tokens = logprobs_data.get("tokens", [])
        token_logprobs = logprobs_data.get("token_logprobs", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])
        
        result_list = []
        for i, (token_str, logprob_str) in enumerate(zip(tokens, token_logprobs)):
            if logprob_str is None:
                logprob = -1000.0  # placeholder for None logprob
            else:
                logprob = float(logprob_str)
            token_id = self._token_str_to_id(token_str)
            
            top_list = []
            if i < len(top_logprobs_list) and top_logprobs_list[i]:
                for top_token_str, top_logprob_str in top_logprobs_list[i].items():
                    if top_logprob_str is None:
                        continue
                    top_logprob = float(top_logprob_str)
                    top_token_id = self._token_str_to_id(top_token_str)
                    top_list.append(TokenLogprob(
                        token_id=top_token_id,
                        token_str=top_token_str,
                        logprob=top_logprob
                    ))
            
            result_list.append(PositionLogprobs(
                generated_token=TokenLogprob(token_id, token_str, logprob),
                top_logprobs=top_list,
                position=i
            ))
        return result_list
    
    async def health_check(self) -> bool:
        try:
            await self._ensure_loaded()
            return self.llm is not None
        except Exception:
            return False
    
    async def close(self):
        if self.llm:
            self.llm.close()
            self.llm = None