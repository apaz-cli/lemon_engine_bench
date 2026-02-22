"""
Token utilities for consistent token ID mapping across engines.
Uses HuggingFace tokenizer as ground truth.
"""

from typing import Dict, Optional
from transformers import AutoTokenizer


class TokenMapper:
    """
    Maps token strings to consistent token IDs using a reference tokenizer.
    Handles normalization of token string representations (e.g., leading spaces to ▁ prefix).
    """
    def __init__(self, tokenizer_name: Optional[str] = None, tokenizer=None):
        if tokenizer is None and tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        self._cache: Dict[str, int] = {}
    
    def normalize_token_str(self, token_str: str) -> str:
        """
        Normalize token string to match HuggingFace tokenizer's representation.
        Converts leading spaces to ▁ prefix, handles special cases.
        """
        if token_str == ' ':
            return '▁'
        if token_str.startswith(' '):
            stripped = token_str.lstrip()
            if stripped == '':
                return token_str
            token_str = '▁' + stripped
        if token_str.startswith('▁'):
            # Already normalized
            pass
        # Strip trailing spaces (should not happen)
        token_str = token_str.rstrip()
        return token_str
    
    def token_str_to_id(self, token_str: str) -> int:
        """
        Map token string to token ID using reference tokenizer's vocabulary.
        Returns -1 if mapping fails.
        """
        if token_str in self._cache:
            return self._cache[token_str]
        
        if self.tokenizer is None:
            return -1
        
        normalized = self.normalize_token_str(token_str)
        vocab = self.tokenizer.get_vocab()
        if normalized in vocab:
            token_id = vocab[normalized]
            self._cache[token_str] = token_id
            return token_id
        
        # Fallback: try to tokenize the string (may produce multiple tokens)
        try:
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                # Use first token ID
                token_id = token_ids[0]
                self._cache[token_str] = token_id
                return token_id
        except:
            pass
        
        # Ultimate fallback
        self._cache[token_str] = -1
        return -1
    
    def batch_token_str_to_id(self, token_strings: list[str]) -> list[int]:
        """Map multiple token strings to IDs."""
        return [self.token_str_to_id(s) for s in token_strings]
    
    def token_id_to_str(self, token_id: int) -> str:
        """Convert token ID to string using reference tokenizer."""
        if self.tokenizer is None:
            return f"token_{token_id}"
        try:
            return self.tokenizer.decode([token_id])
        except:
            return f"token_{token_id}"