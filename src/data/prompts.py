from transformers import AutoTokenizer
from typing import List, Optional

def apply_chat_template(model_name: str, messages: List[dict]) -> str:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    # Fallback: concatenate messages
    text = ""
    for msg in messages:
        text += f"{msg['role']}: {msg['content']}\n"
    return text.strip()

def prepare_prompt(model_name: str, text: str, task: Optional[str] = None) -> str:
    if task == "humaneval":
        # Code completion: no special formatting
        return text
    elif task == "gsm8k":
        messages = [{"role": "user", "content": f"Solve the following math problem: {text}"}]
    elif task == "multilingual":
        messages = [{"role": "user", "content": text}]
    else:
        # General text continuation
        messages = [{"role": "user", "content": text}]
    
    return apply_chat_template(model_name, messages)

def prepare_prefixes(full_text: str, fractions: List[float] = [0.25, 0.5, 0.75]) -> List[str]:
    tokens = full_text.split()
    prefixes = []
    for frac in fractions:
        length = int(len(tokens) * frac)
        prefix = " ".join(tokens[:length])
        prefixes.append(prefix)
    return prefixes

if __name__ == "__main__":
    sample = "This is a sample text with multiple words."
    print(prepare_prefixes(sample))