from datasets import load_dataset
import random
from typing import List, Dict, Any, Optional

def load_wikitext(split: str = "test", min_tokens: int = 50, max_samples: int = 100) -> List[str]:
    dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)
    paragraphs = []
    for example in dataset:
        text = example["text"].strip()
        if len(text.split()) >= min_tokens and text:
            paragraphs.append(text)
            if len(paragraphs) >= max_samples:
                break
    return paragraphs

def load_humaneval(split: str = "test", max_samples: int = 50) -> List[str]:
    dataset = load_dataset("openai_humaneval", split=split)
    prompts = []
    for example in dataset:
        prompt = example["prompt"].strip()
        prompts.append(prompt)
        if len(prompts) >= max_samples:
            break
    return prompts

def load_gsm8k(split: str = "test", max_samples: int = 50) -> List[str]:
    dataset = load_dataset("gsm8k", "main", split=split)
    questions = []
    for example in dataset:
        question = example["question"].strip()
        questions.append(question)
        if len(questions) >= max_samples:
            break
    return questions

def load_multilingual_wikipedia(languages: List[str] = ["de", "fr", "zh", "ja", "es"], samples_per_lang: int = 10) -> List[str]:
    texts = []
    for lang in languages:
        try:
            dataset = load_dataset("wikipedia", f"20240301.{lang}", split="train", streaming=True)
            count = 0
            for example in dataset:
                text = example["text"].strip()
                if text:
                    texts.append(text)
                    count += 1
                    if count >= samples_per_lang:
                        break
        except Exception as e:
            print(f"Failed to load Wikipedia {lang}: {e}")
            continue
    return texts

def load_all_datasets() -> Dict[str, List[str]]:
    return {
        "wikitext": load_wikitext(),
        "humaneval": load_humaneval(),
        "gsm8k": load_gsm8k(),
        "multilingual": [],  # Wikipedia dataset scripts deprecated
    }

if __name__ == "__main__":
    data = load_all_datasets()
    for key, val in data.items():
        print(f"{key}: {len(val)} samples")