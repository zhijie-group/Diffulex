"""
Benchmark Datasets - Dataset loaders for benchmark evaluation
Supports common evaluation datasets such as GSM8K, HumanEval, etc.
"""

from typing import List, Dict, Any, Optional, Callable
from datasets import load_dataset


def load_gsm8k(
    split: str = "test",
    limit: Optional[int] = None,
    prompt_template: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset
    
    Args:
        split: Dataset split, default "test"
        limit: Limit number of samples, None means all
        prompt_template: Prompt template function that takes question string and returns full prompt
        
    Returns:
        List of dataset items, each containing 'prompt' and 'answer' fields
    """
    dataset = load_dataset("gsm8k", "main", split=split)
    
    if limit:
        dataset = dataset[:limit]
    
    results = []
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        
        if prompt_template:
            prompt = prompt_template(question)
        else:
            # Default template
            prompt = f"Question: {question}\nAnswer:"
        
        results.append({
            'prompt': prompt,
            'answer': answer,
            'question': question,
        })
    
    return results


def load_humaneval(
    limit: Optional[int] = None,
    prompt_template: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load HumanEval dataset
    
    Args:
        limit: Limit number of samples, None means all
        prompt_template: Prompt template function that takes prompt string and returns full prompt
        
    Returns:
        List of dataset items, each containing 'prompt', 'test', 'entry_point' fields
    """
    dataset = load_dataset("openai/humaneval", split="test")
    
    if limit:
        dataset = dataset[:limit]
    
    results = []
    for item in dataset:
        prompt = item["prompt"]
        test = item["test"]
        entry_point = item["entry_point"]
        
        if prompt_template:
            full_prompt = prompt_template(prompt)
        else:
            full_prompt = prompt
        
        results.append({
            'prompt': full_prompt,
            'original_prompt': prompt,
            'test': test,
            'entry_point': entry_point,
            'task_id': item.get('task_id', ''),
        })
    
    return results


def load_benchmark_dataset(
    dataset_name: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Unified dataset loading interface
    
    Args:
        dataset_name: Dataset name, supports "gsm8k", "humaneval"
        **kwargs: Arguments passed to the specific dataset loader
        
    Returns:
        List of dataset items
    """
    loaders = {
        'gsm8k': load_gsm8k,
        'humaneval': load_humaneval,
    }
    
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(loaders.keys())}"
        )
    
    return loaders[dataset_name](**kwargs)

