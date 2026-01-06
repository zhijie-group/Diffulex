"""
Benchmark Metrics - Evaluation metrics computation
"""

import re
from typing import List, Dict, Any, Optional
import json


def extract_number(text: str) -> Optional[float]:
    """
    Extract number from text (for GSM8K and other math problems)
    
    Args:
        text: Input text
        
    Returns:
        Extracted number, or None if not found
    """
    # Try to match #### number format (GSM8K standard format)
    pattern = r'####\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    
    # Try to match the last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None


def gsm8k_accuracy(
    predictions: List[str],
    ground_truths: List[str],
) -> float:
    """
    Calculate GSM8K accuracy
    
    Args:
        predictions: List of predicted texts
        ground_truths: List of ground truth answers (including full solution process)
        
    Returns:
        Accuracy (0-1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground_truths must have the same length")
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_num = extract_number(pred)
        gt_num = extract_number(gt)
        
        if pred_num is not None and gt_num is not None:
            if abs(pred_num - gt_num) < 1e-6:
                correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def humaneval_pass_at_k(
    results: List[Dict[str, Any]],
    k: int = 1,
) -> float:
    """
    Calculate HumanEval Pass@k metric
    
    Args:
        results: List of results, each should contain 'output', 'test', 'entry_point' fields
        k: k value, default 1
        
    Returns:
        Pass@k score
    """
    # Note: Full HumanEval evaluation requires code execution, this is just a framework
    # In practice, need to integrate code execution environment (e.g., Docker)
    # Returns None, actual evaluation requires implementing code execution logic
    return None


def compute_metrics(
    outputs: List[Dict[str, Any]],
    ground_truths: Optional[List[str]] = None,
    dataset_name: str = "gsm8k",
) -> Dict[str, Any]:
    """
    Compute evaluation metrics
    
    Args:
        outputs: List of generation results
        ground_truths: List of ground truth answers (optional)
        dataset_name: Dataset name, used to select appropriate evaluation method
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic statistics
    total_tokens = sum(len(o.get('token_ids', [])) for o in outputs)
    avg_diff_steps = sum(o.get('n_diff_steps', 0) for o in outputs) / len(outputs) if outputs else 0
    total_time = sum(o.get('generation_time', 0) for o in outputs)
    
    metrics['num_samples'] = len(outputs)
    metrics['total_tokens'] = total_tokens
    metrics['avg_tokens_per_sample'] = total_tokens / len(outputs) if outputs else 0
    metrics['avg_diff_steps'] = avg_diff_steps
    metrics['total_time'] = total_time
    metrics['throughput_tok_s'] = total_tokens / total_time if total_time > 0 else 0
    
    # Dataset-specific metrics
    if ground_truths and dataset_name == "gsm8k":
        predictions = [o.get('text', '') for o in outputs]
        metrics['accuracy'] = gsm8k_accuracy(predictions, ground_truths)
    elif ground_truths and dataset_name == "humaneval":
        # HumanEval requires code execution, this is just a framework
        metrics['pass_at_1'] = None  # Need to implement code execution logic
        metrics['note'] = "HumanEval evaluation requires code execution environment"
    
    return metrics

