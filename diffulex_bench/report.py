"""
Benchmark Report - Report generation for benchmark results
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


def generate_report(results_file: str, output_file: Optional[str] = None) -> str:
    """
    Generate benchmark report
    
    Args:
        results_file: Path to results JSON file
        output_file: Path to output report file, if None prints to console
        
    Returns:
        Report text
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    config = results['config']
    metrics = results['metrics']
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Diffulex Benchmark Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Configuration:")
    report_lines.append(f"  Model: {config.get('model_path', 'N/A')}")
    report_lines.append(f"  Model Name: {config.get('model_name', 'N/A')}")
    report_lines.append(f"  Decoding Strategy: {config.get('decoding_strategy', 'N/A')}")
    report_lines.append(f"  Dataset: {config.get('dataset_name', 'N/A')}")
    report_lines.append(f"  Tensor Parallel Size: {config.get('tensor_parallel_size', 'N/A')}")
    report_lines.append(f"  Data Parallel Size: {config.get('data_parallel_size', 'N/A')}")
    report_lines.append("")
    report_lines.append("Metrics:")
    report_lines.append(f"  Number of Samples: {metrics.get('num_samples', 'N/A')}")
    report_lines.append(f"  Total Tokens: {metrics.get('total_tokens', 'N/A')}")
    report_lines.append(f"  Average Tokens per Sample: {metrics.get('avg_tokens_per_sample', 0):.2f}")
    report_lines.append(f"  Average Diffusion Steps: {metrics.get('avg_diff_steps', 0):.2f}")
    report_lines.append(f"  Total Time: {metrics.get('total_time', 0):.2f} seconds")
    report_lines.append(f"  Throughput: {metrics.get('throughput_tok_s', 0):.2f} tokens/s")
    
    if 'accuracy' in metrics and metrics['accuracy'] is not None:
        report_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    report_lines.append("")
    report_lines.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save or output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    else:
        print(report_text)
    
    return report_text


def compare_results(result_files: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Compare multiple benchmark results
    
    Args:
        result_files: List of result file paths
        output_file: Path to output CSV file, if None only returns DataFrame
        
    Returns:
        DataFrame with comparison results
    """
    rows = []
    
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        config = results['config']
        metrics = results['metrics']
        
        row = {
            'model_path': config.get('model_path', 'N/A'),
            'model_name': config.get('model_name', 'N/A'),
            'decoding_strategy': config.get('decoding_strategy', 'N/A'),
            'dataset': config.get('dataset_name', 'N/A'),
            'num_samples': metrics.get('num_samples', 0),
            'total_tokens': metrics.get('total_tokens', 0),
            'avg_tokens_per_sample': metrics.get('avg_tokens_per_sample', 0),
            'avg_diff_steps': metrics.get('avg_diff_steps', 0),
            'throughput_tok_s': metrics.get('throughput_tok_s', 0),
            'accuracy': metrics.get('accuracy', None),
            'timestamp': results.get('timestamp', 'N/A'),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Comparison saved to: {output_file}")
    
    return df

