"""
Benchmark Runner - Benchmark runner that wraps Diffulex inference engine
Provides a unified interface for benchmarking
"""

import time
from typing import List, Dict, Any, Optional

from diffulex import Diffulex, SamplingParams
from transformers import AutoTokenizer
from diffulex.logger import get_logger


class BenchmarkRunner:
    """
    Benchmark runner that wraps the Diffulex inference engine
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        wait_ready: bool = True,
        **diffulex_kwargs
    ):
        """
        Initialize the benchmark runner
        
        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer, if None uses model_path
            wait_ready: Whether to wait for engine to be fully initialized before returning
            **diffulex_kwargs: Additional arguments to pass to Diffulex
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.logger = get_logger(__name__)
        
        # Initialize Diffulex engine
        self.logger.info("Initializing Diffulex engine...")
        self.llm = Diffulex(model_path, **diffulex_kwargs)
        
        # Wait for engine to be ready if requested
        if wait_ready:
            self._wait_for_ready()
        
        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True
        )
        self.logger.success("Tokenizer loaded successfully")
    
    def _wait_for_ready(self, timeout: float = 300.0, check_interval: float = 0.5):
        """
        Wait for the Diffulex engine to be fully initialized and ready
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Interval between readiness checks in seconds
        """
        start_time = time.time()
        
        # Check if it's a DP worker (has _ask method) or TP worker
        if hasattr(self.llm, '_ask'):
            # DP worker: wait for all child processes to be ready
            # by sending a lightweight command to each
            dp_size = getattr(self.llm, 'dp_size', 1)
            self.logger.info(f"[DiffulexDPWorker (DP={dp_size})]: Waiting for {dp_size} DiffulexTPWorker subprocesses to be ready...")
            
            while time.time() - start_time < timeout:
                try:
                    # Try to send a lightweight command to check readiness
                    # Use is_finished as a lightweight check
                    for i in range(dp_size):
                        self.llm._ask(i, "is_finished")
                    self.logger.success("All DiffulexTPWorker subprocesses are ready")
                    return
                except (EOFError, RuntimeError, AttributeError, ConnectionError) as e:
                    # Process not ready yet, wait and retry
                    elapsed = time.time() - start_time
                    if elapsed < timeout:
                        time.sleep(check_interval)
                    else:
                        raise RuntimeError(
                            f"Timeout waiting for DP workers to be ready after {elapsed:.1f}s: {e}"
                        ) from e
        else:
            # TP worker: wait for all subprocesses to be ready
            # Check if subprocesses are alive and wait a bit for initialization
            if hasattr(self.llm, 'ps') and self.llm.ps:
                num_subprocesses = len(self.llm.ps)
                self.logger.info(f"Waiting for {num_subprocesses} TP subprocess(es) to be ready...")
                
                while time.time() - start_time < timeout:
                    # Check if all subprocesses are alive
                    all_alive = all(p.is_alive() for p in self.llm.ps)
                    
                    if all_alive:
                        # Give subprocesses a bit more time to complete initialization
                        # The main process initialization is synchronous, but subprocesses
                        # may still be initializing (model loading, warmup, etc.)
                        # Subprocesses will synchronize via barrier in ModelRunnerBase.__init__
                        # So we just need to wait a bit for them to complete initialization
                        time.sleep(2.0)  # Wait a bit for subprocess initialization
                        self.logger.success("All TP subprocesses are ready")
                        return
                    else:
                        # Some process died, check which one
                        dead_processes = [
                            i for i, p in enumerate(self.llm.ps) if not p.is_alive()
                        ]
                        exit_codes = [
                            self.llm.ps[i].exitcode for i in dead_processes
                        ]
                        raise RuntimeError(
                            f"TP subprocess(es) {dead_processes} terminated during initialization. "
                            f"Exit code(s): {exit_codes}"
                        )
                
                elapsed = time.time() - start_time
                raise RuntimeError(
                    f"Timeout waiting for TP subprocesses to be ready after {elapsed:.1f}s"
                )
            else:
                # Single process TP worker, should be ready immediately
                # Main process initialization is synchronous
                self.logger.success("TP worker is ready")
                return
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate text
        
        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of generation results, each containing text, token_ids, n_diff_steps
        """
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        end_time = time.time()
        
        # Add timing information
        total_time = end_time - start_time
        for output in outputs:
            output['generation_time'] = total_time / len(outputs) if outputs else 0
        
        return outputs
    
    def evaluate_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        use_tqdm: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of prompts
        
        Args:
            prompts: List of input prompts
            sampling_params: Sampling parameters
            use_tqdm: Whether to show progress bar
            
        Returns:
            Evaluation result dictionary containing generation results and statistics
        """
        outputs = self.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        
        # Calculate statistics
        total_tokens = sum(len(o['token_ids']) for o in outputs)
        total_time = sum(o.get('generation_time', 0) for o in outputs)
        avg_diff_steps = sum(o.get('n_diff_steps', 0) for o in outputs) / len(outputs) if outputs else 0
        
        return {
            'outputs': outputs,
            'num_samples': len(outputs),
            'total_tokens': total_tokens,
            'total_time': total_time,
            'avg_tokens_per_sample': total_tokens / len(outputs) if outputs else 0,
            'avg_diff_steps': avg_diff_steps,
            'throughput_tok_s': total_tokens / total_time if total_time > 0 else 0,
        }

