#!/usr/bin/env python3
"""
Example usage of the WER evaluation tool.

This script demonstrates how to use the WEREvaluator class programmatically.
"""

from wer_evaluator import WEREvaluator


def example_api_evaluation():
    """Example: Evaluate using an API endpoint."""
    print("Example 1: API-based evaluation")
    print("=" * 60)
    
    evaluator = WEREvaluator(
        api_url="http://localhost:8002",
        language="de",
        batch_size=1
    )
    
    results = evaluator.evaluate(
        dataset_name="NeurologyAI/neuro-whisper-v1",
        split="validation",
        limit=10,  # Limit for quick testing
        output_file="example_results_api.json"
    )
    
    print(f"\nWER: {results['wer_percent']:.2f}%")
    return results


def example_model_evaluation():
    """Example: Evaluate using a HuggingFace model directly."""
    print("\nExample 2: Direct model evaluation")
    print("=" * 60)
    
    evaluator = WEREvaluator(
        model="NeurologyAI/neuro-parakeet",
        language="de",
        batch_size=1
    )
    
    results = evaluator.evaluate(
        dataset_name="NeurologyAI/neuro-whisper-v1",
        split="validation",
        limit=10,  # Limit for quick testing
        output_file="example_results_model.json"
    )
    
    print(f"\nWER: {results['wer_percent']:.2f}%")
    return results


if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # Example 1: API-based evaluation
    # example_api_evaluation()
    
    # Example 2: Direct model evaluation
    # example_model_evaluation()
    
    print("Uncomment one of the examples above to run it.")
    print("\nNote: Make sure you have:")
    print("  1. An ASR server running (for API evaluation), OR")
    print("  2. A HuggingFace model available (for direct evaluation)")

