#!/usr/bin/env python3
"""
Medical ASR Evaluator

A standalone tool for evaluating Automatic Speech Recognition (ASR) models,
particularly optimized for medical/clinical speech recognition, using Word Error Rate (WER) metric.
Supports evaluation via API endpoints or direct model inference.

Usage:
    # Evaluate via API endpoint
    python wer_evaluator.py --dataset NeurologyAI/neuro-whisper-v1 --split validation --api-url http://localhost:8002
    
    # Evaluate with HuggingFace model directly
    python wer_evaluator.py --dataset NeurologyAI/neuro-whisper-v1 --split validation --model NeurologyAI/neuro-parakeet
    
    # Evaluate with limit (for quick testing)
    python wer_evaluator.py --dataset NeurologyAI/neuro-whisper-v1 --split validation --limit 100 --output results.json
"""

import argparse
import io
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Suppress transformers warnings about missing ML frameworks when using API mode
warnings.filterwarnings("ignore", message=".*PyTorch.*")
warnings.filterwarnings("ignore", message=".*TensorFlow.*")
warnings.filterwarnings("ignore", message=".*Flax.*")

import evaluate
import librosa
import numpy as np
import requests
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def clean_text_for_wer(text: str) -> str:
    """
    Clean text for WER computation.
    
    Normalizes text by:
    - Converting to lowercase
    - Stripping whitespace
    - Removing trailing punctuation
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text string
    """
    if text is None:
        return ""
    return text.lower().strip().rstrip('.').rstrip(',').rstrip('!').rstrip('?').strip()


def decode_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decode audio from raw bytes using soundfile/librosa.
    
    Args:
        audio_bytes: Raw audio bytes
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buffer)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio, sr


def transcribe_audio_api(audio_array: np.ndarray, sample_rate: int, api_url: str, 
                         model: str = "parakeet-tdt-0.6b-v3", language: str = "de") -> str:
    """
    Send audio to an ASR API endpoint and get transcription.
    
    Args:
        audio_array: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        api_url: Base URL of the ASR API
        model: Model name to use
        language: Language code (default: "de" for German)
        
    Returns:
        Transcribed text
    """
    # Convert audio to bytes (WAV format)
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='WAV')
    buffer.seek(0)
    
    # Send to API
    endpoint = f"{api_url.rstrip('/')}/v1/audio/transcriptions"
    files = {
        'file': ('audio.wav', buffer, 'audio/wav')
    }
    data = {
        'model': model,
        'language': language,
        'response_format': 'json'
    }
    
    try:
        response = requests.post(endpoint, files=files, data=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get('text', '')
    except requests.exceptions.RequestException as e:
        print(f"\nAPI Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Response: {e.response.text}")
            except:
                pass
        return ""


def transcribe_audio_model(audio_array: np.ndarray, sample_rate: int, 
                           pipeline) -> str:
    """
    Transcribe audio using a HuggingFace pipeline directly.
    
    Args:
        audio_array: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        pipeline: HuggingFace ASR pipeline
        
    Returns:
        Transcribed text
    """
    try:
        # Convert to dict format expected by pipeline
        audio_dict = {
            "array": audio_array,
            "sampling_rate": sample_rate
        }
        result = pipeline(audio_dict)
        return result.get('text', '')
    except Exception as e:
        print(f"\nModel inference error: {e}")
        return ""


class WEREvaluator:
    """Word Error Rate Evaluator for ASR models."""
    
    def __init__(self, api_url: Optional[str] = None, model: Optional[str] = None, 
                 language: str = "de", batch_size: int = 1, use_batch_endpoint: bool = False):
        """
        Initialize WER Evaluator.
        
        Args:
            api_url: API endpoint URL (if using API-based evaluation)
            model: HuggingFace model name (if using direct model evaluation)
            language: Language code for transcription
            batch_size: Batch size for parallel processing
            use_batch_endpoint: Whether to use batch API endpoint
        """
        self.api_url = api_url
        self.model = model
        self.language = language
        self.batch_size = batch_size
        self.use_batch_endpoint = use_batch_endpoint
        
        # Initialize pipeline if model is provided
        self.pipeline = None
        if model and not api_url:
            try:
                from transformers import pipeline
                print(f"Loading model: {model}...")
                self.pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    device_map="auto"
                )
                print("✓ Model loaded")
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                sys.exit(1)
        
        # Load WER metric
        self.wer_metric = evaluate.load("wer")
    
    def check_api_availability(self) -> bool:
        """Check if API is available."""
        if not self.api_url:
            return False
        
        try:
            # Try health endpoint first
            health_response = requests.get(f"{self.api_url.rstrip('/')}/", timeout=10)
            health_response.raise_for_status()
            return True
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False
        except:
            # If root endpoint doesn't work, try the transcriptions endpoint
            try:
                test_response = requests.get(f"{self.api_url.rstrip('/')}/v1/audio/transcriptions", timeout=5)
                # Even if it returns an error, the endpoint exists
                return True
            except requests.exceptions.ConnectionError:
                return False
            except:
                return False
    
    def evaluate(self, dataset_name: str, split: str = "validation", 
                 limit: Optional[int] = None, audio_column: str = "audio",
                 text_column: str = "transcription", output_file: Optional[str] = None) -> dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to evaluate
            limit: Limit number of samples (None for all)
            audio_column: Name of audio column in dataset
            text_column: Name of text/reference column in dataset
            output_file: Optional output file for detailed results
            
        Returns:
            Dictionary with evaluation results
        """
        # Check API if using API mode
        if self.api_url:
            if not self.check_api_availability():
                print(f"✗ API not available at {self.api_url}")
                print(f"\nTroubleshooting:")
                print(f"  1. Make sure the ASR server is running")
                print(f"  2. Check that the server is accessible at {self.api_url}")
                print(f"  3. Verify the server is listening on the correct port")
                print(f"  4. Test the connection: curl {self.api_url}/")
                sys.exit(1)
            print(f"✓ API is available at {self.api_url}")
        
        # Load dataset
        print(f"\nLoading dataset: {dataset_name} ({split} split)...")
        try:
            dataset = load_dataset(dataset_name, split=split)
            dataset_arrow = dataset.with_format("arrow")
            print(f"✓ Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"✗ Failed to load dataset: {e}")
            sys.exit(1)
        
        # Auto-detect text column if needed
        if text_column not in dataset.column_names:
            possible_text_columns = ["transcription", "text", "sentence", "transcript"]
            for col in possible_text_columns:
                if col in dataset.column_names:
                    text_column = col
                    print(f"Auto-detected text column: {text_column}")
                    break
            else:
                print(f"✗ Could not find text column. Available columns: {dataset.column_names}")
                sys.exit(1)
        
        # Determine number of samples
        num_samples = len(dataset)
        if limit:
            num_samples = min(limit, num_samples)
            print(f"Evaluating {num_samples} samples (limited)")
        
        # Evaluate
        print(f"\n{'='*60}")
        print("Starting Evaluation")
        print(f"{'='*60}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total samples to process: {num_samples}")
        if self.use_batch_endpoint:
            print(f"Processing mode: Batch (using batch endpoint)")
        elif self.batch_size > 1:
            print(f"Processing mode: Parallel (concurrent requests)")
        else:
            print(f"Processing mode: Sequential")
        print(f"{'='*60}\n")
        
        all_predictions = []
        all_references = []
        detailed_results = []
        
        total_time = 0
        total_audio_duration = 0
        failed_samples = []
        error_counts = defaultdict(int)
        
        arrow_table = dataset_arrow.data
        start_eval_time = time.time()
        
        def process_sample(i):
            """Process a single sample."""
            # Get raw audio bytes
            audio_struct = arrow_table[audio_column][i].as_py()
            audio_bytes = audio_struct.get('bytes')
            
            if audio_bytes is None:
                error_counts["missing_audio_bytes"] += 1
                return None
            
            # Decode audio
            try:
                audio_array, sample_rate = decode_audio_from_bytes(audio_bytes)
            except Exception as e:
                error_counts["audio_decode_error"] += 1
                return {"index": i, "error": f"Audio decode failed: {str(e)}"}
            
            # Get reference transcription
            try:
                reference = arrow_table[text_column][i].as_py()
            except Exception as e:
                error_counts["reference_error"] += 1
                return {"index": i, "error": f"Reference read failed: {str(e)}"}
            
            # Calculate audio duration
            audio_duration = len(audio_array) / sample_rate
            
            # Transcribe
            start_time = time.time()
            try:
                if self.api_url:
                    prediction = transcribe_audio_api(
                        audio_array, sample_rate, self.api_url, 
                        model=self.model or "parakeet-tdt-0.6b-v3",
                        language=self.language
                    )
                else:
                    prediction = transcribe_audio_model(audio_array, sample_rate, self.pipeline)
            except Exception as e:
                error_counts["transcription_error"] += 1
                return {"index": i, "error": f"Transcription failed: {str(e)}", 
                       "audio_duration": audio_duration}
            elapsed = time.time() - start_time
            
            if not prediction:
                error_counts["empty_prediction"] += 1
            
            # Clean texts for WER
            cleaned_prediction = clean_text_for_wer(prediction)
            cleaned_reference = clean_text_for_wer(reference)
            
            return {
                "index": i,
                "reference": reference,
                "prediction": prediction,
                "cleaned_reference": cleaned_reference,
                "cleaned_prediction": cleaned_prediction,
                "audio_duration": audio_duration,
                "inference_time": elapsed
            }
        
        # Process samples
        if self.batch_size > 1 and not self.use_batch_endpoint:
            # Parallel processing
            print("Using parallel processing\n")
            pbar = tqdm(total=num_samples, desc="Transcribing", unit="sample")
            
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                futures = {executor.submit(process_sample, i): i for i in range(num_samples)}
                
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result is None or "error" in result:
                        if result and "error" in result:
                            failed_samples.append(result)
                        pbar.update(1)
                        continue
                    
                    detailed_results.append(result)
                    all_predictions.append(result["cleaned_prediction"])
                    all_references.append(result["cleaned_reference"])
                    total_audio_duration += result["audio_duration"]
                    total_time += result["inference_time"]
                    pbar.update(1)
            
            pbar.close()
        else:
            # Sequential processing
            print("Using sequential processing\n")
            pbar = tqdm(total=num_samples, desc="Transcribing", unit="sample")
            
            for i in range(num_samples):
                result = process_sample(i)
                
                if result is None or "error" in result:
                    if result and "error" in result:
                        failed_samples.append(result)
                    pbar.update(1)
                    continue
                
                detailed_results.append(result)
                all_predictions.append(result["cleaned_prediction"])
                all_references.append(result["cleaned_reference"])
                total_audio_duration += result["audio_duration"]
                total_time += result["inference_time"]
                pbar.update(1)
            
            pbar.close()
        
        # Sort results by index
        detailed_results.sort(key=lambda x: x["index"])
        
        # Calculate evaluation time
        total_eval_time = time.time() - start_eval_time
        
        if len(all_predictions) == 0:
            print("\n✗ No valid predictions generated.")
            if failed_samples:
                print(f"\nFailed sample details (first 10):")
                for sample in failed_samples[:10]:
                    print(f"  Sample {sample.get('index', 'unknown')}: {sample.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Compute WER
        print(f"\n{'='*60}")
        print("Computing Word Error Rate (WER)...")
        print(f"{'='*60}")
        final_wer = self.wer_metric.compute(predictions=all_predictions, references=all_references)
        
        # Calculate statistics
        num_valid_samples = len(detailed_results)
        avg_inference_time = total_time / num_valid_samples if num_valid_samples > 0 else 0
        real_time_factor = total_time / total_audio_duration if total_audio_duration > 0 else 0
        
        inference_times = [r["inference_time"] for r in detailed_results]
        audio_durations = [r["audio_duration"] for r in detailed_results]
        
        if inference_times:
            min_inference = min(inference_times)
            max_inference = max(inference_times)
            median_inference = np.median(inference_times)
        else:
            min_inference = max_inference = median_inference = 0
        
        if audio_durations:
            min_audio = min(audio_durations)
            max_audio = max(audio_durations)
            avg_audio = np.mean(audio_durations)
            total_audio_hours = total_audio_duration / 3600
        else:
            min_audio = max_audio = avg_audio = total_audio_hours = 0
        
        samples_per_second = num_valid_samples / total_eval_time if total_eval_time > 0 else 0
        
        # Print results
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Sample Statistics:")
        print(f"  Total samples: {num_samples}")
        print(f"  Valid samples: {num_valid_samples}")
        print(f"  Failed samples: {len(failed_samples)}")
        print(f"  Success rate: {100*num_valid_samples/num_samples:.2f}%")
        
        print(f"\n⏱️  Timing Statistics:")
        print(f"  Total evaluation time: {total_eval_time:.2f}s ({total_eval_time/60:.2f} min)")
        print(f"  Total audio duration: {total_audio_duration:.2f}s ({total_audio_duration/60:.2f} min, {total_audio_hours:.2f} hours)")
        print(f"  Total inference time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Average inference time per sample: {avg_inference_time:.3f}s")
        print(f"  Real-time factor (RTF): {real_time_factor:.3f}x")
        print(f"  Processing rate: {samples_per_second:.2f} samples/s")
        
        print(f"\n{'='*60}")
        print(f"🎯 Word Error Rate (WER): {100 * final_wer:.2f}%")
        print(f"{'='*60}\n")
        
        # Prepare results dictionary
        results = {
            "dataset": dataset_name,
            "split": split,
            "evaluation_time": datetime.now().isoformat(),
            "total_samples": num_samples,
            "valid_samples": num_valid_samples,
            "failed_samples": len(failed_samples),
            "wer": float(final_wer),
            "wer_percent": float(100 * final_wer),
            "total_audio_duration": total_audio_duration,
            "total_inference_time": total_time,
            "total_evaluation_time": total_eval_time,
            "real_time_factor": real_time_factor,
            "statistics": {
                "avg_inference_time": avg_inference_time,
                "min_inference_time": float(min_inference),
                "max_inference_time": float(max_inference),
                "median_inference_time": float(median_inference),
                "avg_audio_duration": float(avg_audio),
                "min_audio_duration": float(min_audio),
                "max_audio_duration": float(max_audio),
                "samples_per_second": samples_per_second,
            },
            "error_counts": dict(error_counts),
            "results": detailed_results[:100] if output_file else [],  # Limit for JSON output
            "failed_samples": failed_samples[:100] if failed_samples else []
        }
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ Detailed results saved to: {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models using Word Error Rate (WER) metric"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., NeurologyAI/neuro-whisper-v1)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate (default: validation)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="ASR API base URL (e.g., http://localhost:8002). If not provided, --model must be specified."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (e.g., NeurologyAI/neuro-parakeet). Required if --api-url is not provided."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Language code for transcription (default: de)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON format)"
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio",
        help="Name of the audio column in the dataset (default: audio)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="transcription",
        help="Name of the transcription column in the dataset (default: transcription)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent requests for parallel processing (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.api_url and not args.model:
        print("✗ Error: Either --api-url or --model must be specified")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = WEREvaluator(
        api_url=args.api_url,
        model=args.model,
        language=args.language,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        audio_column=args.audio_column,
        text_column=args.text_column,
        output_file=args.output
    )
    
    return results


if __name__ == "__main__":
    main()

