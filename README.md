# WER Evaluation Tool

A standalone tool for evaluating Automatic Speech Recognition (ASR) models using Word Error Rate (WER) metric. This tool supports evaluation via API endpoints or direct HuggingFace model inference.

## Features

- **Flexible Evaluation**: Evaluate models via API endpoints or directly using HuggingFace models
- **Comprehensive Metrics**: Calculate WER along with timing statistics, real-time factors, and detailed per-sample results
- **Parallel Processing**: Support for concurrent requests to speed up evaluation
- **Multiple Dataset Formats**: Works with any HuggingFace dataset containing audio and text columns
- **Detailed Reporting**: Export detailed results in JSON format for further analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd wer-evaluation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Evaluate via API Endpoint

If you have an ASR server running (e.g., OpenAI-compatible API):

```bash
python wer_evaluator.py \
    --dataset NeurologyAI/neuro-whisper-v1 \
    --split validation \
    --api-url http://localhost:8002 \
    --output results.json
```

### Evaluate with HuggingFace Model Directly

If you want to evaluate a model directly without an API:

```bash
python wer_evaluator.py \
    --dataset NeurologyAI/neuro-whisper-v1 \
    --split validation \
    --model NeurologyAI/neuro-parakeet \
    --output results.json
```

### Quick Test with Limited Samples

For quick testing, limit the number of samples:

```bash
python wer_evaluator.py \
    --dataset NeurologyAI/neuro-whisper-v1 \
    --split validation \
    --api-url http://localhost:8002 \
    --limit 100
```

### Parallel Processing

Speed up evaluation by processing multiple samples concurrently:

```bash
python wer_evaluator.py \
    --dataset NeurologyAI/neuro-whisper-v1 \
    --split validation \
    --api-url http://localhost:8002 \
    --batch-size 4
```

## Command Line Arguments

- `--dataset`: HuggingFace dataset name (required)
- `--split`: Dataset split to evaluate (default: `validation`)
- `--api-url`: ASR API base URL (e.g., `http://localhost:8002`). Required if `--model` is not provided.
- `--model`: HuggingFace model name (e.g., `NeurologyAI/neuro-parakeet`). Required if `--api-url` is not provided.
- `--language`: Language code for transcription (default: `de`)
- `--limit`: Limit number of samples to evaluate (default: all)
- `--output`: Output file for detailed results (JSON format)
- `--audio-column`: Name of the audio column in the dataset (default: `audio`)
- `--text-column`: Name of the transcription column in the dataset (default: `transcription`)
- `--batch-size`: Number of concurrent requests for parallel processing (default: `1`)

## Output

The tool provides:

1. **Console Output**: Real-time progress and summary statistics including:
   - Sample statistics (total, valid, failed)
   - Timing statistics (evaluation time, inference time, real-time factor)
   - Word Error Rate (WER)

2. **JSON Output** (if `--output` is specified): Detailed results including:
   - Overall WER and statistics
   - Per-sample predictions and references
   - Timing information for each sample
   - Error counts and failed samples

## Example Output

```
============================================================
EVALUATION RESULTS
============================================================

📊 Sample Statistics:
  Total samples: 5289
  Valid samples: 5289
  Failed samples: 0
  Success rate: 100.00%

⏱️  Timing Statistics:
  Total evaluation time: 1234.56s (20.58 min)
  Total audio duration: 22680.00s (378.00 min, 6.30 hours)
  Total inference time: 952.34s (15.87 min)
  Average inference time per sample: 0.180s
  Real-time factor (RTF): 0.042x
  Processing rate: 4.28 samples/s

============================================================
🎯 Word Error Rate (WER): 1.04%
============================================================
```

## API Compatibility

The tool expects an OpenAI-compatible transcription API endpoint:

- **Endpoint**: `POST /v1/audio/transcriptions`
- **Request**: Multipart form data with `file` (audio file) and optional `model`, `language`, `response_format`
- **Response**: JSON with `text` field containing the transcription

Example API request:
```python
files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
data = {'model': 'parakeet-tdt-0.6b-v3', 'language': 'de', 'response_format': 'json'}
response = requests.post('http://localhost:8002/v1/audio/transcriptions', files=files, data=data)
```

## Text Cleaning

The tool automatically cleans text for WER computation by:
- Converting to lowercase
- Stripping whitespace
- Removing trailing punctuation (`.`, `,`, `!`, `?`)

This normalization ensures fair comparison between predictions and references.

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{wer_evaluation,
  title = {WER Evaluation Tool},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/wer-evaluation}
}
```

