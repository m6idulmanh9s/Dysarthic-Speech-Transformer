# Dysarthric Speech Transformer

An end-to-end, sequence-to-sequence (seq2seq) Automatic Speech Recognition (ASR) system built from scratch in PyTorch. The model maps 200 ms log-spectrogram inputs to character sequences and is specifically designed for dysarthric speech recognition using a two-phase transfer learning framework (LJSpeech pre-training $\rightarrow$ UA-Speech speaker fine-tuning).

Includes a production-ready FastAPI inference engine, low-VRAM hardware optimizations (tested on a 4 GB RTX 2050), and Docker deployment support.

## Key Features

- **Custom Seq2Seq Architecture:** Features a 3-layer CNN temporal downsampling frontend, a 5-block Transformer-2 encoder with depthwise-separable convolutions and dual multi-head attention, and a 3-layer Transformer decoder.
- **Two-Phase Transfer Learning:** Pre-trains general acoustic-to-character representations on healthy speech before applying severity-dependent layer freezing for speaker-specific adaptation.
- **Low-VRAM & High-Throughput Optimization:** Optimized with PyTorch Automatic Mixed Precision (AMP FP16), gradient accumulation, multi-worker data loading, and fast tensor-native SpecAugment to maximize GPU utilization.
- **Robust FastAPI Inference Engine:** Serves real-time transcriptions via a REST API with automatic stereo-to-mono downmixing, 16 kHz resampling, and clean greedy autoregressive decoding.
- **Containerized Deployment:** Includes Docker and Docker Compose configurations for containerized deployment.

## Model Architecture

The model processes 200 ms audio frames converted to linear spectrograms ($N_{\text{fft}} = 3200$, hop length = 160, 1601 frequency bins in dB):

```text
Audio Spectrogram [B, T, 1601]
        │
        ▼
[CNN Frontend] ──> 3 x 2D Convolutions (Stride 2) ──> 8x Temporal Downsampling
        │
        ▼
[Transformer-2 Encoder] ──> 5 Blocks (Dual Attention + Depthwise Separable Convs)
        │
        ▼
[Transformer Decoder] ──> 3 Layers (Masked Self-Attention + Cross-Attention)
        │
        ▼
[Linear Classifier] ──> 42-Token Vocabulary (A-Z, 0-9, Space, ', PAD, SOS, EOS, UNK)
```


## 📚 Research Background & Reference

This project is inspired by the research paper:

**Dysarthric Speech Transformer: A Sequence-to-Sequence Dysarthric Speech Recognition System**

- **Authors:** Seyed Reza Shahamiri, Vanshika Lal, and Dhvani Shah
- **Journal:** IEEE Transactions on Neural Systems and Rehabilitation Engineering
- **Volume:** 31, 2023, pp. 3407–3416
- **DOI:** [10.1109/TNSRE.2023.3307020](https://doi.org/10.1109/TNSRE.2023.3307020)

### Research Inspiration

The paper investigates an end-to-end, speaker-adaptive Transformer architecture for dysarthric speech recognition, addressing the challenges of limited clinical speech data and substantial variation in speech intelligibility.

The following concepts informed the design of this project:

- **Sequence-to-Sequence Recognition:** Mapping speech spectrograms to character sequences through an encoder-decoder architecture.
- **Transformer-2 Architecture:** Using a deeper encoder with additional attention and depthwise-separable convolution blocks to learn complex acoustic representations.
- **Transfer Learning:** Leveraging healthy speech data to initialize model representations before adapting to dysarthric speech.
- **Neural Freezing:** Selectively freezing model layers during fine-tuning to preserve previously learned representations while adapting to individual speakers.
- **Audio Data Augmentation:** Using acoustic perturbations to address limited dysarthric speech training data.

### Implementation in This Project

This repository implements a PyTorch-based adaptation of these research ideas, with a focus on training and inference under constrained hardware resources.

The implementation includes:

- A custom CNN–Transformer encoder-decoder architecture.
- Healthy-speech pre-training using LJSpeech.
- Speaker-specific fine-tuning using UA-Speech.
- Severity-dependent layer-freezing strategies.
- Training evaluation using Character Error Rate (CER) and exact-match Word Recognition Accuracy (WRA).
- A FastAPI inference service and Docker deployment configuration.

**Implementation distinction:** The original paper describes an intermediate control-speaker adaptation stage between healthy-speech pre-training and dysarthric-speaker fine-tuning. The current implementation uses a two-phase pipeline consisting of LJSpeech pre-training followed by UA-Speech speaker-specific fine-tuning. Its experimental setup and results therefore should not be interpreted as a direct reproduction of the paper's reported benchmarks.

### Experimental Results

The results reported in this repository are from the project's own training experiments on UA-Speech speaker F02. They are separate from the results reported in the reference paper.

The current F02 experiment used an 80/20 within-speaker split, with the following metrics at epoch 50:

- **Exact-match WRA:** 36.60%
- **Character Error Rate (CER):** 43.20%

These results are preliminary validation metrics and are not directly comparable to the paper's results, which use a different experimental protocol.

### Reference

Shahamiri, S. R., Lal, V., and Shah, D., “Dysarthric Speech Transformer: A Sequence-to-Sequence Dysarthric Speech Recognition System,” *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 3407–3416, 2023. DOI: [10.1109/TNSRE.2023.3307020](https://doi.org/10.1109/TNSRE.2023.3307020).



## Training Methodology

### Phase 1: Healthy Speech Pre-Training

- **Dataset:** LJSpeech-1.1 (13,100 clean, healthy speech utterances).
- **Objective:** Train the full encoder-decoder model from scratch to map acoustic frames to English character sequences.
- **Configuration:** AdamW optimizer ($\text{lr} = 3 \times 10^{-5}$), Cosine Annealing learning rate schedule, loss-based checkpointing.

### Phase 2: Speaker-Specific Fine-Tuning

- **Dataset:** UA-Speech (Speaker-disjoint, isolated dysarthric words).
- **Severity-Dependent Freezing:** Automatically detects speaker intelligibility (very_low, low, mild, high) and selectively freezes target decoder layers to preserve language structure while adapting the frontend to dysarthric acoustics.
- **Regularization:** Fast additive Gaussian noise injection and SpecAugment (time/frequency masking) prevent overfitting on small individual speaker datasets.

## Results & Performance Metrics

Fine-tuning evaluated on UA-Speech target speaker F02 (Mild Dysarthria):

| Metric | Phase 1 (LJSpeech Base) | Phase 2 (Speaker F02 Fine-Tuned) |
|---|---:|---:|
| Validation Loss | 1.7296 | 0.5851 |
| Character Error Rate (CER) | 84.42% | 43.20% |
| Exact Word Recognition Accuracy (WRA) | 0.00% | 36.60% |

**Sample Prediction (Speaker F02):**

- **Target:** CASUALTIES
- **Predicted:** CASHALTIES

## Project Structure

```text
Dysarthic-Speech-Transformer/
├── configs/
│   ├── model_config.yaml       # Transformer depth, hidden dim, vocab config
│   ├── train_config.yaml       # Hyperparameters, batch size, AMP, schedule
│   └── data_config.yaml        # Audio sample rate, STFT window/hop sizes
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI REST endpoints (/health, /transcribe)
│   │   └── inference_engine.py # Greedily decoded autoregressive inference engine
│   ├── dataset/
│   │   ├── dataloader.py       # PyTorch Dataset, collate_fn, padding masks
│   │   ├── augmentations.py    # Tensor-native SpecAugment & Gaussian noise
│   │   └── index_builder.py    # Dataset index builder for LJSpeech & UA-Speech
│   ├── models/
│   │   └── transformer.py      # CNN frontend & Transformer-2 architecture
│   ├── training/
│   │   ├── trainer.py          # Seq2Seq AMP training loop with grad accumulation
│   │   ├── evaluator.py        # Exact-WRA, CER, and Levenshtein metrics
│   │   └── freeze_utils.py     # Phase 2 severity-dependent layer freezing
│   └── utils/
│       ├── tokenizer.py        # 42-token character mapping & decoding
│       └── logger.py           # Structured logging utility
├── checkpoints/                # Directory containing best base & speaker .pt files
├── train.py                    # Main CLI entry point for Phase 1 & Phase 2 training
├── requirements.txt            # Python dependencies
└── docker/
    ├── Dockerfile              # CPU-optimized PyTorch container build
    └── docker-compose.yml      # Multi-container orchestration config
```

## Installation & Setup

### 1. Clone & Setup Environment

```bash
gh repo clone m6idulmanh9s/Dysarthic-Speech-Transformer
cd Dysarthic-Speech-Transformer

# Create and activate Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Setup

Build the master dataset CSV index after extracting LJSpeech and UA-Speech to the `data/` directory:

```bash
python -m src.dataset.index_builder
```

## Usage Guide

### Phase 1 Training (Pre-Training on LJSpeech)

```bash
python train.py --phase 1
```

### Phase 2 Training (Fine-Tuning on UA-Speech)

```bash
python train.py --phase 2 --speaker F02
```

## API Deployment

### Running Locally with Uvicorn

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "device": "cuda",
  "checkpoint": "checkpoints/best_dysarthric_model_F02.pt",
  "model_loaded": true
}
```

### Audio Transcription Endpoint

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/UA-Speech/F02/F02_M1_1.wav"
```

**Response:**

```json
{
  "filename": "F02_M1_1.wav",
  "transcription": "COMMAND",
  "processing_time_sec": 0.082
}
```

## Docker Deployment

Build and run the containerized inference service using Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The server will start at [http://localhost:8000](http://localhost:8000).

## Tech Stack & Dependencies

- **Core:** PyTorch, Torchaudio, NumPy, Pandas
- **API & Serving:** FastAPI, Uvicorn, PyYAML
- **Experiment Tracking:** Weights & Biases (W&B)
- **DevOps:** Docker, Docker Compose
