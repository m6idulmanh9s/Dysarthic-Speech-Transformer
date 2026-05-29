# Dysarthric Speech Recognition: Speaker-Adaptive Transformer

An end-to-end Sequence-to-Sequence Transformer engineered specifically to transcribe dysarthric speech. This project implements a ground-up PyTorch architecture utilizing a 3-phase transfer learning pipeline to mitigate the extreme data scarcity inherent in clinical voice datasets. 


## Architecture & Methodology

Standard Automatic Speech Recognition (ASR) models fail catastrophically on dysarthric speech due to highly irregular phonetics and severe acoustic distortions. To solve this, this repository implements a multi-stage "Neural Freezing" and adaptation pipeline:

1. **Phase 1: Base Language Modeling (LJ Speech)**
   A 5-encoder/3-decoder Transformer is trained from scratch on 13,000+ healthy utterances. This phase establishes fundamental English phonetics and a robust character-level language model without relying on pre-trained black boxes.
2. **Phase 2: Control Acoustic Adaptation**
   The Decoder (Language Model) is frozen. The unfrozen Encoder is fine-tuned on healthy control speakers from the clinical dataset to adapt to specific microphone hardware, room acoustics, and noise floors.
3. **Phase 3: Speaker-Adaptive Fine-Tuning**
   Targeted at specific dysarthric patients (e.g., severe or moderate cases). Deep encoder layers are selectively frozen based on impairment severity, while shallow layers adapt to the patient's unique vocal tract variations.


## Key Technical Features

* **Custom Acoustic Frontend:** Replaces standard linear embeddings with Depthwise Separable 1D Convolutions to extract localized spectral patterns from 200ms Voicegrams.
* **Mixed-Precision Training (`bfloat16`):** Engineered for absolute mathematical stability and VRAM efficiency, preventing attention softmax overflow on consumer GPUs during deep layer calculations.
* **Synthetic Data Augmentation:** Implements dynamic pitch shifting ($\pm$ 2 steps), speed modification ($\pm$ 10%), and white noise injection to double the effective size of clinical training pools.
* **Dynamic Auto-Regressive Inference:** Includes a custom repetition penalty matrix and dynamic sequence-length scaling based on audio duration to break the "Insertion Hallucination" loops common in severe dysarthria decoding.


## Datasets
This project utilizes two primary datasets. You must download these locally to replicate the training pipeline:

* **LJ Speech (v1.1):** 13,100 short audio clips of a single healthy speaker reading classic literature. Used exclusively for Phase 1 Base Training.

* **TORGO Database of Dysarthric Speech:** Provided by the University of Toronto. Contains aligned acoustic and articulatory recordings from speakers with either cerebral palsy or ALS.
       1. Control Speakers are used for Phase 2.
       2. Dysarthric Speakers (e.g., F01, M02) are used for Phase 3.


## Directory Structure

**Note:** Due to GitHub file size limits, the `data/` and `checkpoints/` directories are tracked via `.gitignore` and are not included in this repository. 

```text
dysarthric-asr-transformer/
│
├── data/                   # (Local Only) Raw .wavs, .npy spectrograms, CSV indices
├── checkpoints/            # (Local Only) Saved .pt model weights
│
├── src/                    
│   ├── project.ipynb       # PyTorch Dataset, Collate functions, and Character Tokenizer. The complete Transformer architecture and Depthwise Convolutions. CSV indexing and the Librosa Spectrogram Factory scripts. 
│                           # Phase 1: LJ Speech base model training loop. Phase 2 & 3: Fine-tuning loops with dynamic Neural Freezing. Batch CER/WER evaluation script using Jiwer 
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Repository shield for massive datasets
└── README.md               # Project documentation
