# Mini LLM from Scratch (PyTorch + BPE)

A lightweight GPT-style language model built from scratch using PyTorch and a Byte Pair Encoding (BPE) tokenizer. This project is designed to run locally on CPU and demonstrates the full pipeline of training, evaluation, and text generation.

---

## Overview

This project implements a minimal Transformer-based language model similar to GPT. It includes tokenizer training, model training, evaluation metrics, and inference for text generation.

The goal is to provide a clear and practical understanding of how modern language models work internally.

---

## Features

* Transformer-based architecture (GPT-style decoder)
* Subword tokenization using BPE
* Training and validation loss tracking
* Perplexity evaluation
* Text generation with custom prompts
* Automatic checkpointing and resume capability
* Structured output per training run

---

## Installation

```bash
pip install torch matplotlib tokenizers
```

---

## Usage

### 1. Prepare dataset

Place your dataset in:

```text
data/input.txt
```

---

### 2. Train tokenizer

```bash
python train_tokenizer.py
```

This generates:

```text
tokenizer.json
```

---

### 3. Train model

```bash
python train.py
```

Training outputs are saved in:

```text
output/run_YYYYMMDD_HHMMSS/
```

---

### 4. Generate text

```bash
python generate.py
```

Example:

```text
Prompt: The future of AI
Output: The future of AI is expected to influence multiple industries...
```

---

## Training Outputs

Each training run produces:

* loss_curve.png
* perplexity_curve.png
* samples.txt
* model.pt
* checkpoint.pt
* config.json

---

## Model Details

| Component    | Description                     |
| ------------ | ------------------------------- |
| Architecture | Transformer (GPT-style decoder) |
| Tokenization | Byte Pair Encoding (BPE)        |
| Objective    | Next-token prediction           |
| Device       | CPU (GPU optional)              |

---

## Example Configuration

```python
batch_size = 16
block_size = 128
max_iters = 10000
learning_rate = 1e-4

n_embd = 256
n_head = 8
n_layer = 6
```

---

## Limitations

* Trained on a relatively small dataset
* Limited coherence compared to large-scale models
* Not instruction-tuned or optimized for conversational tasks

---

## Purpose

This project is intended for:

* Learning how large language models are implemented
* Understanding tokenization and transformer mechanics
* Experimenting with local model training and inference

---

## Future Improvements

* Larger and cleaner datasets
* Tokenizer optimization
* GPU acceleration
* Integration with web applications (e.g., API or frontend)

---

## License

This project is for educational and experimental purposes.
