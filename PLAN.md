# STMC-Transformer Sign Language Translation Implementation Plan

## Summary hi

Implement sign-to-text translation for PHOENIX-2014-T dataset using a Transformer-based architecture with offline feature extraction. This follows the approach from "Better Sign Language Translation with STMC-Transformer" (COLING 2020).

## Key Architectural Decision: Offline Feature Extraction

**Critical**: Features are extracted ONCE and saved to disk. Training loads lightweight tensors, not raw frames.

**Rationale**:
- Training stability (no CPU bottleneck from image decoding)
- Experiment velocity (hyperparameter tuning doesn't re-run CNN)
- Future extensibility (swap ResNet features for STMC features without model changes)

---

## Repository Examination Summary

### Files and Modules Examined

#### kayoyin/transformer-slt (COLING 2020)
- `train.py` - Wrapper that calls `onmt.bin.train.main()`
- `preprocess.py` - Wrapper that calls `onmt.bin.preprocess.main()`
- `onmt/encoders/transformer.py` - Standard Transformer encoder
- `onmt/model_builder.py` - Standard OpenNMT model construction

**Finding**: This repo is primarily for **gloss-to-text** translation using OpenNMT. The STMC in the paper uses pre-extracted STMC features as input.

#### PHOENIX-2014-T Dataset Structure
```
PHOENIX-2014-T-release-v3/
└── PHOENIX-2014-T/
    ├── annotations/manual/
    │   ├── PHOENIX-2014-T.train.corpus.csv  (~7096 samples)
    │   ├── PHOENIX-2014-T.dev.corpus.csv    (~519 samples)
    │   └── PHOENIX-2014-T.test.corpus.csv   (~642 samples)
    └── features/fullFrame-210x260px/
        ├── train/VIDEO_ID/images*.png
        ├── dev/VIDEO_ID/images*.png
        └── test/VIDEO_ID/images*.png
```

CSV columns: `name|video|start|end|speaker|orth|translation`
- `orth`: German sign glosses (e.g., "JETZT WETTER MORGEN")
- `translation`: German text (e.g., "und nun die wettervorhersage")

---

## Phase 1: Offline Feature Extraction

### File: `sentisign/sign_language/preprocess_features.py`

**Purpose**: Extract ResNet-18 features from all video frames, save to disk.

**Input**: Raw PNG frames from `PHOENIX-2014-T/features/fullFrame-210x260px/{train,dev,test}/`

**Output**: `.pt` files with shape `(T, 512)` per video

**Important**: The CNN is **frozen** and used only for feature extraction. ResNet-18 is in `eval()` mode with gradients disabled. It is never trained.

**Preprocessing**: Frames are resized to 224x224 and normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

**Key implementation details**:
```python
# Batch frames for GPU efficiency
model.eval()
for chunk in chunked(frames, batch_size=32):
    with torch.no_grad():
        features = model(chunk.cuda())  # (B, 512, 1, 1)
```

**Output location**: `PHOENIX-2014-T/features/resnet18/{train,dev,test}/<video_id>.pt`

---

## Phase 2: Dataset & Vocabulary

### File: `sentisign/sign_language/dataset.py`

**Classes**:
- `Phoenix2014TDataset`: Loads pre-extracted features + translations
- `collate_fn`: Handles variable-length sequences with padding

**Dataset output format**:
```python
{
    "features": Tensor(T, 512),       # Pre-extracted CNN features
    "feature_length": int,            # For encoder padding mask
    "translation_ids": Tensor(L),     # BPE-tokenized target
    "translation_length": int         # For decoder padding mask
}
```

### File: `sentisign/sign_language/vocabulary.py`

**Tokenization**: Target text is tokenized using **Subword/BPE** via SentencePiece, trained on the training split only.

This module wraps SentencePiece (not word-level tokens) and provides:
- `train_bpe()`: Train BPE model on training translations
- `encode()`: Text → token IDs
- `decode()`: Token IDs → text (with BPE reversal)

**Special tokens**: `<pad>` (0), `<sos>` (1), `<eos>` (2), `<unk>` (3)

---

## Phase 3: Model Architecture

### File: `sentisign/sign_language/model.py`

**Architecture**:
```
Pre-extracted features (T, 512)
    ↓ Linear projection
Features (T, d_model=512)
    ↓ Positional Encoding
    ↓ Transformer Encoder (6 layers)
Encoder output (T, d_model)
    ↓ Transformer Decoder (6 layers)
    ↓ Output projection
Logits (L, vocab_size)
```

**Key components**:
- `PositionalEncoding`: Sinusoidal positional encoding
- `SLTModel`: Main encoder-decoder model
  - `feature_projection`: Linear(512, d_model)
  - `encoder`: nn.TransformerEncoder (not monolithic nn.Transformer, for clarity and control)
  - `decoder`: nn.TransformerDecoder
  - `output_projection`: Linear(d_model, vocab_size)

**Masking**: Padding masks are computed from `feature_length` and `translation_length`, passed to encoder and decoder respectively. Causal mask applied to decoder self-attention.

**Config**:
```python
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
```

---

## Phase 4: Training Loop

### File: `sentisign/sign_language/trainer.py`

**Training configuration**:
- Optimizer: Adam (base lr is controlled by Noam scheduler)
- Scheduler: Noam decay with warmup_steps=4000 (lr scales with d_model^-0.5)
- Loss: CrossEntropyLoss with label_smoothing=0.1
- Gradient clipping: max_norm=1.0

**Features**:
- Checkpoint saving (best + periodic)
- Validation loop with BLEU evaluation
- Early stopping (patience=10)

**BLEU Evaluation**: BLEU is computed on **detokenized text**. BPE tokens are reversed/merged before scoring using SentencePiece's `decode()`.

### File: `train.py` (project root)

Entry point script that:
- Parses config/CLI args
- Initializes dataset, model, trainer
- Runs training loop

---

## Phase 5: Inference & Demo

### File: `sentisign/sign_language/inference.py`

**Features**:
- Load checkpoint
- Greedy decoding
- Beam search (beam_size=5)

### File: `demos/run_sign_to_text.py`

**Demo script**:
- Load trained model
- Run on **pre-extracted features** for a PHOENIX video (not raw frames)
- Print translated German text

---

## Final Directory Structure

```
sentisign/sign_language/
├── __init__.py
├── preprocess_features.py   # Phase 1: Feature extraction
├── dataset.py               # Phase 2: Dataset loader
├── vocabulary.py            # Phase 2: BPE tokenization (SentencePiece)
├── model.py                 # Phase 3: Transformer model
├── trainer.py               # Phase 4: Training loop
└── inference.py             # Phase 5: Inference

train.py                     # Entry point for training
demos/run_sign_to_text.py    # Inference demo
```

---

## Data Flow Summary

```
[One-time preprocessing]
Raw frames (PNG) → ResNet-18 (frozen) → features/*.pt (T, 512)

[Training]
features/*.pt → Dataset → Model → Loss → Optimizer

[Inference]
features/*.pt → Model → Beam Search → BPE decode → German text
```

---

## Critical Files to Create (in order)

1. `sentisign/sign_language/preprocess_features.py`
2. `sentisign/sign_language/vocabulary.py`
3. `sentisign/sign_language/dataset.py`
4. `sentisign/sign_language/model.py`
5. `sentisign/sign_language/trainer.py`
6. `train.py`
7. `sentisign/sign_language/inference.py`
8. `demos/run_sign_to_text.py`

---

## Dependencies to Add

```toml
# pyproject.toml additions
sacrebleu      # BLEU evaluation
sentencepiece  # BPE tokenization
h5py           # Optional: HDF5 storage
tqdm           # Progress bars
```

---

## Future Extension Point (STMC)

The architecture is designed so STMC can be added later by:
1. Creating `preprocess_stmc_features.py` that extracts face/hand/full-frame features
2. Swapping feature files: `resnet18/*.pt` → `stmc/*.pt`
3. Updating `feature_projection` input dimension

No changes to Transformer, training loop, or dataset structure required.

---

## Technical Debt Flags

1. **No STMC feature extraction**: Full SMC/TMC modules would require pose estimation, hand/face detection. Can be added later.

2. **German-only**: Current implementation targets German (PHOENIX-2014-T).

3. **No CTC recognition head**: Joint recognition/translation could improve results but adds complexity.

---

## References

- [Sign Language Transformers (Camgoz et al., CVPR 2020)](https://github.com/neccam/slt)
- [transformer-slt (Yin & Read, COLING 2020)](https://github.com/kayoyin/transformer-slt)
- [STMC Network (Zhou et al., AAAI 2020)](https://arxiv.org/abs/2002.03187)
- [PHOENIX-2014-T Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
