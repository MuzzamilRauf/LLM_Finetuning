"""
Configuration for Qwen2.5-3B LoRA + SFT fine-tuning.

This file contains the main configuration values used by the training
pipeline.

We are fine-tuning:

    Base model:
        Qwen/Qwen2.5-3B

    Dataset:
        Databricks Dolly 15k

    Fine-tuning method:
        Supervised Fine-Tuning (SFT)

    Parameter-efficient method:
        LoRA

    Maximum sequence length:
        1024 tokens

    Hardware:
        Apple Silicon Mac with 16 GB RAM

The first training run will be a small smoke test. Once we confirm
that LoRA + SFT works correctly, we can remove the sample limit and
train on the complete dataset.
"""

MODEL_NAME = "Qwen/Qwen2.5-3B"

DATASET_PATH = "data/processed/dolly-prepared"

OUTPUT_DIR = "outputs/qwen2.5-3b-dolly-lora"


# ============================================================
# Sequence configuration
# ============================================================

MAX_LENGTH = 1024


# ============================================================
# Loss masking
# ============================================================

# Compute the loss only on the assistant's reply. With this off, the
# model is also graded on predicting the user's prompt, which wastes
# capacity on text it will never need to generate.
#
# This is done by feeding TRL a prompt/completion dataset rather than a
# single text blob. TRL masks the prompt and trains on the completion.
#
# Note: we do NOT use SFTConfig(assistant_only_loss=True). That path
# needs {% generation %} markers in the chat template, and TRL cannot
# patch Qwen's stock template -- it raises at trainer construction.

COMPLETION_ONLY_LOSS = True


# ============================================================
# Smoke test
# ============================================================

# Start with a small subset to verify that the complete training
# pipeline works correctly.
#
# Set to None later to train on the complete dataset.

MAX_TRAIN_SAMPLES = 100
MAX_EVAL_SAMPLES = 10


# ============================================================
# LoRA configuration
# ============================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

# Attention projections plus the MLP projections. The MLP holds roughly
# two thirds of a transformer's parameters, so adapting attention alone
# leaves most of the model untouched.

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ============================================================
# Training configuration
# ============================================================

NUM_TRAIN_EPOCHS = 1

PER_DEVICE_TRAIN_BATCH_SIZE = 1

PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03


# ============================================================
# Logging / evaluation / saving
# ============================================================

LOGGING_STEPS = 10

# The smoke test runs only ~13 optimizer steps, so these must stay small
# or evaluation and checkpointing never trigger. Raise both to 50-100
# when MAX_TRAIN_SAMPLES is set back to None.

EVAL_STEPS = 5

SAVE_STEPS = 5

SAVE_TOTAL_LIMIT = 2