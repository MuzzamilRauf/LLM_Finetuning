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

import os

import torch


# True on Colab and any other CUDA box. The batch sizes below differ a
# lot between a real GPU and the 16 GB Mac, so detect it once here.

CUDA = torch.cuda.is_available()


MODEL_NAME = "Qwen/Qwen2.5-3B"

DATASET_PATH = "data/processed/dolly-prepared"

# Override with the OUTPUT_DIR environment variable. On Colab point this
# at Google Drive -- /content is wiped when the session ends, which would
# destroy the checkpoints that resuming depends on.
#
#     os.environ["OUTPUT_DIR"] = "/content/drive/MyDrive/qwen-dolly-lora"

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "outputs/qwen2.5-3b-dolly-lora",
)


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

# Set MAX_TRAIN_SAMPLES back to 100 to re-run the smoke test. With only
# ~13 optimizer steps, also drop EVAL_STEPS/SAVE_STEPS to 5 or neither
# evaluation nor checkpointing will ever fire.

MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = 200


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

# A GPU runs a real batch in roughly the time it takes to run one
# example, so accumulating eight single-example passes wastes most of
# the card. The Mac had no choice: batch 1 was all that fit in 16 GB.
#
# Batch 4 is sized for the 15 GB T4 that free Colab usually gives out.
# Batch 8 without gradient checkpointing runs it out of memory partway
# through the first step. On a 40 GB A100 batch 16 is comfortable.
#
# Every branch gives the same effective batch of 8, so the learning rate
# stays valid either way.

if CUDA:
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 2
else:
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    PER_DEVICE_EVAL_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8


# Recomputing activations in the backward pass costs roughly 30% speed
# and saves a large amount of memory. A 3B model at max_length=1024
# does not fit a 15 GB T4 without it, so it stays on everywhere. Turn it
# off only on a card with plenty of headroom (A100 and up).

GRADIENT_CHECKPOINTING = True

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03


# ============================================================
# Logging / evaluation / saving
# ============================================================

LOGGING_STEPS = 10

# The full run is ~1687 optimizer steps (13,496 examples / effective
# batch 8) on any device -- only the time per step changes. Every 100
# steps gives ~17 evaluations and ~17 checkpoints across the run.
#
# Drop both to 5 for the smoke test, which is only ~13 steps.

EVAL_STEPS = 100

SAVE_STEPS = 100

SAVE_TOTAL_LIMIT = 2


# ============================================================
# Resuming
# ============================================================

# Colab sessions end without warning. When this is True, training picks
# up from the newest checkpoint in OUTPUT_DIR instead of starting over.
# Starting fresh is handled automatically when no checkpoint exists, so
# this is safe to leave on for the first run.

RESUME_FROM_CHECKPOINT = True