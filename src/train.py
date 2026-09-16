"""
Train Qwen2.5-3B using SFT + LoRA.

This script connects all of the components we have prepared so far:

    Prepared Dolly dataset
            ↓
    Qwen chat template
            ↓
       Qwen tokenizer
            ↓
       Qwen2.5-3B
            ↓
          LoRA
            ↓
        SFTTrainer
            ↓
          Training

Important:
    The original Qwen2.5-3B weights are frozen.

    Only the LoRA adapter parameters are trained.

    The loss is computed on the assistant's reply only. Each example is
    fed as a prompt/completion pair, so TRL masks the prompt tokens and
    trains on the completion.

The first run uses a small dataset subset as a smoke test. This allows
us to verify that the training pipeline works correctly on the 16 GB
Apple Silicon Mac before starting a full training run.

After the smoke test succeeds, change:

    MAX_TRAIN_SAMPLES = None
    MAX_EVAL_SAMPLES = None

in config.py to train on the complete dataset.
"""

import os

import torch

from datasets import load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

import config


def get_device():
    """
    Select the best available device.

    CUDA is checked first: when both are somehow present, a real GPU
    always beats Apple Silicon for this workload.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(device):
    """
    Pick the training precision for this device.

    bfloat16 has the same range as float32, so it cannot overflow to inf
    the way float16 does. It needs Ampere or newer, which rules out the
    T4 that free Colab usually hands out.

        CUDA, Ampere+ (L4/A100)  -> bfloat16
        CUDA, older (T4)         -> float16, with the gradient scaler
        MPS                      -> float16
        CPU                      -> float32
    """

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    if device.type == "mps":
        return torch.float16

    return torch.float32


def prepare_dataset(dataset):
    """
    Convert the messages column into prompt/completion columns.
    """

    return dataset.map(
        to_prompt_completion,
        remove_columns=["messages"],
        desc="Building prompt/completion pairs",
    )


def to_prompt_completion(example):
    """
    Split one conversation into a prompt/completion pair.

    TRL applies Qwen's official chat template to each side, then masks
    the prompt so the loss is computed on the assistant's reply only.

    The prompt ends with the assistant header ("<|im_start|>assistant\n")
    and the completion carries the reply plus "<|im_end|>", so the model
    is still trained to emit its stop token.
    """

    user_message, assistant_message = example["messages"]

    return {
        "prompt": [user_message],
        "completion": [assistant_message],
    }


def create_lora_config():
    """
    Create the LoRA configuration.

    LoRA adapters will be added to the attention projection layers.
    """

    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model(device, dtype):
    """
    Load Qwen2.5-3B in the precision chosen for this device.

    Half precision roughly halves the memory the weights occupy compared
    with float32, which is what makes a 3B model trainable here at all.
    """

    print("\n=== LOADING MODEL ===")
    print(f"Precision: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        dtype=dtype,
    )

    model.to(device)

    # Required for training because we are not using the KV cache.
    model.config.use_cache = False

    print("Model loaded successfully.")

    return model


def find_checkpoint(output_dir):
    """
    Return the newest checkpoint in output_dir, or None if there is none.

    Passing resume_from_checkpoint=True to a directory holding no
    checkpoint raises, so the first run has to be told to start fresh.
    """

    if not config.RESUME_FROM_CHECKPOINT:
        return None

    if not os.path.isdir(output_dir):
        return None

    return get_last_checkpoint(output_dir)


def print_device_info(device):
    """
    Print information about the selected hardware.
    """

    print("\n=== DEVICE ===")
    print(f"Device: {device}")

    if device.type == "mps":
        print("Using Apple Silicon MPS.")

    elif device.type == "cuda":
        print("Using NVIDIA CUDA.")

    else:
        print("Using CPU.")


def main():
    print("=== QWEN2.5-3B LoRA + SFT TRAINING ===")

    device = get_device()
    dtype = get_dtype(device)
    print_device_info(device)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\n=== LOADING TOKENIZER ===")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME
    )

    # Qwen needs a padding token for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded successfully.")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(f"Output directory: {config.OUTPUT_DIR}")

    print("\n=== LOADING DATASET ===")

    dataset = load_from_disk(config.DATASET_PATH)

    print(dataset)

    # --------------------------------------------------------
    # Prepare text
    # --------------------------------------------------------

    # The dataset holds a "messages" column in conversational format.
    # We split it into prompt/completion so TRL can mask the prompt, and
    # let TRL apply Qwen's official chat template to each side. The
    # ChatML string is no longer built by hand, so the training format is
    # guaranteed to match what the model sees at inference time.

    print("\n=== PREPARING DATASET ===")

    train_dataset = prepare_dataset(dataset["train"])
    eval_dataset = prepare_dataset(dataset["test"])

    # --------------------------------------------------------
    # Smoke-test subset
    # --------------------------------------------------------

    if config.MAX_TRAIN_SAMPLES is not None:
        train_dataset = train_dataset.select(
            range(
                min(
                    config.MAX_TRAIN_SAMPLES,
                    len(train_dataset),
                )
            )
        )

    if config.MAX_EVAL_SAMPLES is not None:
        eval_dataset = eval_dataset.select(
            range(
                min(
                    config.MAX_EVAL_SAMPLES,
                    len(eval_dataset),
                )
            )
        )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(device, dtype)

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

    print("\n=== CONFIGURING LoRA ===")

    lora_config = create_lora_config()

    print(f"LoRA rank: {config.LORA_R}")
    print(f"LoRA alpha: {config.LORA_ALPHA}")
    print(f"LoRA dropout: {config.LORA_DROPOUT}")
    print(f"Target modules: {config.LORA_TARGET_MODULES}")

    # --------------------------------------------------------
    # SFT configuration
    # --------------------------------------------------------

    print("\n=== CONFIGURING SFT ===")

    training_args = SFTConfig(
        output_dir=config.OUTPUT_DIR,

        num_train_epochs=config.NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=(
            config.PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            config.PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            config.GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY,

        # warmup_ratio=config.WARMUP_RATIO,

        logging_steps=config.LOGGING_STEPS,

        eval_strategy="steps",

        eval_steps=config.EVAL_STEPS,

        save_strategy="steps",

        save_steps=config.SAVE_STEPS,

        save_total_limit=config.SAVE_TOTAL_LIMIT,

        max_length=config.MAX_LENGTH,

        # Grade the model only on the assistant's reply, not the prompt.
        completion_only_loss=config.COMPLETION_ONLY_LOSS,

        packing=False,

        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,

        # Let the Trainer drive autocast on CUDA. fp16 also switches on
        # the gradient scaler, without which fp16 gradients underflow.
        bf16=(device.type == "cuda" and dtype == torch.bfloat16),

        fp16=(device.type == "cuda" and dtype == torch.float16),

        report_to="none",

        use_cpu=False,
    )

    # --------------------------------------------------------
    # Create trainer
    # --------------------------------------------------------

    print("\n=== CREATING SFT TRAINER ===")

    trainer = SFTTrainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,
    )

    # --------------------------------------------------------
    # Verify trainable parameters
    # --------------------------------------------------------

    print("\n=== TRAINABLE PARAMETERS ===")

    trainer.model.print_trainable_parameters()

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    checkpoint = find_checkpoint(config.OUTPUT_DIR)

    if checkpoint:
        print(f"\n=== RESUMING FROM {checkpoint} ===")
    else:
        print("\n=== STARTING TRAINING ===")

    trainer.train(resume_from_checkpoint=checkpoint)

    # --------------------------------------------------------
    # Save adapter
    # --------------------------------------------------------

    print("\n=== SAVING MODEL ===")

    trainer.save_model(config.OUTPUT_DIR)

    tokenizer.save_pretrained(config.OUTPUT_DIR)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Adapter saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()