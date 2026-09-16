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

import torch

from datasets import load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import config


def get_device():
    """
    Select the best available device.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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


def load_model(device):
    """
    Load Qwen2.5-3B.

    float16 is used on Apple Silicon to significantly reduce memory
    usage compared with float32.
    """

    print("\n=== LOADING MODEL ===")

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        dtype=torch.float16,
    )

    model.to(device)

    # Required for training because we are not using the KV cache.
    model.config.use_cache = False

    print("Model loaded successfully.")

    return model


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

    model = load_model(device)

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

        gradient_checkpointing=True,

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

    print("\n=== STARTING TRAINING ===")

    trainer.train()

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