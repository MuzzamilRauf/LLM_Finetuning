from datasets import load_from_disk


INPUT_PATH = "data/processed/dolly-cleaned"
OUTPUT_PATH = "data/processed/dolly-prepared"

TEST_SIZE = 0.10
SEED = 42


def create_messages(example):
    """
    Convert one Dolly example into conversational format.

    Dolly:
        instruction
        context
        response

    Becomes:
        messages = [
            {
                "role": "user",
                "content": ...
            },
            {
                "role": "assistant",
                "content": ...
            }
        ]
    """

    instruction = example["instruction"]
    context = example["context"]
    response = example["response"]

    if context:
        user_content = (
            f"{instruction}\n\n"
            f"Context:\n{context}"
        )
    else:
        user_content = instruction

    return {
        "messages": [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]
    }


def main():
    print("Loading cleaned dataset...")

    dataset = load_from_disk(INPUT_PATH)

    print("\n=== CLEANED DATASET ===")
    print(dataset)
    print(f"Total examples: {len(dataset)}")

    # --------------------------------------------------
    # 1. Create train/evaluation split
    # --------------------------------------------------

    print("\nCreating train/evaluation split...")

    split_dataset = dataset.train_test_split(
        test_size=TEST_SIZE,
        seed=SEED,
    )

    print("\n=== SPLIT ===")
    print(split_dataset)

    print(
        f"Training examples: "
        f"{len(split_dataset['train'])}"
    )

    print(
        f"Evaluation examples: "
        f"{len(split_dataset['test'])}"
    )

    # --------------------------------------------------
    # 2. Convert to conversational format
    # --------------------------------------------------

    print("\nConverting examples to messages format...")

    prepared_dataset = split_dataset.map(
        create_messages,
        remove_columns=dataset.column_names,
    )

    # --------------------------------------------------
    # 3. Show examples
    # --------------------------------------------------

    print("\n=== PREPARED DATASET ===")
    print(prepared_dataset)

    print("\n=== FIRST TRAINING EXAMPLE ===")

    print(
        prepared_dataset["train"][0]
    )

    print("\n=== FIRST EVALUATION EXAMPLE ===")

    print(
        prepared_dataset["test"][0]
    )

    # --------------------------------------------------
    # 4. Save prepared dataset
    # --------------------------------------------------

    print(
        f"\nSaving prepared dataset to: "
        f"{OUTPUT_PATH}"
    )

    prepared_dataset.save_to_disk(
        OUTPUT_PATH
    )

    print("\nPreparation completed successfully.")


if __name__ == "__main__":
    main()

