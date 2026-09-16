"""
Analyze the real token lengths of the prepared Dolly dataset.

Why this script exists:
    Our prepared dataset contains conversational messages:

        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]

    However, the Qwen2.5-3B base model does not provide the same
    conversational chat template as the Instruct model.

    Therefore, for this project we explicitly format each conversation
    as training text and then use the Qwen tokenizer to convert that text
    into token IDs.

    This script:
        1. Loads the prepared Dolly dataset.
        2. Loads the Qwen2.5-3B tokenizer.
        3. Converts each conversation into training text.
        4. Tokenizes the complete training text.
        5. Calculates the real token length.
        6. Reports token-length statistics.
        7. Shows how many examples exceed common max_length values.
        8. Displays the longest examples.

    The results will help us choose an appropriate max_length for
    LoRA/SFT training on the 16 GB Mac.
"""

from statistics import mean

from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B"
DATASET_PATH = "data/processed/dolly-prepared"


def build_training_text(example):
    """
    Convert the structured messages into the text format
    that we will use for this training experiment.
    """

    messages = example["messages"]

    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]

    return (
        "<|im_start|>user\n"
        f"{user_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_content}"
        "<|im_end|>"
    )


def get_token_length(example, tokenizer):
    """
    Build the training text and tokenize it.

    The returned value is the number of tokens that the model
    will receive for this example.
    """

    training_text = build_training_text(example)

    token_ids = tokenizer(
        training_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    return len(token_ids)


def analyze_split(dataset, tokenizer, split_name):
    """
    Analyze token lengths for one dataset split.
    """

    print(f"\n=== {split_name.upper()} TOKEN ANALYSIS ===")

    lengths = []

    for example in dataset[split_name]:
        token_length = get_token_length(example, tokenizer)
        lengths.append(token_length)

    print(f"Examples: {len(lengths)}")
    print(f"Minimum tokens: {min(lengths)}")
    print(f"Average tokens: {mean(lengths):.2f}")
    print(f"Maximum tokens: {max(lengths)}")

    print("\n=== LENGTH THRESHOLDS ===")

    thresholds = [128, 256, 512, 1024, 2048, 4096, 8192]

    for threshold in thresholds:
        count = sum(length > threshold for length in lengths)
        percentage = (count / len(lengths)) * 100

        print(
            f">{threshold:4} tokens: "
            f"{count:5} examples ({percentage:6.2f}%)"
        )

    print("\n=== LENGTH DISTRIBUTION ===")

    ranges = [
        ("0-128", 0, 128),
        ("129-256", 129, 256),
        ("257-512", 257, 512),
        ("513-1024", 513, 1024),
        ("1025-2048", 1025, 2048),
        ("2049-4096", 2049, 4096),
        (">4096", 4097, float("inf")),
    ]

    for label, minimum, maximum in ranges:
        count = sum(
            minimum <= length <= maximum
            for length in lengths
        )

        percentage = (count / len(lengths)) * 100

        print(
            f"{label:>10}: "
            f"{count:5} examples ({percentage:6.2f}%)"
        )

    return lengths


def show_long_examples(dataset, tokenizer, split_name, limit=10):
    """
    Display the longest examples so we can inspect them.
    """

    examples_with_lengths = []

    for index, example in enumerate(dataset[split_name]):
        token_length = get_token_length(example, tokenizer)

        examples_with_lengths.append(
            (token_length, index, example)
        )

    examples_with_lengths.sort(
        reverse=True,
        key=lambda item: item[0],
    )

    print(f"\n=== LONGEST {split_name.upper()} EXAMPLES ===")

    for token_length, index, example in examples_with_lengths[:limit]:
        print(f"\nIndex: {index}")
        print(f"Tokens: {token_length}")

        print("\nUser:")
        print(example["messages"][0]["content"][:300])

        print("\nAssistant:")
        print(example["messages"][1]["content"][:300])


def main():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")

    print("\nLoading prepared dataset...")

    dataset = load_from_disk(DATASET_PATH)

    print("Dataset loaded successfully.")
    print(dataset)

    train_lengths = analyze_split(
        dataset,
        tokenizer,
        "train",
    )

    test_lengths = analyze_split(
        dataset,
        tokenizer,
        "test",
    )

    show_long_examples(
        dataset,
        tokenizer,
        "train",
    )

    print("\n=== FINAL SUMMARY ===")
    print(f"Train examples: {len(train_lengths)}")
    print(f"Test examples: {len(test_lengths)}")
    print(f"Longest train example: {max(train_lengths)} tokens")
    print(f"Longest test example: {max(test_lengths)} tokens")


if __name__ == "__main__":
    main()