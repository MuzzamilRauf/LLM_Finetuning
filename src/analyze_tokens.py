from collections import Counter
from statistics import mean

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B"
DATASET_NAME = "databricks/databricks-dolly-15k"


def build_text(row):
    """
    Combine instruction, optional context, and response
    into the text that represents one training example.
    """

    instruction = str(row["instruction"]).strip()
    context = str(row["context"]).strip()
    response = str(row["response"]).strip()

    if context:
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Response:\n{response}"
        )

    return (
        f"Instruction:\n{instruction}\n\n"
        f"Response:\n{response}"
    )


def analyze_token_lengths(dataset, tokenizer):
    print("\n=== TOKEN LENGTH ANALYSIS ===")

    token_lengths = []

    for row in dataset:
        text = build_text(row)

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        token_lengths.append(len(tokens["input_ids"]))

    print(f"\nNumber of examples: {len(token_lengths)}")
    print(f"Minimum tokens: {min(token_lengths)}")
    print(f"Average tokens: {mean(token_lengths):.2f}")
    print(f"Maximum tokens: {max(token_lengths)}")


def analyze_thresholds(dataset, tokenizer):
    print("\n=== TOKEN LENGTH THRESHOLDS ===")

    token_lengths = []

    for row in dataset:
        text = build_text(row)

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        token_lengths.append(len(tokens["input_ids"]))

    thresholds = [256, 512, 1024, 2048, 4096, 8192]

    for threshold in thresholds:
        count = sum(
            length > threshold
            for length in token_lengths
        )

        percentage = (count / len(token_lengths)) * 100

        print(
            f"> {threshold:4} tokens: "
            f"{count:5} examples "
            f"({percentage:.2f}%)"
        )


def show_long_examples(dataset, tokenizer, limit=10):
    print("\n=== LONGEST EXAMPLES ===")

    examples = []

    for index, row in enumerate(dataset):
        text = build_text(row)

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        token_count = len(tokens["input_ids"])

        examples.append(
            (
                token_count,
                index,
                row["instruction"],
            )
        )

    examples.sort(reverse=True)

    for token_count, index, instruction in examples[:limit]:
        print(f"\nIndex: {index}")
        print(f"Tokens: {token_count}")
        print(f"Instruction: {instruction[:200]}")


def show_token_length_distribution(dataset, tokenizer):
    print("\n=== TOKEN LENGTH DISTRIBUTION ===")

    token_lengths = []

    for row in dataset:
        text = build_text(row)

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        token_lengths.append(len(tokens["input_ids"]))

    buckets = [
        (0, 128),
        (129, 256),
        (257, 512),
        (513, 1024),
        (1025, 2048),
        (2049, 4096),
        (4097, float("inf")),
    ]

    for lower, upper in buckets:
        count = sum(
            lower <= length <= upper
            for length in token_lengths
        )

        if upper == float("inf"):
            label = f"> 4096"
        else:
            label = f"{lower}-{upper}"

        percentage = (count / len(token_lengths)) * 100

        print(
            f"{label:>10} tokens: "
            f"{count:5} examples "
            f"({percentage:.2f}%)"
        )


def main():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print("Dataset loaded.")
    print(f"Total examples: {len(dataset)}")

    analyze_token_lengths(
        dataset,
        tokenizer,
    )

    analyze_thresholds(
        dataset,
        tokenizer,
    )

    show_token_length_distribution(
        dataset,
        tokenizer,
    )

    show_long_examples(
        dataset,
        tokenizer,
    )


if __name__ == "__main__":
    main()