from collections import Counter
from statistics import mean

from datasets import load_dataset


DATASET_NAME = "databricks/databricks-dolly-15k"


def clean_text(value):
    """Convert a value to a normalized string for analysis."""
    if value is None:
        return ""

    return str(value).strip()


def analyze_missing_values(dataset):
    print("\n=== MISSING / EMPTY VALUES ===")

    fields = ["instruction", "context", "response", "category"]

    for field in fields:
        empty_count = 0

        for row in dataset:
            value = clean_text(row[field])

            if not value:
                empty_count += 1

        percentage = (empty_count / len(dataset)) * 100

        print(
            f"{field}: "
            f"{empty_count} empty "
            f"({percentage:.2f}%)"
        )


def analyze_duplicates(dataset):
    print("\n=== DUPLICATES ===")

    seen = set()
    duplicate_count = 0

    for row in dataset:
        instruction = clean_text(row["instruction"])
        context = clean_text(row["context"])
        response = clean_text(row["response"])

        key = (
            instruction,
            context,
            response,
        )

        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)

    percentage = (duplicate_count / len(dataset)) * 100

    print(f"Duplicate examples: {duplicate_count}")
    print(f"Duplicate percentage: {percentage:.2f}%")


def analyze_text_lengths(dataset):
    print("\n=== TEXT LENGTH STATISTICS ===")

    instruction_lengths = []
    context_lengths = []
    response_lengths = []

    for row in dataset:
        instruction = clean_text(row["instruction"])
        context = clean_text(row["context"])
        response = clean_text(row["response"])

        instruction_lengths.append(len(instruction))
        context_lengths.append(len(context))
        response_lengths.append(len(response))

    print("\nInstruction length (characters):")
    print(f"  Min:     {min(instruction_lengths)}")
    print(f"  Average: {mean(instruction_lengths):.2f}")
    print(f"  Max:     {max(instruction_lengths)}")

    print("\nContext length (characters):")
    print(f"  Min:     {min(context_lengths)}")
    print(f"  Average: {mean(context_lengths):.2f}")
    print(f"  Max:     {max(context_lengths)}")

    print("\nResponse length (characters):")
    print(f"  Min:     {min(response_lengths)}")
    print(f"  Average: {mean(response_lengths):.2f}")
    print(f"  Max:     {max(response_lengths)}")


def analyze_short_examples(dataset):
    print("\n=== VERY SHORT EXAMPLES ===")

    # These are only for inspection.
    # We are NOT deleting them yet.
    min_response_length = 20

    short_examples = []

    for index, row in enumerate(dataset):
        instruction = clean_text(row["instruction"])
        response = clean_text(row["response"])

        if len(response) < min_response_length:
            short_examples.append(
                {
                    "index": index,
                    "instruction": instruction,
                    "response": response,
                }
            )

    print(
        f"Responses shorter than "
        f"{min_response_length} characters: "
        f"{len(short_examples)}"
    )

    print("\nFirst 10 short examples:")

    for example in short_examples[:10]:
        print(f"\nIndex: {example['index']}")
        print(f"Instruction: {example['instruction']}")
        print(f"Response: {example['response']}")


def analyze_long_examples(dataset):
    print("\n=== VERY LONG EXAMPLES ===")

    # Again, this is only for inspection.
    max_response_length = 5000

    long_examples = []

    for index, row in enumerate(dataset):
        instruction = clean_text(row["instruction"])
        context = clean_text(row["context"])
        response = clean_text(row["response"])

        total_length = (
            len(instruction)
            + len(context)
            + len(response)
        )

        if total_length > max_response_length:
            long_examples.append(
                {
                    "index": index,
                    "instruction_length": len(instruction),
                    "context_length": len(context),
                    "response_length": len(response),
                    "total_length": total_length,
                }
            )

    print(
        f"Examples longer than "
        f"{max_response_length} characters: "
        f"{len(long_examples)}"
    )

    print("\nFirst 10 long examples:")

    for example in long_examples[:10]:
        print(
            f"\nIndex: {example['index']}"
            f"\nInstruction length: {example['instruction_length']}"
            f"\nContext length: {example['context_length']}"
            f"\nResponse length: {example['response_length']}"
            f"\nTotal length: {example['total_length']}"
        )


def analyze_categories(dataset):
    print("\n=== CATEGORY DISTRIBUTION ===")

    categories = Counter()

    for row in dataset:
        category = clean_text(row["category"])
        categories[category] += 1

    for category, count in categories.most_common():
        percentage = (count / len(dataset)) * 100

        print(
            f"{category}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


def show_empty_examples(dataset):
    print("\n=== EXAMPLES WITH EMPTY INSTRUCTION OR RESPONSE ===")

    found = 0

    for index, row in enumerate(dataset):
        instruction = clean_text(row["instruction"])
        response = clean_text(row["response"])

        if not instruction or not response:
            print(f"\nIndex: {index}")
            print(f"Instruction: {instruction}")
            print(f"Response: {response}")

            found += 1

            if found >= 10:
                break

    if found == 0:
        print("No examples found.")


def main():
    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print("\nDataset loaded successfully.")
    print(f"Total records: {len(dataset)}")

    analyze_missing_values(dataset)
    analyze_duplicates(dataset)
    analyze_text_lengths(dataset)
    analyze_short_examples(dataset)
    analyze_long_examples(dataset)
    analyze_categories(dataset)
    show_empty_examples(dataset)


if __name__ == "__main__":
    main()