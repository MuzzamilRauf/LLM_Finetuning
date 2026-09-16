from collections import Counter

from datasets import Dataset, load_dataset


DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_PATH = "data/processed/dolly-cleaned"


def clean_text(value):
    """
    Normalize basic whitespace.

    Example:
        "   Explain   Python   "
    becomes:
        "Explain Python"
    """

    if value is None:
        return ""

    return " ".join(str(value).split())


def clean_dataset(dataset):
    """
    Clean the Dolly dataset.

    We:
    - remove empty instructions
    - remove empty responses
    - normalize whitespace
    - remove exact duplicate examples

    We intentionally do NOT remove:
    - short valid responses
    - long valid examples
    - examples with empty context
    """

    cleaned_records = []
    removed_records = []
    seen = set()

    for index, row in enumerate(dataset):
        instruction = clean_text(row["instruction"])
        context = clean_text(row["context"])
        response = clean_text(row["response"])
        category = clean_text(row["category"])

        # Remove examples with no instruction.
        if not instruction:
            removed_records.append(
                (index, "empty_instruction")
            )
            continue

        # Remove examples with no response.
        if not response:
            removed_records.append(
                (index, "empty_response")
            )
            continue

        # Identify exact duplicate examples.
        duplicate_key = (
            instruction,
            context,
            response,
        )

        if duplicate_key in seen:
            removed_records.append(
                (index, "duplicate")
            )
            continue

        seen.add(duplicate_key)

        cleaned_records.append(
            {
                "instruction": instruction,
                "context": context,
                "response": response,
                "category": category,
            }
        )

    return cleaned_records, removed_records


def print_summary(
    original_count,
    cleaned_count,
    removed_records,
):
    print("\n=== CLEANING SUMMARY ===")

    removed_count = len(removed_records)

    print(f"Original records: {original_count}")
    print(f"Cleaned records:  {cleaned_count}")
    print(f"Removed records:  {removed_count}")

    if original_count > 0:
        percentage = (
            removed_count / original_count
        ) * 100

        print(
            f"Removal percentage: "
            f"{percentage:.2f}%"
        )

    print("\n=== REMOVAL REASONS ===")

    reasons = Counter(
        reason
        for _, reason in removed_records
    )

    if not reasons:
        print("No records were removed.")
        return

    for reason, count in reasons.most_common():
        print(f"{reason}: {count}")


def main():
    print("Loading Dolly dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print(
        f"Original dataset size: "
        f"{len(dataset)}"
    )

    print("\nCleaning dataset...")

    cleaned_records, removed_records = clean_dataset(
        dataset
    )

    cleaned_dataset = Dataset.from_list(
        cleaned_records
    )

    print_summary(
        original_count=len(dataset),
        cleaned_count=len(cleaned_dataset),
        removed_records=removed_records,
    )

    print(
        f"\nSaving cleaned dataset to: "
        f"{OUTPUT_PATH}"
    )

    cleaned_dataset.save_to_disk(
        OUTPUT_PATH
    )

    print("\nCleaning completed successfully.")

    print("\n=== CLEANED DATASET ===")
    print(cleaned_dataset)

    print("\n=== FIRST CLEANED EXAMPLE ===")
    print(cleaned_dataset[0])


if __name__ == "__main__":
    main()
