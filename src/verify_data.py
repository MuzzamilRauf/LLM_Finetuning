from datasets import load_from_disk


DATASET_PATH = "data/processed/dolly-prepared"


def verify_dataset(dataset):
    print("\n=== DATASET STRUCTURE ===")
    print(dataset)

    for split in ["train", "test"]:
        print(f"\n=== {split.upper()} ===")
        print(f"Examples: {len(dataset[split])}")
        print(f"Columns: {dataset[split].column_names}")

    print("\n=== CHECKING MESSAGE FORMAT ===")

    problems = []

    for split in ["train", "test"]:
        for index, example in enumerate(dataset[split]):
            messages = example["messages"]

            # Must contain messages
            if not messages:
                problems.append(
                    (split, index, "empty_messages")
                )
                continue

            # We expect exactly two messages:
            # user -> assistant
            if len(messages) != 2:
                problems.append(
                    (split, index, "unexpected_message_count")
                )
                continue

            user_message = messages[0]
            assistant_message = messages[1]

            # Check roles
            if user_message["role"] != "user":
                problems.append(
                    (split, index, "first_message_not_user")
                )

            if assistant_message["role"] != "assistant":
                problems.append(
                    (split, index, "second_message_not_assistant")
                )

            # Check content
            if not user_message["content"].strip():
                problems.append(
                    (split, index, "empty_user_content")
                )

            if not assistant_message["content"].strip():
                problems.append(
                    (split, index, "empty_assistant_content")
                )

    if problems:
        print(
            f"Found {len(problems)} problems."
        )

        print("\nFirst 10 problems:")

        for problem in problems[:10]:
            print(problem)

    else:
        print(
            "All examples passed the message format checks."
        )


def show_examples(dataset):
    print("\n=== SAMPLE TRAINING EXAMPLES ===")

    for index in range(min(3, len(dataset["train"]))):
        example = dataset["train"][index]

        print(f"\n--- Example {index + 1} ---")

        for message in example["messages"]:
            print(
                f"\nRole: {message['role']}"
            )

            print(
                f"Content:\n{message['content']}"
            )


def main():
    print("Loading prepared dataset...")

    dataset = load_from_disk(DATASET_PATH)

    verify_dataset(dataset)
    show_examples(dataset)


if __name__ == "__main__":
    main()