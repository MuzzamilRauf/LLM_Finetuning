from datasets import load_dataset


DATASET_NAME = "databricks/databricks-dolly-15k"


def main():
    dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    print("\n=== DATASET ===")
    print(dataset)

    print("\n=== COLUMNS ===")
    print(dataset.column_names)

    print("\n=== FIRST EXAMPLE ===")
    print(dataset[0])

    print("\n=== NUMBER OF RECORDS ===")
    print(len(dataset))

    print("\n=== CATEGORIES ===")
    categories = dataset.unique("category")
    print(categories)


if __name__ == "__main__":
    main()