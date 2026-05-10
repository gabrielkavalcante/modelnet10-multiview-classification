from pathlib import Path


DATASET_PATH = Path("ModelNet10_36views_224x224")
EXPECTED_VIEWS = 36


def verify_dataset(dataset_path, expected_views):
    problems = []
    total_objects = 0
    total_images = 0

    for split in ["train", "test"]:
        split_path = dataset_path / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split folder not found: {split_path}")

        for class_path in sorted(split_path.iterdir()):
            if not class_path.is_dir():
                continue

            for object_path in sorted(class_path.iterdir()):
                if not object_path.is_dir():
                    continue

                images = sorted([
                    file for file in object_path.iterdir()
                    if file.suffix.lower() == ".png"
                ])

                total_objects += 1
                total_images += len(images)

                if len(images) != expected_views:
                    problems.append({
                        "split": split,
                        "class": class_path.name,
                        "object": object_path.name,
                        "num_views": len(images)
                    })

    return total_objects, total_images, problems


def main():
    print("Checking rendered dataset")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Expected views per object: {EXPECTED_VIEWS}")

    total_objects, total_images, problems = verify_dataset(
        dataset_path=DATASET_PATH,
        expected_views=EXPECTED_VIEWS
    )

    print(f"Total objects: {total_objects}")
    print(f"Total images: {total_images}")
    print(f"Folders with problems: {len(problems)}")

    if problems:
        print("Examples of problems:")
        for problem in problems[:20]:
            print(problem)
    else:
        print("All object folders have the expected number of views")


if __name__ == "__main__":
    main()