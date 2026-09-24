from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "disease_small"

splits = ["train", "val", "test"]

classes = sorted([
    folder.name
    for folder in (DATASET_DIR / "train").iterdir()
    if folder.is_dir()
])

for class_name in classes:
    print("\n" + "=" * 50)
    print(class_name)

    for split in splits:
        class_dir = DATASET_DIR / split / class_name

        if class_dir.exists():
            count = len([
                p for p in class_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ])
        else:
            count = 0

        print(f"{split:5}: {count}")

print("\nSplit check completed.")