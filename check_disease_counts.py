from pathlib import Path

# Prepared dataset folder
DATASET_DIR = Path("dataset/disease_small")

print("Checking prepared disease dataset...")
print("=" * 60)

if not DATASET_DIR.exists():
    print(f"ERROR: Folder '{DATASET_DIR}' not found.")
    exit()

splits = ["train", "val", "test"]

total_all = 0
all_classes = set()

for split in splits:
    split_dir = DATASET_DIR / split

    print(f"\n{split.upper()}")
    print("=" * 60)

    if not split_dir.exists():
        print(f"ERROR: {split} folder not found.")
        continue

    split_total = 0

    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():

            count = sum(
                1
                for file in class_dir.iterdir()
                if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            )

            print(f"{class_dir.name:<40} {count}")

            split_total += count
            all_classes.add(class_dir.name)

    print("-" * 60)
    print(f"{split.upper()} TOTAL: {split_total}")

    total_all += split_total

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Total images: {total_all}")
print(f"Total classes: {len(all_classes)}")

print("\nClasses:")
for class_name in sorted(all_classes):
    print(f"- {class_name}")

print("=" * 60)