from datasets import load_dataset
from pathlib import Path
from collections import defaultdict
import random
import shutil

# -----------------------------
# Paths
# -----------------------------
CACHE_ROOT = Path(
    r"C:\Users\Maheshwari\.cache\huggingface\hub\datasets--DigiGreen--Crop_Disease_Images"
    r"\snapshots\2b18be861bddeb525c83cf2d7e07eb7f649dfed9"
)

OUTPUT_ROOT = Path(
    r"D:\AI_Agriculture_Project\dataset\disease_small"
)

# -----------------------------
# Selected classes
# -----------------------------
CLASSES = [
    "Healthy",
    "Leaf Curl Virus",
    "Fall Armyworm",
    "Aphids",
    "Thrips",
    "Powdery Mildew",
    "Mites",
    "Shoot and Fruit Borer",
    "Anthracnose",
    "Mosaic Virus",
]

CLASS_NAMES = {
    "Healthy": "Healthy",
    "Leaf Curl Virus": "Leaf_Curl_Virus",
    "Fall Armyworm": "Fall_Armyworm",
    "Aphids": "Aphids",
    "Thrips": "Thrips",
    "Powdery Mildew": "Powdery_Mildew",
    "Mites": "Mites",
    "Shoot and Fruit Borer": "Shoot_and_Fruit_Borer",
    "Anthracnose": "Anthracnose",
    "Mosaic Virus": "Mosaic_Virus",
}

# -----------------------------
# Load metadata
# -----------------------------
print("Loading dataset metadata...")

ds = load_dataset(
    "DigiGreen/Crop_Disease_Images",
    split="train"
)

# -----------------------------
# Collect unique image records
# -----------------------------
records = []
seen_images = set()

for row in ds:
    diagnosis = row["diagnosis"]
    image_file = row["image_file"]

    # Only clean single-diagnosis records
    if ";" in diagnosis:
        continue

    if diagnosis not in CLASSES:
        continue

    # Prevent the same image from appearing more than once
    if image_file in seen_images:
        continue

    source = CACHE_ROOT / image_file

    if not source.exists():
        print("WARNING: Missing:", source)
        continue

    seen_images.add(image_file)

    records.append({
        "image_file": image_file,
        "diagnosis": diagnosis,
        "source": source
    })

print(f"Usable unique images: {len(records)}")

# -----------------------------
# Group by class
# -----------------------------
by_class = defaultdict(list)

for record in records:
    by_class[record["diagnosis"]].append(record)

print("\nImages per class:")
for cls in CLASSES:
    print(f"{cls}: {len(by_class[cls])}")

# -----------------------------
# Create output folders
# -----------------------------
if OUTPUT_ROOT.exists():
    print("\nRemoving previous disease_small dataset...")
    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (OUTPUT_ROOT / split / CLASS_NAMES[cls]).mkdir(
            parents=True,
            exist_ok=True
        )

# -----------------------------
# Reproducible split
# -----------------------------
random.seed(42)

for cls in CLASSES:

    items = by_class[cls].copy()

    random.shuffle(items)

    total = len(items)

    train_count = int(total * 0.70)
    val_count = int(total * 0.15)

    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]

    splits = {
        "train": train_items,
        "val": val_items,
        "test": test_items
    }

    for split_name, split_items in splits.items():

        destination_dir = (
            OUTPUT_ROOT
            / split_name
            / CLASS_NAMES[cls]
        )

        for index, item in enumerate(split_items):

            source = item["source"]

            # Preserve original extension
            extension = source.suffix.lower()

            destination = (
                destination_dir
                / f"{index:04d}{extension}"
            )

            shutil.copy2(source, destination)

# -----------------------------
# Print final distribution
# -----------------------------
print("\n================================")
print("DATASET PREPARATION COMPLETE")
print("================================")

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    total = 0

    for cls in CLASSES:

        folder = OUTPUT_ROOT / split / CLASS_NAMES[cls]

        count = len(list(folder.glob("*")))

        total += count

        print(f"{CLASS_NAMES[cls]:25} {count}")

    print(f"Total: {total}")

print("\nDataset location:")
print(OUTPUT_ROOT)