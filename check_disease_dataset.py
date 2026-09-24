from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "disease_clean"

image_files = list(DATASET_DIR.rglob("*.jpg"))

print("Total JPG images:", len(image_files))
print()

# =========================================
# 1. Check image properties
# =========================================

corrupted = []
small_images = []
modes = defaultdict(int)

for file in image_files:

    try:
        with Image.open(file) as img:

            width, height = img.size

            modes[img.mode] += 1

            if width < 200 or height < 200:
                small_images.append(
                    (file, img.size)
                )

            img.verify()

    except Exception as e:

        corrupted.append(
            (file, str(e))
        )


print("IMAGE MODES")
print("===========")

for mode, count in modes.items():
    print(f"{mode}: {count}")


print()
print("CORRUPTED IMAGES")
print("================")

if corrupted:

    for file, error in corrupted:
        print(file, error)

else:

    print("None")


print()
print("VERY SMALL IMAGES")
print("=================")

if small_images:

    for file, size in small_images:
        print(file, size)

else:

    print("None")


# =========================================
# 2. Find exact duplicate images
# =========================================

print()
print("CHECKING EXACT DUPLICATES...")
print()

hashes = defaultdict(list)

for file in image_files:

    try:

        data = file.read_bytes()

        file_hash = hashlib.md5(
            data
        ).hexdigest()

        hashes[file_hash].append(file)

    except Exception:
        pass


duplicate_groups = [
    files
    for files in hashes.values()
    if len(files) > 1
]


print(
    "Duplicate groups:",
    len(duplicate_groups)
)


if duplicate_groups:

    for i, group in enumerate(
        duplicate_groups,
        start=1
    ):

        print()
        print(
            f"Duplicate group {i}:"
        )

        for file in group:
            print(
                " ",
                file
            )


# =========================================
# 3. Check duplicates across classes
# =========================================

print()
print("DUPLICATES ACROSS DIFFERENT CLASSES")
print("===================================")

cross_class_count = 0

for group in duplicate_groups:

    classes = {
        file.parent.name
        for file in group
    }

    if len(classes) > 1:

        cross_class_count += 1

        print()

        print(
            "Classes:",
            ", ".join(sorted(classes))
        )

        for file in group:
            print(
                " ",
                file
            )


print()
print(
    "Cross-class duplicate groups:",
    cross_class_count
)

print()
print("Dataset check completed.")