from pathlib import Path
import hashlib
import shutil
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "dataset" / "disease_small"
CLEAN_DIR = BASE_DIR / "dataset" / "disease_clean"

# Remove old clean dataset if it exists
if CLEAN_DIR.exists():
    shutil.rmtree(CLEAN_DIR)

# Copy original dataset first
shutil.copytree(SOURCE_DIR, CLEAN_DIR)

print("Original dataset copied to:")
print(CLEAN_DIR)
print()

# =========================================
# Find exact duplicates
# =========================================

image_files = list(CLEAN_DIR.rglob("*.jpg"))

hashes = defaultdict(list)

for file in image_files:
    data = file.read_bytes()
    file_hash = hashlib.md5(data).hexdigest()
    hashes[file_hash].append(file)

duplicate_groups = [
    files
    for files in hashes.values()
    if len(files) > 1
]

print("Duplicate groups found:", len(duplicate_groups))
print()

removed = []

# =========================================
# Remove duplicates
# =========================================

for group in duplicate_groups:

    # Group files by class
    class_groups = defaultdict(list)

    for file in group:
        class_name = file.parent.name
        class_groups[class_name].append(file)

    # -----------------------------------------
    # Cross-class duplicate
    # -----------------------------------------

    if len(class_groups) > 1:

        print("CROSS-CLASS DUPLICATE:")
        for class_name, files in class_groups.items():
            for file in files:
                print(" ", class_name, "->", file)

        # Keep the first class alphabetically
        keep_class = sorted(class_groups.keys())[0]

        print("Keeping class:", keep_class)

        for class_name, files in class_groups.items():

            if class_name != keep_class:

                for file in files:
                    file.unlink()
                    removed.append(file)
                    print("Removed:", file)

        print()

    # -----------------------------------------
    # Same-class duplicates
    # -----------------------------------------

    for class_name, files in class_groups.items():

        # Re-check files that still exist
        existing_files = [
            file for file in files
            if file.exists()
        ]

        if len(existing_files) > 1:

            # Keep first file
            keep = existing_files[0]

            for file in existing_files[1:]:

                if file.exists():
                    file.unlink()
                    removed.append(file)

                    print(
                        f"Removed duplicate from {class_name}:"
                    )
                    print(" ", file)

print()
print("========================================")
print("CLEANING COMPLETED")
print("========================================")

print("Files removed:", len(removed))
print("Clean dataset:", CLEAN_DIR)