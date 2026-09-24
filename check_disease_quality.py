from pathlib import Path
from PIL import Image


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR /
    "dataset" /
    "disease_small"
)


# =========================
# Settings
# =========================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# =========================
# Counters
# =========================

total_images = 0
valid_images = 0
invalid_images = 0

very_small_images = 0

widths = []
heights = []


# =========================
# Check dataset
# =========================

print("DISEASE DATASET QUALITY CHECK")
print("=" * 60)


for split in ["train", "val", "test"]:

    split_dir = DATASET_DIR / split

    print(f"\n{split.upper()}")
    print("=" * 60)

    for class_dir in sorted(split_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        class_total = 0
        class_invalid = 0
        class_small = 0

        for image_path in class_dir.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            total_images += 1
            class_total += 1

            try:

                with Image.open(image_path) as img:

                    width, height = img.size

                    widths.append(width)
                    heights.append(height)

                    valid_images += 1

                    # Flag extremely small images
                    if width < 100 or height < 100:

                        very_small_images += 1
                        class_small += 1

            except Exception as e:

                invalid_images += 1
                class_invalid += 1

                print(
                    "Invalid image:",
                    image_path
                )

        print(
            f"{class_dir.name:<35} "
            f"Images: {class_total:<4} "
            f"Invalid: {class_invalid:<3} "
            f"Very small: {class_small}"
        )


# =========================
# Summary
# =========================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"Total images: {total_images}"
)

print(
    f"Valid images: {valid_images}"
)

print(
    f"Invalid images: {invalid_images}"
)

print(
    f"Very small images (<100x100): "
    f"{very_small_images}"
)


if widths:

    print(
        f"\nSmallest width: {min(widths)}"
    )

    print(
        f"Largest width: {max(widths)}"
    )

    print(
        f"Smallest height: {min(heights)}"
    )

    print(
        f"Largest height: {max(heights)}"
    )


print("\nDataset quality check completed.")