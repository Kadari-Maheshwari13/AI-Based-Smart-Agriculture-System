from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR /
    "dataset" /
    "disease_small" /
    "train"
)

OUTPUT_DIR = BASE_DIR / "dataset_inspection"

OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# Settings
# =========================

THUMBNAIL_SIZE = 160
COLUMNS = 5
PADDING = 20
LABEL_HEIGHT = 35


# =========================
# Process each class
# =========================

class_dirs = sorted(
    [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]
)


print("Classes found:")
print()

for class_dir in class_dirs:

    images = sorted(
        [
            file
            for file in class_dir.iterdir()
            if file.suffix.lower() in
            [".jpg", ".jpeg", ".png"]
        ]
    )

    print(
        f"{class_dir.name}: "
        f"{len(images)} images"
    )

    if not images:
        continue


    rows = math.ceil(
        len(images) / COLUMNS
    )


    sheet_width = (
        COLUMNS *
        (THUMBNAIL_SIZE + PADDING)
        + PADDING
    )

    sheet_height = (
        rows *
        (THUMBNAIL_SIZE + LABEL_HEIGHT + PADDING)
        + PADDING
    )


    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )


    draw = ImageDraw.Draw(sheet)


    # -------------------------
    # Add images
    # -------------------------

    for index, image_path in enumerate(images):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                (
                    THUMBNAIL_SIZE,
                    THUMBNAIL_SIZE
                )
            )


            column = index % COLUMNS
            row = index // COLUMNS


            x = (
                PADDING
                + column *
                (THUMBNAIL_SIZE + PADDING)
            )

            y = (
                PADDING
                + row *
                (
                    THUMBNAIL_SIZE
                    + LABEL_HEIGHT
                    + PADDING
                )
            )


            # Center image
            image_x = (
                x
                + (
                    THUMBNAIL_SIZE
                    - image.width
                ) // 2
            )

            image_y = (
                y
                + (
                    THUMBNAIL_SIZE
                    - image.height
                ) // 2
            )


            sheet.paste(
                image,
                (
                    image_x,
                    image_y
                )
            )


            # Image number
            draw.text(
                (
                    x,
                    y + THUMBNAIL_SIZE + 5
                ),
                str(index + 1),
                fill="black"
            )


        except Exception as e:

            print(
                f"Could not open "
                f"{image_path}: {e}"
            )


    # -------------------------
    # Save contact sheet
    # -------------------------

    output_file = (
        OUTPUT_DIR /
        f"{class_dir.name}.jpg"
    )


    sheet.save(
        output_file,
        quality=90
    )


    print(
        f"  Saved: {output_file}"
    )


print()
print(
    "Dataset inspection completed."
)

print(
    "Open this folder:"
)

print(
    OUTPUT_DIR
)