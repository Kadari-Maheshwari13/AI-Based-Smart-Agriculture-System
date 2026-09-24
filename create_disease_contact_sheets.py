from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset" / "disease_clean" / "train"

OUTPUT_DIR = BASE_DIR / "disease_contact_sheets"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Settings
# ============================================================

THUMBNAIL_SIZE = 160
COLUMNS = 5

BACKGROUND = "white"
TEXT_COLOR = "black"


# ============================================================
# Create sheets
# ============================================================

class_dirs = sorted(
    [
        p for p in DATA_DIR.iterdir()
        if p.is_dir()
    ]
)


for class_dir in class_dirs:

    image_files = sorted(
        list(class_dir.glob("*.jpg")) +
        list(class_dir.glob("*.jpeg")) +
        list(class_dir.glob("*.png"))
    )

    if not image_files:
        continue


    rows = math.ceil(
        len(image_files) / COLUMNS
    )


    cell_width = THUMBNAIL_SIZE
    cell_height = THUMBNAIL_SIZE + 30


    sheet_width = COLUMNS * cell_width

    sheet_height = rows * cell_height


    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        BACKGROUND
    )


    draw = ImageDraw.Draw(sheet)


    for index, image_path in enumerate(image_files):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image.thumbnail(
                (
                    THUMBNAIL_SIZE - 10,
                    THUMBNAIL_SIZE - 10
                )
            )


            x = (
                index % COLUMNS
            ) * cell_width

            y = (
                index // COLUMNS
            ) * cell_height


            image_x = (
                x
                + (cell_width - image.width) // 2
            )

            image_y = y + 5


            sheet.paste(
                image,
                (
                    image_x,
                    image_y
                )
            )


            draw.text(
                (
                    x + 5,
                    y + THUMBNAIL_SIZE
                ),
                image_path.stem,
                fill=TEXT_COLOR
            )


        except Exception as e:

            print(
                "Could not open:",
                image_path,
                e
            )


    output_path = (
        OUTPUT_DIR
        / f"{class_dir.name}.jpg"
    )


    sheet.save(
        output_path,
        quality=90
    )


    print(
        f"Created: {output_path}"
    )


print()
print("Contact sheets created.")
print("Folder:", OUTPUT_DIR)