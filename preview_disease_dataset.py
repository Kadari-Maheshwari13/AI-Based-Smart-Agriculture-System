from pathlib import Path
import random

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "disease_small" / "train"

classes = sorted([
    folder.name
    for folder in TRAIN_DIR.iterdir()
    if folder.is_dir()
])

random.seed(42)

fig, axes = plt.subplots(
    10, 3,
    figsize=(10, 28)
)

for row, class_name in enumerate(classes):

    class_dir = TRAIN_DIR / class_name

    images = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    selected = random.sample(
        images,
        min(3, len(images))
    )

    for col in range(3):

        ax = axes[row, col]
        ax.axis("off")

        if col < len(selected):

            image = Image.open(selected[col]).convert("RGB")

            ax.imshow(image)

            if col == 0:
                ax.set_title(
                    class_name,
                    fontsize=10,
                    fontweight="bold"
                )


plt.tight_layout()

output_path = BASE_DIR / "disease_dataset_preview.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Preview created successfully:")
print(output_path)