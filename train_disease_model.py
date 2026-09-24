from pathlib import Path
import json
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.metrics import f1_score


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset" / "disease_clean"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


MODEL_PATH = MODEL_DIR / "disease_mobilenetv3_exp6.pth"
CLASS_PATH = MODEL_DIR / "disease_classes_exp6.json"


# ============================================================
# Settings
# ============================================================

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 16

EPOCHS = 30
PATIENCE = 7

LEARNING_RATE = 0.00005

NUM_WORKERS = 0


# ============================================================
# Transforms
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# Datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)


class_names = train_dataset.classes

print()
print("Classes:")
for i, name in enumerate(class_names):
    print(i, name)

print()
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print()


# ============================================================
# Balanced sampler
# ============================================================

class_counts = [0] * len(class_names)

for _, label in train_dataset.samples:
    class_counts[label] += 1


print("Class counts:")
for name, count in zip(class_names, class_counts):
    print(f"{name}: {count}")

print()


class_weights = [
    1.0 / count
    for count in class_counts
]


sample_weights = [
    class_weights[label]
    for _, label in train_dataset.samples
]


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# Model
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(
    weights=weights
)


# Freeze everything first

for parameter in model.features.parameters():
    parameter.requires_grad = False


# Unfreeze the last 6 feature blocks

for block in model.features[-6:]:
    for parameter in block.parameters():
        parameter.requires_grad = True


# Replace classifier

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names)
)


model = model.to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# Training
# ============================================================

best_f1 = 0.0
best_state = None

patience_counter = 0


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        total_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_accuracy = (
        correct / total
    ) * 100


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_predictions = []
    val_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_labels.extend(
                labels.numpy()
            )


    val_accuracy = (
        sum(
            p == y
            for p, y in zip(
                val_predictions,
                val_labels
            )
        )
        / len(val_labels)
    ) * 100


    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro",
        zero_division=0
    )


    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"Macro F1: {val_f1:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        best_state = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

        print(
            f"  New best model! "
            f"Macro F1 = {best_f1:.4f}"
        )

    else:

        patience_counter += 1


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print()
        print("Early stopping.")

        break


# ============================================================
# Save model
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )


with open(
    CLASS_PATH,
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )


# ============================================================
# Finished
# ============================================================

print()
print("==============================")
print("Training completed")
print("==============================")
print("Best Validation Macro F1:", round(best_f1, 4))
print("Model saved:", MODEL_PATH)
print("Classes saved:", CLASS_PATH)