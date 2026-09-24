from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset" / "disease_clean" / "test"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = (
    MODEL_DIR /
    "disease_mobilenetv3_weightedloss.pth"
)

CLASS_PATH = (
    MODEL_DIR /
    "disease_classes_weightedloss.json"
)

CONFUSION_MATRIX_PATH = (
    BASE_DIR /
    "disease_confusion_matrix_weightedloss.png"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(f"{i}: {class_name}")

num_classes = len(class_names)


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=test_transform
)

print("\nTest images:", len(test_dataset))

print("\nDataset classes:")
print(test_dataset.classes)


# ============================================================
# DATA LOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# ============================================================
# CREATE MODEL
# ============================================================

model = mobilenet_v3_small(weights=None)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)


# ============================================================
# CHECK CHECKPOINT CLASSIFIER STRUCTURE
# ============================================================

print("\nCheckpoint classifier keys:")

for key in checkpoint.keys():

    if key.startswith("classifier"):

        if hasattr(checkpoint[key], "shape"):
            print(key, tuple(checkpoint[key].shape))

        else:
            print(key)


# ============================================================
# DETERMINE CLASSIFIER STRUCTURE
# ============================================================

if (
    "classifier.3.weight" in checkpoint
    and
    "classifier.3.bias" in checkpoint
):

    print("\nDetected standard classifier structure.")

    input_features = checkpoint["classifier.3.weight"].shape[1]

    model.classifier[3] = nn.Linear(
        input_features,
        num_classes
    )


elif (
    "classifier.3.1.weight" in checkpoint
    and
    "classifier.3.1.bias" in checkpoint
):

    print("\nDetected Sequential classifier structure.")

    input_features = checkpoint["classifier.3.1.weight"].shape[1]

    model.classifier[3] = nn.Sequential(
        nn.Identity(),
        nn.Linear(
            input_features,
            num_classes
        )
    )


else:

    print("\nERROR: Unknown classifier structure.")

    print("\nAvailable classifier keys:")

    for key in checkpoint.keys():

        if key.startswith("classifier"):

            if hasattr(checkpoint[key], "shape"):
                print(
                    key,
                    tuple(checkpoint[key].shape)
                )

            else:
                print(key)

    raise RuntimeError(
        "Could not determine the classifier structure."
    )


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []


print("\nRunning evaluation...")


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_labels = np.array(all_labels)

all_predictions = np.array(all_predictions)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# ============================================================
# MACRO F1
# ============================================================

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# WEIGHTED F1
# ============================================================

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print("WEIGHTED LOSS MODEL RESULTS")

print("=" * 60)

print(
    f"\nAccuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1    : {macro_f1:.4f}"
)

print(
    f"Weighted F1 : {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
)


print("\nConfusion Matrix:\n")

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(12, 10)
)

plt.imshow(cm)

plt.title(
    "Disease Detection - Weighted Loss Model"
)

plt.colorbar()

plt.xticks(
    range(num_classes),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(num_classes),
    class_names
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")


# Add numbers inside cells

for i in range(num_classes):

    for j in range(num_classes):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()


print(
    f"\nConfusion matrix saved to:"
)

print(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# FINAL
# ============================================================

print("\nEvaluation completed successfully.")