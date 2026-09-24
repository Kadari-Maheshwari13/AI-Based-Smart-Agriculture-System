from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small


# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset" / "disease_clean"
MODEL_PATH = BASE_DIR / "models" / "disease_mobilenetv3_exp5.pth"


# =========================
# Settings
# =========================

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224


# =========================
# Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# Dataset
# =========================

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)

class_names = test_dataset.classes


# =========================
# Model
# =========================

model = mobilenet_v3_small(
    weights=None
)

num_classes = len(class_names)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# =========================
# Confusion summary
# =========================

confusions = Counter()

correct_count = 0
wrong_count = 0


with torch.no_grad():

    for index in range(len(test_dataset)):

        image, true_label = test_dataset[index]

        image = image.unsqueeze(0).to(DEVICE)

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_label = torch.argmax(
            probabilities,
            dim=1
        ).item()


        actual = class_names[true_label]
        predicted = class_names[predicted_label]


        if predicted_label == true_label:

            correct_count += 1

        else:

            wrong_count += 1

            confusions[
                (actual, predicted)
            ] += 1


# =========================
# Print results
# =========================

print()
print("DISEASE CONFUSION SUMMARY")
print("==========================")
print()

print("Correct predictions:", correct_count)
print("Wrong predictions  :", wrong_count)

print()
print("Most common confusions:")
print("------------------------")

for (actual, predicted), count in confusions.most_common():

    print(
        f"{actual:25s} -> "
        f"{predicted:25s} : {count}"
    )

print()
print("==========================")