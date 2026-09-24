from datasets import load_dataset
from collections import Counter

print("Loading original dataset metadata...")

ds = load_dataset(
    "DigiGreen/Crop_Disease_Images",
    split="train"
)

counts = Counter()

for row in ds:
    diagnosis = row["diagnosis"]

    if ";" not in diagnosis:
        counts[diagnosis] += 1

print("\nOriginal dataset class counts:")
print("=" * 50)

for diagnosis, count in counts.most_common():
    print(f"{diagnosis:40} {count}")

print("\nTotal single-diagnosis records:", sum(counts.values()))