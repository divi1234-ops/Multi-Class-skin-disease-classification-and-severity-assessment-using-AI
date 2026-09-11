import os
import pandas as pd
import matplotlib.pyplot as plt
ROOT = "Processed_Dataset"

METADATA = os.path.join(ROOT, "metadata.csv")
TRAIN = os.path.join(ROOT, "train.csv")
VAL = os.path.join(ROOT, "val.csv")
TEST = os.path.join(ROOT, "test.csv")

REPORT_DIR = os.path.join(ROOT, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("Loading CSV Files")
print("=" * 60)

metadata = pd.read_csv(METADATA)
train = pd.read_csv(TRAIN)
val = pd.read_csv(VAL)
test = pd.read_csv(TEST)

print("Loaded Successfully\n")

summary = []

summary.append("DATASET SUMMARY\n")

summary.append(f"Total Images : {len(metadata)}")
summary.append(f"Training Images : {len(train)}")
summary.append(f"Validation Images : {len(val)}")
summary.append(f"Testing Images : {len(test)}")

summary.append(f"\nNumber of Classes : {metadata['label'].nunique()}")

summary.append(
    f"\nHAM10000 Images : {(metadata['dataset']=='HAM10000').sum()}"
)

summary.append(
    f"ISIC2018 Images : {(metadata['dataset']=='ISIC2018').sum()}"
)


summary.append("\nMISSING VALUES\n")

summary.append(str(metadata.isnull().sum()))


duplicate_images = metadata["image"].duplicated().sum()

summary.append("\n DUPLICATES \n")

summary.append(f"Duplicate Image IDs : {duplicate_images}")



missing = 0

if "image_path" in metadata.columns:

    for path in metadata["image_path"]:

        if not os.path.exists(path):
            missing += 1

summary.append(f"Missing Image Files : {missing}")


class_dist = metadata["label"].value_counts()

class_dist.to_csv(
    os.path.join(REPORT_DIR, "class_distribution.csv")
)

summary.append("\n CLASS DISTRIBUTION\n")

summary.append(str(class_dist))

dataset_dist = metadata["dataset"].value_counts()

dataset_dist.to_csv(
    os.path.join(REPORT_DIR, "dataset_distribution.csv")
)

summary.append("\nDATASET DISTRIBUTION \n")

summary.append(str(dataset_dist))



split = pd.Series({

    "Train": len(train),
    "Validation": len(val),
    "Test": len(test)

})



with open(
    os.path.join(REPORT_DIR, "summary.txt"),
    "w"
) as f:

    for line in summary:

        f.write(str(line))
        f.write("\n")

print("\nSummary saved.")

plt.figure(figsize=(10, 6))

class_dist.plot(kind="bar")

plt.title("Class Distribution")

plt.xlabel("Disease")

plt.ylabel("Images")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "class_distribution.png"),
    dpi=300
)

plt.close()

plt.figure(figsize=(6, 6))

dataset_dist.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")

plt.title("Dataset Contribution")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "dataset_distribution.png"),
    dpi=300
)

plt.close()
plt.figure(figsize=(7, 5))

split.plot(kind="bar")
plt.title("Train / Validation / Test Split")
plt.ylabel("Images")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORT_DIR, "dataset_split.png"),
    dpi=300
)
plt.close()
print("=" * 60)
print("DATASET ANALYSIS COMPLETED")
print("=" * 60)
print("\nReports saved inside:")
print(REPORT_DIR)