import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from configs.config import *
from classification.dataset import SkinDataset
from classification.model import SkinDiseaseClassifier
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
test_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()

])

test_dataset = SkinDataset(

    TEST_CSV,

    transform=test_transform

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()

)
model = SkinDiseaseClassifier(
    num_classes=7
).to(DEVICE)

model.load_state_dict(

    torch.load(

        os.path.join(

            MODELS_DIR,

            "classifier_best.pth"

        ),

        map_location=DEVICE

    )

)

model.eval()

print("Model Loaded Successfully.")

@torch.no_grad()

def evaluate():

    predictions = []

    targets = []

    for images, labels in tqdm(

        test_loader,

        desc="Testing"

    ):

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(

            outputs,

            dim=1

        )

        predictions.extend(

            preds.cpu().numpy()

        )

        targets.extend(

            labels.cpu().numpy()

        )

    return np.array(targets), np.array(predictions)

def print_metrics(targets, predictions):

    accuracy = accuracy_score(

        targets,

        predictions

    )

    precision = precision_score(

        targets,

        predictions,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        targets,

        predictions,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        targets,

        predictions,

        average="weighted",

        zero_division=0

    )

    print("\nTEST RESULTS \n")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")

    class_names = [

        "AKIEC",

        "BCC",

        "BKL",

        "DF",

        "MEL",

        "NV",

        "VASC"

    ]

    print(

        classification_report(

            targets,

            predictions,

            target_names=class_names,

            zero_division=0

        )

    )

def plot_confusion_matrix(

    targets,

    predictions

):

    cm = confusion_matrix(

        targets,

        predictions

    )

    class_names = [

        "AKIEC",

        "BCC",

        "BKL",

        "DF",

        "MEL",

        "NV",

        "VASC"

    ]

    plt.figure(figsize=(8,6))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=class_names,

        yticklabels=class_names

    )

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.title("Confusion Matrix")

    os.makedirs(

        "outputs",

        exist_ok=True

    )

    plt.savefig(

        "outputs/confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()

if __name__ == "__main__":

    targets, predictions = evaluate()

    print_metrics(

        targets,

        predictions

    )

    plot_confusion_matrix(

        targets,

        predictions

    )

    print("\nEvaluation Completed Successfully!")

    print("\nConfusion Matrix Saved In:")

    print("outputs/confusion_matrix.png")

    