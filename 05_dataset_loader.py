import os
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

TRAIN_CSV = "Processed_Dataset/train.csv"
VAL_CSV = "Processed_Dataset/val.csv"
TEST_CSV = "Processed_Dataset/test.csv"

LABEL_ENCODER_PATH = "Processed_Dataset/label_encoder.pkl"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

train_df = pd.read_csv(TRAIN_CSV)

label_encoder = LabelEncoder()
label_encoder.fit(train_df["label"])

joblib.dump(label_encoder, LABEL_ENCODER_PATH)

print("\nLabel Encoder Saved")

print(label_encoder.classes_)

encoded_labels = label_encoder.transform(train_df["label"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(encoded_labels),
    y=encoded_labels
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float
)

print("\nClass Weights")

print(class_weights)

train_transform = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.Rotate(limit=20, p=0.5),

    A.RandomBrightnessContrast(p=0.5),

    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()

])

val_transform = A.Compose([

    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()

])

class SkinDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = row["image_path"]

        image = cv2.imread(image_path)

        if image is None:

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        label = label_encoder.transform(
            [row["label"]]
        )[0]

        if self.transform:

            image = self.transform(
                image=image
            )["image"]

        return image, torch.tensor(
            label,
            dtype=torch.long
        )

def get_dataloaders():

    train_dataset = SkinDataset(
        TRAIN_CSV,
        train_transform
    )

    val_dataset = SkinDataset(
        VAL_CSV,
        val_transform
    )

    test_dataset = SkinDataset(
        TEST_CSV,
        val_transform
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True

    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
        label_encoder
    )

if __name__ == "__main__":

    train_loader, val_loader, test_loader, weights, encoder = get_dataloaders()
    print("Dataset Information")
    print(f"Training Images   : {len(train_loader.dataset)}")
    print(f"Validation Images : {len(val_loader.dataset)}")
    print(f"Testing Images    : {len(test_loader.dataset)}")

    print()
    print("Classes")
    print(encoder.classes_)
    print()
    images, labels = next(iter(train_loader))
    print("Image Batch Shape :", images.shape)
    print("Label Batch Shape :", labels.shape)
    print("\nDataset Loader Ready!")