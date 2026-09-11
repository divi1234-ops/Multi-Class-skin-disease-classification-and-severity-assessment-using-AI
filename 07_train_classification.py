import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
from configs.config import *
from classification.dataset import HAM10000Dataset
from classification.model import SkinDiseaseClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
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
def create_dataloaders():

    train_dataset = HAM10000Dataset(
        TRAIN_CSV,
        train_transform
    )

    val_dataset = HAM10000Dataset(
        VAL_CSV,
        val_transform
    )

    test_dataset = HAM10000Dataset(
        TEST_CSV,
        val_transform
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()

    )

    return train_loader, val_loader, test_loader

def get_class_weights():

    df = pd.read_csv(TRAIN_CSV)

    labels = df["label"].str.lower()

    class_map = {
        "akiec":0,
        "bcc":1,
        "bkl":2,
        "df":3,
        "mel":4,
        "nv":5,
        "vasc":6
    }

    y = labels.map(class_map).values

    weights = compute_class_weight(

        class_weight="balanced",

        classes=np.arange(7),

        y=y

    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    ).to(DEVICE)

model = SkinDiseaseClassifier(
    num_classes=7
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=get_class_weights()
)

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4

)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=EPOCHS,

    eta_min=1e-6

)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)

writer = SummaryWriter(
    LOG_DIR + "/classification"
)

class EarlyStopping:

    def __init__(

        self,

        patience=10,

        delta=0

    ):

        self.patience = patience

        self.delta = delta

        self.best_loss = np.inf

        self.counter = 0

        self.stop = False

    def __call__(self, loss):

        if loss < self.best_loss - self.delta:

            self.best_loss = loss

            self.counter = 0

        else:

            self.counter += 1

            print(
                f"EarlyStopping Counter: "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.stop = True

def save_model():

    torch.save(

        model.state_dict(),

        os.path.join(

            MODELS_DIR,

            "classifier_best.pth"

        )

    )

    print("\nBest Model Saved.")

def train_one_epoch(loader):

    model.train()

    running_loss = 0

    predictions = []
    targets = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

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

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(loader)

    epoch_acc = accuracy_score(
        targets,
        predictions
    )

    return epoch_loss, epoch_acc

@torch.no_grad()

def validate(loader):

    model.eval()

    running_loss = 0

    predictions = []
    targets = []

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False
    )

    for images, labels in progress:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += loss.item()

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

    epoch_loss = running_loss / len(loader)

    epoch_acc = accuracy_score(
        targets,
        predictions
    )

    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    for images, labels in tqdm(loader, desc="Testing"):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)

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

    print("\n Test Results ")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":

    print("=" * 60)
    print("Skin Disease Classification Training")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders()

    best_val_loss = np.inf

    early_stopping = EarlyStopping(
        patience=10
    )

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            train_loader
        )

        val_loss, val_acc = validate(
            val_loader
        )

        scheduler.step()

        writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/Train",
            train_acc,
            epoch
        )

        writer.add_scalar(
            "Accuracy/Validation",
            val_acc,
            epoch
        )

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_model()

        early_stopping(val_loss)

        if early_stopping.stop:

            print("\nEarly Stopping Triggered.")

            break

    writer.close()

    print("\nTraining Completed Successfully!")

    print(f"\nBest Validation Loss : {best_val_loss:.4f}")

    print("\nModel Saved At:")

    print(
        os.path.join(
            MODELS_DIR,
            "classifier_best.pth"
        )
    )
