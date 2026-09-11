import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import *
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),

    A.VerticalFlip(p=0.5),

    A.Rotate(limit=20, p=0.5),

    A.RandomBrightnessContrast(p=0.5),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
])
import pandas as pd
class SegmentationDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["mask_path"]

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise FileNotFoundError(
                f"Mask not found:\n{mask_path}"
            )

        mask = mask.astype(np.float32) / 255.0

        if self.transform:

            transformed = self.transform(
                image=image,
                mask=mask
            )

            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.unsqueeze(0)

        return image, mask.float()
def create_dataloaders():

    train_dataset = SegmentationDataset(
        SEG_TRAIN_CSV,
        train_transform
    )

    val_dataset = SegmentationDataset(
        SEG_VAL_CSV,
        val_transform
    )

    test_dataset = SegmentationDataset(
        SEG_TEST_CSV,
        val_transform
    )
    print("Train Dataset :", len(train_dataset))
    print("Validation Dataset :", len(val_dataset))
    print("Test Dataset :", len(test_dataset))
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
def build_model():

    model = smp.Unet(

        encoder_name="resnet34",

        encoder_weights="imagenet",

        in_channels=3,

        classes=1,

        activation=None

    )

    return model.to(DEVICE)
class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, prediction, target):

        prediction = torch.sigmoid(prediction)

        prediction = prediction.contiguous().view(-1)

        target = target.contiguous().view(-1)

        intersection = (prediction * target).sum()

        dice = (

            (2.0 * intersection + 1)

            /

            (prediction.sum() + target.sum() + 1)

        )

        return 1 - dice
class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, prediction, target):

        bce_loss = self.bce(

            prediction,

            target

        )

        dice_loss = self.dice(

            prediction,

            target

        )

        return bce_loss + dice_loss
def dice_score(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > 0.5).float()

    prediction = prediction.view(-1)

    target = target.view(-1)

    intersection = (prediction * target).sum()

    dice = (

        (2 * intersection + 1)

        /

        (

            prediction.sum()

            +

            target.sum()

            +

            1

        )

    )

    return dice.item()
def iou_score(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > 0.5).float()

    prediction = prediction.view(-1)

    target = target.view(-1)

    intersection = (

        prediction * target

    ).sum()

    union = (

        prediction + target

    ).sum() - intersection

    iou = (

        intersection + 1

    ) / (

        union + 1

    )

    return iou.item()
def build_optimizer(model):

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS

    )

    return optimizer, scheduler
def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, masks in progress_bar:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        running_dice += dice_score(outputs, masks)

        running_iou += iou_score(outputs, masks)

        progress_bar.set_postfix({

            "Loss": f"{loss.item():.4f}"

        })

    epoch_loss = running_loss / len(loader)

    epoch_dice = running_dice / len(loader)

    epoch_iou = running_iou / len(loader)

    return epoch_loss, epoch_dice, epoch_iou
def validate_one_epoch(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            loader,
            desc="Validation",
            leave=False
        )

        for images, masks in progress_bar:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            running_loss += loss.item()

            running_dice += dice_score(outputs, masks)

            running_iou += iou_score(outputs, masks)

    epoch_loss = running_loss / len(loader)

    epoch_dice = running_dice / len(loader)

    epoch_iou = running_iou / len(loader)

    return epoch_loss, epoch_dice, epoch_iou
def save_checkpoint(model, optimizer, epoch, loss):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "loss": loss

    }

    torch.save(

        checkpoint,

        os.path.join(

            CHECKPOINT_DIR,

            f"checkpoint_epoch_{epoch}.pth"

        )

    )
def save_best_model(model):

    torch.save(

        model.state_dict(),

        SEGMENTATION_MODEL

    )

    print("\nBest Model Saved")

class EarlyStopping:

    def __init__(

            self,

            patience=10

    ):

        self.patience = patience

        self.counter = 0

        self.best_loss = np.inf

        self.stop = False

    def __call__(

            self,

            val_loss

    ):

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            self.counter = 0

        else:

            self.counter += 1

            print(

                f"EarlyStopping Counter: "

                f"{self.counter}/{self.patience}"

            )

            if self.counter >= self.patience:

                self.stop = True

writer = SummaryWriter(LOG_DIR)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)

if __name__ == "__main__":

    print("=" * 60)
    print("Lesion Segmentation Training")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders()

    model = build_model()

    criterion = BCEDiceLoss()

    optimizer, scheduler = build_optimizer(model)

    early_stopping = EarlyStopping(patience=10)

    best_val_loss = float("inf")

    print(f"\nTraining on {DEVICE}")
    print(f"Epochs : {EPOCHS}")
    print(f"Batch Size : {BATCH_SIZE}")
    print("-" * 60)

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_dice, train_iou = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss, val_dice, val_iou = validate_one_epoch(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        writer.add_scalar("Dice/Train", train_dice, epoch)
        writer.add_scalar("Dice/Validation", val_dice, epoch)

        writer.add_scalar("IoU/Train", train_iou, epoch)
        writer.add_scalar("IoU/Validation", val_iou, epoch)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Train Dice : {train_dice:.4f}")
        print(f"Val Dice   : {val_dice:.4f}")

        print(f"Train IoU  : {train_iou:.4f}")
        print(f"Val IoU    : {val_iou:.4f}")

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_best_model(model)

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss
            )

        early_stopping(val_loss)

        if early_stopping.stop:

            print("\nEarly stopping triggered.")

            break

    writer.close()

    print("\nTraining Completed Successfully!")

    print(f"\nBest Validation Loss : {best_val_loss:.4f}")

    print(f"\nModel Saved At:\n{SEGMENTATION_MODEL}")