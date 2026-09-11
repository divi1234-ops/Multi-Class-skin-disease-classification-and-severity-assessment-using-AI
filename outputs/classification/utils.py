import os
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):

        self.sum += value * n

        self.count += n

        self.avg = self.sum / self.count

def calculate_metrics(targets, predictions):

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

    return accuracy, precision, recall, f1

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

    def __call__(self, val_loss):

        if val_loss < self.best_loss - self.delta:

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

def save_checkpoint(

    model,

    optimizer,

    epoch,

    val_loss,

    path

):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "val_loss": val_loss

    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint Saved : {path}"
    )


def load_checkpoint(

    model,

    optimizer,

    path,

    device

):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]

    val_loss = checkpoint["val_loss"]

    print(
        f"Checkpoint Loaded : {path}"
    )

    return model, optimizer, epoch, val_loss