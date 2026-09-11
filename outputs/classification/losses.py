import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_weighted_cross_entropy(train_labels, device):

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion

def get_label_smoothing_loss():

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    return criterion

def get_weighted_label_smoothing_loss(
        train_labels,
        device
):

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1
    )

    return criterion

class FocalLoss(nn.Module):

    def __init__(
            self,
            alpha=1,
            gamma=2
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
            self,
            inputs,
            targets
    ):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * \
            (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()