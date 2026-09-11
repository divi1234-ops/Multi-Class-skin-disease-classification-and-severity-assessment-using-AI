import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
class SkinDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)

        self.transform = transform

        self.class_to_idx = {
            "akiec": 0,
            "bcc": 1,
            "bkl": 2,
            "df": 3,
            "mel": 4,
            "nv": 5,
            "vasc": 6
        }

        self.idx_to_class = {
            v: k
            for k, v in self.class_to_idx.items()
        }

        self.data["label_encoded"] = (
            self.data["label"]
            .str.lower()
            .map(self.class_to_idx)
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image = cv2.imread(row["image_path"])

        if image is None:
            raise FileNotFoundError(
                f"Image not found:\n{row['image_path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        label = int(row["label_encoded"])

        if self.transform:

            transformed = self.transform(
                image=image
            )

            image = transformed["image"]

        return image, torch.tensor(label, dtype=torch.long)