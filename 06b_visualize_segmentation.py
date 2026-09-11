import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from configs.config import *
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "outputs/segmentation_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)
df = pd.read_csv(SEG_TEST_CSV)
print(f"Test Images : {len(df)}")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(
    torch.load(
        "models/attention_unet_best.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Model Loaded Successfully.")

def predict(image):

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype(np.float32) / 255.0

    image = torch.tensor(image).permute(2,0,1).unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        pred = model(image)

        pred = torch.sigmoid(pred)

        pred = pred.squeeze().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)

    return pred

samples = random.sample(range(len(df)), min(10, len(df)))

for i, idx in enumerate(samples):

    row = df.iloc[idx]

    image = cv2.imread(row["image_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(
        row["mask_path"],
        cv2.IMREAD_GRAYSCALE
    )

    prediction = predict(image)

    image_show = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    mask_show = cv2.resize(
        mask,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    fig = plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(image_show)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask_show, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

    save_path = os.path.join(
        SAVE_DIR,
        f"prediction_{i+1}.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    
print("\nVisualization Complete.")
print(f"Results saved in:\n{SAVE_DIR}")