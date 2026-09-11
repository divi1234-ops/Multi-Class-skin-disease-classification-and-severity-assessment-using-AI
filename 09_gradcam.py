import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from configs.config import *

from classification.model import SkinDiseaseClassifier
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SkinDiseaseClassifier(
    num_classes=7
)

model.load_state_dict(

    torch.load(

        os.path.join(
            MODELS_DIR,
            "classifier_best.pth"
        ),

        map_location=DEVICE

    )

)

model.to(DEVICE)

model.eval()

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])


df = pd.read_csv(TEST_CSV)

IMAGE_PATH = df.iloc[0]["image_path"]

print("Using image:", IMAGE_PATH)

image = Image.open(IMAGE_PATH).convert("RGB")

rgb = np.array(image).astype(np.float32) / 255.0

input_tensor = transform(image).unsqueeze(0).to(DEVICE)

target_layers = [
    model.model.features[-1]
]

cam = GradCAM(

    model=model,

    target_layers=target_layers
)
grayscale_cam = cam(
    input_tensor=input_tensor
)[0]

visualization = show_cam_on_image(

    rgb,

    grayscale_cam,

    use_rgb=True
)

os.makedirs(
    "outputs/gradcam",
    exist_ok=True
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(visualization)
plt.title("Grad-CAM")
plt.axis("off")

plt.savefig(

    "outputs/gradcam/gradcam.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

