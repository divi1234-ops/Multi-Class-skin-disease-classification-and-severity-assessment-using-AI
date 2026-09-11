import os
import cv2
import numpy as np
import pandas as pd

import torch
import segmentation_models_pytorch as smp

from torchvision import transforms

from classification.model import SkinDiseaseClassifier
from configs.config import *

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CLASS_NAMES = [

    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "VASC"

]


seg_model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=1

)

seg_model.load_state_dict(

    torch.load(

        os.path.join(

            MODELS_DIR,

            "attention_unet_best.pth"

        ),

        map_location=DEVICE

    )

)

seg_model.to(DEVICE)

seg_model.eval()

cls_model = SkinDiseaseClassifier(
    num_classes=7
)

cls_model.load_state_dict(

    torch.load(

        os.path.join(

            MODELS_DIR,

            "classifier_best.pth"

        ),

        map_location=DEVICE

    )

)

cls_model.to(DEVICE)

cls_model.eval()


transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(

        (IMAGE_SIZE, IMAGE_SIZE)

    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

@torch.no_grad()

def predict_mask(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    pred = seg_model(tensor)

    pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8)

    return pred


@torch.no_grad()

def predict_disease(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    output = cls_model(tensor)

    probs = torch.softmax(
        output,
        dim=1
    )

    confidence, index = torch.max(
        probs,
        dim=1
    )

    return (

        CLASS_NAMES[index.item()],

        confidence.item()

    )


def lesion_features(mask):

    area = np.sum(mask)

    contours, _ = cv2.findContours(

        mask.astype(np.uint8),

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    if len(contours) == 0:

        return area, 0, 0

    contour = max(

        contours,

        key=cv2.contourArea

    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    circularity = 0

    if perimeter > 0:

        circularity = (

            4*np.pi*area

        )/(perimeter*perimeter)

    return area, perimeter, circularity

def assess_severity(

    disease,

    confidence,

    area,

    circularity

):

    score = 0

    if disease == "MEL":

        score += 3

    elif disease in [

        "BCC",

        "AKIEC"

    ]:

        score += 2

    if confidence > 0.90:

        score += 2

    if area > 8000:

        score += 2

    if circularity < 0.60:

        score += 1

    if score >= 6:

        return "High"

    elif score >= 4:

        return "Moderate"

    return "Low"

if __name__ == "__main__":

    df = pd.read_csv(TEST_CSV)

    image_path = df.iloc[0]["image_path"]

    image = cv2.imread(image_path)

    image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )

    mask = predict_mask(image)

    disease, confidence = predict_disease(image)

    area, perimeter, circularity = lesion_features(mask)

    severity = assess_severity(

        disease,

        confidence,

        area,

        circularity

    )

    print("\n RESULT \n")

    print("Disease :", disease)

    print(f"Confidence : {confidence:.4f}")

    print(f"Lesion Area : {area}")

    print(f"Perimeter : {perimeter:.2f}")

    print(f"Circularity : {circularity:.3f}")

    print("Severity :", severity)
    