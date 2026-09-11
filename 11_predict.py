import cv2
import os
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tenth_severity_assessment import (
    predict_mask,
    predict_disease,
    lesion_features,
    assess_severity
)
Tk().withdraw()
image_path = askopenfilename(
    title="Select Skin Image",
    filetypes=[
        ("Images","*.jpg *.png *.jpeg")
    ]
)
if image_path == "":
    print("No Image Selected.")
    exit()
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

print("\n    RESULT   \n")

print("Image :", os.path.basename(image_path))

print("Disease :", disease)

print(f"Confidence : {confidence:.4f}")

print(f"Lesion Area : {area}")

print(f"Perimeter : {perimeter:.2f}")

print(f"Circularity : {circularity:.3f}")

print("Severity :", severity)


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.imshow(image)

plt.title("Original Image")

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(mask, cmap="gray")

plt.title("Predicted Lesion Mask")

plt.axis("off")

plt.tight_layout()

plt.show()


def recommendation(disease, severity):

    if severity == "High":
        return "Consult a dermatologist immediately."

    elif severity == "Moderate":
        return "Dermatology consultation is recommended."

    else:

        if disease == "NV":
            return "Benign lesion. Continue regular skin monitoring."

        elif disease == "BKL":
            return "Likely benign. Monitor for changes in size, color, or shape."

        elif disease == "DF":
            return "Generally benign. Follow up if the lesion changes."

        elif disease == "VASC":
            return "Usually low risk, but seek medical advice if it grows or bleeds."

        else:
            return "Please consult a dermatologist."
        

print("\nRecommendation :")

print(recommendation(
    disease,
    severity
))