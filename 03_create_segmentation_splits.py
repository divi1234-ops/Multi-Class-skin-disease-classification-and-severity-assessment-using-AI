import os
import random
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
IMAGE_DIR = "Processed_Dataset/segmentation/images"
MASK_DIR = "Processed_Dataset/segmentation/masks"
OUTPUT_DIR = "Processed_Dataset/segmentation"

RANDOM_SEED = 42

image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))

if len(image_paths) == 0:
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
pairs = []
for image_path in image_paths:

    filename = os.path.basename(image_path)

    image_name = os.path.splitext(filename)[0]

    image_name = image_name.replace("_segmentation", "")

    mask_name = image_name + "_segmentation.png"

    mask_path = os.path.join(MASK_DIR, mask_name)

    if os.path.exists(mask_path):

        pairs.append({

            "image_path": image_path,

            "mask_path": mask_path

        })

print(f"\nTotal Valid Pairs : {len(pairs)}")
df = pd.DataFrame(pairs)
train_df, temp_df = train_test_split(

    df,

    test_size=0.30,

    random_state=RANDOM_SEED,

    shuffle=True

)
val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=RANDOM_SEED,

    shuffle=True

)
train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)
val_df.to_csv(
    os.path.join(OUTPUT_DIR, "val.csv"),
    index=False
)
test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)
print("\n")
print("Segmentation Dataset Summary")
print(f"Total Images : {len(df)}")
print(f"Train Images : {len(train_df)}")
print(f"Validation Images : {len(val_df)}")
print(f"Test Images : {len(test_df)}")
print("\nCSV files created successfully.")