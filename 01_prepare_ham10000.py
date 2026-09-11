import os
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

HAM_ROOT = r"dataset2"
PART1 = os.path.join(HAM_ROOT, "HAM10000_images_part_1")
PART2 = os.path.join(HAM_ROOT, "HAM10000_images_part_2")
METADATA = os.path.join(HAM_ROOT, "HAM10000_metadata.csv")
OUTPUT = "Processed_Dataset"

OUTPUT_IMAGES = os.path.join(
    OUTPUT,
    "HAM10000",
    "images"
)

OUTPUT_METADATA = os.path.join(
    OUTPUT,
    "HAM10000",
    "metadata.csv"
)

IMAGE_SIZE = (224, 224)
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
print("Reading metadata...")
df = pd.read_csv(METADATA)
print(df.head())
print("\nTotal Metadata Entries :", len(df))

def locate_image(image_id):

    file_name = image_id + ".jpg"

    p1 = os.path.join(PART1, file_name)

    if os.path.exists(p1):
        return p1

    p2 = os.path.join(PART2, file_name)

    if os.path.exists(p2):
        return p2

    return None

new_records = []
missing = 0
processed = 0
print("\nProcessing Images...\n")

for _, row in tqdm(df.iterrows(), total=len(df)):

    image_id = row["image_id"]

    image_path = locate_image(image_id)

    if image_path is None:

        missing += 1

        continue

    image = cv2.imread(image_path)

    if image is None:

        missing += 1

        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE)

    save_name = image_id + ".jpg"

    save_path = os.path.join(
        OUTPUT_IMAGES,
        save_name
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    new_records.append({

        "image_id": image_id,

        "image_path": save_path,

        "label": row["dx"],

        "lesion_id": row["lesion_id"],

        "dx_type": row["dx_type"],

        "age": row["age"],

        "sex": row["sex"],

        "localization": row["localization"]

    })

    processed += 1

processed_df = pd.DataFrame(new_records)

processed_df.to_csv(
    OUTPUT_METADATA,
    index=False
)
print("Processing Complete")
print("Processed Images :", processed)
print("Missing Images :", missing)
print("Metadata Saved :", OUTPUT_METADATA)