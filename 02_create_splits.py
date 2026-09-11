import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
INPUT_CSV = "Processed_Dataset/HAM10000/metadata.csv"
OUTPUT_DIR = "Processed_Dataset"
TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")

RANDOM_STATE = 42

print("=" * 60)
print("Loading Metadata...")
print("=" * 60)
df = pd.read_csv(INPUT_CSV)
print(f"Total Images : {len(df)}")
print(f"Unique Lesions : {df['lesion_id'].nunique()}")

gss = GroupShuffleSplit(
    n_splits=1,
    train_size=0.70,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(
    gss.split(
        df,
        groups=df["lesion_id"]
    )
)

train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df = df.iloc[temp_idx].reset_index(drop=True)


gss = GroupShuffleSplit(
    n_splits=1,
    train_size=0.50,
    random_state=RANDOM_STATE
)

val_idx, test_idx = next(
    gss.split(
        temp_df,
        groups=temp_df["lesion_id"]
    )
)

val_df = temp_df.iloc[val_idx].reset_index(drop=True)
test_df = temp_df.iloc[test_idx].reset_index(drop=True)


train_groups = set(train_df["lesion_id"])
val_groups = set(val_df["lesion_id"])
test_groups = set(test_df["lesion_id"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("\nNo lesion leakage detected.")



train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print("\n==============================")
print("Dataset Split Summary")
print("==============================")

print(f"Train Images      : {len(train_df)}")
print(f"Validation Images : {len(val_df)}")
print(f"Test Images       : {len(test_df)}")

print()

print("Train Distribution")
print(train_df["label"].value_counts())

print()

print("Validation Distribution")
print(val_df["label"].value_counts())

print()

print("Test Distribution")
print(test_df["label"].value_counts())

print("\nCSV files saved successfully.")

print(TRAIN_CSV)
print(VAL_CSV)
print(TEST_CSV)