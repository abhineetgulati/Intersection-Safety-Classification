import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CHANGE THIS TO YOUR SSD PATH
# ==========================================
ssd_path = r'D:/nyc_raw_images/preprocessed_images'

# ==========================================
# Load CSV
# ==========================================
intersection_df = pd.read_csv(
    "refined_nyc_intersections_FINAL_FINAL.csv"
)

# ==========================================
# Keep only rows with actual images
# ==========================================
df_ready = intersection_df[
    intersection_df['image_exists'] == True
].copy()

# ==========================================
# Encode labels
#
# LabelEncoder gives:
# High   -> 0
# Low    -> 1
# Medium -> 2
# ==========================================
le = LabelEncoder()
df_ready['label_num'] = le.fit_transform(
    df_ready['refined_risk_level']
)

print("Classes assigned:")
print(dict(zip(
    le.classes_,
    le.transform(le.classes_)
)))

# ==========================================
# Required counts FIRST
#
# High   -> 7000
# Low    -> 7000
# Medium -> ALL (~4000)
# ==========================================
required_counts = {
    0: 7000,   # High
    1: 7000,   # Low
    2: None    # Medium = all
}

label_to_folder = {
    0: "High",
    1: "Low",
    2: "Medium"
}

# ==========================================
# Create subset BEFORE splitting
# ==========================================
final_subset_list = []

for label, count in required_counts.items():
    class_df = df_ready[
        df_ready['label_num'] == label
    ]

    if count is None:
        selected_df = class_df.copy()
        print(
            f"{label_to_folder[label]}: "
            f"using ALL {len(class_df)}"
        )

    else:
        actual_count = min(count, len(class_df))

        print(
            f"{label_to_folder[label]}: "
            f"requested={count}, "
            f"available={len(class_df)}, "
            f"using={actual_count}"
        )

        selected_df = class_df.sample(
            n=actual_count,
            random_state=42
        )

    final_subset_list.append(selected_df)

# Final balanced dataset
final_df = pd.concat(
    final_subset_list
).reset_index(drop=True)

print("\nFinal subset distribution:")
print(
    final_df['label_num']
    .value_counts()
    .sort_index()
)

# ==========================================
# NOW split into Train / Val / Test
#
# 64% Train
# 16% Val
# 20% Test
# ==========================================
X = final_df['image_path']
y = final_df['label_num']

# First split → Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split → Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42,
    stratify=y_train_val
)

print("\nFinal split sizes:")
print(f"Train: {len(X_train)}")
print(f"Val: {len(X_val)}")
print(f"Test: {len(X_test)}")

# ==========================================
# Create DataFrames for each split
# ==========================================
train_df = pd.DataFrame({
    "path": X_train,
    "label": y_train
})

val_df = pd.DataFrame({
    "path": X_val,
    "label": y_val
})

test_df = pd.DataFrame({
    "path": X_test,
    "label": y_test
})

# ==========================================
# Create Folder Structure
# ==========================================
for split in ['train', 'val', 'test']:
    for folder in ['Low', 'Medium', 'High']:
        os.makedirs(
            os.path.join(
                ssd_path,
                split,
                folder
            ),
            exist_ok=True
        )

# ==========================================
# Copy function
# ==========================================
def move_to_ssd(df, split_type):
    print(
        f"\nTransferring {len(df)} images "
        f"to SSD ({split_type})..."
    )

    copied = 0
    missing = 0

    for _, row in df.iterrows():
        src = row['path']
        label = row['label']

        if os.path.exists(src):
            folder_name = label_to_folder[label]

            dest_dir = os.path.join(
                ssd_path,
                split_type,
                folder_name
            )

            filename = os.path.basename(src)
            dest = os.path.join(
                dest_dir,
                filename
            )

            try:
                shutil.copy2(src, dest)
                copied += 1

            except Exception as e:
                print(
                    f"Error copying {filename}: {e}"
                )

        else:
            missing += 1

    print(f"Copied: {copied}")
    print(f"Missing: {missing}")

# ==========================================
# Run transfer
# ==========================================
move_to_ssd(train_df, 'train')
move_to_ssd(val_df, 'val')
move_to_ssd(test_df, 'test')

print("\nTransfer complete!")
print(f"Saved to: {ssd_path}")
