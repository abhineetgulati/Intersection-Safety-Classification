from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

intersection_df = pd.read_csv("refined_nyc_intersections (1).csv")
# 1. Filter to keep only intersections with actual images
df_ready = intersection_df[intersection_df['image_exists'] == True].copy()

# 2. Encode your labels (Low -> 0, Medium -> 1, High -> 2)
le = LabelEncoder()
df_ready['label_num'] = le.fit_transform(df_ready['refined_risk_level'])
print(f"Classes assigned: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Define X (paths) and y (numeric labels)
X = df_ready['image_path']
y = df_ready['label_num']

# 4. First split: Separate out the Test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Second split: Split the remaining 80% into Train (80%) and Val (20%)
# This results in: 64% Train, 16% Val, 20% Test
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
)

print(f"Final Counts -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")