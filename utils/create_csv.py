import os
import pandas as pd


# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = (
    "data/raw/TESS Toronto emotional speech set data"
)


# ==========================================
# DATA STORAGE
# ==========================================

paths = []

texts = []

emotions = []


# ==========================================
# LOOP THROUGH DATASET
# ==========================================

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(
        DATASET_PATH,
        folder
    )

    if not os.path.isdir(folder_path):

        continue

    emotion = folder.split("_")[-1]

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):

            full_path = os.path.join(
                folder_path,
                file
            )

            # IMPORTANT FIX
            full_path = full_path.replace(
                "\\",
                "/"
            )

            word = file.split("_")[1]

            paths.append(full_path)

            texts.append(word)

            emotions.append(emotion)


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame({

    "path": paths,
    "text": texts,
    "emotion": emotions
})


# ==========================================
# SAVE CSV
# ==========================================

os.makedirs(
    "data/csv",
    exist_ok=True
)

df.to_csv(
    "data/csv/dataset.csv",
    index=False
)

print(df.head())

print("\nCSV created successfully!")

print(
    f"Total samples: {len(df)}"
)