import pandas as pd
import torch

from torch.utils.data import Dataset

from transformers import BertTokenizer

from utils.label_encoder import (
    emotion_map,
    normalize_emotion
)


# ==========================================
# TOKENIZER
# ==========================================

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ==========================================
# TEXT DATASET
# ==========================================

class TextDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = str(row["text"])

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )

        emotion = normalize_emotion(
            row["emotion"]
        )

        label = emotion_map[emotion]

        return {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "label":
                torch.tensor(label)
        }