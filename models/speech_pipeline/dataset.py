import pandas as pd
import librosa
import torch

from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor

from utils.label_encoder import (
    emotion_map,
    normalize_emotion
)


# ==========================================
# WAV2VEC2 PROCESSOR
# ==========================================

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)


# ==========================================
# SPEECH DATASET
# ==========================================

class SpeechDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]

        # ===============================
        # LOAD AUDIO
        # ===============================

        audio, sr = librosa.load(
            path,
            sr=16000
        )

        # ===============================
        # NORMALIZE AUDIO
        # ===============================

        audio = librosa.util.normalize(audio)

        # ===============================
        # WAV2VEC PROCESSING
        # ===============================

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        # ===============================
        # LABEL
        # ===============================

        emotion = normalize_emotion(
            row["emotion"]
        )

        label = emotion_map[emotion]

        return {

            "input_values":
                inputs.input_values.squeeze(0),

            "label":
                torch.tensor(label)
        }


# ==========================================
# COLLATE FUNCTION
# ==========================================

def collate_fn(batch):

    input_values = [
        item["input_values"]
        for item in batch
    ]

    labels = torch.tensor([
        item["label"]
        for item in batch
    ])

    padded_inputs = processor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt"
    )

    return {
        "input_values":
            padded_inputs.input_values,

        "labels":
            labels
    }