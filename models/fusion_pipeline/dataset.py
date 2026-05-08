import pandas as pd
import librosa
import torch

from torch.utils.data import Dataset

from transformers import (
    Wav2Vec2Processor,
    BertTokenizer
)

from utils.label_encoder import (
    emotion_map,
    normalize_emotion
)


# ==========================================
# PROCESSORS
# ==========================================

speech_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

text_tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)


# ==========================================
# DATASET
# ==========================================

class FusionDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ==================================
        # AUDIO
        # ==================================

        audio, sr = librosa.load(
            row["path"],
            sr=16000
        )

        audio = librosa.util.normalize(audio)

        speech_inputs = speech_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        # ==================================
        # TEXT
        # ==================================

        text = str(row["text"])

        text_inputs = text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )

        # ==================================
        # LABEL
        # ==================================

        emotion = normalize_emotion(
            row["emotion"]
        )

        label = emotion_map[emotion]

        return {

            "input_values":
                speech_inputs.input_values.squeeze(0),

            "input_ids":
                text_inputs["input_ids"].squeeze(0),

            "attention_mask":
                text_inputs["attention_mask"].squeeze(0),

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

    padded_audio = speech_processor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt"
    )

    input_ids = torch.stack([
        item["input_ids"]
        for item in batch
    ])

    attention_mask = torch.stack([
        item["attention_mask"]
        for item in batch
    ])

    return {

        "input_values":
            padded_audio.input_values,

        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "labels":
            labels
    }