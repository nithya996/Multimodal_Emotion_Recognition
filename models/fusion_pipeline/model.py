import torch
import torch.nn as nn

from transformers import (
    Wav2Vec2Model,
    BertModel
)


class FusionEmotionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # =====================================
        # SPEECH MODEL
        # =====================================

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.speech_lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # =====================================
        # TEXT MODEL
        # =====================================

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        # =====================================
        # FUSION
        # =====================================

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(
            256 + 768,
            256
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            256,
            7
        )

    def forward(
        self,
        input_values,
        input_ids,
        attention_mask
    ):

        # =====================================
        # SPEECH
        # =====================================

        speech_outputs = self.wav2vec(
            input_values
        )

        speech_hidden = (
            speech_outputs.last_hidden_state
        )

        speech_lstm_out, _ = self.speech_lstm(
            speech_hidden
        )

        speech_embedding = (
            speech_lstm_out[:, -1, :]
        )

        # =====================================
        # TEXT
        # =====================================

        text_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_embedding = (
            text_outputs.pooler_output
        )

        # =====================================
        # FUSION
        # =====================================

        combined = torch.cat(
            (
                speech_embedding,
                text_embedding
            ),
            dim=1
        )

        out = self.dropout(combined)

        out = self.fc1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out