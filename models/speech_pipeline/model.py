import torch
import torch.nn as nn

from transformers import Wav2Vec2Model


class SpeechEmotionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # =====================================
        # WAV2VEC2
        # =====================================

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # =====================================
        # BiLSTM
        # =====================================

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # =====================================
        # DROPOUT
        # =====================================

        self.dropout = nn.Dropout(0.3)

        # =====================================
        # CLASSIFIER
        # =====================================

        self.fc1 = nn.Linear(
            256,
            128
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            7
        )

    def forward(self, input_values):

        # =====================================
        # WAV2VEC FEATURES
        # =====================================

        outputs = self.wav2vec(
            input_values
        )

        hidden_states = outputs.last_hidden_state

        # =====================================
        # LSTM
        # =====================================

        lstm_out, _ = self.lstm(
            hidden_states
        )

        # =====================================
        # LAST TIME STEP
        # =====================================

        out = lstm_out[:, -1, :]

        # =====================================
        # CLASSIFIER
        # =====================================

        out = self.dropout(out)

        out = self.fc1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out