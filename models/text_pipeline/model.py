import torch.nn as nn

from transformers import BertModel


class TextEmotionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # =====================================
        # BERT
        # =====================================

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        # =====================================
        # DROPOUT
        # =====================================

        self.dropout = nn.Dropout(0.3)

        # =====================================
        # CLASSIFIER
        # =====================================

        self.fc1 = nn.Linear(
            768,
            128
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            7
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output

        out = self.dropout(
            pooled_output
        )

        out = self.fc1(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        return out