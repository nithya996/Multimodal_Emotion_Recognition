import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.text_pipeline.dataset import (
    TextDataset
)

from models.text_pipeline.model import (
    TextEmotionModel
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}\n")


# ==========================================
# DATASET
# ==========================================

dataset = TextDataset(
    "data/csv/dataset.csv"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# ==========================================
# MODEL
# ==========================================

model = TextEmotionModel().to(device)


# ==========================================
# LOSS
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)


# ==========================================
# TRAINING
# ==========================================

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    progress_bar = tqdm(loader)

    for batch in progress_bar:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        # ==========================
        # FORWARD
        # ==========================

        outputs = model(
            input_ids,
            attention_mask
        )

        loss = criterion(
            outputs,
            labels
        )

        # ==========================
        # BACKPROP
        # ==========================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ==========================
        # ACCURACY
        # ==========================

        _, predicted = torch.max(
            outputs,
            1
        )

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

        accuracy = 100 * correct / total

        total_loss += loss.item()

        progress_bar.set_description(
            f"Epoch {epoch+1}"
        )

        progress_bar.set_postfix(
            loss=loss.item(),
            accuracy=accuracy
        )

    avg_loss = total_loss / len(loader)

    epoch_accuracy = 100 * correct / total

    print("\n=================================")

    print(f"Epoch: {epoch+1}")

    print(f"Loss: {avg_loss:.4f}")

    print(f"Accuracy: {epoch_accuracy:.2f}%")

    print("=================================\n")


# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "Results/models/text_model.pth"
)

print("\nText model saved successfully!\n")