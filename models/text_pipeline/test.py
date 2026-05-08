import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from models.text_pipeline.dataset import (
    TextDataset
)

from models.text_pipeline.model import (
    TextEmotionModel
)

from utils.label_encoder import (
    reverse_emotion_map
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
    shuffle=False
)


# ==========================================
# LOAD MODEL
# ==========================================

model = TextEmotionModel().to(device)

model.load_state_dict(
    torch.load(
        "Results/models/text_model.pth",
        map_location=device
    )
)

model.eval()


# ==========================================
# TESTING
# ==========================================

all_labels = []

all_preds = []

correct = 0

total = 0


with torch.no_grad():

    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        outputs = model(
            input_ids,
            attention_mask
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            predicted.cpu().numpy()
        )


# ==========================================
# ACCURACY
# ==========================================

accuracy = 100 * correct / total

print(f"\nAccuracy: {accuracy:.2f}%\n")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

target_names = [
    reverse_emotion_map[i]
    for i in range(7)
]

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=target_names
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Text Emotion Confusion Matrix"
)

plt.savefig(
    "Results/plots/text_confusion_matrix.png"
)

plt.show()

print(
    "\nConfusion matrix saved!\n"
)