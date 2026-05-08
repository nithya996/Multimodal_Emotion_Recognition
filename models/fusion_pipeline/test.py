import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from models.fusion_pipeline.dataset import (
    FusionDataset,
    collate_fn
)

from models.fusion_pipeline.model import (
    FusionEmotionModel
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

dataset = FusionDataset(
    "data/csv/dataset.csv"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)


# ==========================================
# LOAD MODEL
# ==========================================

model = FusionEmotionModel().to(device)

model.load_state_dict(
    torch.load(
        "Results/models/fusion_model.pth",
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

        input_values = batch[
            "input_values"
        ].to(device)

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        outputs = model(
            input_values,
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
    cmap="Reds",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Fusion Emotion Confusion Matrix"
)

plt.savefig(
    "Results/plots/fusion_confusion_matrix.png"
)

plt.show()

print(
    "\nFusion confusion matrix saved!\n"
)