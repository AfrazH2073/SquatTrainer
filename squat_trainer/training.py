"""Training and evaluation utilities."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .bootstrap import ensure_local_dependencies

ensure_local_dependencies()

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    random_seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, dataloader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
    }


def train_model(
    model,
    train_dataset,
    val_dataset,
    class_names: list[str],
    config: TrainConfig,
    device,
    output_dir: str | Path,
):
    set_seed(config.random_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_loader = create_data_loader(train_dataset, config.batch_size, True, config.num_workers)
    val_loader = create_data_loader(val_dataset, config.batch_size, False, config.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []
    best_state = None
    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_metrics)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": best_state,
        "class_names": class_names,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "config": asdict(config),
    }
    torch.save(checkpoint, output_path / "best_model.pt")
    with (output_path / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return history, checkpoint


def evaluate_model(model, dataset, class_names: list[str], batch_size: int, num_workers: int, device):
    dataloader = create_data_loader(dataset, batch_size, False, num_workers)
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy().tolist()
            pred_labels.extend(predictions)
            true_labels.extend(labels.numpy().tolist())

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, pred_labels)
    accuracy = float(np.mean(np.array(true_labels) == np.array(pred_labels)))

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
    }


def plot_history(history):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Accuracy")
    axes[1].plot(epochs, val_acc, label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_confusion_matrix(matrix, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

