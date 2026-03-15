"""Generate the project notebook with executable cells."""

from __future__ import annotations

import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


NOTEBOOK = {
    "cells": [
        md_cell(
            """
            # SquatTrainer

            This notebook implements the full squat posture classification pipeline for the dataset in `Data/train` and `Data/test`.

            Important constraint: the current dataset is organized as labeled still images, so the implemented model is a **frame classifier** rather than a temporal video model. The webcam demo therefore predicts posture from each live frame and smooths predictions across recent frames.
            """
        ),
        md_cell(
            """
            ## 1. Setup

            This cell loads the project-local dependencies from `.deps` and imports the reusable modules used throughout the notebook.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys

            PROJECT_ROOT = Path.cwd()
            DEPS_DIR = PROJECT_ROOT / ".deps"
            if DEPS_DIR.exists():
                sys.path.insert(0, str(DEPS_DIR))

            import json
            import random

            import matplotlib.pyplot as plt
            import torch

            from squat_trainer.data import (
                DISPLAY_NAMES,
                SquatImageDataset,
                class_names_from_records,
                scan_split,
                stratified_train_val_split,
                summarize_records,
            )
            from squat_trainer.inference import load_checkpoint, predict_image_path, run_webcam_inference
            from squat_trainer.modeling import build_model, build_transforms, resolve_device
            from squat_trainer.training import TrainConfig, evaluate_model, plot_confusion_matrix, plot_history, train_model

            device = resolve_device()
            print(f"Using device: {device}")
            """
        ),
        md_cell(
            """
            ## 2. Load and Inspect the Dataset

            The project uses the provided train/test split. A validation split is created from the training portion for model selection.
            """
        ),
        code_cell(
            """
            data_root = PROJECT_ROOT / "Data"

            train_records_full = scan_split(data_root, "train")
            test_records = scan_split(data_root, "test")
            train_records, val_records = stratified_train_val_split(train_records_full, val_ratio=0.15, random_state=42)

            class_names = class_names_from_records(train_records_full)

            print("Class names:", class_names)
            print("Train split:", summarize_records(train_records))
            print("Validation split:", summarize_records(val_records))
            print("Test split:", summarize_records(test_records))
            """
        ),
        md_cell(
            """
            ## 3. Build PyTorch Datasets

            Training uses moderate augmentation. Validation and test data use deterministic preprocessing.
            """
        ),
        code_cell(
            """
            train_transform, eval_transform = build_transforms(image_size=224)

            train_dataset = SquatImageDataset(train_records, transform=train_transform)
            val_dataset = SquatImageDataset(val_records, transform=eval_transform)
            test_dataset = SquatImageDataset(test_records, transform=eval_transform)

            len(train_dataset), len(val_dataset), len(test_dataset)
            """
        ),
        md_cell(
            """
            ## 4. Configure and Train the Model

            The current implementation uses a small ResNet-18 classifier trained from scratch because the environment should not depend on downloading pretrained weights at runtime.
            """
        ),
        code_cell(
            """
            output_dir = PROJECT_ROOT / "outputs" / "resnet18_squat"
            output_dir.mkdir(parents=True, exist_ok=True)

            config = TrainConfig(
                batch_size=16,
                num_epochs=8,
                learning_rate=1e-3,
                weight_decay=1e-4,
                num_workers=0,
                random_seed=42,
            )

            model = build_model(num_classes=len(class_names), dropout=0.2).to(device)

            history, checkpoint = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                class_names=class_names,
                config=config,
                device=device,
                output_dir=output_dir,
            )

            print("Best epoch:", checkpoint["best_epoch"])
            print("Best validation accuracy:", round(checkpoint["best_val_accuracy"], 4))
            print("Checkpoint saved to:", output_dir / "best_model.pt")
            """
        ),
        md_cell(
            """
            ## 5. Review Training Curves
            """
        ),
        code_cell(
            """
            plot_history(history)
            plt.show()
            """
        ),
        md_cell(
            """
            ## 6. Evaluate on the Held-Out Test Set

            This section measures final performance on the provided test folder and visualizes the confusion matrix.
            """
        ),
        code_cell(
            """
            test_metrics = evaluate_model(
                model=model,
                dataset=test_dataset,
                class_names=class_names,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                device=device,
            )

            print("Test accuracy:", round(test_metrics["accuracy"], 4))
            print(json.dumps(test_metrics["classification_report"], indent=2))
            """
        ),
        code_cell(
            """
            plot_confusion_matrix(test_metrics["confusion_matrix"], class_names)
            plt.show()
            """
        ),
        md_cell(
            """
            ## 7. Run Sample Predictions

            This cell runs a few example predictions from the test set and prints the predicted class and confidence.
            """
        ),
        code_cell(
            """
            checkpoint_path = output_dir / "best_model.pt"
            inference_model, inference_class_names, checkpoint_data = load_checkpoint(checkpoint_path)
            inference_model = inference_model.to(device)

            sample_paths = [record.image_path for record in random.sample(test_records, k=min(6, len(test_records)))]
            for path in sample_paths:
                prediction = predict_image_path(inference_model, inference_class_names, path, device=device)
                print(path.name, "->", prediction["label_display"], f"({prediction['confidence'] * 100:.1f}%)")
            """
        ),
        md_cell(
            """
            ## 8. Webcam Inference

            Run the next cell after training finishes to open a webcam window. Press `q` in the OpenCV window to exit.
            """
        ),
        code_cell(
            """
            # checkpoint_path = PROJECT_ROOT / "outputs" / "resnet18_squat" / "best_model.pt"
            # run_webcam_inference(checkpoint_path=checkpoint_path, camera_index=0, image_size=224, smoothing_window=5)
            """
        ),
        md_cell(
            """
            ## 9. Notes

            - `good` means the frame looks like correct squat posture.
            - `bad_back` means the model sees back-position issues.
            - `bad_heel` means the model sees heel-position issues.
            - If you later collect true video clips with repetition-level labels, the next upgrade should be a temporal model over frame sequences rather than this frame-by-frame classifier.
            """
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    output_path = Path("SquatTrainer.ipynb")
    output_path.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
