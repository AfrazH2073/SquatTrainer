from __future__ import annotations

from pathlib import Path

from squat_trainer.inference import run_webcam_inference


def main() -> None:
    checkpoint_path = Path("outputs") / "resnet18_squat" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Train the model in SquatTrainer.ipynb first."
        )
    run_webcam_inference(checkpoint_path=checkpoint_path, camera_index=0, image_size=224, smoothing_window=5)


if __name__ == "__main__":
    main()
