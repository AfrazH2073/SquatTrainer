"""Inference helpers for single images and webcam streams."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from .bootstrap import ensure_local_dependencies

ensure_local_dependencies()

import cv2
import numpy as np
import torch
from PIL import Image

from .modeling import build_model, build_transforms, resolve_device


DISPLAY_LABELS = {
    "bad_back": "Bad Back",
    "bad_heel": "Bad Heel",
    "good": "Good",
}


def load_checkpoint(checkpoint_path: str | Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), **checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, checkpoint


def predict_pil_image(model, class_names, image, device=None, image_size: int = 224, use_tta: bool = True):
    if device is None:
        device = resolve_device()
    _, eval_transform = build_transforms(image_size=image_size)
    pil_image = image.convert("RGB")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        tensors = [eval_transform(pil_image).unsqueeze(0)]
        if use_tta:
            tensors.append(eval_transform(pil_image.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0))
        batch = torch.cat(tensors, dim=0).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
    best_idx = int(probs.argmax())
    return {
        "label_name": class_names[best_idx],
        "label_display": DISPLAY_LABELS.get(class_names[best_idx], class_names[best_idx]),
        "confidence": float(probs[best_idx]),
        "probabilities": {class_names[idx]: float(prob) for idx, prob in enumerate(probs)},
    }


def predict_image_path(model, class_names, image_path: str | Path, device=None, image_size: int = 224):
    with Image.open(image_path) as image:
        return predict_pil_image(model, class_names, image, device=device, image_size=image_size)


def overlay_prediction(frame, prediction, confidence_threshold: float = 0.6):
    label_name = prediction["label_name"]
    confidence = prediction["confidence"]
    is_confident = confidence >= confidence_threshold
    is_good = label_name == "good" and is_confident
    if not is_confident:
        banner_color = (80, 80, 80)
        summary = "NEEDS CLEARER VIEW"
        detail = "Low confidence"
    else:
        banner_color = (0, 180, 0) if is_good else (0, 0, 220)
        summary = "GOOD POSTURE" if is_good else "BAD POSTURE"
        detail = prediction["label_display"]

    cv2.rectangle(frame, (10, 10), (420, 125), banner_color, thickness=-1)
    cv2.putText(frame, summary, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Class: {detail}", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return frame


def run_webcam_inference(
    checkpoint_path: str | Path,
    camera_index: int = 0,
    image_size: int | None = None,
    smoothing_window: int = 5,
    confidence_threshold: float = 0.6,
    display_mode: str = "window",
    frame_delay: float = 0.03,
):
    model, class_names, checkpoint = load_checkpoint(checkpoint_path)
    device = resolve_device()
    model = model.to(device)
    if image_size is None:
        image_size = checkpoint.get("config", {}).get("image_size", 224)
    _, eval_transform = build_transforms(image_size=image_size)

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    history = deque(maxlen=smoothing_window)
    use_notebook_display = display_mode == "notebook"
    clear_output = None
    display = None
    IPythonImage = None
    if use_notebook_display:
        from IPython.display import Image as IPythonImage
        from IPython.display import clear_output, display

        print("Rendering webcam frames inline in the notebook. Interrupt the cell to stop.")
    else:
        print("Press 'q' to quit the webcam window.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            tensor = eval_transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            history.append(probs)
            mean_probs = np.mean(np.stack(history), axis=0)
            best_idx = int(np.argmax(mean_probs))
            prediction = {
                "label_name": class_names[best_idx],
                "label_display": DISPLAY_LABELS.get(class_names[best_idx], class_names[best_idx]),
                "confidence": float(mean_probs[best_idx]),
                "probabilities": {class_names[idx]: float(prob) for idx, prob in enumerate(mean_probs)},
            }

            overlay_prediction(frame, prediction, confidence_threshold=confidence_threshold)
            if use_notebook_display:
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError("Could not encode webcam frame for notebook display.")
                clear_output(wait=True)
                display(IPythonImage(data=encoded.tobytes()))
                time.sleep(frame_delay)
            else:
                cv2.imshow("SquatTrainer Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Stopped webcam inference.")
    finally:
        capture.release()
        if use_notebook_display and clear_output is not None:
            clear_output(wait=True)
            print("Webcam inference stopped.")
        else:
            cv2.destroyAllWindows()
