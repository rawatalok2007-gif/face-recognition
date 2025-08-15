# train_lbph.py
import cv2
import os
import numpy as np
import pickle
from pathlib import Path

DATASET_DIR = Path("dataset")
MODEL_PATH = "model.yml"
LABELS_PATH = "labels.pickle"
CASCADE_PATH = "haarcascade_frontalface_default.xml"


def load_images_and_labels():
    X = []  # images (grayscale)
    y = []  # integer labels
    label_map = {}  # name -> id
    current_id = 0

    if not DATASET_DIR.exists():
        raise RuntimeError("Dataset directory does not exist. Run capture_faces.py first.")

    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        if name not in label_map:
            label_map[name] = current_id
            current_id += 1

        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Normalize size just in case
            img = cv2.resize(img, (200, 200))
            X.append(img)
            y.append(label_map[name])

    if len(X) == 0:
        raise RuntimeError("No training images found in dataset. Collect images first.")

    return np.array(X), np.array(y), label_map


def main():
    print("[i] Loading dataset...")
    X, y, label_map = load_images_and_labels()
    print(f"[i] Loaded {len(X)} images across {len(label_map)} classes: {list(label_map.keys())}")

    # Create the LBPH recognizer (requires opencv-contrib-python)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    print("[i] Training model (LBPH)...")
    recognizer.train(list(X), y)
    recognizer.save(MODEL_PATH)
    print(f"[+] Saved model to {MODEL_PATH}")

    with open(LABELS_PATH, "wb") as f:
        pickle.dump({v: k for k, v in label_map.items()}, f)  # store id->name
    print(f"[+] Saved labels to {LABELS_PATH}")


if __name__ == "__main__":
    main()