import cv2
import pickle
import urllib.request
import os

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
MODEL_PATH = "model.yml"
LABELS_PATH = "labels.pickle"


def ensure_cascade():
    if not os.path.exists(CASCADE_PATH):
        print("[i] Haar cascade not found. Downloading...")
        try:
            urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
            print("[i] Downloaded Haar cascade.")
        except Exception as e:
            print("[!] Failed to download cascade:", e)
            print("[!] Please download it manually and place it next to this script.")


def main():
    ensure_cascade()

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("[!] Missing model or labels. Train first with train_lbph.py")
        return

    with open(LABELS_PATH, "rb") as f:
        id_to_name = pickle.load(f)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not access the webcam.")
        return

    print("[i] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
            label_id, confidence = recognizer.predict(roi)

            # Lower confidence = better match (LBPH). We'll map to a percentage-like score.
            # Typical LBPH confidence ranges roughly from ~0 to ~100+ depending on params.
            conf_text = f"{max(0, min(100, int(100 - confidence)))}%"

            name = id_to_name.get(label_id, "Unknown")
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{name} {conf_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition (LBPH)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()