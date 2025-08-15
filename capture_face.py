import cv2
import os
import time
import urllib.request

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"


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
    name = input("Enter person name (folder-safe, e.g., 'sujal_rana'): ").strip()
    if not name:
        print("[!] Name cannot be empty.")
        return

    save_dir = os.path.join("dataset", name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not access the webcam.")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    print("[i] Press 'c' to capture, 'q' to quit. Aim for 20–50 images with different angles/lighting.")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Samples: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Capture Faces", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save cropped faces (if multiple faces, save them all)
            if len(faces) == 0:
                print("[!] No face detected. Try again.")
                continue
            for i, (x, y, w, h) in enumerate(faces):
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))
                filename = os.path.join(save_dir, f"{int(time.time())}_{i}.jpg")
                cv2.imwrite(filename, face_img)
                count += 1
                print(f"[+] Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()