# Face Recognition & Finger Counting in Python

A Python project that detects and recognizes faces using OpenCV's LBPH algorithm,  
plus a hand-tracking module to count the number of fingers raised.

## 📦 Requirements
```bash
pip install opencv-python opencv-contrib-python
pip install mediapipe
 
Collect Dataset

python capture_face.py


Train Model

python train_lbph.py


Run Realtime Recognition & Finger Counting

python recognize.py