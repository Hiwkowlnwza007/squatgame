SquatDetection — webcam 2D prototype (not yet Stereo Camera)

1) Create Python 3.11 or 3.12 virtual environment.
2) Run: python -m pip install mediapipe opencv-python numpy
3) Download the official MediaPipe Pose Landmarker Lite model:
   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   Place it in this folder named pose_landmarker_lite.task
4) Run: python squat_detector.py
5) Stand facing the camera with your hips, knees and ankles visible.
6) Press C; remain still and standing during 20 calibration frames.
7) Slowly squat and return to standing; one JSON result prints per completed rep.
8) Press Q to close.

Example console output:
{"rep_id": 1, "result": "CORRECT"}
{"rep_id": 2, "result": "KNEE_VALGUS"}

IMPORTANT:
This is a non-validated, single RGB camera heuristic. It is NOT a true stereo depth
implementation, not a clinical/biomechanical diagnosis, and not a complete squat
technique assessor. CORRECT means no valgus-like reduction in knee spacing was
detected under an experimental threshold. Thresholds MUST be evaluated with
labeled videos from multiple people and expert annotation before making claims.

Python video-mode API docs:
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
