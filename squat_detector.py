"""Front-view squat prototype (2D webcam only, NOT stereo depth).

C: calibrate while standing still; Q: quit.
Requires mediapipe Tasks model: pose_landmarker_lite.task in same folder.
"""
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")
CAMERA_INDEX = 0

# Starting values for DEMO only. Tune with labeled videos and expert review.
DOWN_LEVEL = 0.80       # hip-ankle vertical distance / standing value
UP_LEVEL = 0.93
KNEE_DROP = 0.15        # how much knee/ankle gap ratio decreases from standing
MIN_BAD_FRAMES = 3
MIN_DOWN_FRAMES = 3

# MediaPipe IDs
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
JOINTS = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
          LEFT_ANKLE, RIGHT_ANKLE)


def measure(points):
    """Return 2D squat metrics, or None if landmarks are missing/occluded."""
    if any((getattr(points[i], "visibility", 1.0) < 0.6 or
            getattr(points[i], "presence", 1.0) < 0.6 or
            not (0 <= points[i].x <= 1 and 0 <= points[i].y <= 1))
           for i in JOINTS):
        return None

    hip_y = (points[LEFT_HIP].y + points[RIGHT_HIP].y) / 2
    ankle_y = (points[LEFT_ANKLE].y + points[RIGHT_ANKLE].y) / 2
    hip_height = ankle_y - hip_y

    ankle_gap = abs(points[LEFT_ANKLE].x - points[RIGHT_ANKLE].x)
    knee_gap = abs(points[LEFT_KNEE].x - points[RIGHT_KNEE].x)
    if hip_height <= 0.03 or ankle_gap < 0.04:
        return None

    return hip_height, knee_gap / ankle_gap


def draw_skeleton(frame, points):
    h, w = frame.shape[:2]
    links = ((23, 25), (25, 27), (24, 26), (26, 28), (23, 24))
    for i, j in links:
        a = (int(points[i].x * w), int(points[i].y * h))
        b = (int(points[j].x * w), int(points[j].y * h))
        cv2.line(frame, a, b, (80, 220, 80), 2)
    for i in JOINTS:
        p = (int(points[i].x * w), int(points[i].y * h))
        cv2.circle(frame, p, 6, (0, 180, 255), -1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Download pose_landmarker_lite.task to {MODEL_PATH}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera. Try CAMERA_INDEX = 1.")

    baseline_height = None
    baseline_knee = None
    calibration_samples = []
    calibrating = False
    phase = "READY"
    down_frames = 0
    bad_frames = 0
    invalid_during_rep = False
    rep_id = 0
    last_result = "--"
    started_down = 0.0
    last_rep_time = -100.0
    timer_start = time.monotonic()
    last_timestamp_ms = -1

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ms = int((time.monotonic() - timer_start) * 1000)
                ms = max(ms, last_timestamp_ms + 1)
                last_timestamp_ms = ms
                result = detector.detect_for_video(mp_image, ms)
                points = result.pose_landmarks[0] if result.pose_landmarks else None
                metrics = measure(points) if points is not None else None
                if points is not None:
                    draw_skeleton(frame, points)

                if calibrating:
                    if metrics is None:
                        calibration_samples.clear()
                    else:
                        calibration_samples.append(metrics)
                        if len(calibration_samples) >= 20:
                            baseline_height = float(np.mean([m[0] for m in calibration_samples]))
                            baseline_knee = float(np.mean([m[1] for m in calibration_samples]))
                            calibrating = False
                            phase = "READY"
                            last_result = "CALIBRATED"

                elif baseline_height is not None:
                    if metrics is None:
                        if phase == "DOWN":
                            invalid_during_rep = True
                            down_frames = 0
                    else:
                        height, knee_ratio = metrics
                        relative_height = height / baseline_height
                        now = time.monotonic()

                        if phase == "READY" and now - last_rep_time > 0.8:
                            if relative_height < DOWN_LEVEL:
                                down_frames += 1
                                if down_frames >= MIN_DOWN_FRAMES:
                                    phase = "DOWN"
                                    started_down = now
                                    bad_frames = 0
                                    invalid_during_rep = False
                            else:
                                down_frames = 0

                        elif phase == "DOWN":
                            if knee_ratio < baseline_knee - KNEE_DROP:
                                bad_frames += 1
                            if relative_height > UP_LEVEL and now - started_down > 0.3:
                                rep_id += 1
                                if invalid_during_rep:
                                    last_result = "UNKNOWN"
                                elif bad_frames >= MIN_BAD_FRAMES:
                                    last_result = "KNEE_VALGUS"
                                else:
                                    last_result = "CORRECT"
                                payload = {"rep_id": rep_id, "result": last_result}
                                print(json.dumps(payload), flush=True)
                                phase = "READY"
                                down_frames = 0
                                last_rep_time = now

                status = "Press C to calibrate while standing" if baseline_height is None else f"Phase: {phase}"
                if calibrating:
                    status = f"Calibrating: {len(calibration_samples)}/20"

                    # วาดพื้นหลังสีดำ
                cv2.rectangle(
                    frame,
                    (10, 10),       # มุมซ้ายบน
                    (350, 140),     # มุมขวาล่าง
                    (0, 0, 0),      # สีดำ (BGR)
                    -1              # เติมสีเต็มพื้นที่
                )
                cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Reps: {rep_id}  Result: {last_result}",
                            (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "C = recalibrate   Q = quit", (15, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow("Squat detector - 2D prototype", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    calibrating = True
                    calibration_samples.clear()
                    baseline_height = None
                    baseline_knee = None
                    phase = "READY"
                    down_frames = 0
                    last_result = "CALIBRATING"
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
