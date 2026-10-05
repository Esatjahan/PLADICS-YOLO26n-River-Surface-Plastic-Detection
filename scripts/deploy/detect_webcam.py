from ultralytics import YOLO
import cv2
import os
import time
import csv
from datetime import datetime

# =========================================
# PLADICS CLEAN WEBCAM DETECTION SCRIPT
# Purpose:
# Real-world deployment evaluation for clean single-class plastic models
# =========================================

MODEL_CONFIGS = {
    "1": {
        "model_name": "yolo26n_clean",
        "model_path": r"D:\PLADICS\experiments_clean\yolo26n_clean\weights\best.pt",
        "conf": 0.35,
    },
    "2": {
        "model_name": "yolo11n_clean",
        "model_path": r"D:\PLADICS\experiments_clean\yolo11n_clean\weights\best.pt",
        "conf": 0.35,
    },
}

IMG_SIZE = 640
CAMERA_INDEX = 0

SCREENSHOT_DIR = r"D:\PLADICS\results\webcam_tests\screenshots"
VIDEO_DIR = r"D:\PLADICS\results\webcam_tests\videos"
LOG_DIR = r"D:\PLADICS\results\webcam_tests\logs"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def get_target_info(x1, y1, x2, y2):
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    area = int((x2 - x1) * (y2 - y1))
    return x_center, y_center, area


def get_direction(frame_width, x_center, area, area_stop_threshold=90000):
    left_zone = frame_width // 3
    right_zone = 2 * frame_width // 3

    if area >= area_stop_threshold:
        return "STOP / COLLECT"

    if x_center < left_zone:
        return "TURN LEFT"
    elif x_center > right_zone:
        return "TURN RIGHT"
    else:
        return "MOVE FORWARD"


def draw_zone_lines(frame):
    h, w = frame.shape[:2]
    left_zone = w // 3
    right_zone = 2 * w // 3

    cv2.line(frame, (left_zone, 0), (left_zone, h), (255, 255, 0), 1)
    cv2.line(frame, (right_zone, 0), (right_zone, h), (255, 255, 0), 1)

    return left_zone, right_zone


print("=====================================")
print("PLADICS Clean Webcam Detection")
print("Choose model:")
print("  1 = YOLO26n clean")
print("  2 = YOLO11n clean")
print("=====================================")

choice = input("Enter choice (1/2): ").strip()

if choice not in MODEL_CONFIGS:
    raise ValueError("Invalid choice. Use 1 or 2.")

config = MODEL_CONFIGS[choice]
MODEL_NAME = config["model_name"]
MODEL_PATH = config["model_path"]
CONF_THRESHOLD = config["conf"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}\n\n"
        "Train the clean model first, then run this script."
    )

print("=====================================")
print("Selected model    :", MODEL_NAME)
print("Model path        :", MODEL_PATH)
print("Confidence thresh :", CONF_THRESHOLD)
print("Image size        :", IMG_SIZE)
print("=====================================")

model = YOLO(MODEL_PATH)
print("Model classes:", model.names)
print("=====================================")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Webcam open hoy nai. Camera index check koro.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"{MODEL_NAME}_{timestamp}_webcam_log.csv")

log_file = open(log_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(log_file)

csv_writer.writerow([
    "frame",
    "timestamp",
    "fps",
    "detection_count",
    "best_confidence",
    "best_x_center",
    "best_y_center",
    "best_area",
    "action"
])

print("Controls:")
print("  q = quit")
print("  s = save screenshot")
print("  v = start/stop video recording")
print("Log file:", log_path)
print("=====================================")

video_writer = None
recording = False
video_path = None

prev_time = time.time()
frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame read hoy nai.")
        break

    frame_id += 1
    start_time = time.time()

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=0.45,
        verbose=False
    )

    inference_time = time.time() - start_time
    fps = 1.0 / max(inference_time, 1e-6)

    annotated = frame.copy()
    draw_zone_lines(annotated)

    detection_count = 0
    best_target = None
    best_conf = -1.0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])

            cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else str(cls_id)

            x_center, y_center, area = get_target_info(x1, y1, x2, y2)

            detection_count += 1

            if conf > best_conf:
                best_conf = conf
                best_target = {
                    "cls_name": cls_name,
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "x_center": x_center,
                    "y_center": y_center,
                    "area": area,
                }

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (x_center, y_center), 4, (0, 0, 255), -1)

            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

    action = "NO PLASTIC"
    best_x = ""
    best_y = ""
    best_area = ""
    best_confidence = ""

    if best_target is not None:
        action = get_direction(
            frame_width=annotated.shape[1],
            x_center=best_target["x_center"],
            area=best_target["area"]
        )

        best_x = best_target["x_center"]
        best_y = best_target["y_center"]
        best_area = best_target["area"]
        best_confidence = round(best_target["conf"], 4)

    csv_writer.writerow([
        frame_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        round(fps, 2),
        detection_count,
        best_confidence,
        best_x,
        best_y,
        best_area,
        action
    ])

    # =========================================
    # DISPLAY STATUS
    # =========================================
    cv2.putText(annotated, f"Model: {MODEL_NAME}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(annotated, f"Threshold: {CONF_THRESHOLD:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    cv2.putText(annotated, f"Detections: {detection_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(annotated, f"Action: {action}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

    if best_target is not None:
        cv2.putText(annotated, f"Best Conf: {best_target['conf']:.2f}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(annotated, f"Center: ({best_target['x_center']}, {best_target['y_center']})", (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(annotated, f"Area: {best_target['area']}", (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    record_text = "REC: ON" if recording else "REC: OFF"
    record_color = (0, 0, 255) if recording else (180, 180, 180)

    cv2.putText(annotated, record_text, (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, record_color, 2)

    if recording and video_writer is not None:
        video_writer.write(annotated)

    cv2.imshow(f"PLADICS Clean Detection - {MODEL_NAME}", annotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        shot_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(
            SCREENSHOT_DIR,
            f"{MODEL_NAME}_{shot_time}.png"
        )
        cv2.imwrite(screenshot_path, annotated)
        print("[OK] Screenshot saved:", screenshot_path)

    elif key == ord("v"):
        if not recording:
            video_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(
                VIDEO_DIR,
                f"{MODEL_NAME}_{video_time}.mp4"
            )

            frame_h, frame_w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_w, frame_h))

            recording = True
            print("[OK] Recording started:", video_path)

        else:
            recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print("[OK] Recording stopped:", video_path)

cap.release()

if video_writer is not None:
    video_writer.release()

log_file.close()
cv2.destroyAllWindows()

print("\n Webcam detection finished")
print("Log saved:", log_path)