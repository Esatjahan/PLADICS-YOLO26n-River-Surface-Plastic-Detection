import os
import re
import csv
import time
import yaml
import json
import platform
import cv2
import threading
import argparse

from collections import deque, Counter
from flask import Flask, Response, jsonify
from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO

# ==========================================================
# PLADICS dual-drive motor controller
# ==========================================================
#
# Raspberry Pi:
#     Imports the real L298N dual-motor controller.
#
# Windows:
#     RPi.GPIO is unavailable, so deployment remains usable
#     for camera/model/dashboard development without motors.
# ==========================================================

DualMotorController = None
MOTOR_IMPORT_ERROR = None

try:
    from hardware.dual_motor_controller import (
        DualMotorController,
    )
except ImportError as motor_import_error:
    MOTOR_IMPORT_ERROR = motor_import_error

try:
   from picamera2 import Picamera2
except ImportError:
   Picamera2 = None
try:
    import psutil
except ImportError:
    psutil = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_CONFIG_PATH = os.path.join(
    BASE_DIR,
    "configs",
    "deployment_config.yaml"
)
MISSIONS_DIR = os.path.join(BASE_DIR, "missions")

DEFAULT_PI_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
# ==========================================================
# PLADICS tracking and navigation stability settings
# ==========================================================

TRACK_LOST_TOLERANCE_FRAMES = 6
TRACK_RECOVERY_DISTANCE_PX = 100
TRACK_RECOVERY_AREA_RATIO_MIN = 0.45
TRACK_RECOVERY_AREA_RATIO_MAX = 2.20

ACTION_HISTORY_LENGTH = 5
MIN_STABLE_TRACK_FRAMES = 3
MIN_COLLECTION_STABLE_FRAMES = 4
# ==========================================================
# MJPEG Streaming Server
# ==========================================================


MJPEG_HOST = "0.0.0.0"
MJPEG_PORT = 5000
MJPEG_JPEG_QUALITY = 80

mjpeg_app = Flask(__name__)

_mjpeg_frame_lock = threading.Lock()
_mjpeg_latest_jpeg = None
_mjpeg_server_started = False

def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    
def is_windows_os():
    return platform.system().lower().startswith("win")


def resolve_model_path(config):
    if is_windows_os():
        model_path = config["paths"].get("model_path_windows", "")
        if os.path.exists(model_path):
            return model_path

        fallback = os.path.join(PROJECT_ROOT, "models", "best.pt")
        if os.path.exists(fallback):
            return fallback

        return model_path

    model_path = config["paths"].get("model_path_pi", "")
    if os.path.exists(model_path):
        return model_path

    if os.path.exists(DEFAULT_PI_MODEL_PATH):
        return DEFAULT_PI_MODEL_PATH

    return model_path


def resolve_camera_source(config):
    if is_windows_os():
        return config["camera"].get("source_windows", 0)

    return config["camera"].get("source_pi", 0)


def display_available():
    if is_windows_os():
        return True

    return bool(os.environ.get("DISPLAY"))
def update_mjpeg_frame(frame):
    """
    Convert the latest annotated OpenCV frame to JPEG and make it
    available to connected MJPEG clients.
    """
    global _mjpeg_latest_jpeg

    if frame is None:
        return

    try:
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            int(MJPEG_JPEG_QUALITY)
        ]

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            encode_params
        )

        if not success:
            return

        jpeg_bytes = encoded.tobytes()

        with _mjpeg_frame_lock:
            _mjpeg_latest_jpeg = jpeg_bytes

    except Exception as exc:
        print(f"[MJPEG] Frame encode failed: {exc}")


def mjpeg_frame_generator():
    """
    Continuously send the most recent JPEG frame using the
    multipart/x-mixed-replace MJPEG format.
    """
    while True:
        with _mjpeg_frame_lock:
            frame_bytes = _mjpeg_latest_jpeg

        if frame_bytes is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Pragma: no-cache\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

        time.sleep(0.02)


@mjpeg_app.route("/")
def mjpeg_home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PLADICS Live Stream</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            html, body {
                margin: 0;
                width: 100%;
                height: 100%;
                background: #050b14;
                color: #eaf7ff;
                font-family: Arial, sans-serif;
            }

            .viewer {
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            h2 {
                margin: 12px 0;
                color: #00e5ff;
            }

            img {
                display: block;
                max-width: 96vw;
                max-height: 88vh;
                width: auto;
                height: auto;
                object-fit: contain;
                background: #000;
                border: 1px solid rgba(0, 229, 255, 0.45);
                border-radius: 12px;
            }
        </style>
    </head>

    <body>
        <div class="viewer">
            <h2>PLADICS LIVE CAMERA & AI DETECTIONS</h2>
            <img src="/video_feed">
        </div>
    </body>
    </html>
    """


@mjpeg_app.route("/video_feed")
def mjpeg_video_feed():
    return Response(
        mjpeg_frame_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@mjpeg_app.route("/health")
def mjpeg_health():
    with _mjpeg_frame_lock:
        frame_ready = _mjpeg_latest_jpeg is not None

    return jsonify({
        "service": "PLADICS MJPEG Stream",
        "status": "online",
        "frame_ready": frame_ready,
        "port": MJPEG_PORT
    })


def _run_mjpeg_server():
    try:
        mjpeg_app.run(
            host=MJPEG_HOST,
            port=MJPEG_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as exc:
        print(f"[MJPEG] Server failed: {exc}")


def start_mjpeg_server():
    global _mjpeg_server_started

    if _mjpeg_server_started:
        return

    server_thread = threading.Thread(
        target=_run_mjpeg_server,
        name="pladics-mjpeg-server",
        daemon=True
    )

    server_thread.start()
    _mjpeg_server_started = True

    print(
        f"[MJPEG] Stream server started on "
        f"http://0.0.0.0:{MJPEG_PORT}/video_feed"
    )
def open_camera(config):
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    fps = int(config["camera"].get("fps", 30))

    if is_windows_os():
        source = config["camera"].get("source_windows", 0)

        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            raise RuntimeError("Windows camera open hoy nai.")

        return "opencv", cap

    if Picamera2 is None:
        raise RuntimeError(
            "Picamera2 import hoy nai. Raspberry Pi camera backend unavailable."
        )

    picam2 = Picamera2()

    camera_config = picam2.create_video_configuration(
        main={
            "size": (width, height),
            "format": "RGB888"
        },
        buffer_count=4
    )

    picam2.configure(camera_config)
    picam2.start()

    time.sleep(2)

    return "picamera2", picam2


def read_camera(camera_backend, camera_device):
    if camera_backend == "picamera2":
        frame_rgb = camera_device.capture_array()

        if frame_rgb is None:
            return False, None

        frame_bgr = cv2.cvtColor(
            frame_rgb,
            cv2.COLOR_RGB2BGR
        )

        return True, frame_bgr

    return camera_device.read()


def close_camera(camera_backend, camera_device):
    if camera_device is None:
        return

    if camera_backend == "picamera2":
        try:
            camera_device.stop()
        except Exception:
            pass
    else:
        try:
            camera_device.release()
        except Exception:
            pass

def bd_now():
    return datetime.now(ZoneInfo("Asia/Dhaka"))


def folder_time():
    return bd_now().strftime("%Y-%m-%d_%H-%M-%S_BDT")


def csv_time():
    return bd_now().strftime("%Y-%m-%d %H:%M:%S")


def get_next_mission_number():
    os.makedirs(MISSIONS_DIR, exist_ok=True)
    nums = []
    for name in os.listdir(MISSIONS_DIR):
        m = re.match(r"mission_(\d+)_", name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def create_mission(config):
    mission_no = get_next_mission_number()
    mission_id = f"{mission_no:03d}"
    mission_name = f"mission_{mission_id}_{folder_time()}"
    mission_path = os.path.join(MISSIONS_DIR, mission_name)

    paths = {
        "mission": mission_path,
        "logs": os.path.join(mission_path, "logs"),
        "screenshots": os.path.join(mission_path, "screenshots"),
        "videos": os.path.join(mission_path, "videos"),
        "graphs": os.path.join(mission_path, "graphs"),
        "telemetry": os.path.join(mission_path, "telemetry"),
        "report": os.path.join(mission_path, "report"),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    now = bd_now()
    mission_info = {
        "mission_id": mission_id,
        "mission_folder": mission_name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "Asia/Dhaka",
        "timezone_label": "BDT",
        "status": "Running",
        "project": config["project"]["name"],
        "mode": config["project"]["mode"],
        "model": config["model"]["name"],
        "imgsz": config["model"]["imgsz"],
        "confidence_threshold": config["model"]["confidence_threshold"],
        "iou_threshold": config["model"]["iou_threshold"],
        "platform_config": config["hardware"]["platform"],
        "camera_config": config["hardware"]["camera"],
        "collector": config["hardware"]["collector"],
        "motor_driver": config["hardware"]["motor_driver"],
        "running_os": platform.system()
    }

    info_path = os.path.join(mission_path, "mission_info.yaml")
    with open(info_path, "w", encoding="utf-8") as f:
        yaml.dump(mission_info, f, sort_keys=False)

    return mission_id, mission_name, mission_path, paths, info_path


def update_mission_status(info_path, status):
    if not os.path.exists(info_path):
        return

    with open(info_path, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    now = bd_now()
    info["status"] = status
    info["last_updated_date"] = now.strftime("%Y-%m-%d")
    info["last_updated_time"] = now.strftime("%H:%M:%S")
    info["last_updated_timezone"] = "BDT"

    with open(info_path, "w", encoding="utf-8") as f:
        yaml.dump(info, f, sort_keys=False)


def get_cpu():
    if psutil is None:
        return ""
    return round(psutil.cpu_percent(interval=None), 2)


def get_ram():
    if psutil is None:
        return ""
    return round(psutil.virtual_memory().percent, 2)


def get_temp():
    temp_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r") as f:
                return round(float(f.read()) / 1000.0, 2)
        except Exception:
            return ""
    return ""


def get_box_info(x1, y1, x2, y2):
    bw = int(x2 - x1)
    bh = int(y2 - y1)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    area = int(bw * bh)
    return cx, cy, bw, bh, area


def get_action(frame_width, center_x, area, config):
    left = int(frame_width * config["navigation"]["left_zone_ratio"])
    right = int(frame_width * config["navigation"]["right_zone_ratio"])
    collect_area = int(config["navigation"]["collect_area_threshold"])

    if area >= collect_area:
        return "STOP"

    if center_x < left:
        return "TURN_LEFT"

    if center_x > right:
        return "TURN_RIGHT"

    return "MOVE_FORWARD"


def get_mission_state(action, detection_count):
    if detection_count == 0:
        return "SEARCHING"
    if action in ["TURN_LEFT", "TURN_RIGHT"]:
        return "TRACKING"
    if action == "MOVE_FORWARD":
        return "MOVING"
    if action == "STOP":
        return "COLLECTING"
    return "IDLE"


def get_conveyor_state(action):
    return "ON" if action == "STOP" else "OFF"

def setup_conveyor(config):
    conveyor_config = config.get("conveyor", {})

    if not conveyor_config.get("enabled", False):
        print("[CONVEYOR] Disabled in config.")
        return None

    if GPIO is None:
        print("[CONVEYOR] RPi.GPIO unavailable.")
        return None

    control_pin = int(
        conveyor_config.get("control_pin", 24)
    )

    active_state = str(
        conveyor_config.get("active_state", "HIGH")
    ).strip().upper()

    GPIO.setwarnings(False)

    try:
        GPIO.setmode(GPIO.BCM)
    except ValueError:
        # BCM mode may already have been selected
        # by the drive-motor controller.
        pass

    GPIO.setup(
        control_pin,
        GPIO.OUT,
        initial=GPIO.LOW
        if active_state == "HIGH"
        else GPIO.HIGH
    )

    print(
        f"[CONVEYOR] GPIO initialized. "
        f"Control pin: GPIO{control_pin}, "
        f"active state: {active_state}"
    )

    return {
        "pin": control_pin,
        "active_state": active_state,
    }


def set_conveyor_output(
    conveyor_controller,
    turn_on
):
    if conveyor_controller is None:
        return

    pin = conveyor_controller["pin"]
    active_state = conveyor_controller["active_state"]

    if active_state == "HIGH":
        output_value = (
            GPIO.HIGH if turn_on else GPIO.LOW
        )
    else:
        output_value = (
            GPIO.LOW if turn_on else GPIO.HIGH
        )

    GPIO.output(pin, output_value)


def cleanup_conveyor(conveyor_controller):
    if conveyor_controller is None:
        return

    try:
        set_conveyor_output(
            conveyor_controller,
            False
        )

        GPIO.cleanup(
            conveyor_controller["pin"]
        )

        print("[CONVEYOR] Safe shutdown complete.")

    except Exception as conveyor_error:
        print(
            "[CONVEYOR] Shutdown warning:",
            conveyor_error
        )


def setup_mqtt(config):
    if mqtt is None:
        print("[MQTT] paho-mqtt not installed. MQTT disabled.")
        return None

    if not config.get("iot", {}).get("enabled", False):
        print("[MQTT] Disabled in config.")
        return None

    broker = config["iot"]["broker"]
    port = int(config["iot"]["port"])

    try:
        client = mqtt.Client()
        client.connect(broker, port, 60)
        client.loop_start()
        
        print(f"[MQTT] Connected to {broker}:{port}")
        return client
    
    except Exception as e:
        print("[MQTT] Connection failed:", e)
        return None


def publish_mqtt(client, config, payload):
    if client is None:
        return

    topic = config["iot"]["topic_live"]

    try:
        client.publish(topic, json.dumps(payload))
    except Exception as e:
        print("[MQTT] Publish failed:", e)


def draw_zones(frame, config):
    h, w = frame.shape[:2]
    left = int(w * config["navigation"]["left_zone_ratio"])
    right = int(w * config["navigation"]["right_zone_ratio"])

    cv2.line(frame, (left, 0), (left, h), (255, 255, 0), 1)
    cv2.line(frame, (right, 0), (right, h), (255, 255, 0), 1)
    cv2.drawMarker(
        frame,
        (w // 2, h // 2),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2
    )


def draw_status(frame, status):
    """
    Draw only essential real-time information so the camera
    view remains clear for plastic detection.
    """

    active_track_id = status.get(
        "active_track_id",
        -1
    )

    if (
        active_track_id is not None
        and int(active_track_id) >= 0
    ):
        track_text = str(active_track_id)
    else:
        track_text = "--"

    confidence = status.get(
        "confidence",
        0.0
    )

    lines = [
        f"Mission: {status['mission_id']}",

        (
            f"FPS: {status['fps']} | "
            f"Infer: {status['inference_ms']} ms"
        ),

        (
            f"Plastic: {status['plastic_count']} | "
            f"Track: {track_text} | "
            f"Conf: {float(confidence):.2f}"
        ),

        f"Action: {status['action']}",

        f"State: {status['mission_state']}",

        (
            f"Motor: "
            f"{'ON' if status['motor_enabled'] else 'OFF'}"
            f" | GPIO: "
            f"{'ON' if status['gpio_enabled'] else 'OFF'}"
        ),

        f"Conveyor: {status['conveyor_state']}",

        f"Temp: {status['temp']} C",
    ]

    x = 15
    y = 25
    line_height = 22

    # Compact dark background behind status text.
    panel_width = 360
    panel_height = (
        len(lines) * line_height + 16
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (5, 5),
        (panel_width, panel_height),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame
    )

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += line_height

CSV_HEADER = [
    "timestamp",
    "frame",
    "plastic_id",
    "plastic_count",
    "class_name",
    "confidence",
    "center_x",
    "center_y",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "fps",
    "inference_ms",
    "cpu_usage_percent",
    "ram_usage_percent",
    "cpu_temperature",
    "distance_cm",
    "action",
    "mission_state",
    "conveyor_state",
    "obstacle_detected",
    "collection_count",
]    


def init_csv(log_path):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)


def append_csv(log_path, row):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def center_distance(obj_a, obj_b):
    if obj_a is None or obj_b is None:
        return float("inf")

    dx = float(obj_a["center_x"]) - float(obj_b["center_x"])
    dy = float(obj_a["center_y"]) - float(obj_b["center_y"])

    return (dx * dx + dy * dy) ** 0.5


def area_ratio_compatible(previous_obj, current_obj):
    if previous_obj is None or current_obj is None:
        return False

    previous_area = max(float(previous_obj["bbox_area"]), 1.0)
    current_area = max(float(current_obj["bbox_area"]), 1.0)

    ratio = current_area / previous_area

    return (
        TRACK_RECOVERY_AREA_RATIO_MIN
        <= ratio
        <= TRACK_RECOVERY_AREA_RATIO_MAX
    )


def select_primary_target(
    tracked_objects,
    previous_target=None,
    frame_width=None
):
    if not tracked_objects:
        return None

    frame_center = (
        float(frame_width) / 2.0
        if frame_width is not None
        else 0.0
    )

    max_area = max(
        float(obj["bbox_area"])
        for obj in tracked_objects
    )

    best_target = None
    best_score = -1.0

    for obj in tracked_objects:
        confidence = float(obj["confidence"])
        area_score = float(obj["bbox_area"]) / max(max_area, 1.0)

        if frame_width is not None:
            center_error = abs(
                float(obj["center_x"]) - frame_center
            )

            alignment_score = max(
                0.0,
                1.0 - center_error / max(frame_center, 1.0)
            )
        else:
            alignment_score = 0.0

        continuity_score = 0.0

        if previous_target is not None:
            same_tracker_id = (
                int(obj.get("track_id", -1)) >= 0
                and int(previous_target.get("track_id", -1))
                == int(obj.get("track_id", -1))
            )

            if same_tracker_id:
                continuity_score = 1.0
            else:
                distance = center_distance(
                    previous_target,
                    obj
                )

                if (
                    distance <= TRACK_RECOVERY_DISTANCE_PX
                    and area_ratio_compatible(
                        previous_target,
                        obj
                    )
                ):
                    continuity_score = 0.65

        score = (
            0.40 * confidence
            + 0.30 * area_score
            + 0.15 * alignment_score
            + 0.15 * continuity_score
        )

        if score > best_score:
            best_score = score
            best_target = obj.copy()
            best_target["target_score"] = round(score, 4)

    return best_target


def smooth_action(action_history, proposed_action):
    action_history.append(proposed_action)

    counts = Counter(action_history)
    highest_count = max(counts.values())

    candidates = {
        action
        for action, count in counts.items()
        if count == highest_count
    }

    for action in reversed(action_history):
        if action in candidates:
            return action

    return proposed_action

def main():
    parser = argparse.ArgumentParser(
        description="PLADICS deployment runner"
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to deployment YAML configuration"
    )

    args = parser.parse_args()
    config = load_config(args.config)
    conveyor_controller = setup_conveyor(
        config
    )

    conveyor_running = False
    conveyor_started_time = 0.0

    conveyor_run_time = float(
        config.get(
            "conveyor",
            {}
        ).get(
            "run_time_seconds",
            5.0
        )
    )

    # ==========================================================
    # Initialize motor controller
    # ==========================================================

    drive_motors = None

    if DualMotorController is not None:
        try:
            drive_motors = DualMotorController(config)
            drive_motors.setup()

            print("[MOTOR] Controller initialized.")

        except Exception as motor_error:
            print(
                f"[MOTOR] Initialization failed: {motor_error}"
            )

            drive_motors = None

    else:
        print(
            "[MOTOR] DualMotorController unavailable."
        )

    camera_backend = None
    camera_device = None

    start_mjpeg_server()

        # ============================
    # Tracking state variables
    # ============================

    previous_target = None
    active_target = None

    target_lost_frames = 0
    stable_track_frames = 0

    active_track_id = None

    action_history = deque(maxlen=ACTION_HISTORY_LENGTH)

    collected_track_ids = set()  

    mission_id, mission_name, mission_path, paths, info_path = create_mission(config)
    log_path = os.path.join(paths["logs"], "mission_log.csv")
    init_csv(log_path)

    model_path = resolve_model_path(config)
    camera_source = resolve_camera_source(config)
    show_window = display_available()
    mqtt_client = setup_mqtt(config)

    print("====================================")
    print("PLADICS Deployment v2 + MQTT")
    print("====================================")
    print("Mission ID:", mission_id)
    print("Mission folder:", mission_path)
    print("Log file:", log_path)
    print("Model:", model_path)
    print("Camera:", camera_source)
    print("MQTT:", "ON" if mqtt_client else "OFF")
    print("====================================")

    if not os.path.exists(model_path):
        update_mission_status(info_path, "Failed_Model_Not_Found")
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(
       model_path,
       task="detect"
    )

    print("[MODEL] Detection model loaded successfully.")
    print("[MODEL] Classes:", model.names)
    target_class_name = str(
        config.get("model", {}).get(
            "target_class_name",
            "plastic"
        )
    ).strip().lower()

    print(
        "[MODEL] Required target class:",
        target_class_name
    )

    try:
       camera_backend, camera_device = open_camera(config)
    except Exception:
       update_mission_status(info_path, "Failed_Camera_Not_Opened")
       raise

    print("Camera backend:", camera_backend)

    frame_id = 0
    collection_count = 0
    recording = False
    video_writer = None
    
    final_status = "Running"

    previous_action = "IDLE"

    # Last action physically sent to the drive motor controller.
    # This prevents the same GPIO command from being repeated
    # during every camera frame.
    last_motor_action = None

    last_collection_time = 0.0
    collection_cooldown = float(
        config.get("collection", {}).get("cooldown_seconds", 5.0)
    )
    
    

    last_publish_time = 0.0
    publish_interval = float(
        config.get("iot", {}).get("publish_interval_seconds", 1.0)
    )
    

    print("Controls: q=quit | s=screenshot | v=video start/stop")

    try:
        while True:
            ret, frame = read_camera(camera_backend, camera_device)

            if not ret:
                final_status = "Failed_Frame_Read"
                print("[CAMERA] Frame read failed.")
                break

            frame_id += 1
            start = time.time()

            results = model.track(
                source=frame,
                imgsz=int(config["model"]["imgsz"]),
                conf=float(
                    config["model"]["confidence_threshold"]
                ),
                iou=float(
                    config["model"]["iou_threshold"]
                ),
                tracker=config["model"].get(
                    "tracker",
                    "bytetrack.yaml"
                ),
                persist=True,
                verbose=False
            )


            inference_sec = time.time() - start
            inference_ms = round(inference_sec * 1000, 2)
            fps = round(1.0 / max(inference_sec, 1e-6), 2)

            annotated = frame.copy()
            draw_zones(annotated, config)

            detection_count = 0
            best = None

            tracked_objects = []
            active_track_id = None

            for r in results:
                if (
                    r.boxes is None
                    or len(r.boxes) == 0
                ):
                    continue

                boxes = r.boxes

                # Debug information only every 30 frames.
                # Detection processing still happens every frame.
                if frame_id % 30 == 0:
                    print(
                        "[TRACK DEBUG]",
                        "detections=",
                        len(boxes),
                        "ids=",
                        (
                            None
                            if boxes.id is None
                            else boxes.id.int().cpu().tolist()
                        )
                    )

                if boxes.id is not None:
                    track_ids = (
                        boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )
                else:
                    track_ids = [-1] * len(boxes)

                xyxy_list = (
                    boxes.xyxy
                    .int()
                    .cpu()
                    .tolist()
                )

                conf_list = (
                    boxes.conf
                    .cpu()
                    .tolist()
                )

                cls_list = (
                    boxes.cls
                    .int()
                    .cpu()
                    .tolist()
                )

                for (
                    xyxy,
                    conf,
                    cls_id,
                    track_id
                ) in zip(
                    xyxy_list,
                    conf_list,
                    cls_list,
                    track_ids
                ):
                    x1, y1, x2, y2 = map(
                        int,
                        xyxy
                    )

                    if isinstance(
                        model.names,
                        dict
                    ):
                        class_name = model.names.get(
                            cls_id,
                            str(cls_id)
                        )
                    else:
                        class_name = model.names[
                            cls_id
                        ]

                    normalized_class_name = str(
                        class_name
                    ).strip().lower()

                    if (
                        normalized_class_name
                        != target_class_name
                    ):
                        continue

                    cx, cy, bw, bh, area = (
                        get_box_info(
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )

                    frame_area = (
                        annotated.shape[0]
                        * annotated.shape[1]
                    )

                    area_ratio = (
                        float(area)
                        / max(
                            float(frame_area),
                            1.0
                        )
                    )

                    acceptance_confidence = float(
                        config["model"].get(
                            "acceptance_confidence",
                            0.40
                        )
                    )

                    minimum_bbox_area = int(
                        config["model"].get(
                            "minimum_bbox_area",
                            1200
                        )
                    )

                    maximum_bbox_area_ratio = float(
                        config["model"].get(
                            "maximum_bbox_area_ratio",
                            0.55
                        )
                    )

                    accepted_detection = (
                        float(conf)
                        >= acceptance_confidence
                        and int(area)
                        >= minimum_bbox_area
                        and area_ratio
                        <= maximum_bbox_area_ratio
                    )

                    if not accepted_detection:
                        continue

                    detection_count += 1

                    object_data = {
                        "track_id": int(track_id),
                        "class_name": class_name,
                        "confidence": round(
                            float(conf),
                            4
                        ),
                        "center_x": int(cx),
                        "center_y": int(cy),
                        "bbox_width": int(bw),
                        "bbox_height": int(bh),
                        "bbox_area": int(area),
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }

                    tracked_objects.append(
                        object_data
                    )

                    if track_id >= 0:
                        box_color = (
                            0,
                            255,
                            0
                        )

                        label = (
                            f"{class_name} "
                            f"ID:{track_id} "
                            f"{float(conf):.2f}"
                        )
                    else:
                        box_color = (
                            0,
                            200,
                            255
                        )

                        label = (
                            f"{class_name} "
                            f"untracked "
                            f"{float(conf):.2f}"
                        )

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    cv2.circle(
                        annotated,
                        (cx, cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (
                            x1,
                            max(y1 - 10, 20)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        box_color,
                        2,
                        cv2.LINE_AA
                    )

                                     
            candidate_target = select_primary_target(
                tracked_objects=tracked_objects,
                previous_target=previous_target,
                frame_width=annotated.shape[1]
            )

            if candidate_target is not None:
                active_target = candidate_target
                target_lost_frames = 0

                same_valid_track = (
                    previous_target is not None
                    and int(active_target.get("track_id", -1)) >= 0
                    and int(previous_target.get("track_id", -1))
                    == int(active_target.get("track_id", -1))
                )

                if same_valid_track:
                    stable_track_frames += 1
                else:
                    stable_track_frames = 1

                active_target["recovered"] = False
                previous_target = active_target.copy()

            else:
                target_lost_frames += 1

                if (
                    previous_target is not None
                    and target_lost_frames
                    <= TRACK_LOST_TOLERANCE_FRAMES
                ):
                    active_target = previous_target.copy()
                    active_target["recovered"] = True
                else:
                    active_target = None
                    previous_target = None
                    stable_track_frames = 0

            best = active_target

            active_track_id = (
                int(best.get("track_id", -1))
                if best is not None
                else None
            )

            if best is None:
                proposed_action = "SEARCH"

                action = smooth_action(
                    action_history,
                    proposed_action
                )

                mission_state = "SEARCHING"
                conveyor_state = "OFF"

                plastic_id = ""
                class_name = ""
                confidence = ""
                center_x = ""
                center_y = ""
                bbox_width = ""
                bbox_height = ""
                bbox_area = ""

            else:
                proposed_action = get_action(
                    frame_width=annotated.shape[1],
                    center_x=best["center_x"],
                    area=best["bbox_area"],
                    config=config
                )

                # STOP must be immediate.
                # Other movement actions remain smoothed.
                if proposed_action == "STOP":
                    action_history.clear()
                    action_history.append("STOP")
                    action = "STOP"
                else:
                    action = smooth_action(
                        action_history,
                        proposed_action
                    )

                mission_state = get_mission_state(
                    action,
                    1
                )

                # Never activate collection from a stale,
                # temporarily recovered target.
                if candidate_target is None:
                    conveyor_state = "OFF"

                plastic_id = best.get(
                    "track_id",
                    -1
                )

                class_name = best.get(
                    "class_name",
                    ""
                )

                confidence = best.get(
                    "confidence",
                    0.0
                )

                center_x = best.get(
                    "center_x",
                    ""
                )

                center_y = best.get(
                    "center_y",
                    ""
                )

                bbox_width = best.get(
                    "bbox_width",
                    ""
                )

                bbox_height = best.get(
                    "bbox_height",
                    ""
                )

                bbox_area = best.get(
                    "bbox_area",
                    ""
                )
            # ==================================================
            # Send AI navigation action to drive motors
            # ==================================================
            #
            # The command is sent only when the action changes.
            # With gpio.enabled: false, the controller remains
            # disabled and no physical motor movement occurs.
            # ==================================================

            if (
                drive_motors is not None
                and action != last_motor_action
            ):
                try:
                    drive_motors.execute_action(action)

                    print(
                        f"[MOTOR] Action changed: "
                        f"{last_motor_action} -> {action}"
                    )

                    last_motor_action = action

                except Exception as motor_action_error:
                    print(
                        f"[MOTOR] Action execution failed "
                        f"for {action}: {motor_action_error}"
                    )

                    # Attempt an immediate safe stop if any
                    # motor command fails.
                    try:
                        drive_motors.emergency_stop()
                    except Exception:
                        pass

                    # Do not mark the failed action as completed.
                    last_motor_action = None

            current_time = time.time()

            
            current_track_id = (
                int(active_track_id)
                if active_track_id is not None
                else -1
            )

            target_is_currently_detected = (
                candidate_target is not None
            )

            stable_collection_target = (
                target_is_currently_detected
                and best is not None
                and not best.get("recovered", False)
                and stable_track_frames
                >= MIN_COLLECTION_STABLE_FRAMES
            )

            not_previously_collected = (
                current_track_id < 0
                or current_track_id not in collected_track_ids
            )

            new_collection_event = (
                action == "STOP"
                and stable_collection_target
                and not_previously_collected
                and (
                    current_time - last_collection_time
                ) >= collection_cooldown
            )

            if new_collection_event:
                collection_count += 1
                last_collection_time = current_time

                if current_track_id >= 0:
                    collected_track_ids.add(
                        current_track_id
                    )

                print(
                    f"[COLLECTION] New collection event. "
                    f"Track ID: {current_track_id}. "
                    f"Total collected: {collection_count}"
                )

                if conveyor_controller is not None:
                    set_conveyor_output(
                        conveyor_controller,
                        True
                    )

                    conveyor_running = True
                    conveyor_started_time = current_time

                    print(
                        "[CONVEYOR] ON - "
                        f"running for "
                        f"{conveyor_run_time} seconds."
                    )

            if (
                conveyor_running
                and (
                    current_time
                    - conveyor_started_time
                ) >= conveyor_run_time
            ):
                set_conveyor_output(
                    conveyor_controller,
                    False
                )

                conveyor_running = False

                print("[CONVEYOR] OFF")

            conveyor_state = (
                "ON"
                if conveyor_running
                else "OFF"
            )

            previous_action = action
                 

            cpu = get_cpu()
            ram = get_ram()
            temp = get_temp()

            distance_cm = ""
            obstacle_detected = "NO"

            row = [
                csv_time(),
                frame_id,
                plastic_id,
                detection_count,
                class_name,
                confidence,
                center_x,
                center_y,
                bbox_width,
                bbox_height,
                bbox_area,
                fps,
                inference_ms,
                cpu,
                ram,
                temp,
                distance_cm,
                action,
                mission_state,
                conveyor_state,
                obstacle_detected,
                collection_count
            ]

            append_csv(log_path, row)

            telemetry_payload = {
                "mission_id": mission_id,
                "mission_name": mission_name,
                "timestamp": csv_time(),
                "frame": frame_id,
                "plastic_count": detection_count,
                "active_track_id": (
                    active_track_id
                    if active_track_id is not None
                    else -1
                ),
                "tracked_objects": len(tracked_objects),
                "stable_track_frames": stable_track_frames,
                "target_lost_frames": target_lost_frames,
                "target_recovered": (
                    bool(best.get("recovered", False))
                    if best is not None
                    else False
                ),
                "target_score": (
                    best.get("target_score", 0.0)
                    if best is not None
                    else 0.0
                ),
                "confidence": confidence if confidence != "" else 0,
                "fps": fps,
                "inference_ms": inference_ms,
                "action": action,
                "mission_state": mission_state,
                "conveyor_state": conveyor_state,
                "obstacle_detected": obstacle_detected,
                "collection_count": collection_count,
                "cpu_usage_percent": cpu if cpu != "" else 0,
                "ram_usage_percent": ram if ram != "" else 0,
                "cpu_temperature": temp if temp != "" else 0,
                "motor_enabled": bool(
                    drive_motors is not None
                    and config.get(
                         "gpio",
                        {}
                    ).get(
                        "enabled",
                        False
                    )
                    and config.get(
                        "motor",
                        {}
                    ).get(
                        "enabled",
                        False
                    )
                ),

                "gpio_enabled": bool(
                    config.get(
                        "gpio",
                        {}
                    ).get(
                        "enabled",
                        False
                    )
                 ),
       

                "last_motor_action": (
                    last_motor_action
                    if last_motor_action is not None
                    else "NONE"
                )
            }

            now_time = time.time()
            if mqtt_client is not None and (now_time - last_publish_time) >= publish_interval:
                publish_mqtt(mqtt_client, config, telemetry_payload)
                last_publish_time = now_time

            status = {
                "mission_id": mission_id,
                "model": config["model"]["name"],
                "fps": fps,
                "inference_ms": inference_ms,
                "plastic_count": detection_count,
                "confidence": (
                    confidence
                    if confidence != ""
                    else 0.0
                ),

                "active_track_id": (
                    active_track_id
                    if active_track_id is not None
                    else -1
                ),
                "tracked_objects": len(tracked_objects),
                "stable_track_frames": stable_track_frames,
                "target_lost_frames": target_lost_frames,
                "target_recovered": (
                    bool(best.get("recovered", False))
                    if best is not None
                    else False
                ),
                "target_score": (
                    best.get("target_score", 0.0)
                    if best is not None
                    else 0.0
                ),

                "action": action,
                "mission_state": mission_state,
                "conveyor_state": conveyor_state,
                "mqtt_status": "ON" if mqtt_client else "OFF",
                "cpu": cpu,
                "ram": ram,
                "temp": temp,
                "motor_enabled": bool(
                    drive_motors is not None
                    and config.get(
                        "gpio",
                        {}
                    ).get(
                        "enabled",
                        False
                    )
                    and config.get(
                        "motor",
                        {}
                    ).get(
                        "enabled",
                        False
                    )
                ),

               "gpio_enabled": bool(
                    
                   config.get(
                       "gpio",
                       {}
                   ).get(
                       "enabled",
                       False
                   )
                ),

                "last_motor_action": (
                     last_motor_action
                     if last_motor_action is not None
                     else "--"
                 ),
            }

            draw_status(annotated, status)

            # Save latest annotated frame for Node-RED dashboard camera panel
            assets_dir = os.path.join(BASE_DIR, "assets")
            os.makedirs(assets_dir, exist_ok=True)

            latest_frame_path = os.path.join(
                assets_dir,
                "latest_frame.jpg"
            )

            # MJPEG stream every processed frame-এ update হবে
            update_mjpeg_frame(annotated)

            # Disk write কমাতে JPEG প্রতি 5 frame-এ save হবে
            if frame_id % 5 == 0:
                cv2.imwrite(
                    latest_frame_path,
                    annotated
                )

            if recording and video_writer is not None:
                video_writer.write(annotated)

            if show_window:
                cv2.imshow(
                    "PLADICS Deployment v2 MQTT",
                    annotated
                )

                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            if key == ord("q"):
                final_status = "Completed"
                break
            elif key == ord("s"):
                shot_name = f"screenshot_{folder_time()}.png"
                shot_path = os.path.join(paths["screenshots"], shot_name)
                cv2.imwrite(shot_path, annotated)
                print("[OK] Screenshot:", shot_path)

            elif key == ord("v"):
                if not recording:
                    video_name = f"video_{folder_time()}.mp4"
                    video_path = os.path.join(paths["videos"], video_name)

                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))

                    recording = True
                    print("[OK] Recording started:", video_path)

                else:
                    recording = False
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("[OK] Recording stopped.")

    except KeyboardInterrupt:
        final_status = "Interrupted"
        print("\n[MISSION] Interrupted by user.")
        
    except Exception as error:
        final_status = "Failed_Runtime_Error"
        print(f"\n[ERROR] Deployment failed: {error}")
        raise

    finally:
        cleanup_conveyor(
            conveyor_controller
        )
        if drive_motors is not None:
            try:
                drive_motors.emergency_stop()
                drive_motors.cleanup()

                print(
                    "[MOTOR] Safe shutdown complete."
                )

            except Exception as motor_shutdown_error:
                print(
                    "[MOTOR] Shutdown warning:",
                    motor_shutdown_error,
                )
        close_camera(camera_backend, camera_device)

        if video_writer is not None:
            video_writer.release()

        if mqtt_client is not None:
            try:
               mqtt_client.loop_stop()
               mqtt_client.disconnect()
               print("[MQTT] Disconnected safely.")
               
            except Exception as mqtt_error:
                print(f"[MQTT] Shutdown warning: {mqtt_error}")
            
        if show_window:
            cv2.destroyAllWindows()
        
        if os.path.exists(info_path):
            update_mission_status(info_path, final_status)

        print("\nDeployment stopped.")
        print("Final status:", final_status)
        print("Mission folder:", mission_path)
        print("Mission log:", log_path)


if __name__ == "__main__":
    main()