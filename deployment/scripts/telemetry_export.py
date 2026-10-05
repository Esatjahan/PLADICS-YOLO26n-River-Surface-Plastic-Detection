import os
import re
import json
import pandas as pd

BASE_DIR = r"D:\PLADICS\deployment"
MISSIONS_DIR = os.path.join(BASE_DIR, "missions")


def find_latest_mission():
    missions = []

    for name in os.listdir(MISSIONS_DIR):
        path = os.path.join(MISSIONS_DIR, name)
        if os.path.isdir(path) and name.startswith("mission_"):
            match = re.match(r"mission_(\d+)_", name)
            if match:
                missions.append((int(match.group(1)), name, path))

    if not missions:
        return None

    missions.sort(key=lambda x: x[0])
    return missions[-1]


def main():
    mission = find_latest_mission()

    if mission is None:
        print("No mission found.")
        return

    mission_no, mission_name, mission_path = mission
    log_path = os.path.join(mission_path, "logs", "mission_log.csv")
    telemetry_dir = os.path.join(mission_path, "telemetry")

    os.makedirs(telemetry_dir, exist_ok=True)

    if not os.path.exists(log_path):
        print("mission_log.csv not found.")
        return

    df = pd.read_csv(log_path)

    if df.empty:
        print("mission_log.csv empty.")
        return

    latest = df.iloc[-1].fillna("").to_dict()

    telemetry = {
        "mission": mission_name,
        "timestamp": latest.get("timestamp", ""),
        "plastic_count": latest.get("plastic_count", ""),
        "confidence": latest.get("confidence", ""),
        "fps": latest.get("fps", ""),
        "inference_ms": latest.get("inference_ms", ""),
        "action": latest.get("action", ""),
        "mission_state": latest.get("mission_state", ""),
        "conveyor_state": latest.get("conveyor_state", ""),
        "cpu_usage_percent": latest.get("cpu_usage_percent", ""),
        "ram_usage_percent": latest.get("ram_usage_percent", ""),
        "cpu_temperature": latest.get("cpu_temperature", "")
    }

    out_path = os.path.join(telemetry_dir, "latest_telemetry.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=4)

    print("[OK] Telemetry exported:")
    print(out_path)
    print(json.dumps(telemetry, indent=4))


if __name__ == "__main__":
    main()