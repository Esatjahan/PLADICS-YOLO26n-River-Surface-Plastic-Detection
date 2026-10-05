import os
import re
import json
import time
import paho.mqtt.client as mqtt

BASE_DIR = r"D:\PLADICS\deployment"
MISSIONS_DIR = os.path.join(BASE_DIR, "missions")

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "pladics/live"


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

    telemetry_path = os.path.join(
        mission_path,
        "telemetry",
        "latest_telemetry.json"
    )

    if not os.path.exists(telemetry_path):
        print("Telemetry JSON not found.")
        print("Run telemetry_export.py first.")
        return

    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    print("====================================")
    print("PLADICS MQTT Publisher")
    print("Mission:", mission_name)
    print("Topic:", MQTT_TOPIC)
    print("Press CTRL+C to stop")
    print("====================================")

    try:
        while True:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                telemetry = json.load(f)

            payload = json.dumps(telemetry)

            client.publish(MQTT_TOPIC, payload)

            print("[PUBLISHED]", payload)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMQTT publisher stopped.")

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()