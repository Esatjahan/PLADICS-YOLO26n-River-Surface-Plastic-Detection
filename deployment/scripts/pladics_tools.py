import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\PLADICS\deployment"
MISSIONS_DIR = os.path.join(BASE_DIR, "missions")


def find_missions():
    missions = []

    if not os.path.exists(MISSIONS_DIR):
        return missions

    for name in os.listdir(MISSIONS_DIR):
        path = os.path.join(MISSIONS_DIR, name)

        if os.path.isdir(path) and name.startswith("mission_"):
            match = re.match(r"mission_(\d+)_", name)
            if match:
                missions.append((int(match.group(1)), name, path))

    missions.sort(key=lambda x: x[0])
    return missions


def select_mission():
    missions = find_missions()

    if not missions:
        print("No mission found.")
        return None

    print("\nAvailable missions:")
    for no, name, _ in missions:
        print(f"{no}: {name}")

    print("\n1 = Current / Latest Mission")
    print("2 = Select Mission Number")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        return missions[-1]

    elif choice == "2":
        num = input("Enter mission number: ").strip()

        if not num.isdigit():
            print("Invalid mission number.")
            return None

        num = int(num)

        for mission in missions:
            if mission[0] == num:
                return mission

        print("Mission not found.")
        return None

    else:
        print("Invalid choice.")
        return None


def load_mission_log(mission):
    _, mission_name, mission_path = mission

    log_path = os.path.join(mission_path, "logs", "mission_log.csv")

    if not os.path.exists(log_path):
        print("mission_log.csv not found:", log_path)
        return None

    df = pd.read_csv(log_path)

    if df.empty:
        print("mission_log.csv empty.")
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = [
        "plastic_count",
        "confidence",
        "fps",
        "inference_ms",
        "cpu_usage_percent",
        "ram_usage_percent",
        "cpu_temperature"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_line_graph(df, y_col, title, ylabel, output_path):
    if y_col not in df.columns:
        print(f"[SKIP] Missing column: {y_col}")
        return

    data = df.dropna(subset=["timestamp", y_col])

    if data.empty:
        print(f"[SKIP] No valid data for: {y_col}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(data["timestamp"], data[y_col], linewidth=2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("[OK]", output_path)


def generate_graphs():
    mission = select_mission()

    if mission is None:
        return

    mission_no, mission_name, mission_path = mission
    df = load_mission_log(mission)

    if df is None:
        return

    graphs_dir = os.path.join(mission_path, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    print("\nGenerating graphs for:", mission_name)

    save_line_graph(
        df,
        "plastic_count",
        "Detection Count vs Time",
        "Detection Count",
        os.path.join(graphs_dir, "detection_count_vs_time.png")
    )

    save_line_graph(
        df,
        "confidence",
        "Detection Confidence vs Time",
        "Confidence",
        os.path.join(graphs_dir, "confidence_vs_time.png")
    )

    save_line_graph(
        df,
        "fps",
        "FPS vs Time",
        "FPS",
        os.path.join(graphs_dir, "fps_vs_time.png")
    )

    save_line_graph(
        df,
        "inference_ms",
        "Inference Time vs Time",
        "Inference Time (ms)",
        os.path.join(graphs_dir, "inference_time_vs_time.png")
    )

    save_line_graph(
        df,
        "cpu_usage_percent",
        "CPU Usage vs Time",
        "CPU Usage (%)",
        os.path.join(graphs_dir, "cpu_usage_vs_time.png")
    )

    save_line_graph(
        df,
        "ram_usage_percent",
        "RAM Usage vs Time",
        "RAM Usage (%)",
        os.path.join(graphs_dir, "ram_usage_vs_time.png")
    )

    save_line_graph(
        df,
        "cpu_temperature",
        "CPU Temperature vs Time",
        "CPU Temperature (C)",
        os.path.join(graphs_dir, "cpu_temperature_vs_time.png")
    )

    if "mission_state" in df.columns:
        state_counts = df["mission_state"].dropna().value_counts()

        if not state_counts.empty:
            plt.figure(figsize=(7, 7))
            plt.pie(
                state_counts.values,
                labels=state_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )
            plt.title("Mission State Distribution")
            plt.tight_layout()

            out_path = os.path.join(graphs_dir, "mission_state_distribution.png")
            plt.savefig(out_path, dpi=300)
            plt.close()

            print("[OK]", out_path)

    print("\nGraph generation complete.")


def export_latest_telemetry():
    mission = select_mission()

    if mission is None:
        return

    mission_no, mission_name, mission_path = mission
    df = load_mission_log(mission)

    if df is None:
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

    telemetry_dir = os.path.join(mission_path, "telemetry")
    os.makedirs(telemetry_dir, exist_ok=True)

    out_path = os.path.join(telemetry_dir, "latest_telemetry.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=4, default=str)

    print("\n[OK] Telemetry exported:")
    print(out_path)
    print(json.dumps(telemetry, indent=4, default=str))


def show_mission_summary():
    mission = select_mission()

    if mission is None:
        return

    mission_no, mission_name, mission_path = mission
    df = load_mission_log(mission)

    if df is None:
        return

    total_frames = len(df)
    avg_fps = pd.to_numeric(df["fps"], errors="coerce").mean()
    avg_inf = pd.to_numeric(df["inference_ms"], errors="coerce").mean()
    avg_conf = pd.to_numeric(df["confidence"], errors="coerce").mean()
    max_conf = pd.to_numeric(df["confidence"], errors="coerce").max()

    print("\n==============================")
    print("Mission Summary")
    print("==============================")
    print("Mission:", mission_name)
    print("Total frames:", total_frames)
    print("Average FPS:", round(avg_fps, 2) if pd.notna(avg_fps) else "N/A")
    print("Average inference:", round(avg_inf, 2) if pd.notna(avg_inf) else "N/A", "ms")
    print("Average confidence:", round(avg_conf, 3) if pd.notna(avg_conf) else "N/A")
    print("Max confidence:", round(max_conf, 3) if pd.notna(max_conf) else "N/A")

    if "mission_state" in df.columns:
        print("\nMission states:")
        print(df["mission_state"].value_counts())

    print("==============================")


def main():
    while True:
        print("\n====================================")
        print("PLADICS Deployment Tools")
        print("====================================")
        print("1 = Generate graphs")
        print("2 = Export latest telemetry JSON")
        print("3 = Show mission summary")
        print("4 = Generate graphs + telemetry")
        print("5 = Exit")
        print("====================================")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            generate_graphs()

        elif choice == "2":
            export_latest_telemetry()

        elif choice == "3":
            show_mission_summary()

        elif choice == "4":
            generate_graphs()
            export_latest_telemetry()

        elif choice == "5":
            print("Exit.")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()