import os
import re
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


def get_latest_mission():
    missions = find_missions()
    if not missions:
        return None
    return missions[-1]


def get_mission_by_number(num):
    missions = find_missions()
    for mission in missions:
        if mission[0] == num:
            return mission
    return None


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


def generate_graphs(mission):
    mission_no, mission_name, mission_path = mission

    log_path = os.path.join(mission_path, "logs", "mission_log.csv")
    graphs_dir = os.path.join(mission_path, "graphs")

    os.makedirs(graphs_dir, exist_ok=True)

    if not os.path.exists(log_path):
        print("mission_log.csv paoa jay nai:", log_path)
        return

    df = pd.read_csv(log_path)

    if df.empty:
        print("mission_log.csv empty.")
        return

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

    print("====================================")
    print("Generating graphs")
    print("Mission:", mission_name)
    print("Rows:", len(df))
    print("Output:", graphs_dir)
    print("====================================")

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

    print("====================================")
    print("Graph generation complete.")
    print("====================================")


def main():
    print("====================================")
    print("PLADICS Graph Generator")
    print("====================================")
    print("1 = Current / Latest Mission")
    print("2 = Select Mission Number")
    print("3 = Exit")
    print("====================================")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        mission = get_latest_mission()
        if mission is None:
            print("No mission found.")
            return
        generate_graphs(mission)

    elif choice == "2":
        num = input("Enter mission number, example 1 or 2: ").strip()

        if not num.isdigit():
            print("Invalid mission number.")
            return

        mission = get_mission_by_number(int(num))

        if mission is None:
            print("Mission not found.")
            return

        generate_graphs(mission)

    elif choice == "3":
        print("Exit.")

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()