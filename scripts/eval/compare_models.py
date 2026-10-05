import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None


BASE_DIR = r"D:\PLADICS\experiments"
RESULTS_DIR = r"D:\PLADICS\results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
REPORT_DIR = os.path.join(RESULTS_DIR, "reports")

MODELS = ["yolov8n", "yolov10n", "yolo11n", "yolo26n"]

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def bytes_to_mb(num_bytes):
    return round(num_bytes / (1024 * 1024), 2)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_args_yaml(args_path):
    if not os.path.exists(args_path) or yaml is None:
        return {}
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_training_duration_hours(folder_path):
    """
    Approximate duration using earliest and latest file modification times
    inside the experiment folder.
    """
    timestamps = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                timestamps.append(os.path.getmtime(full_path))
            except Exception:
                pass

    if len(timestamps) < 2:
        return 0.0, None, None

    start_ts = min(timestamps)
    end_ts = max(timestamps)

    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    duration_hours = round((end_ts - start_ts) / 3600, 2)

    return duration_hours, start_dt, end_dt


def find_metric_column(df, possible_names):
    for col in possible_names:
        if col in df.columns:
            return col
    return None


def build_summary():
    rows = []

    for model_name in MODELS:
        exp_dir = os.path.join(BASE_DIR, model_name)
        csv_path = os.path.join(exp_dir, "results.csv")
        args_path = os.path.join(exp_dir, "args.yaml")
        best_weight_path = os.path.join(exp_dir, "weights", "best.pt")

        if not os.path.exists(exp_dir):
            print(f"[WARN] Missing experiment folder: {exp_dir}")
            continue

        if not os.path.exists(csv_path):
            print(f"[WARN] Missing results.csv for {model_name}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
            continue

        if df.empty:
            print(f"[WARN] Empty CSV: {csv_path}")
            continue

        last = df.iloc[-1]
        args = load_args_yaml(args_path)

        precision_col = find_metric_column(df, [
            "metrics/precision(B)",
            "metrics/precision"
        ])
        recall_col = find_metric_column(df, [
            "metrics/recall(B)",
            "metrics/recall"
        ])
        map50_col = find_metric_column(df, [
            "metrics/mAP50(B)",
            "metrics/mAP50"
        ])
        map5095_col = find_metric_column(df, [
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95"
        ])
        epoch_col = find_metric_column(df, [
            "epoch"
        ])
        train_box_loss_col = find_metric_column(df, [
            "train/box_loss"
        ])
        val_box_loss_col = find_metric_column(df, [
            "val/box_loss"
        ])

        precision = safe_float(last[precision_col]) if precision_col else 0.0
        recall = safe_float(last[recall_col]) if recall_col else 0.0
        map50 = safe_float(last[map50_col]) if map50_col else 0.0
        map5095 = safe_float(last[map5095_col]) if map5095_col else 0.0
        last_epoch = int(last[epoch_col]) + 1 if epoch_col else args.get("epochs", 0)

        train_box_loss = safe_float(last[train_box_loss_col]) if train_box_loss_col else 0.0
        val_box_loss = safe_float(last[val_box_loss_col]) if val_box_loss_col else 0.0

        if os.path.exists(best_weight_path):
            weight_size_mb = bytes_to_mb(os.path.getsize(best_weight_path))
            best_weight_exists = True
        else:
            weight_size_mb = 0.0
            best_weight_exists = False

        duration_hours, start_dt, end_dt = get_training_duration_hours(exp_dir)

        imgsz = args.get("imgsz", "")
        batch = args.get("batch", "")
        device = args.get("device", "")
        epochs_cfg = args.get("epochs", "")

        rows.append({
            "Model": model_name,
            "Epochs_Run": last_epoch,
            "Epochs_Config": epochs_cfg,
            "ImgSize": imgsz,
            "Batch": batch,
            "Device": device,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map5095, 4),
            "Train_Box_Loss": round(train_box_loss, 4),
            "Val_Box_Loss": round(val_box_loss, 4),
            "BestWeightExists": best_weight_exists,
            "BestWeightSize_MB": weight_size_mb,
            "TrainDuration_Hours": duration_hours,
            "StartTime": start_dt.strftime("%Y-%m-%d %H:%M:%S") if start_dt else "",
            "EndTime": end_dt.strftime("%Y-%m-%d %H:%M:%S") if end_dt else "",
            "ExperimentPath": exp_dir,
            "BestWeightPath": best_weight_path if best_weight_exists else ""
        })

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    # Ranking score: emphasize mAP50, then mAP50_95, then precision, then recall
    summary["Score"] = (
        summary["mAP50"] * 0.40 +
        summary["mAP50_95"] * 0.30 +
        summary["Precision"] * 0.20 +
        summary["Recall"] * 0.10
    ).round(4)

    summary = summary.sort_values(by=["Score", "mAP50", "Precision"], ascending=False).reset_index(drop=True)
    summary["Rank"] = range(1, len(summary) + 1)

    return summary


def save_tables(summary):
    csv_path = os.path.join(TABLE_DIR, "model_comparison_advanced.csv")
    xlsx_path = os.path.join(TABLE_DIR, "model_comparison_advanced.xlsx")

    summary.to_csv(csv_path, index=False)

    try:
        summary.to_excel(xlsx_path, index=False)
        print(f"[OK] Saved Excel: {xlsx_path}")
    except Exception as e:
        print(f"[WARN] Could not save Excel file: {e}")

    print(f"[OK] Saved CSV: {csv_path}")


def plot_metric(summary, metric_col, title, output_name):
    plt.figure(figsize=(9, 5))
    plt.bar(summary["Model"], summary[metric_col])
    plt.title(title)
    plt.ylabel(metric_col)
    if metric_col in ["Precision", "Recall", "mAP50", "mAP50_95", "Score"]:
        plt.ylim(0, 1)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, output_name)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved figure: {out_path}")


def save_report(summary):
    if summary.empty:
        return

    best_row = summary.iloc[0]
    report_path = os.path.join(REPORT_DIR, "best_model_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PLADICS YOLO Model Comparison Report\n")
        f.write("===================================\n\n")
        f.write("Models compared:\n")
        for model in summary["Model"].tolist():
            f.write(f"- {model}\n")

        f.write("\nRanking Table:\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")

        f.write("Best Model Summary\n")
        f.write("------------------\n")
        f.write(f"Selected Best Model : {best_row['Model']}\n")
        f.write(f"Rank                : {best_row['Rank']}\n")
        f.write(f"Score               : {best_row['Score']:.4f}\n")
        f.write(f"Precision           : {best_row['Precision']:.4f}\n")
        f.write(f"Recall              : {best_row['Recall']:.4f}\n")
        f.write(f"mAP@0.5             : {best_row['mAP50']:.4f}\n")
        f.write(f"mAP@0.5:0.95        : {best_row['mAP50_95']:.4f}\n")
        f.write(f"Best Weight Size MB : {best_row['BestWeightSize_MB']:.2f}\n")
        f.write(f"Training Duration H : {best_row['TrainDuration_Hours']:.2f}\n")
        f.write(f"Best Weight Path    : {best_row['BestWeightPath']}\n")
        f.write("\n")

        f.write("Selection Logic\n")
        f.write("---------------\n")
        f.write("Score = 0.40*mAP50 + 0.30*mAP50_95 + 0.20*Precision + 0.10*Recall\n")
        f.write("The model with the highest score is selected as the best overall model.\n")

    print(f"[OK] Saved report: {report_path}")


def print_console_summary(summary):
    if summary.empty:
        print("No experiment results found.")
        return

    print("\n=== PLADICS MODEL COMPARISON SUMMARY ===\n")
    display_cols = [
        "Rank", "Model", "Precision", "Recall", "mAP50", "mAP50_95",
        "Score", "BestWeightSize_MB", "TrainDuration_Hours"
    ]
    print(summary[display_cols].to_string(index=False))

    best = summary.iloc[0]
    print("\n=== BEST MODEL ===")
    print(f"Model          : {best['Model']}")
    print(f"Precision      : {best['Precision']:.4f}")
    print(f"Recall         : {best['Recall']:.4f}")
    print(f"mAP@0.5        : {best['mAP50']:.4f}")
    print(f"mAP@0.5:0.95   : {best['mAP50_95']:.4f}")
    print(f"Score          : {best['Score']:.4f}")
    print(f"Weight Size MB : {best['BestWeightSize_MB']:.2f}")
    print(f"Train Hours    : {best['TrainDuration_Hours']:.2f}")


def main():
    summary = build_summary()

    if summary.empty:
        print("No valid model results found in experiments folder.")
        return

    save_tables(summary)

    plot_metric(summary, "Precision", "Precision Comparison", "precision_comparison.png")
    plot_metric(summary, "Recall", "Recall Comparison", "recall_comparison.png")
    plot_metric(summary, "mAP50", "mAP@0.5 Comparison", "map50_comparison.png")
    plot_metric(summary, "mAP50_95", "mAP@0.5:0.95 Comparison", "map5095_comparison.png")
    plot_metric(summary, "Score", "Overall Ranking Score Comparison", "overall_score_comparison.png")
    plot_metric(summary, "BestWeightSize_MB", "Best Weight File Size Comparison (MB)", "weight_size_comparison.png")
    plot_metric(summary, "TrainDuration_Hours", "Training Duration Comparison (Hours)", "training_duration_comparison.png")

    save_report(summary)
    print_console_summary(summary)


if __name__ == "__main__":
    main()