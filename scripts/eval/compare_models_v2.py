import os
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
    timestamps = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                timestamps.append(os.path.getmtime(full_path))
            except Exception:
                pass

    if len(timestamps) < 2:
        return 0.0, "", ""

    start_ts = min(timestamps)
    end_ts = max(timestamps)

    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    duration_hours = round((end_ts - start_ts) / 3600, 2)

    return duration_hours, start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S")


def find_metric_column(df, candidates):
    for col in candidates:
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
            print(f"[WARN] Missing folder: {exp_dir}")
            continue

        if not os.path.exists(csv_path):
            print(f"[WARN] Missing results.csv for {model_name}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Could not read {csv_path}: {e}")
            continue

        if df.empty:
            print(f"[WARN] Empty CSV for {model_name}")
            continue

        args = load_args_yaml(args_path)
        last = df.iloc[-1]

        precision_col = find_metric_column(df, ["metrics/precision(B)", "metrics/precision"])
        recall_col = find_metric_column(df, ["metrics/recall(B)", "metrics/recall"])
        map50_col = find_metric_column(df, ["metrics/mAP50(B)", "metrics/mAP50"])
        map5095_col = find_metric_column(df, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
        epoch_col = find_metric_column(df, ["epoch"])

        precision = safe_float(last[precision_col]) if precision_col else 0.0
        recall = safe_float(last[recall_col]) if recall_col else 0.0
        map50 = safe_float(last[map50_col]) if map50_col else 0.0
        map5095 = safe_float(last[map5095_col]) if map5095_col else 0.0
        epochs_run = int(last[epoch_col]) + 1 if epoch_col else args.get("epochs", 0)

        if os.path.exists(best_weight_path):
            weight_size_mb = bytes_to_mb(os.path.getsize(best_weight_path))
            best_weight_exists = True
        else:
            weight_size_mb = 0.0
            best_weight_exists = False

        train_duration_hours, start_time, end_time = get_training_duration_hours(exp_dir)

        rows.append({
            "Model": model_name,
            "Epochs_Run": epochs_run,
            "Epochs_Config": args.get("epochs", ""),
            "ImgSize": args.get("imgsz", ""),
            "Batch": args.get("batch", ""),
            "Device": args.get("device", ""),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map5095, 4),
            "BestWeightExists": best_weight_exists,
            "BestWeightSize_MB": weight_size_mb,
            "TrainDuration_Hours": train_duration_hours,
            "StartTime": start_time,
            "EndTime": end_time,
            "ExperimentPath": exp_dir,
            "BestWeightPath": best_weight_path if best_weight_exists else ""
        })

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    summary["OverallScore"] = (
        summary["mAP50"] * 0.40 +
        summary["mAP50_95"] * 0.30 +
        summary["Precision"] * 0.20 +
        summary["Recall"] * 0.10
    ).round(4)

    return summary


def save_summary_tables(summary):
    csv_path = os.path.join(TABLE_DIR, "model_comparison_v2.csv")
    xlsx_path = os.path.join(TABLE_DIR, "model_comparison_v2.xlsx")

    summary.to_csv(csv_path, index=False)

    try:
        summary.to_excel(xlsx_path, index=False)
        print(f"[OK] Saved Excel: {xlsx_path}")
    except Exception as e:
        print(f"[WARN] Excel save failed: {e}")

    print(f"[OK] Saved CSV: {csv_path}")


def save_rankings(summary):
    best_precision = summary.sort_values(by="Precision", ascending=False).reset_index(drop=True)
    best_recall = summary.sort_values(by="Recall", ascending=False).reset_index(drop=True)
    best_map50 = summary.sort_values(by="mAP50", ascending=False).reset_index(drop=True)
    best_overall = summary.sort_values(by="OverallScore", ascending=False).reset_index(drop=True)

    best_precision.to_csv(os.path.join(TABLE_DIR, "ranking_by_precision.csv"), index=False)
    best_recall.to_csv(os.path.join(TABLE_DIR, "ranking_by_recall.csv"), index=False)
    best_map50.to_csv(os.path.join(TABLE_DIR, "ranking_by_map50.csv"), index=False)
    best_overall.to_csv(os.path.join(TABLE_DIR, "ranking_by_overall.csv"), index=False)

    return {
        "precision": best_precision,
        "recall": best_recall,
        "map50": best_map50,
        "overall": best_overall
    }


def plot_metric(summary, metric_col, title, filename, ylim01=False):
    plt.figure(figsize=(9, 5))
    plt.bar(summary["Model"], summary[metric_col])
    plt.title(title)
    plt.ylabel(metric_col)
    if ylim01:
        plt.ylim(0, 1)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved figure: {out_path}")


def save_reports(rankings):
    best_precision = rankings["precision"].iloc[0]
    best_recall = rankings["recall"].iloc[0]
    best_map50 = rankings["map50"].iloc[0]
    best_overall = rankings["overall"].iloc[0]

    report_path = os.path.join(REPORT_DIR, "best_model_report_v2.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PLADICS YOLO MODEL COMPARISON REPORT V2\n")
        f.write("======================================\n\n")

        f.write("BEST BY PRECISION\n")
        f.write("-----------------\n")
        f.write(f"Model: {best_precision['Model']}\n")
        f.write(f"Precision: {best_precision['Precision']:.4f}\n")
        f.write(f"Recall: {best_precision['Recall']:.4f}\n")
        f.write(f"mAP50: {best_precision['mAP50']:.4f}\n")
        f.write(f"mAP50_95: {best_precision['mAP50_95']:.4f}\n\n")

        f.write("BEST BY RECALL\n")
        f.write("--------------\n")
        f.write(f"Model: {best_recall['Model']}\n")
        f.write(f"Precision: {best_recall['Precision']:.4f}\n")
        f.write(f"Recall: {best_recall['Recall']:.4f}\n")
        f.write(f"mAP50: {best_recall['mAP50']:.4f}\n")
        f.write(f"mAP50_95: {best_recall['mAP50_95']:.4f}\n\n")

        f.write("BEST BY mAP@0.5\n")
        f.write("---------------\n")
        f.write(f"Model: {best_map50['Model']}\n")
        f.write(f"Precision: {best_map50['Precision']:.4f}\n")
        f.write(f"Recall: {best_map50['Recall']:.4f}\n")
        f.write(f"mAP50: {best_map50['mAP50']:.4f}\n")
        f.write(f"mAP50_95: {best_map50['mAP50_95']:.4f}\n\n")

        f.write("BEST OVERALL\n")
        f.write("------------\n")
        f.write(f"Model: {best_overall['Model']}\n")
        f.write(f"OverallScore: {best_overall['OverallScore']:.4f}\n")
        f.write(f"Precision: {best_overall['Precision']:.4f}\n")
        f.write(f"Recall: {best_overall['Recall']:.4f}\n")
        f.write(f"mAP50: {best_overall['mAP50']:.4f}\n")
        f.write(f"mAP50_95: {best_overall['mAP50_95']:.4f}\n")
        f.write(f"Weight Size (MB): {best_overall['BestWeightSize_MB']:.2f}\n")
        f.write(f"Training Hours: {best_overall['TrainDuration_Hours']:.2f}\n")
        f.write(f"Best Weight Path: {best_overall['BestWeightPath']}\n\n")

        f.write("Overall scoring formula:\n")
        f.write("OverallScore = 0.40*mAP50 + 0.30*mAP50_95 + 0.20*Precision + 0.10*Recall\n")

    print(f"[OK] Saved report: {report_path}")


def print_best_models(rankings):
    print("\n=== BEST BY PRECISION ===")
    print(rankings["precision"].iloc[0][["Model", "Precision", "Recall", "mAP50", "mAP50_95"]])

    print("\n=== BEST BY RECALL ===")
    print(rankings["recall"].iloc[0][["Model", "Precision", "Recall", "mAP50", "mAP50_95"]])

    print("\n=== BEST BY mAP50 ===")
    print(rankings["map50"].iloc[0][["Model", "Precision", "Recall", "mAP50", "mAP50_95"]])

    print("\n=== BEST OVERALL ===")
    print(rankings["overall"].iloc[0][["Model", "OverallScore", "Precision", "Recall", "mAP50", "mAP50_95"]])


def main():
    summary = build_summary()

    if summary.empty:
        print("No valid experiment data found.")
        return

    save_summary_tables(summary)
    rankings = save_rankings(summary)

    plot_metric(summary.sort_values(by="Precision", ascending=False), "Precision",
                "Precision Comparison", "precision_comparison_v2.png", ylim01=True)

    plot_metric(summary.sort_values(by="Recall", ascending=False), "Recall",
                "Recall Comparison", "recall_comparison_v2.png", ylim01=True)

    plot_metric(summary.sort_values(by="mAP50", ascending=False), "mAP50",
                "mAP@0.5 Comparison", "map50_comparison_v2.png", ylim01=True)

    plot_metric(summary.sort_values(by="mAP50_95", ascending=False), "mAP50_95",
                "mAP@0.5:0.95 Comparison", "map5095_comparison_v2.png", ylim01=True)

    plot_metric(summary.sort_values(by="OverallScore", ascending=False), "OverallScore",
                "Overall Best Model Score Comparison", "overall_score_comparison_v2.png", ylim01=True)

    plot_metric(summary.sort_values(by="BestWeightSize_MB", ascending=False), "BestWeightSize_MB",
                "Model Weight Size Comparison (MB)", "weight_size_comparison_v2.png", ylim01=False)

    plot_metric(summary.sort_values(by="TrainDuration_Hours", ascending=False), "TrainDuration_Hours",
                "Training Duration Comparison (Hours)", "training_duration_comparison_v2.png", ylim01=False)

    save_reports(rankings)
    print_best_models(rankings)


if __name__ == "__main__":
    main()