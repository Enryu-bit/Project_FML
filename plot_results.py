import csv
from collections import defaultdict

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

TRUE_DRIFT_INDEX = 25000

MODEL_LABELS = {
    "HAT": "Hoeffding Adaptive Tree",
    "Scaler+LogReg": "Logistic Regression",
}


def load_telemetry(filepath):
    series = defaultdict(lambda: {"sample": [], "accuracy": [], "drift_flags": []})
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model_name"]
            s = series[model]
            s["sample"].append(int(row["sample_index"]))
            s["accuracy"].append(float(row["accuracy"]))
            if row["drift_detected_flag"] == "True":
                s["drift_flags"].append(int(row["sample_index"]))
    return series


def final_accuracy_and_latency(filepath):
    sums = defaultdict(float)
    counts = defaultdict(int)
    finals = {}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model_name"]
            sums[model] += float(row["latency_us"])
            counts[model] += 1
            finals[model] = float(row["accuracy"])
    mean_latency = {model: sums[model] / counts[model] for model in sums}
    return finals, mean_latency


def plot_drift_recovery():
    series = load_telemetry("telemetry_sea.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    colors = {"HAT": "tab:blue", "Scaler+LogReg": "tab:orange"}
    for model, data in series.items():
        ax.plot(data["sample"], data["accuracy"], label=MODEL_LABELS.get(model, model),
                color=colors.get(model), linewidth=1.2)

    ax.axvline(TRUE_DRIFT_INDEX, color="red", linewidth=2.5, linestyle="-",
               label="True Concept Drift")

    for model, data in series.items():
        if not data["drift_flags"]:
            continue
        detected_index = data["drift_flags"][0]
        delay = detected_index - TRUE_DRIFT_INDEX
        ax.axvline(detected_index, color="green", linewidth=1.5, linestyle="--",
                    label=f"Drift Detected ({MODEL_LABELS.get(model, model)}, delay={delay})")

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Rolling Accuracy (Window=500)")
    ax.set_title("Drift Recovery Curve (SEA Stream)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig("figure1_recovery.png", dpi=300)
    plt.close(fig)


def plot_pareto_frontier():
    datasets = {
        "SEA": "telemetry_sea.csv",
        "Elec2": "telemetry_elec2.csv",
    }

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    markers = {"SEA": "o", "Elec2": "^"}
    colors = {"HAT": "tab:blue", "Scaler+LogReg": "tab:orange"}

    for dataset_name, filepath in datasets.items():
        finals, mean_latency = final_accuracy_and_latency(filepath)
        for model in finals:
            ax.scatter(mean_latency[model], finals[model],
                       color=colors.get(model), marker=markers[dataset_name],
                       s=80, edgecolors="black", linewidths=0.5,
                       label=f"{MODEL_LABELS.get(model, model)} ({dataset_name})")

    ax.set_xscale("log")
    ax.set_xlabel("Mean Latency (µs/sample, log scale)")
    ax.set_ylabel("Final Rolling Accuracy (Window=500)")
    ax.set_title("Pareto Frontier: Latency vs. Accuracy")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig("figure2_pareto.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_drift_recovery()
    print("Saved figure1_recovery.png")

    plot_pareto_frontier()
    print("Saved figure2_pareto.png")
