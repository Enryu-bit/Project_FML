import csv
import time

from river import linear_model, metrics, preprocessing, tree, utils
from river.compose import Pipeline

from page_hinkley import PageHinkleyDetector
from stream_dataset import stream_dataset


def reset_logreg_pipeline(model):
    """Adaptation action for the Scaler+LogReg pipeline on drift.

    LogisticRegression/StandardScaler have no built-in forgetting mechanism,
    so the only way to wipe their memory is to replace both steps with
    fresh instances.
    """
    model.steps["StandardScaler"] = preprocessing.StandardScaler()
    model.steps["LogisticRegression"] = linear_model.LogisticRegression()


def reset_hat(model):
    """Adaptation action for HAT on drift.

    HoeffdingAdaptiveTreeClassifier already prunes/replaces branches whose
    ADWIN-tracked error exceeds an alternate subtree's, so no hard global
    reset is needed here.
    """
    pass


def evaluate(model, model_name, stream, snapshot_interval=5,
             detector=None, on_drift=None):
    accuracy = utils.Rolling(metrics.Accuracy, window_size=500)
    f1 = metrics.F1()
    history = []
    drift_points = []
    total_latency_ns = 0

    for i, (x, y) in enumerate(stream, start=1):
        start = time.perf_counter_ns()

        y_pred = model.predict_one(x)
        accuracy.update(y, y_pred)
        f1.update(y, y_pred)
        error = 0 if y_pred == y else 1
        model.learn_one(x, y)

        elapsed_ns = time.perf_counter_ns() - start
        total_latency_ns += elapsed_ns

        if detector is not None and detector.update(error):
            drift_points.append(i)
            print(f"[{i}] {model_name} | DRIFT DETECTED (PH={detector.ph_statistic:.3f}) -> adapting")
            if on_drift is not None:
                on_drift(model)
            detector.reset()

        history.append({
            "sample": i,
            "model": model_name,
            "accuracy": accuracy.get(),
            "f1": f1.get(),
            "error": error,
            "latency_us": elapsed_ns / 1000,
        })

        if i % snapshot_interval == 0:
            avg_latency_us = (total_latency_ns / i) / 1000
            print(f"[{i}] {model_name} | Acc: {accuracy.get():.4f} | F1: {f1.get():.4f} | Avg Latency: {avg_latency_us:.2f} µs/sample")

    avg_latency_us = (total_latency_ns / i) / 1000
    return {
        "model": model_name,
        "final_accuracy": accuracy.get(),
        "final_f1": f1.get(),
        "avg_latency_us": avg_latency_us,
        "history": history,
        "drift_points": drift_points,
    }


def export_telemetry(results, filepath="telemetry_results.csv"):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "sample", "accuracy", "latency_us", "drift_detected"])
        for r in results:
            drift_samples = set(r["drift_points"])
            for row in r["history"]:
                writer.writerow([
                    row["model"],
                    row["sample"],
                    row["accuracy"],
                    row["latency_us"],
                    row["sample"] in drift_samples,
                ])


if __name__ == "__main__":
    filepath = "sample_stream.csv"
    target_column = "target"
    snapshot_interval = 5

    model_configs = {
        "HAT": (tree.HoeffdingAdaptiveTreeClassifier(), reset_hat),
        "Scaler+LogReg": (Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression()), reset_logreg_pipeline),
    }

    results = []
    for name, (model, on_drift) in model_configs.items():
        print(f"\n--- Evaluating {name} ---")
        detector = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=0.9999)
        result = evaluate(model, name, stream_dataset(filepath, target_column), snapshot_interval,
                           detector=detector, on_drift=on_drift)
        results.append(result)
        print(f"Drift points for {name}: {result['drift_points']}")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['model']:15s} | Final Acc: {r['final_accuracy']:.4f} | Final F1: {r['final_f1']:.4f} | Avg Latency: {r['avg_latency_us']:.2f} µs/sample")

    export_telemetry(results)
    print("\nTelemetry exported to telemetry_results.csv")
