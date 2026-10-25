import csv
import itertools
import math
import random

from river import datasets, linear_model, preprocessing, tree
from river.compose import Pipeline

from page_hinkley import PageHinkleyDetector
from stream_evaluation import evaluate, reset_hat, reset_logreg_pipeline

SNAPSHOT_INTERVAL = 100


def concept_drift_stream(stream, drift_stream, position, width, seed=None):
    """Same sigmoid-weighted mixing as river's ConceptDriftStream, but with a
    numerically stable sigmoid.

    river's own implementation computes ``1 / (1 + exp(v))`` directly, which
    overflows ``math.exp`` (and raises) once ``abs(v)`` gets large -- exactly
    what happens with a narrow width (e.g. width=1) far from ``position``.
    The formula here is mathematically identical, just evaluated in a form
    that never overflows.
    """
    rng = random.Random(seed)
    stream_generator = iter(stream)
    drift_stream_generator = iter(drift_stream)
    sample_idx = 0

    while True:
        sample_idx += 1
        v = -4.0 * float(sample_idx - position) / float(width)
        if v >= 0:
            ev = math.exp(-v)
            probability_drift = ev / (1.0 + ev)
        else:
            ev = math.exp(v)
            probability_drift = 1.0 / (1.0 + ev)
        try:
            if rng.random() > probability_drift:
                x, y = next(stream_generator)
            else:
                x, y = next(drift_stream_generator)
        except StopIteration:
            break
        yield x, y


def export_telemetry(results, filepath):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "model_name", "accuracy", "latency_us", "drift_detected_flag"])
        for r in results:
            drift_samples = set(r["drift_points"])
            for row in r["history"]:
                writer.writerow([
                    row["sample"],
                    row["model"],
                    row["accuracy"],
                    row["latency_us"],
                    row["sample"] in drift_samples,
                ])


def make_model_configs():
    return {
        "HAT": (tree.HoeffdingAdaptiveTreeClassifier(), reset_hat),
        "Scaler+LogReg": (Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression()), reset_logreg_pipeline),
    }


def run_benchmark(stream_factory, n_samples, output_path):
    results = []
    for name, (model, on_drift) in make_model_configs().items():
        print(f"\n--- Evaluating {name} ---")
        detector = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=0.9999)
        stream = stream_factory()
        if n_samples is not None:
            stream = itertools.islice(stream, n_samples)
        result = evaluate(model, name, stream, SNAPSHOT_INTERVAL,
                           detector=detector, on_drift=on_drift)
        results.append(result)
        print(f"Drift points for {name}: {result['drift_points']}")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['model']:15s} | Final Acc: {r['final_accuracy']:.4f} | Final F1: {r['final_f1']:.4f} | Avg Latency: {r['avg_latency_us']:.2f} µs/sample")

    export_telemetry(results, output_path)
    print(f"\nTelemetry exported to {output_path}")


if __name__ == "__main__":
    print("\n### Benchmark 1: Synthetic SEA Stream ###")
    sea_stream = lambda: concept_drift_stream(
        stream=datasets.synth.SEA(seed=42, variant=0),
        drift_stream=datasets.synth.SEA(seed=42, variant=1),
        position=25000,
        width=1,
        seed=42,
    )
    run_benchmark(sea_stream, n_samples=50000, output_path="telemetry_sea.csv")

    print("\n### Benchmark 2: Electricity Market (Elec2) ###")
    run_benchmark(lambda: datasets.Elec2(), n_samples=None, output_path="telemetry_elec2.csv")
