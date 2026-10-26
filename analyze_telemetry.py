"""Derived metrics for the report, computed from the exported telemetry CSVs.

The telemetry stores the rolling accuracy (window W=500) after every sample.
Because the window is known, the per-sample correctness indicator c_t can be
recovered exactly:

    t <= W:  c_t = round(t * acc_t) - sum(c_1 .. c_{t-1})
    t >  W:  c_t = round(W * acc_t) - sum(c_{t-W+1} .. c_{t-1})

From c_t we obtain the overall prequential accuracy, replay the Page-Hinkley
detector on the error signal e_t = 1 - c_t (to check that it reproduces the
logged alarms), and measure post-drift recovery on SEA.

Usage:
    python analyze_telemetry.py          # metrics from the CSVs only
    python analyze_telemetry.py --f1     # also F1 (needs river datasets for labels)
"""
import csv
import itertools
import statistics
import sys
from collections import defaultdict

from page_hinkley import PageHinkleyDetector

WINDOW = 500
TRUE_DRIFT_INDEX = 25000
RECOVERY_LEVEL = 0.95


def load_telemetry(filepath):
    series = defaultdict(lambda: {"accuracy": [], "latency_us": [], "drift_points": []})
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            s = series[row["model_name"]]
            s["accuracy"].append(float(row["accuracy"]))
            s["latency_us"].append(float(row["latency_us"]))
            if row["drift_detected_flag"] == "True":
                s["drift_points"].append(int(row["sample_index"]))
    return series


def reconstruct_correct(accuracy, window=WINDOW):
    correct = []
    for t, acc in enumerate(accuracy, start=1):
        n = min(t, window)
        c = round(n * acc) - sum(correct[-(n - 1):] if n > 1 else [])
        if c not in (0, 1):
            raise ValueError(f"Telemetry is not consistent with window={window} at sample {t}")
        correct.append(c)
    return correct


def replay_detector(errors):
    detector = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=0.9999)
    alarms = []
    for i, e in enumerate(errors, start=1):
        if detector.update(e):
            alarms.append(i)
            detector.reset()
    return alarms


def percentile(values, q):
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def recovery_stats(accuracy, alarm, drift=TRUE_DRIFT_INDEX, level=RECOVERY_LEVEL):
    n = len(accuracy)
    trough = min(range(drift, n), key=lambda i: accuracy[i])
    first_back = next(i for i in range(trough, n) if accuracy[i] >= level) + 1
    last_below = max(i for i in range(drift, n) if accuracy[i] < level) + 1
    return {
        "trough_acc": accuracy[trough],
        "trough_at": trough + 1,
        "first_back": first_back,
        "sustained": last_below + 1,
        "delay": alarm - drift,
    }


def sea_labels(n=50000):
    from river import datasets
    from river_benchmarks import concept_drift_stream
    stream = concept_drift_stream(
        stream=datasets.synth.SEA(seed=42, variant=0),
        drift_stream=datasets.synth.SEA(seed=42, variant=1),
        position=TRUE_DRIFT_INDEX,
        width=1,
        seed=42,
    )
    return [y for _, y in itertools.islice(stream, n)]


def elec2_labels():
    from river import datasets
    return [y for _, y in datasets.Elec2()]


def f1_from_errors(labels, correct):
    from river import metrics
    f1 = metrics.F1()
    for y, c in zip(labels, correct):
        f1.update(y, y if c else not y)
    return f1.get()


def analyze(name, filepath, labels=None):
    print(f"\n### {name} ({filepath}) ###")
    for model, s in load_telemetry(filepath).items():
        correct = reconstruct_correct(s["accuracy"])
        errors = [1 - c for c in correct]
        replayed = replay_detector(errors)
        lat = s["latency_us"]
        n = len(correct)

        print(f"\n{model}  (N={n})")
        print(f"  overall prequential accuracy : {sum(correct) / n:.4f}")
        print(f"  final rolling accuracy (W={WINDOW}): {s['accuracy'][-1]:.4f}")
        if labels is not None:
            print(f"  cumulative F1 (positive=True): {f1_from_errors(labels, correct):.4f}")
        print(f"  latency us  mean {statistics.mean(lat):.2f} | median {statistics.median(lat):.2f} | "
              f"p95 {percentile(lat, 0.95):.2f} | p99 {percentile(lat, 0.99):.2f}")
        print(f"  logged alarms   : {s['drift_points']}")
        print(f"  replayed alarms : {replayed}  (match: {replayed == s['drift_points']})")

        if name == "SEA" and s["drift_points"]:
            alarm = s["drift_points"][0]
            pre = correct[:TRUE_DRIFT_INDEX]
            gap = errors[TRUE_DRIFT_INDEX:alarm]
            r = recovery_stats(s["accuracy"], alarm)
            print(f"  accuracy samples 1-{TRUE_DRIFT_INDEX}: {sum(pre) / len(pre):.4f}")
            print(f"  error rate between drift and alarm: {sum(gap) / len(gap):.4f}")
            print(f"  detection delay: {r['delay']}")
            print(f"  trough rolling acc {r['trough_acc']:.3f} at sample {r['trough_at']}")
            print(f"  first back >= {RECOVERY_LEVEL}: sample {r['first_back']} "
                  f"(+{r['first_back'] - TRUE_DRIFT_INDEX})")
            print(f"  sustained >= {RECOVERY_LEVEL} from: sample {r['sustained']} "
                  f"(+{r['sustained'] - TRUE_DRIFT_INDEX})")


if __name__ == "__main__":
    with_f1 = "--f1" in sys.argv
    analyze("SEA", "telemetry_sea.csv", sea_labels() if with_f1 else None)
    analyze("Elec2", "telemetry_elec2.csv", elec2_labels() if with_f1 else None)
