import csv, random

with open("sample_stream.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["feature_a", "feature_b", "feature_c", "target"])
    for _ in range(20):
        w.writerow([round(random.uniform(0, 10), 2), round(random.uniform(0, 10), 2), round(random.uniform(0, 10), 2), random.randint(0, 1)])
