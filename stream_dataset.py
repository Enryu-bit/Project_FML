import csv


def _cast(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def stream_dataset(filepath: str, target_column: str):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = row.pop(target_column)
            x = {k: _cast(v) for k, v in row.items()}
            yield x, _cast(y)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    target = sys.argv[2] if len(sys.argv) > 2 else "class"

    for i, (x, y) in enumerate(stream_dataset(path, target)):
        if i >= 5:
            break
        print(f"Row {i}: x={x}, y={y}")
