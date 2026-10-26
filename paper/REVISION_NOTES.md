# Report revision notes

The report (`conference_101719.tex` / `.pdf`) was audited against the code and
the telemetry exported from the Mac runs (`telemetry_sea.csv`,
`telemetry_elec2.csv`). Every derived number can be regenerated with
`python analyze_telemetry.py` (add `--f1` for F1, which needs the river datasets).

## Corrections to results

| Claim in previous draft | What the repo shows |
|---|---|
| LR beats HAT on Elec2 (91.2% vs 79.4%) | Those are only the last-500-sample rolling accuracies. Overall prequential accuracy is 0.837 (LR) vs 0.840 (HAT); LR's rolling accuracy exceeds HAT's 50.0% of the time. |
| Both models recover to ≥0.95 within 1,200–1,400 samples on SEA | 1,213/1,356 are the *detection delays*. LR returns to ≥0.95 at +953 (before its alarm) and stays ≥0.95 from +1,837; HAT first returns at +2,823 and stays ≥0.95 only from +6,519. |
| SEA pre-drift accuracy "near-perfect" for both | LR 0.994, HAT 0.971 (HAT's rolling accuracy dips to 0.936). |
| LR's longer delay due to "smoother probability updates" | LR's post-change error rate was lower (4.9% vs 7.4%) because SGD had already begun adapting; delay ≈ λ / (p − ē − δ). |
| Macro F1 | `river.metrics.F1()` is binary F1 of the positive class, cumulative. Values: SEA 0.978 / 0.994, Elec2 0.807 / 0.799 (HAT / LR). |

## Corrections to methodology

* **Order of operations** in `evaluate()`: predict → metrics → learn → (timer stops) → detector update → adapt. The previous draft placed the detector and adaptation before learning.
* The **latency timer** covers predict, the two metric updates, and learn. It excludes the detector.
* **Detector reset** sets α^t to 1, not 0. The detector holds 7 state scalars, not 6.
* **SEA labels**: `True` iff f1+f2 > θ. The classes are 68%/32% before the change and 60%/40% after it, not balanced. The change relabels 8.4% of instances. Both generators use seed 42, so the post-change segment replays the same feature vectors.
* **Elec2 in river** is the normalized version (features in [0,1] except day of week). The split is UP 42.5% / DOWN 57.5%.
* **HAT's alarm callback is a no-op**, so for HAT the external detector only monitors. HAT is unseeded, so its runs are not repeatable.
* **Telemetry** is written for every sample. The interval of 100 applies only to console printing.
* The project code does not use **numpy or pandas**.
* **Hyperparameters** (δ=0.005, λ=50, α=0.9999) are the standard Page-Hinkley defaults, not values "set from signal-to-noise margins".

## Verification performed

* Per-sample correctness was reconstructed exactly from the rolling accuracy. Replaying the detector reproduces all 22 logged alarm indices.
* Re-running SEA (river 0.26.1) reproduces the LR rolling-accuracy trace identically on all 50,000 samples. Five HAT re-runs each gave one alarm, with delays of 1,056–1,356.

## Items for the authors to confirm

* **Hardware:** every logged latency is a multiple of 41.67 ns. That is the 24 MHz timer of Apple-silicon Macs; Intel Macs report 1 ns. The draft's "Intel Core i5-1038NG7" is therefore inconsistent with the telemetry. The report now says "Apple-silicon Mac". Insert the exact chip and RAM.
* **Software versions:** the previous versions (Python 3.12/3.14, river 0.21.0, numpy 2.4.1, pandas 2.3.3) could not be verified from the repo (river 0.21.0 predates Python 3.14 and numpy 2.x, so this combination is unlikely). Run `pip freeze` on the Mac and add the real versions if needed.
* **Use of AI Tools** paragraph: updated to mention the audit and `analyze_telemetry.py`. Review the wording.
