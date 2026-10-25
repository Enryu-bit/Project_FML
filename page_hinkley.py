"""Page-Hinkley concept drift detector, implemented from first principles.

No dependency on river.drift or any other drift-detection library — the
recurrence relations below are derived directly from Page's (1954)
cumulative-sum change-point test, adapted to the streaming/online setting
used for concept drift detection over a model's error signal.
"""


class PageHinkleyDetector:
    r"""Sequential change-point (concept drift) detector.

    Mathematical formulation
    -------------------------
    Let x_t be the t-th value of an observed error signal (e.g. 0.0 for a
    correct prediction, 1.0 for a misclassification). Three quantities are
    maintained online:

    1. A bias-corrected exponentially-fading running mean of the signal,
       used as the "expected" (baseline) value against which new
       observations are compared:

           S_t     = alpha * S_{t-1} + (1 - alpha) * x_t,      S_0 = 0
           mean_t  = S_t / (1 - alpha^t)

       S_t is a plain exponential moving average (EMA) accumulator; dividing
       by (1 - alpha^t) removes the startup bias that a raw EMA has while
       t is still small (the classic bias-correction trick, cf. Adam's
       moment estimates). Consequently:
         - for t << 1 / (1 - alpha), mean_t behaves like the ordinary
           cumulative (arithmetic) mean of x_1 .. x_t;
         - as t grows, alpha^t -> 0 and mean_t smoothly relaxes into a pure
           EMA with effective memory ~ 1 / (1 - alpha) samples.
       This lets the detector track a slowly-drifting baseline without ever
       "chasing" a single early outlier.

    2. The Page-Hinkley cumulative sum of deviations from that baseline,
       offset by a tolerance delta:

           U_t = U_{t-1} + (x_t - mean_t - delta),     U_0 = 0

       delta is the minimum magnitude of change worth penalizing (a
       dead-band): it biases U_t to drift slightly downward under a
       stationary process, so that ordinary noise does not accumulate into
       a false alarm.

    3. The running historical minimum of U_t, and the test statistic itself:

           m_t  = min(U_1, ..., U_t),     m_0 = 0
           PH_t = U_t - m_t

    Drift is signaled the instant PH_t > threshold (lambda).

    Physical intuition
    -------------------
    U_t is a random walk with a small negative drift (-delta per step) as
    long as the process is stationary and mean_t is correctly tracking the
    signal's true average: E[x_t - mean_t - delta] ~= -delta < 0. A
    downward-drifting random walk keeps returning to (and setting new)
    historical minima, so U_t stays close to m_t and PH_t = U_t - m_t stays
    small — no alarm.

    When the error rate abruptly *increases* (a real performance drop),
    x_t jumps to a higher regime while mean_t — being a slow, fading
    average — takes time to catch up. For many consecutive steps
    (x_t - mean_t - delta) is then strongly positive, so U_t climbs steeply
    away from its frozen historical minimum m_t (m_t stops updating because
    U_t is no longer making new lows). The growing gap PH_t = U_t - m_t is
    exactly the size of that climb since the last low point — i.e. "how far
    has the cumulative error deviation risen above its best-ever level" —
    and once it exceeds threshold, the detector concludes the increase is
    too large and too sustained to be noise, and signals drift.

    Parameters
    ----------
    delta : float
        Tolerance / minimum magnitude of change to penalize (dead-band on
        the per-step deviation). Larger delta => less sensitive to small
        drifts, fewer false alarms.
    threshold : float
        Decision threshold (lambda) on PH_t. Larger threshold => fewer
        false alarms but longer detection delay.
    alpha : float
        Fading factor for the running mean, 0 < alpha <= 1. Values close
        to 1 make the baseline mean forget very slowly (long memory);
        smaller values adapt the baseline faster.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 < alpha <= 1")

        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self.reset()

    def reset(self) -> None:
        """Reinitialize all running state in place (no reallocation)."""
        self.n = 0
        self._ema = 0.0          # S_t: raw fading-mean accumulator
        self._alpha_pow = 1.0    # alpha^t, updated incrementally
        self.mean = 0.0          # mean_t: bias-corrected running mean
        self.sum = 0.0           # U_t: cumulative sum statistic
        self.min_sum = 0.0       # m_t: historical minimum of U_t
        self.ph_statistic = 0.0  # PH_t: current test statistic

    def update(self, value: float) -> bool:
        """Ingest one error-signal observation x_t and update all state in O(1).

        Recurrence relations applied, in order:
            n      <- n + 1
            S_n    <- alpha * S_{n-1} + (1 - alpha) * value
            mean_n <- S_n / (1 - alpha^n)
            U_n    <- U_{n-1} + (value - mean_n - delta)
            m_n    <- min(m_{n-1}, U_n)
            PH_n   <- U_n - m_n

        Returns
        -------
        bool
            True if PH_n > threshold (drift signaled), else False.
        """
        self.n += 1

        self._alpha_pow *= self.alpha
        self._ema = self.alpha * self._ema + (1.0 - self.alpha) * value
        self.mean = self._ema / (1.0 - self._alpha_pow)

        self.sum += value - self.mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)
        self.ph_statistic = self.sum - self.min_sum

        return self.ph_statistic > self.threshold


if __name__ == "__main__":
    import random

    random.seed(42)

    n_pre, n_post = 1000, 1000
    p_pre, p_post = 0.05, 0.45
    change_point = n_pre

    stream = [1.0 if random.random() < p_pre else 0.0 for _ in range(n_pre)]
    stream += [1.0 if random.random() < p_post else 0.0 for _ in range(n_post)]

    detector = PageHinkleyDetector(delta=0.005, threshold=50.0, alpha=0.9999)

    detected_at = None
    max_ph_before_change = 0.0

    for i, x in enumerate(stream, start=1):
        drift = detector.update(x)

        if i <= change_point:
            max_ph_before_change = max(max_ph_before_change, detector.ph_statistic)

        if drift and detected_at is None:
            detected_at = i
            print(f"[sample {i}] DRIFT DETECTED  |  PH={detector.ph_statistic:.3f} > "
                  f"threshold={detector.threshold}  |  running mean={detector.mean:.3f}")
            break

    print("\n--- Summary ---")
    print(f"Stream: {n_pre} samples ~ Bernoulli(p={p_pre}) then "
          f"{n_post} samples ~ Bernoulli(p={p_post}); true change point = index {change_point}")
    print(f"Max PH statistic observed during stable phase (no drift, should stay well "
          f"under threshold): {max_ph_before_change:.3f}")

    if detected_at is not None:
        delay = detected_at - change_point
        print(f"Detected drift at index {detected_at}  =>  detection delay = {delay} samples")
    else:
        print("No drift detected within the simulated stream.")
