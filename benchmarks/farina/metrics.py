"""One-to-one spike scoring with an alignment fixed before evaluation."""

import numpy as np


def score_spikes(reference, predicted, *, tolerance, lag=0, start=0, stop=None):
    """Score sample indices; add lag to predictions BEFORE selecting the time window.

    Sorted, unique indices are required. Empty denominators score zero and counts
    are retained so silent reference windows can be excluded explicitly.
    """
    if tolerance < 0 or (stop is not None and stop <= start):
        raise ValueError("Invalid tolerance or time window")
    reference = np.asarray(reference, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    for train in (reference, predicted):
        if train.ndim != 1 or np.any(np.diff(train) <= 0):
            raise ValueError("Spike indices must be one-dimensional, sorted and unique")
    predicted = predicted + lag
    reference = reference[reference >= start]
    predicted = predicted[predicted >= start]
    if stop is not None:
        reference = reference[reference < stop]
        predicted = predicted[predicted < stop]
    i = j = common = 0
    while i < len(reference) and j < len(predicted):
        if abs(reference[i] - predicted[j]) <= tolerance:
            common += 1
            i += 1
            j += 1
        elif reference[i] < predicted[j]:
            i += 1
        else:
            j += 1
    false_positive = len(predicted) - common
    false_negative = len(reference) - common

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return {
        "tp": common,
        "fp": false_positive,
        "fn": false_negative,
        "n_reference": len(reference),
        "n_predicted": len(predicted),
        "precision": ratio(common, len(predicted)),
        "recall": ratio(common, len(reference)),
        "f1": ratio(2 * common, len(reference) + len(predicted)),
        "roa": ratio(common, common + false_positive + false_negative),
    }
