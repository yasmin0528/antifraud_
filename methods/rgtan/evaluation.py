import numpy as np


def best_macro_f1_threshold(labels, scores):
    """Select the exact best binary macro-F1 threshold in O(N log N).

    Predictions use ``score >= threshold``. Ties are resolved exactly like the
    previous ascending exhaustive search: the smallest threshold attaining the
    maximum macro-F1 is returned.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if len(labels) == 0:
        raise ValueError("cannot select a threshold from an empty validation set")
    if not np.isfinite(scores).all():
        raise ValueError("validation scores must all be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("best_macro_f1_threshold supports only binary labels 0/1")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_positive = (labels[order] == 1).astype(np.int64)
    positive_prefix = np.concatenate(([0], np.cumsum(sorted_positive)))
    candidates = np.unique(np.concatenate(([0.0], sorted_scores, [1.0])))

    # Samples before each insertion point have score < threshold and are
    # predicted negative; all remaining samples are predicted positive.
    split = np.searchsorted(sorted_scores, candidates, side="left")
    false_negative = positive_prefix[split]
    true_negative = split - false_negative
    total_positive = positive_prefix[-1]
    true_positive = total_positive - false_negative
    false_positive = (len(labels) - split) - true_positive

    positive_denominator = 2 * true_positive + false_positive + false_negative
    negative_denominator = 2 * true_negative + false_positive + false_negative
    positive_f1 = np.divide(
        2 * true_positive, positive_denominator,
        out=np.zeros_like(candidates, dtype=np.float64), where=positive_denominator != 0)
    negative_f1 = np.divide(
        2 * true_negative, negative_denominator,
        out=np.zeros_like(candidates, dtype=np.float64), where=negative_denominator != 0)
    # sklearn's average="macro" averages over labels present in either y_true
    # or y_pred. Usually both are present, but preserve that behavior for
    # degenerate one-class validation fixtures as well.
    predicted_positive = len(labels) - split
    predicted_negative = split
    positive_present = (total_positive > 0) | (predicted_positive > 0)
    total_negative = len(labels) - total_positive
    negative_present = (total_negative > 0) | (predicted_negative > 0)
    present_count = positive_present.astype(np.int64) + negative_present.astype(np.int64)
    macro_f1 = (
        positive_f1 * positive_present + negative_f1 * negative_present
    ) / present_count
    best_index = int(np.argmax(macro_f1))
    return float(candidates[best_index]), float(macro_f1[best_index])
