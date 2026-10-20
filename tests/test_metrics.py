import pytest

from app.metrics import threshold_metrics


def test_threshold_metrics_known_answer():
    # 4 frauds, 4 legit; threshold 0.5 flags 3 frauds + 1 legit
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.1]

    result = threshold_metrics(y_true, y_score, threshold=0.5)

    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (3, 1, 1, 3)
    assert result["precision"] == pytest.approx(0.75)  # 3 / (3 + 1)
    assert result["recall"] == pytest.approx(0.75)     # 3 / (3 + 1)


def test_nothing_flagged_gives_zero_precision():
    # Threshold above every score -> no alerts -> precision must be 0, not a crash
    result = threshold_metrics([0, 1, 1], [0.1, 0.2, 0.3], threshold=0.99)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_single_class_does_not_crash():
    # No fraud and no alerts at all -> confusion matrix must still unpack
    result = threshold_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert result["tn"] == 3