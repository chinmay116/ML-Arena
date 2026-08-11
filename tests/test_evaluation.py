import numpy as np

from ml_arena.evaluation import evaluate_classification

def test_evaluate_binary_classification():
    y_test = np.array([0, 0, 1, 1, 1, 0, 1, 0])

    predictions = np.array([0, 1, 1, 1, 0, 0, 1, 0])

    probabilities = np.array([
        [0.90, 0.10],
        [0.40, 0.60],
        [0.20, 0.80],
        [0.10, 0.90],
        [0.70, 0.30],
        [0.80, 0.20],
        [0.30, 0.70],
        [0.95, 0.05],
    ])

    results = evaluate_classification(
        y_test,
        predictions,
        probabilities,
    )

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "roc_auc" in results
    assert "confusion_matrix" in results


def test_evaluate_binary_classification_metrics_are_valid():
    y_test = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])

    probabilities = np.array([
        [0.90, 0.10],
        [0.40, 0.60],
        [0.20, 0.80],
        [0.10, 0.90],
    ])

    results = evaluate_classification(
        y_test,
        predictions,
        probabilities,
    )

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1
    assert 0 <= results["roc_auc"] <= 1


def test_confusion_matrix_shape():
    y_test = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])

    probabilities = np.array([
        [0.90, 0.10],
        [0.40, 0.60],
        [0.20, 0.80],
        [0.10, 0.90],
    ])

    results = evaluate_classification(
        y_test,
        predictions,
        probabilities,
    )

    assert results["confusion_matrix"].shape == (2, 2)