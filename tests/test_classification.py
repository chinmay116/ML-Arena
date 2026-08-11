import numpy as np

from ml_arena.classification import run_classification


def test_run_classification():

    # Small artificial binary classification dataset
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
    ])

    y_train = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    X_test = np.array([
        [1.5, 2.5],
        [3.5, 4.5],
        [6.5, 7.5],
        [8.5, 9.5],
    ])

    y_test = np.array([
        0, 0, 1, 1,
    ])

    results = run_classification(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    expected_models = [
        "Random Forest",
        "SVM",
        "Decision Tree",
        "KNN",
        "Logistic Regression",
    ]

    # Check that every model was executed
    for model_name in expected_models:
        assert model_name in results

    # Check that we have exactly the expected models
    assert set(results.keys()) == set(expected_models)

    # Check the metrics returned by every model
    for model_name, metrics in results.items():

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics

        # Metrics should be between 0 and 1
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

        # Binary classification → 2 x 2 confusion matrix
        assert metrics["confusion_matrix"].shape == (2, 2)