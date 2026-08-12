import pandas as pd

from ml_arena.main import benchmark

def test_classification_benchmark():

    df = pd.read_csv("data/data.csv")

    results = benchmark(df, target="target", problem_type="classification")

    expected_models = {
        "Random Forest",
        "SVM",
        "Decision Tree",
        "KNN",
        "Logistic Regression",
    }

    assert set(results.keys()) == expected_models

    for model_name, metrics in results.items():

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics