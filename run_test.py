import pandas as pd

from ml_arena.main import benchmark

df = pd.read_csv("data/data.csv")

results = benchmark(data=df, target="target", problem_type="classification")

for model_name, metrics in results.items():

    print(f"\n{model_name}")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

# print(df.shape)
# print(df["target"].value_counts())
# print(df.corr()["target"].sort_values(ascending=False))