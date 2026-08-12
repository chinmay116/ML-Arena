from .classification import run_classification
from sklearn.model_selection import train_test_split

def benchmark(data, target, problem_type):          # data -> df, target -> target_column, problem_type -> classification/regression

    if target not in data.columns:
        raise ValueError(f"Target columns '{target}' not found in dataset")
    
    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)       # X-> Features,  y-> Target

    if problem_type == "classification":
        results = run_classification(X_train, X_test, y_train, y_test)

    return results

    # else:
    #     results = run_regression()