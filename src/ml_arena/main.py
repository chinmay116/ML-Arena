from classification import run_classification
from sklearn.model_selection import train_test_split

def benchmark(data, target, problem_type):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)       # X-> Features,  y-> Target

    if problem_type == "classification":
        results = run_classification(X_train, X_test, y_train, y_test)

    # else:
    #     results = run_regression()