from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

from evaluation import evaluate_regression

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor()
}

def run_regression(X_train, X_test, y_train, y_test):

    results = {}

    for name, model in REGRESSION_MODELS:
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        probabilities = model.predict_proba(X_test)

        metrics = evaluate_regression(y_test, prediction, probabilities)

        results[name] = metrics

        return results