from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate_classification

CLASSIFICATION_MODELS = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression()
}

def run_classification(X_train, X_test, y_train, y_test):

    results = {}

    for name, model in CLASSIFICATION_MODELS:
        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        probabilities = model.predict_proba(X_test)

        metrics = evaluate_classification(y_test, prediction, probabilities)

        results[name] = metrics

        return results