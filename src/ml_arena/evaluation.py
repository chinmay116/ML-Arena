from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_classification(y_test, predictions, probabilities):

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions, average='macro')

    recall = recall_score(y_test, predictions, average='macro')

    f1 = f1_score(y_test, predictions, average='macro')

    no_of_classes = len(set(y_test))

    if no_of_classes == 2:
        roc_auc = roc_auc_score(y_test, probabilities[:,1])
    else:
        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr')

    cm = confusion_matrix(y_test, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }

# def evaluate_regression()