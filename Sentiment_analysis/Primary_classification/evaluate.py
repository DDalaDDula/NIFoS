from sklearn.metrics import accuracy_score, classification_report

def print_evaluation_metrics(test_accuracy_mean, true_labels, predicted_labels):
    print(f"Test Accuracy: {test_accuracy_mean}")
    print(classification_report(true_labels, predicted_labels, target_names=['긍정', '부정', '중립']))