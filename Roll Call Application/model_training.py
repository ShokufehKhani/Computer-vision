# model_training.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import joblib

def train_svm(features, labels):
    param_grid = {
        'C': [1e-3, 1e-2, 1e-1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid_search.fit(features, labels)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo')

    # Decode labels
    labels_decoded = label_encoder.inverse_transform(np.unique(y_test))

    return accuracy, cm, roc_auc

def visualize_results(images, true_labels, predicted_labels):
    plt.figure(figsize=(10, 5))
    for i, (image, true_label, predicted_label) in enumerate(zip(images, true_labels, predicted_labels)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)
