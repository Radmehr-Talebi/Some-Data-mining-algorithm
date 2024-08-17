
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import time


# Data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
data = pd.read_csv(url, header=None)

# Labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC()
    }
    for name, model in models.items():
        # Train Start Time
        train_start_time = time.time()
        # Train
        model.fit(X_train, y_train)
        # Train End Time
        train_end_time = time.time()

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        # Test End Time
        test_end_time = time.time()
        # Accuracy
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        train_time = train_end_time - train_start_time
        test_time = test_end_time - train_end_time
        
        print(f"{name} - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")
        print(f"{name} - Train Time: {train_time:.2f} seconds, Test Time: {test_time:.2f} seconds\n")

# Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# PCA
pca = PCA(n_components=15)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Results using PCA:")
evaluate_models(X_train_pca, X_test_pca, y_train, y_test)

# SelectKBest
k_best = SelectKBest(score_func=f_classif, k=15) 
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

print("Results using SelectKBest:")
evaluate_models(X_train_kbest, X_test_kbest, y_train, y_test)

# None
print("Result without any Reducing")
evaluate_models(X_train, X_test, y_train, y_test)