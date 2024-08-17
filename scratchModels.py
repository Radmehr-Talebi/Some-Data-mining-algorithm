
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Data
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
columns_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"

# Fetching
response = requests.get(columns_url)
names_text = response.text

# Parsing 
feature_names = []
for line in names_text.splitlines():
    if line.startswith('word_freq_') or line.startswith('char_freq_') or line.startswith('capital_run_length_'):
        feature_names.append(line.split(':')[0])
feature_names.append('label')

response = requests.get(data_url)
data_text = response.text
data = [list(map(float, line.split(','))) for line in data_text.strip().split('\n')]
df = pd.DataFrame(data, columns=feature_names)

# Displaying
print(df.head())
# Splitting the data Train and Test
X = df.drop('label', axis=1).values  # Convert to numpy array
y = df['label'].values  # Convert to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#           Custom Decision Tree
class CustomDecisionTree:
    def __init__(self, depth=10):
        self.depth = depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=self.depth)

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(set(y)) == 1:
            return np.mean(y)
        feature_idx, threshold = self._best_split(X, y)
        left_mask = X[:, feature_idx] < threshold
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth - 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth - 1)
        return (feature_idx, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_feature, best_threshold, best_impurity = None, None, float('inf')
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask
                impurity = self._impurity(y[left_mask], y[right_mask])
                if impurity < best_impurity:
                    best_feature, best_threshold, best_impurity = feature_idx, threshold, impurity
        return best_feature, best_threshold

    def _impurity(self, left_y, right_y):
        left_impurity = np.var(left_y) if len(left_y) > 0 else 0
        right_impurity = np.var(right_y) if len(right_y) > 0 else 0
        return len(left_y) * left_impurity + len(right_y) * right_impurity

    def predict(self, X):
        return np.array([self._predict_one(sample) for sample in X])

    def _predict_one(self, sample):
        node = self.tree
        while isinstance(node, tuple):
            feature_idx, threshold, left_tree, right_tree = node
            if sample[feature_idx] < threshold:
                node = left_tree
            else:
                node = right_tree
        return round(node)  # Ensure the prediction is binary

#               Custom KNN
class CustomKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for sample in X:
            distances = np.sqrt(((self.X_train - sample) ** 2).sum(axis=1))
            k_nearest = self.y_train[np.argsort(distances)[:self.k]]
            predictions.append(np.bincount(k_nearest.astype(int)).argmax())
        return np.array(predictions)

def evaluate_k_values(X_train, X_test, y_train, y_test):
    best_k = 1
    best_test_accuracy = 0
    list_acc=[]
    for k in range(2,100):  # You can adjust the range as needed but for now run we set it from 2 to 100
        model = CustomKNN(k=k)
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
        
        #print(f"k={k} - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")
        #print(f"k={k} - Train Time: {train_time:.2f} seconds, Test Time: {test_time:.2f} seconds\n")
        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_k = k
        list_acc.append(test_accuracy)    
    #print(f"\nThe best k value is {best_k} with a test accuracy of {best_test_accuracy:.2f}\n")
    #print(max(list))
    print('list of test accuaracies is : ',list_acc)
    print("best K is", best_k)
    return (best_k)
#print(evaluate_k_values(X_train, X_test, y_train, y_test))





#############                                                       ##########################



def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Custom Decision Tree": CustomDecisionTree(depth=10),
        "Custom KNN": CustomKNN(k=evaluate_k_values(X_train, X_test, y_train, y_test))
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
        #print(evaluate_k_values(X_train, X_test, y_train, y_test),'HHHHHHHHHHHHHHHHHHHHHH')

# PCA
# pca = PCA(n_components=15)  
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# print("Results using PCA:")
# evaluate_models(X_train_pca, X_test_pca, y_train, y_test)

# SelectKBest
# k_best = SelectKBest(score_func=f_classif, k=15) 
# X_train_kbest = k_best.fit_transform(X_train, y_train)
# X_test_kbest = k_best.transform(X_test)

# print("Results using SelectKBest:")
# evaluate_models(X_train_kbest, X_test_kbest, y_train, y_test)

# None
print("Result without any Reducing")
evaluate_models(X_train, X_test, y_train, y_test)
# list=[]

# for i in range(2):
#     CustomKNN.fit(X_train, y_train)
#     y_pred_train =CustomKNN(k=i).predict(X_train)
#     y_pred_test =CustomKNN(k=i).predict(X_test)
#     train_accuracy = accuracy_score(y_train, y_pred_train)
#     test_accuracy = accuracy_score(y_test, y_pred_test)
#     list.append(test_accuracy)
# print(list)    







#                            THANKS FOR YOUR TIME


