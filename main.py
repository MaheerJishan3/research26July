# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from svm_model import LinearSVM

# Load and preprocess data
data = pd.read_csv('breast-cancer.csv')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop('id', axis=1)
X = data.drop('diagnosis', axis=1).values
y = data['diagnosis'].values
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# Visualize decision boundary in 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
svm_2d = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
svm_2d.fit(X_train_pca, y_train)

def plot_decision_boundary(X, y, model):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('SVM Decision Boundary (2D PCA Projection)')
    plt.show()

plot_decision_boundary(X_test_pca, y_test, svm_2d)