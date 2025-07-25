# svm_model.py
import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Convert labels to {-1, 1} for SVM
        y_ = np.where(y <= 0, -1, 1)

        # Gradient descent
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    # Update weights for correctly classified points (only regularization term)
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Update weights and bias for misclassified points
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(np.sign(linear_output) == 1, 1, 0)