import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Gradient Descent
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def sigmoid(self, x):

        x = np.array(x, dtype=np.float64)
        out = np.empty_like(x)
        positive_mask = x >= 0
        negative_mask = ~positive_mask
        # For positive x
        out[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))
        # For negative x, use exp(x) to avoid overflow of exp(-x)
        exp_x = np.exp(x[negative_mask])
        out[negative_mask] = exp_x / (1.0 + exp_x)
        return out
