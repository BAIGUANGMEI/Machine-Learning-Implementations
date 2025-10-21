# Using Gini Index to build a simple Decision Tree(CART)
import numpy as np

class DecisionTreeCART:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _calculate_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        gini = 1.0 - sum((count / len(y)) ** 2 for count in counts)
        return gini
    
    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                gini_left = self._calculate_gini(y[left_indices])
                gini_right = self._calculate_gini(y[right_indices])
                weighted_gini = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / n_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        return best_split
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        if num_classes == 1 or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._majority_class(y)
            return {'type': 'leaf', 'class': leaf_value}

        split = self._best_split(X, y)
        if split is None:
            leaf_value = self._majority_class(y)
            return {'type': 'leaf', 'class': leaf_value}

        left_subtree = self._build_tree(X[split['left_indices']], y[split['left_indices']], depth + 1)
        right_subtree = self._build_tree(X[split['right_indices']], y[split['right_indices']], depth + 1)

        return {
            'type': 'node',
            'feature_index': split['feature_index'],
            'threshold': split['threshold'],
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        return majority_class
    
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
    
    def _predict_sample(self, sample, tree):
        if tree['type'] == 'leaf':
            return tree['class']
        
        feature_value = sample[tree['feature_index']]
        if feature_value <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])
        
    def importance(self):
        feature_importances = {}
        self._calculate_importance(self.tree, feature_importances, 1.0)
        return sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_importance(self, node, feature_importances, weight):
        if node['type'] == 'leaf':
            return
        
        feature_index = node['feature_index']
        if feature_index not in feature_importances:
            feature_importances[feature_index] = 0.0
        feature_importances[feature_index] += weight

        self._calculate_importance(node['left'], feature_importances, weight * 0.5)
        self._calculate_importance(node['right'], feature_importances, weight * 0.5)
        
    def accuracy(self, y_true, y_pred):
        return np.sum(y_pred == y_true) / len(y_true)
