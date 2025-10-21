# Using Gini Index to build a simple Decision Tree(CART)
import numpy as np

class DecisionTreeCART:

    # Define the constructor with max_depth parameter
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    # Fit the model to the data
    def fit(self, X, y):
        '''
        Fit the decision tree model to the training data.
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Class labels.
        '''
        self.tree = self._build_tree(X, y, depth=0)

    def _calculate_gini(self, y):
        '''
        Calculate the Gini impurity for a given array of class labels.
        Parameters:
        y (array-like): Array of class labels.
        Returns:
        float: Gini impurity.
        '''
        classes, counts = np.unique(y, return_counts=True)
        gini = 1.0 - sum((count / len(y)) ** 2 for count in counts)
        return gini
    
    def _best_split(self, X, y):
        '''
        Find the best feature and threshold to split the data to minimize Gini impurity.
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Class labels.
        Returns:
        dict: Best split information containing feature index, threshold, and indices for left and right splits.
        '''
        best_gini = float('inf')
        best_split = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):

            # Get unique thresholds for the feature
            thresholds = np.unique(X[:, feature_index])

            # Handle binary features separately
            if len(np.unique(X[:, feature_index])) <= 2:
                threshold = 0.5
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
            
            # Non-binary features
            else:
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
        '''
        Recursively build the decision tree.
        Parameters:
        X (array-like): Feature matrix.
        y (array-like): Class labels.
        depth (int): Current depth of the tree.
        Returns:
        dict: Tree node.
        '''
        num_classes = len(np.unique(y))


        # Stop Splitting When All Samples Belong to One Class or Max Depth is Reached
        if num_classes == 1 or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._majority_class(y)
            return {'type': 'leaf', 'class': leaf_value}

        split = self._best_split(X, y)
        # Stop if no valid split is found
        if split is None:
            leaf_value = self._majority_class(y)
            return {'type': 'leaf', 'class': leaf_value}

        # Recursively build left and right subtrees
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
        '''
        Determine the majority class in an array of class labels.
        Parameters:
        y (array-like): Array of class labels.
        Returns:
        int/str: Majority class label.
        '''
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        return majority_class
    
    def predict(self, X):
        '''
        Predict class labels for the input samples.
        Parameters:
        X (array-like): Feature matrix.
        Returns:
        array: Predicted class labels.
        '''
        return np.array([self._predict_sample(sample, X, self.tree) for sample in X])

    def _predict_sample(self, sample, X, tree):
        '''
        Predict the class label for a single sample by traversing the tree.
        Parameters:
        sample (array-like): Single input sample.
        tree (dict): Current tree node.
        Returns:
        int/str: Predicted class label.
        '''
        if tree['type'] == 'leaf':
            return tree['class']
        
        feature_value = sample[tree['feature_index']]
        # Binary feature
        if len(np.unique(X[:, tree['feature_index']])) <= 2:
            threshold = 0.5
            if feature_value <= threshold:
                return self._predict_sample(sample, X, tree['left'])
            else:
                return self._predict_sample(sample, X, tree['right'])
        # Non-binary feature
        else:  
            if feature_value <= tree['threshold']:
                return self._predict_sample(sample, X, tree['left'])
            else:
                return self._predict_sample(sample, X, tree['right'])

    def importance(self):
        '''
        Calculate feature importance based on the decision tree structure.
        Returns:
        list: Sorted list of tuples (feature_index, importance_score).
        '''
        feature_importances = {}
        self._calculate_importance(self.tree, feature_importances, 1.0)
        return sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_importance(self, node, feature_importances, weight):
        '''
        Recursively calculate feature importance.
        Parameters:
        node (dict): Current tree node.
        feature_importances (dict): Dictionary to store feature importance scores.
        weight (float): Weight of the current node.
        '''
        if node['type'] == 'leaf':
            return
        
        feature_index = node['feature_index']
        if feature_index not in feature_importances:
            feature_importances[feature_index] = 0.0
        feature_importances[feature_index] += weight

        self._calculate_importance(node['left'], feature_importances, weight * 0.5)
        self._calculate_importance(node['right'], feature_importances, weight * 0.5)
        
    def accuracy(self, y_true, y_pred):
        '''
        Calculate the accuracy of predictions.
        Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        Returns:
        float: Accuracy score.
        '''
        return np.sum(y_pred == y_true) / len(y_true)