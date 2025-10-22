
import numpy as np

class LinearRegression:

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        '''
        Fit the linear regression model to the training data using the Normal Equation.
        Parameters:
        X : numpy array of shape (n_samples, n_features)
            Training data.
        y : numpy array of shape (n_samples,)
            Target values.
        Returns:
        None
        '''
        if np.nan in X or np.nan in y or '' in X or '' in y or None in X or None in y:
            raise ValueError("Input data contains NaN values. Please handle missing data before fitting the model.")
        
        # Add bias term (intercept) to the features
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance

        try:

            theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

            self.intercept = theta_best[0]
            self.coefficients = theta_best[1:]
        except Exception as e:
            print(f"Error during model fitting: {e}")
            raise

    def predict(self, X):
        '''
        Predict using the linear model.
        Parameters:
        X : array-like
            Samples.
        Returns:
        array
            Returns predicted values.
        '''
        return X.dot(self.coefficients) + self.intercept
    
    def accuracy(self, y_pred, y):
        '''
        Calculate the R-squared accuracy of the model.
        Parameters:
        y_pre : numpy array of shape (n_samples,)
            Predicted target values.
        y : numpy array of shape (n_samples,)
            Actual target values.
        Returns:
        float
            R-squared accuracy score.
        '''
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        return r_squared