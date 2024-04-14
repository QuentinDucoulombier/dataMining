#=============================================================================================================
# Import Libraries
import numpy as np

#=============================================================================================================
# Linear Regression Model Class
class LinearRegression:
    def __init__(self, regularization_strength=0.1):
        self.weights = None
        self.regularization_strength = regularization_strength

    def fit(self, X, y):
        """
        Fit the linear regression model using the Normal Equation with regularization:
        w = (X^T * X + lambda * I)^-1 * X^T * y
        where X is the matrix of features, y is the vector of targets, w is the vector of weights,
        lambda is the regularization strength, and I is the identity matrix.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0  # Do not regularize the bias term
        self.weights = np.linalg.inv(X_b.T.dot(X_b) + self.regularization_strength * I).dot(X_b.T).dot(y)

    def predict(self, X):
        """
        Make predictions using the linear regression model.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance
        return X_b.dot(self.weights)

    def calculate_rmse(self, predictions, targets):
        """
        Calculate the Root Mean Square Error (RMSE) between predictions and targets.
        """
        return np.sqrt(np.mean((predictions - targets) ** 2))

#=============================================================================================================
# Example usage of the Linear Regression Model
if __name__ == "__main__":
    # Example data
    # X represents the feature matrix, y represents the targets
    # Let's assume we have some dummy data here:
    X = np.array([[1, 2], [2, 3], [4, 5], [6, 7], [8, 9]])
    y = np.array([1, 3, 5, 7, 9])

    # Create an instance of LinearRegression
    model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # Predict using the model
    predictions = model.predict(X)

    # Calculate and print RMSE
    rmse = model.calculate_rmse(predictions, y)
    print("RMSE:", rmse)

    # Print predicted values
    print("Predictions:", predictions)
