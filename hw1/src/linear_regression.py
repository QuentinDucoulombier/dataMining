#=============================================================================================================
# Import Libraries
import numpy as np
import pandas as pd

#=============================================================================================================
# Linear Regression Model Class
class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, regularization_strength=1e-5):
        """
        Fit the linear regression model using the Normal Equation with regularization.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance
        identity_matrix = np.eye(X_b.shape[1])
        identity_matrix[0, 0] = 0  # Exclude the bias term from regularization
        self.weights = np.linalg.inv(X_b.T.dot(X_b) + regularization_strength * identity_matrix).dot(X_b.T).dot(y)


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
# Data Loading Function
def load_data(filepath):
    """
    Load and prepare data from a CSV file, handling NaN values.
    """
    data = pd.read_csv(filepath)
    # Check for NaN values
    if data.isnull().values.any():
        print("NaN values found in", filepath, "Handling...")
        # Option to fill NaNs
        data = data.fillna(method='ffill').fillna(method='bfill')
    X = data.drop('PM2.5', axis=1).values  # Assuming 'PM2.5' is the target column
    y = data['PM2.5'].values
    return X, y


#=============================================================================================================
# Main Execution Block
if __name__ == "__main__":
    # Load training and testing data
    X_train, y_train = load_data('./processed_train.csv')
    X_test, y_test = load_data('./processed_test.csv')

    # Create an instance of LinearRegression
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict using the model
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    # Calculate and print RMSE
    rmse_train = model.calculate_rmse(predictions_train, y_train)
    rmse_test = model.calculate_rmse(predictions_test, y_test)
    print("Training RMSE:", rmse_train)
    print("Testing RMSE:", rmse_test)

    # Print predicted values
    print("Train Predictions:", predictions_train)
    print("Test Predictions:", predictions_test)
