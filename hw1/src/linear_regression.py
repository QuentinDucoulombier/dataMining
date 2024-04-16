import numpy as np
import pandas as pd
from data_preprocessing import prepare_data

# Load data
features, labels, testing_data, test_feature_array = prepare_data()
class LinearModel:
    def __init__(self):
        self.coefficients = None

    def train(self, X, y):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

    def predict(self, X):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_with_bias @ self.coefficients

class RidgeModel:
    def __init__(self):
        self.coefficients = None

    def train(self, X, y, lambda_val=1e-5):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        I = np.eye(X_with_bias.shape[1])
        I[0, 0] = 0  # Exclude bias from regularization
        self.coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias + lambda_val * I) @ X_with_bias.T @ y

    def predict(self, X):
        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_with_bias @ self.coefficients

def calculate_rmse(actual, predicted):
    mean_squared_error = np.mean((actual - predicted) ** 2)
    return np.sqrt(mean_squared_error)

def split_dataset(features, targets, split_ratio=0.2, seed=42):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(features))
    test_set_size = int(len(features) * split_ratio)
    train_indices = shuffled_indices[:-test_set_size]
    test_indices = shuffled_indices[-test_set_size:]
    return features[train_indices], targets[train_indices], features[test_indices], targets[test_indices]

def train_evaluate_predict(model_kind, training_features, training_labels, validation_features, validation_labels, identifier, future_features, regularization=None):
    if model_kind == 'linear':
        model = LinearModel()
    elif model_kind == 'ridge':
        model = RidgeModel()
    else:
        raise ValueError("Model type not supported, select 'linear' or 'ridge'")

    model.train(training_features, training_labels, regularization if model_kind == 'ridge' else 0.01)
    val_predictions = model.predict(validation_features)
    validation_rmse = calculate_rmse(validation_labels, val_predictions)

    model.train(features, labels, regularization if model_kind == 'ridge' else 0.01)
    future_predictions = model.predict(future_features)

    prediction_data = [[idx, prediction] for idx, prediction in zip(identifier, future_predictions)]
    results_df = pd.DataFrame(prediction_data, columns=['Index', 'Prediction'])
    results_df.to_csv("results.csv", index=False)

    return validation_rmse

identifiers = testing_data['index_0'].unique()

train_X, train_Y, val_X, val_Y = split_dataset(features, labels)
model_selected = 'ridge'
lambda_value = 0.01
validation_rmse = train_evaluate_predict(model_selected, train_X, train_Y, val_X, val_Y, identifiers, test_feature_array, lambda_value)

print("Validation RMSE:", validation_rmse)