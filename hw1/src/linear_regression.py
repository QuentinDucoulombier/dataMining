import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import prepare_data


features, labels, testing_data, test_feature_array, _ = prepare_data()
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

def train_evaluate_for_sizes(features, labels, sizes, model_kind='linear', regularization=1e-5):
    train_sizes = sizes
    train_rmses = []
    val_rmses = []

    for size in train_sizes:
        subset_indices = np.random.choice(np.arange(len(features)), int(len(features) * size), replace=False)
        subset_features = features[subset_indices]
        subset_labels = labels[subset_indices]

        train_X, train_Y, val_X, val_Y = split_dataset(subset_features, subset_labels)

        model, validation_rmse = train_evaluate_predict(model_kind, train_X, train_Y, val_X, val_Y, identifiers, test_feature_array, regularization, return_model=True)
        train_predictions = model.predict(train_X)
        train_rmse = calculate_rmse(train_Y, train_predictions)

        train_rmses.append(train_rmse)
        val_rmses.append(validation_rmse)

    return train_sizes, train_rmses, val_rmses

# Adjusted function to return the model as well
def train_evaluate_predict(model_kind, training_features, training_labels, validation_features, validation_labels, identifier, future_features, regularization=None, return_model=False):
    if model_kind == 'linear':
        model = LinearModel()
        model.train(training_features, training_labels)  # No regularization parameter
    elif model_kind == 'ridge':
        model = RidgeModel()
        model.train(training_features, training_labels, regularization)  # Pass regularization parameter
    else:
        raise ValueError("Model type not supported, select 'linear' or 'ridge'")

    val_predictions = model.predict(validation_features)
    validation_rmse = calculate_rmse(validation_labels, val_predictions)

    if return_model:
        return model, validation_rmse
    return validation_rmse

def plot_impact_amounts(training_sizes, train_rmses, val_rmses):
    plt.figure(figsize=(10, 6))
    plt.plot(training_sizes, train_rmses, marker='o', label='Training RMSE')
    plt.plot(training_sizes, val_rmses, marker='o', color='red', label='Validation RMSE')
    plt.title('Impact of Training Data Size on RMSE')
    plt.xlabel('Percentage of Training Data Used')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_evaluate_for_regularization(features, labels, model_kind, lambda_values):
    val_rmses = []

    for lambda_val in lambda_values:
        # Split the data
        train_X, train_Y, val_X, val_Y = split_dataset(features, labels)

        # Train and evaluate the model with regularization
        model, validation_rmse = train_evaluate_predict(model_kind, train_X, train_Y, val_X, val_Y, identifiers, test_feature_array, lambda_val, return_model=True)
        val_rmses.append(validation_rmse)

    return lambda_values, val_rmses

def plot_regularization_impact(lambda_values, val_rmses):
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, val_rmses, marker='o', label='Validation RMSE')
    plt.title('Impact of Regularization on PM2.5 Prediction Accuracy')
    plt.xlabel('Regularization Strength (Lambda)')
    plt.ylabel('Validation RMSE')
    plt.xscale('log')  # Use logarithmic scale if lambda_values vary exponentially
    plt.legend()
    plt.grid(True)
    plt.show()

identifiers = testing_data['index'].unique()

train_X, train_Y, val_X, val_Y = split_dataset(features, labels)
model_selected = 'linear'
lambda_value = 0.1
validation_rmse = train_evaluate_predict(model_selected, train_X, train_Y, val_X, val_Y, identifiers, test_feature_array, lambda_value)

print("Validation RMSE:", validation_rmse)

sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
train_sizes, train_rmses, val_rmses = train_evaluate_for_sizes(features, labels, sizes)

plot_impact_amounts(train_sizes, train_rmses, val_rmses)

lambda_values = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
lambda_vals, validation_rmses = train_evaluate_for_regularization(features, labels, 'ridge', lambda_values)
plot_regularization_impact(lambda_vals, validation_rmses)