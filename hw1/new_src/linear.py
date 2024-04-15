import pandas as pd
import numpy as np

def load_and_prepare_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df = train_df.drop(["Location"], axis=1)  # Assuming location is not needed
    return train_df, test_df

def pivot_data(df, hours):
    pivoted_data = []
    for hour in hours:
        relevant_data = df[['Date', 'ItemName', hour]]
        pivot = relevant_data.pivot(index='Date', columns='ItemName', values=hour)
        pivoted_data.append(pivot)
    return pd.concat(pivoted_data, ignore_index=True, axis=0)

def clean_and_impute_data(df):
    replace_dict = {'#                              ': np.nan, 'x                              ': np.nan,
                    '*                              ': np.nan, 'A                              ': np.nan}
    df = df.replace(replace_dict)
    df.drop(['RAINFALL            ', 'NO                  ', 'WIND_SPEED          ', 'O3                  ',
             'AMB_TEMP            ', 'NMHC                '], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df = df.astype(float)
    return df

def calculate_impute_values(df):
    impute_values = {col: df[col].median() for col in df.columns}
    return impute_values

# Constants
HOURS = [str(i) for i in range(24)]  # More Pythonic way to create the hour list

# Main Execution
train_df, test_df = load_and_prepare_data('./data/train.csv', './data/test.csv')
print(train_df.shape)

new_data = pivot_data(train_df, HOURS)
new_data = new_data.rename_axis(None, axis=1)
new_data = clean_and_impute_data(new_data)

impute_values = calculate_impute_values(new_data)
print(new_data.info())
print(new_data.describe(include='all'))
print(new_data.shape)
print(impute_values)


def impute_nans(df, vals_to_impute, item_name):
    # Define all values to be replaced with NaN in a dictionary outside the loop
    replace_dict = {
        '#                              ': np.nan,
        'x                              ': np.nan,
        '*                              ': np.nan,
        'A                              ': np.nan,
        'WIND_DIR+D2070EC          ': np.nan
    }

    # Loop through each key in the vals_to_impute dictionary
    for key in vals_to_impute:
        # Select rows where the item_name matches key, and replace according to replace_dict
        mask = df[item_name] == key
        df.loc[mask] = df.loc[mask].replace(replace_dict)

    return df



# Testing dataset preprocessing
# Test targets are used as the validation data

def interpolate_nan(arr, prev):
    rows, cols = arr.shape
    for i in range(rows):
        nan_mask = np.isnan(arr[i])
        not_nan_indices = np.where(~nan_mask)[0]
        if len(not_nan_indices) == 0 and prev is not None:
            # If all values are NaN, take from previous data if available
            prev_nan_mask = np.isnan(prev[i])
            prev_not_nan_indices = np.where(~prev_nan_mask)[0]
            input_values = prev[i][prev_not_nan_indices]
        else:
            input_values = arr[i][not_nan_indices]

        arr[i, nan_mask] = np.interp(np.where(nan_mask)[0], not_nan_indices, input_values)
    return arr, np.copy(arr)


def preprocess_df(df, columns_to_remove):
    for col in columns_to_remove:
        df = df[df['NEW_COL             '] != col].reset_index(drop=True)
    return df

def interpolate_and_replace(df, days, impute_values, column_subset):
    month_arr = df[df['index_0'] == days[0]].iloc[:, column_subset].to_numpy().astype(float)
    for day in days[1:]:
        to_concat = df[df['index_0'] == day].iloc[:, column_subset].to_numpy().astype(float)
        month_arr = np.concatenate((month_arr, to_concat), axis=1)

    interpolated_data, prev = interpolate_nan(month_arr, None)
    for i, day in enumerate(days):
        start_idx = i * 9
        end_idx = start_idx + 9
        df.loc[df['index_0'] == day, df.columns[column_subset]] = interpolated_data[:, start_idx:end_idx]
    return df, prev

def extract_features_and_targets(df, unique_dates):
    features, targets = [], []
    for date in unique_dates:
        date_df = df[df['index_0'] == date]
        feature = date_df.iloc[:, 2:11].to_numpy().astype(float)
        target = date_df[date_df['NEW_COL             '] == 'PM2.5               '].iloc[:, 10].to_numpy()
        if target.size > 0:
            targets.append(float(target[0]))
            features.append(feature.flatten())
    return np.array(features), np.array(targets)



# Main script starts here
print(impute_values)
columns_to_remove = ['RAINFALL            ', 'NO                  ', 'WIND_SPEED          ', 'O3                  ', 'AMB_TEMP            ', 'NMHC                ']
modified_test_df = preprocess_df(test_df.copy(), columns_to_remove)
modified_test_df = impute_nans(modified_test_df, impute_values, 'NEW_COL             ')
print(modified_test_df.shape)

days = list(modified_test_df['index_0'].unique())
modified_test_df, _ = interpolate_and_replace(modified_test_df, days, impute_values, range(2, 11))

test_features, test_targets = extract_features_and_targets(modified_test_df, modified_test_df['index_0'].unique())
print(test_features.shape)

print(train_df.columns)
train_df


def remove_unwanted_columns(df, columns):
    for column in columns:
        df = df[df['ItemName'] != column].reset_index(drop=True)
    return df

def impute_and_interpolate(df, date_column, item_column, columns_range):
    days = df[date_column].unique()
    all_arr = df[df[date_column] == days[0]].iloc[:, columns_range].to_numpy().astype(float)
    for day in days[1:]:
        to_concat = df[df[date_column] == day].iloc[:, columns_range].to_numpy().astype(float)
        all_arr = np.concatenate((all_arr, to_concat), axis=1)

    interpolated_data, _ = interpolate_nan(all_arr, None)
    for i, day in enumerate(days):
        start_idx = i * 24
        end_idx = start_idx + 24
        df.loc[df[date_column] == day, df.columns[columns_range]] = interpolated_data[:, start_idx:end_idx]
    return df

def create_features_and_targets(df, days_per_month, start_col, target_col_offset):
    features, targets = [], []
    for month in range(days_per_month.shape[0] // 20):
        month_days = days_per_month[month * 20:(month + 1) * 20]
        month_arr = df[df['Date'] == month_days[0]].iloc[:, start_col:(start_col + 26)].to_numpy().astype(float)
        for day in month_days[1:]:
            to_concat = df[df['Date'] == day].iloc[:, start_col:(start_col + 26)].to_numpy().astype(float)
            month_arr = np.concatenate((month_arr, to_concat), axis=1)

        for hour in range(20 * 24):
            start_idx = hour
            end_idx = start_idx + 9
            if end_idx >= 20 * 24:
                break
            feature = month_arr[:, start_idx:end_idx]
            target = month_arr[5, end_idx]  # Assuming target is always at the same column offset

            features.append(feature.flatten())
            targets.append(target)
    return np.array(features), np.array(targets)

# Main script starts here
print(train_df.columns)
columns_to_remove = ['RAINFALL            ', 'NO                  ', 'WIND_SPEED          ', 'O3                  ', 'AMB_TEMP            ', 'NMHC                ']
modified_df = remove_unwanted_columns(train_df.copy(), columns_to_remove)
modified_df = impute_nans(modified_df, impute_values, 'ItemName')

days = modified_df['Date'].unique()
no_months = days.shape[0] // 20
print(no_months)

modified_df = impute_and_interpolate(modified_df, 'Date', 'ItemName', range(2, 26))

features, targets = create_features_and_targets(modified_df, days, 2, 5) 
print(len(features))
print(features.shape)
print(len(targets))


class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        # Add a bias column of ones to the input feature array
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Calculate weights using the Normal Equation
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        # Add a bias column of ones to the input feature array
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Return predictions
        return X_b @ self.weights

class RidgeRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, regularization_strength=1e-5):
        # Add a bias column of ones to the input feature array
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Create an identity matrix of the shape (n_features + 1, n_features + 1)
        identity_matrix = np.eye(X_b.shape[1])
        # Set the first diagonal element to 0 to exclude the bias term from regularization
        identity_matrix[0, 0] = 0
        # Calculate weights using the Ridge Regression formula (Normal Equation with regularization)
        self.weights = np.linalg.inv(X_b.T @ X_b + regularization_strength * identity_matrix) @ X_b.T @ y

    def predict(self, X):
        # Add a bias column of ones to the input feature array
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Return predictions
        return X_b @ self.weights

def compute_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


############################################################################################################
import numpy as np
import pandas as pd

def split_data(features, targets, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    indices = np.random.permutation(len(features))
    split_point = int((1 - test_size) * len(features))
    return (features[indices[:split_point]], targets[indices[:split_point]],
            features[indices[split_point:]], targets[indices[split_point:]])

def train_and_evaluate_regression(model_type, train_features, train_targets, val_features, val_targets, indexes, test_features, regularization_strength=None):
    # Initialize the appropriate model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = RidgeRegression()
    else:
        raise ValueError("Unsupported model type: choose 'linear' or 'ridge'")

    # Fit the model
    model.fit(train_features, train_targets, regularization_strength if model_type == 'ridge' and regularization_strength is not None else 0.01)
    
    # Predict on validation data
    predictions = model.predict(val_features)
    rmse = compute_rmse(val_targets, predictions)

    # Fit on entire dataset and predict on test set
    model.fit(features, targets, regularization_strength if model_type == 'ridge' and regularization_strength is not None else 0.01)
    test_predictions = model.predict(test_features)

    # Save predictions to CSV
    data = [[index, pred] for index, pred in zip(indexes, test_predictions)]
    df = pd.DataFrame(data, columns=['index', 'answer'])
    df.to_csv("submission.csv", index=False)

    return rmse

# Data loading and preparation
indexes = test_df['index_0'].unique()  # Assuming test_df and its structure is predefined

# Splitting data
train_features, train_targets, val_features, val_targets = split_data(features, targets)

# Model training and evaluation
model_type = 'ridge'  # Can be 'linear' or 'ridge'
regularization_strength = 0.01  # Used only if model_type is 'ridge'
rmse = train_and_evaluate_regression(model_type, train_features, train_targets, val_features, val_targets, indexes, test_features, regularization_strength)

# Output results
print("RMSE:", rmse)

