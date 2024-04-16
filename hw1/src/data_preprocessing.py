import pandas as pd
import numpy as np

excluded_columns = ['NO                  ',
                    'WIND_SPEED          ',
                    'NMHC                ', 
                    'O3                  ', 
                    'NO2                 ', 
                    'SO2                 ',]

# To test if the result changes and have a third submission
additional_excluded_columns = ['NO                  ', 
                            'RAINFALL            ', 
                            'WIND_SPEED          ', 
                            'O3                  ', 
                            'AMB_TEMP            ', 
                            'NMHC                ']

#excluded_columns = additional_excluded_columns

TIME_SLOTS = [str(i) for i in range(24)]


def fetch_and_process_data(path_to_train, path_to_test):
    test_headers = ['index', 'TYPE             ', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9']
    training_data = pd.read_csv(path_to_train)
    testing_data = pd.read_csv(path_to_test)
    testing_data = pd.read_csv(path_to_test, names=test_headers)
    training_data = training_data.drop(["Location"], axis=1)
    return training_data, testing_data

def transform_data(dataset, time_slots):
    transformed_data = []
    for time_slot in time_slots:
        relevant_dataset = dataset[['Date', 'ItemName', time_slot]]
        transformed = relevant_dataset.pivot(index='Date', columns='ItemName', values=time_slot)
        transformed_data.append(transformed)
    return pd.concat(transformed_data, ignore_index=True, axis=0)

def sanitize_and_fill(dataset):
    mapping_dict = {'#                              ': np.nan, 'x                              ': np.nan,
                    '*                              ': np.nan, 'A                              ': np.nan}
    dataset = dataset.replace(mapping_dict)
    dataset.drop(excluded_columns, axis=1, inplace=True)
    dataset.dropna(inplace=True)
    dataset.reset_index(inplace=True, drop=True)
    dataset = dataset.astype(float)
    return dataset

def compute_fill_values(dataset):
    fill_values = {column: dataset[column].median() for column in dataset.columns}
    return fill_values

def fill_missing_values(data, fill_dict, category_column):
    replacement_dict = {
        '#                              ': np.nan,
        'x                              ': np.nan,
        '*                              ': np.nan,
        'A                              ': np.nan,
        'WIND_DIR+D2070EC          ': np.nan
    }

    for key in fill_dict:
        data.loc[data[category_column] == key] = data.loc[data[category_column] == key].replace(replacement_dict)

    return data

def interpolate_missing_values(array, previous):
    row_count, _ = array.shape
    for i in range(row_count):
        nan_indices = np.isnan(array[i])
        valid_indices = np.where(~nan_indices)[0]
        if len(valid_indices) == 0 and previous is not None:
            previous_valid_indices = np.where(~np.isnan(previous[i]))[0]
            previous_values = previous[i][previous_valid_indices]
        else:
            previous_values = array[i][valid_indices]

        array[i, nan_indices] = np.interp(np.where(nan_indices)[0], valid_indices, previous_values)
    return array, np.copy(array)

def preprocess(dataset, excluded_columns):
    for column in excluded_columns:
        dataset = dataset[dataset['TYPE             '] != column].reset_index(drop=True)
    return dataset

def preprocess_and_interpolate(dataset, days_list, column_indices):
    first_day_data = dataset[dataset['index'] == days_list[0]].iloc[:, column_indices].to_numpy().astype(float)
    for day in days_list[1:]:
        daily_data = dataset[dataset['index'] == day].iloc[:, column_indices].to_numpy().astype(float)
        first_day_data = np.concatenate((first_day_data, daily_data), axis=1)

    interpolated_data, _ = interpolate_missing_values(first_day_data, None)
    for i, day in enumerate(days_list):
        start_idx = i * 9
        end_idx = start_idx + 9
        dataset.loc[dataset['index'] == day, dataset.columns[column_indices]] = interpolated_data[:, start_idx:end_idx]
    return dataset, _

def extract_features_targets(dataset, unique_days):
    features, targets = [], []
    for day in unique_days:
        day_data = dataset[dataset['index'] == day]
        feature = day_data.iloc[:, 2:11].to_numpy().astype(float)
        target = day_data[day_data['TYPE             '] == 'PM2.5               '].iloc[:, 10].to_numpy()
        if target.size > 0:
            targets.append(float(target[0]))
            features.append(feature.flatten())
    return np.array(features), np.array(targets)

def filter_out_columns(dataframe, unwanted_columns):
    for column in unwanted_columns:
        dataframe = dataframe[dataframe['ItemName'] != column].reset_index(drop=True)
    return dataframe

def fill_and_smooth_data(dataframe, date_col, column_indices):
    unique_days = dataframe[date_col].unique()
    accumulated_data = dataframe[dataframe[date_col] == unique_days[0]].iloc[:, column_indices].to_numpy().astype(float)
    for subsequent_day in unique_days[1:]:
        day_data = dataframe[dataframe[date_col] == subsequent_day].iloc[:, column_indices].to_numpy().astype(float)
        accumulated_data = np.concatenate((accumulated_data, day_data), axis=1)

    smoothed_data, _ = interpolate_missing_values(accumulated_data, None)
    for i, day in enumerate(unique_days):
        start_index = i * 24
        end_index = start_index + 24
        dataframe.loc[dataframe[date_col] == day, dataframe.columns[column_indices]] = smoothed_data[:, start_index:end_index]
    return dataframe

def generate_features_and_labels(dataframe, day_counts, start_column):
    feature_set, label_set = [], []
    for month in range(day_counts.shape[0] // 20):
        month_days = day_counts[month * 20:(month + 1) * 20]
        month_data = dataframe[dataframe['Date'] == month_days[0]].iloc[:, start_column:(start_column + 26)].to_numpy().astype(float)
        for day in month_days[1:]:
            day_data = dataframe[dataframe['Date'] == day].iloc[:, start_column:(start_column + 26)].to_numpy().astype(float)
            month_data = np.concatenate((month_data, day_data), axis=1)

        for hour in range(20 * 24):
            start_idx = hour
            end_idx = start_idx + 9
            if end_idx >= 20 * 24:
                break
            feature = month_data[:, start_idx:end_idx]
            target = month_data[5, end_idx]  # Assuming target is always at the same column offset

            feature_set.append(feature.flatten())
            label_set.append(target)
    return np.array(feature_set), np.array(label_set)

def prepare_data():
    training_data, testing_data = fetch_and_process_data('./data/train.csv', './data/test.csv')

    processed_data = transform_data(training_data, TIME_SLOTS)
    processed_data = sanitize_and_fill(processed_data)
    fill_values = compute_fill_values(processed_data)

    processed_df = filter_out_columns(training_data.copy(), excluded_columns)
    processed_df = fill_missing_values(processed_df, fill_values, 'ItemName')
    processed_df = fill_and_smooth_data(processed_df, 'Date', range(2, 26))
    features, labels = generate_features_and_labels(processed_df, processed_df['Date'].unique(), 2)

    cleaned_testing_data = preprocess(testing_data.copy(), excluded_columns)
    cleaned_testing_data = fill_missing_values(cleaned_testing_data, fill_values, 'TYPE             ')
    cleaned_testing_data, _ = preprocess_and_interpolate(cleaned_testing_data, cleaned_testing_data['index'].unique(), range(2, 11))
    test_feature_array, test_target_array = extract_features_targets(cleaned_testing_data, cleaned_testing_data['index'].unique())

    return features, labels, testing_data, test_feature_array, test_target_array


if __name__ == "__main__":
    # Prepare data
    features, labels, testing_data, test_feature_array, test_target_array = prepare_data()

    # Displaying some basic information about the prepared data
    print("Features shape:", features.shape)
    print("Labels length:", len(labels))
    print("Testing data shape:", testing_data.shape)
    print("Test feature array shape:", test_feature_array.shape)
    print("Test target array length:", len(test_target_array))

