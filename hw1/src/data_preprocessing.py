#=============================================================================================================
# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#=============================================================================================================
# Functions to load and preprocess data

def load_and_preprocess_data():
    # Import Train and Test Data
    data_train = pd.read_csv("./data/train.csv")
    data_test = pd.read_csv("./data/test.csv", header=None)

    # Labels for the features in the datasets
    labels = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
              'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']

    train_data = []
    test_data = []

    # Constants for the shape of the data matrix
    nbrDataHorizontal_train = 24       # Maximum of DataHorizontal for train
    nbrDataVertical_train = 240        # Maximum of DataVertical for train
    nbrDataHorizontal_test = 9         # Maximum of DataHorizontal for test
    nbrDataVertical_test = 244         # Maximum of DataVertical for test

    # Data Extraction
    for i in range(nbrDataVertical_train):
        for j in range(nbrDataHorizontal_train):
            tempDataPoint = []
            for k in range(18):
                data_point = data_train.iloc[18*i+k, 3+j].strip()
                if data_point in ['#', 'x', 'A', '*']:
                    tempDataPoint.append(np.nan)
                else:
                    tempDataPoint.append(float(data_point))
            train_data.append(tempDataPoint)

    for i in range(nbrDataVertical_test):
        for j in range(nbrDataHorizontal_test):
            tempDataPoint = []
            for k in range(18):
                data_point = data_test.iloc[18*i+k, 2+j].strip()
                if data_point in ['#', 'x', 'A', '*', 'WIND_DIR+D2070EC']:
                    tempDataPoint.append(np.nan)
                else:
                    tempDataPoint.append(float(data_point))
            test_data.append(tempDataPoint)

    # Create DataFrame from train_data and test_data with column names as labels
    df_train = pd.DataFrame(train_data, columns=labels)
    df_test = pd.DataFrame(test_data, columns=labels)

    return df_train, df_test

def save_data(df_train, df_test):
    df_train.to_csv('./processed_train.csv', index=False)
    df_test.to_csv('./processed_test.csv', index=False)

def main():
    df_train, df_test = load_and_preprocess_data()
    save_data(df_train, df_test)

    # Data Analysis (Optional: You might move this to another script or notebook for clarity)
    correlation_matrix = df_train.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
    sns.pairplot(df_train.dropna(), diag_kind='kde', plot_kws={'alpha': 0.2, 's': 10}, height=0.6)
    plt.show()

if __name__ == '__main__':
    main()
