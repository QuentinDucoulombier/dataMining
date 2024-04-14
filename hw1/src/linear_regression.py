import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, regularization_strength=1e-5):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Ajouter x0 = 1 à chaque instance
        identity_matrix = np.eye(X_b.shape[1])
        identity_matrix[0, 0] = 0  # Exclure le terme de biais de la régularisation
        self.weights = np.linalg.inv(X_b.T @ X_b + regularization_strength * identity_matrix) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Ajouter x0 = 1 à chaque instance
        return X_b @ self.weights

    def calculate_rmse(self, predictions, targets):
        return np.sqrt(np.mean((predictions - targets) ** 2))

def load_data(filepath):
    data = pd.read_csv(filepath)
    if 'PM2.5' not in data.columns:
        raise ValueError("Data must contain 'PM2.5' column.")
    X = data.drop('PM2.5', axis=1).values
    y = data['PM2.5'].values
    return X, y

if __name__ == "__main__":
    # Charger les données d'entraînement et de test
    X_train, y_train = load_data('./processed_train.csv')
    X_test, y_test = load_data('./processed_test.csv')

    # Créer une instance de LinearRegression
    model = LinearRegression()

    # Ajuster le modèle
    model.fit(X_train, y_train)

    # Prédire en utilisant le modèle
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    # Calculer et afficher RMSE
    rmse_train = model.calculate_rmse(predictions_train, y_train)
    rmse_test = model.calculate_rmse(predictions_test, y_test)

    # Tracer les prédictions par rapport aux valeurs réelles pour le jeu de test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    plt.show()

    print(f"RMSE on training set: {rmse_train:.10f}")
    print(f"RMSE on test set: {rmse_test:.10f}")
