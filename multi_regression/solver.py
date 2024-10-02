import pandas as pd
import numpy as np
import sys

def drop_too_old_cars(data, year = 1970):
    indices = data[data['Godina proizvodnje'] <= year].index
    return data.drop(indices, inplace=False)

def drop_older_pricey_cars(data, year = 1990, price = 20000):
    indices = data[(data['Godina proizvodnje'] < year) & (data['Cena'] >= price)].index
    return data.drop(indices, inplace=False)

def drop_enormous_price_cars(data, price = 100000):
    indices = data[data['Cena'] >= price].index
    return data.drop(indices, inplace=False)

def feature_min_max_normalization(data, columns):
    mins = data[columns].min().to_dict()
    maxs = data[columns].max().to_dict()
    normalized_data = data.copy()
    for column in columns:
        normalized_data[column] = (data[column] - mins[column]) / (maxs[column]-mins[column])
    
    stats_dict = {column: {'min': mins[column], 'max': maxs[column]} for column in columns}
    
    return normalized_data, stats_dict

def min_max_normalize_test_set(data, coeffs, columns):
    normalized_data = data.copy()
    for column in columns:
        normalized_data[column] = (data[column] - coeffs[column]['min']) / (coeffs[column]['max']-coeffs[column]['min'])
    return normalized_data

class TargetEncoder:
    def __init__(self, cathegoric_columns):
        self.cathegoric_columns = cathegoric_columns
        self.mapping = {}

    def fit(self, data, target):
        self.data = data
        for col in self.cathegoric_columns:
            self.mapping[col] = data.groupby(col)[target.name].mean().to_dict()

    def transform(self, data):
        data_encoded = data.copy()
        for col in self.cathegoric_columns:
            data_encoded[col] = data_encoded[col].map(lambda x: self.mapping[col].get(x, 0))
        return data_encoded
    
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def euclidan_distance(self, x_test):
        return np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
    
    def predict(self, X_test):
        y_pred = np.zeros(len(X_test))
        for i, x_test in enumerate(X_test):
            distances = self.euclidan_distance(x_test)
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:self.k]
            k_nearest_labels = self.Y_train[k_nearest_indices]
            y_pred[i] = np.mean(k_nearest_labels)
        return y_pred
    

def knn_prediction(train_data, test_data, k):
    normalized_columns = ['Godina proizvodnje', 'Karoserija', 'Gorivo',
            'Konjske snage', 'Menjac', 'Marka']
    x_drops = ['Cena', 'Zapremina motora', 'Grad', 'Kilometraza']

    te = TargetEncoder(['Karoserija', 'Menjac', 'Gorivo', 'Marka'])
    te.fit(train_data, train_data['Cena'])

    train_data = te.transform(train_data)
    X_train = train_data.drop(columns=x_drops)
    y_train = train_data['Cena']

    test_data = te.transform(test_data)
    X_test = test_data.drop(columns=x_drops)
    y_test = test_data['Cena']

    X_train, train_min_max_coeffs = feature_min_max_normalization(X_train, normalized_columns)
    X_train['Godina proizvodnje'] = np.exp(X_train['Godina proizvodnje'])
    X_train['Konjske snage'] = np.exp(X_train['Konjske snage'])
    X_train, y_train = X_train.values, y_train.values

    X_test = min_max_normalize_test_set(X_test, train_min_max_coeffs, normalized_columns)
    X_test['Godina proizvodnje'] = np.exp(X_test['Godina proizvodnje'])
    X_test['Konjske snage'] = np.exp(X_test['Konjske snage'])
    X_test, y_test = X_test.values, y_test.values

    knn = KNN(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error = y_pred - y_test
    mse = np.average(error**2)
    rmse = np.sqrt(mse)
    
    return rmse


if __name__ == "__main__":
    _, train_dataset_path, test_dataset_path = sys.argv[0], sys.argv[1], sys.argv[2]
    train_data = pd.read_csv(train_dataset_path, delimiter='\t')
    test_data = pd.read_csv(test_dataset_path, delimiter='\t')

    train_data = drop_too_old_cars(train_data)
    train_data = train_data.reset_index(drop=True)
    train_data = drop_older_pricey_cars(train_data)
    train_data = train_data.reset_index(drop=True)
    train_data = drop_enormous_price_cars(train_data)
    train_data = train_data.reset_index(drop=True)
    train_data = train_data.drop_duplicates(inplace=False)
    train_data = train_data.reset_index(drop=True)
    rmse = knn_prediction(train_data, test_data, 5)
    print(rmse)

