import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def load_datasets(train_dataset_path, test_dataset_path):
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    X_train, Y_train = train_data['X'].values.reshape(-1,1), train_data['Y'].values
    X_test, Y_test = test_data['X'].values.reshape(-1,1), test_data['Y'].values

    return X_train, Y_train, X_test, Y_test

def remove_outliers(x_train, y_train, outliers_x):
    valid_indices= np.where(~np.isin(x_train.flatten(), outliers_x))
    return x_train[valid_indices], y_train[valid_indices]

def compute_cost(X, y, theta):
    m = len(y)
    y_pred = np.dot(X, theta)
    err_sq = (y_pred-y)**2
    return err_sq.sum()/(2*m)

def gradient_descent(X, y, theta, learning_rate, num_iters):
    m = len(y)
    error = []

    x0 = np.ones((len(X),1))
    X = np.append(x0, X, axis=1)

    for _ in range(num_iters):
        y_pred = np.dot(X, theta)
        cost = compute_cost(X, y, theta)
        error.append(cost)

        delta = learning_rate * np.dot(X.T, (y_pred - y))/m
        theta -= delta
    
    return theta, error

def fit(X, Y, learning_rate=0.1, num_iters=1000):
    theta = np.zeros((2))
    theta, error = gradient_descent(X, Y, theta, learning_rate, num_iters)

    return theta, error

def predict(X, theta):
    x0 = np.ones((len(X),1))
    X = np.append(x0, X, axis=1)
    
    return np.dot(X, theta)

def calc_rmse_error(y, ypred):
    error = ypred - y
    mse = np.average(error**2)
    return np.sqrt(mse)

def normalize_min_max(x, minn, maxx):
    return (x-minn)/(maxx-minn)

def invert_normalize_min_max(x, minn, maxx):
    return x*(maxx-minn)+minn

if __name__ == "__main__":
    _, train_dataset_path, test_dataset_path = sys.argv[0], sys.argv[1], sys.argv[2]
    
    X_train, Y_train, X_test, Y_test = load_datasets(train_dataset_path, test_dataset_path)
    train_data = pd.read_csv(train_dataset_path)
    X_train, X_test, Y_train, Y_test = train_test_split(train_data['X'].values.reshape(-1, 1), train_data['Y'].values, test_size=0.2, random_state=10)
    
    
    outliers_x = [[9.75521505002886],
                [2.532916025397821],
                [7.038885835403663],
                [7.142412995491114],
                [9.194826137446736],
                [3.1194499547960186],
                [5.210366062041293],
                [7.29990562424058],
                [0.2010754618749355]]
    
    X_train_filtered, Y_train_filtered = remove_outliers(X_train, Y_train, outliers_x)

    min_x, max_x, min_y, max_y = np.min(X_train_filtered), np.max(X_train_filtered), np.min(Y_train_filtered), np.max(Y_train_filtered)

    theta, error = fit(normalize_min_max(X_train_filtered, min_x, max_x)**2, normalize_min_max(Y_train_filtered, min_y, max_y))
    Y_pred = predict(normalize_min_max(X_test, min_x, max_x)**2, theta)
    rmse_non_eq = calc_rmse_error(Y_test, invert_normalize_min_max(Y_pred, min_y, max_y))

    # theta, error = fit(X_train_filtered, Y_train_filtered)
    # Y_pred = predict(X_test, theta)
    # rmse_non_eq = calc_rmse_error(Y_test, Y_pred)
    
    print(rmse_non_eq)

    plt.plot(error)
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


