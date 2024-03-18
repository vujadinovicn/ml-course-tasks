import sys, os, glob
import pandas as pd
import numpy as np

def predict_theta(x, y):
    x0 = np.ones((len(x),1))
    x =  np.append(x0, x, axis=1)
    inv = np.linalg.inv(x.T@x)
    multi = x.T@y
    theta = inv@multi
    return theta

def predict_y(X,theta):
    x0 = np.ones((len(X),1))
    X = np.append(x0, X, axis=1)
    return np.dot(X,theta)

def calc_rmse_error(y,ypred):
    error = ypred - y
    mse = np.average(error**2)
    return np.sqrt(mse)

if __name__ == "__main__":
    _, train_dataset_path, test_dataset_path = sys.argv[0], sys.argv[1], sys.argv[2]
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    X_data = train_data['X']
    Y_data = train_data['Y']

    outliers_x = [[9.75521505002886],
                [2.532916025397821],
                [7.038885835403663],
                [7.142412995491114],
                [9.194826137446736],
                [3.1194499547960186],
                [5.210366062041293],
                [7.29990562424058],
                [0.2010754618749355]]

    X_train, Y_train = train_data['X'].values.reshape(-1,1), train_data['Y'].values
    X_test, Y_test = test_data['X'].values.reshape(-1,1), test_data['Y'].values
    valid_indices= np.where(~np.isin(X_train.flatten(), outliers_x))

    X_train_filtered = X_train[valid_indices]
    Y_train_filtered = Y_train[valid_indices]

    ### KOD ZA NON EQ 
    theta = predict_theta(X_train_filtered**2, Y_train_filtered)
    print(theta)
    Y_pred = predict_y(X_test**2, theta)
    rmse_non_eq = calc_rmse_error(Y_test, Y_pred)
    print(rmse_non_eq)



