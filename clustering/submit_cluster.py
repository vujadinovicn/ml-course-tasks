import sys 
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import v_measure_score

def label_encode_data(train_data, test_data):
    encoder = LabelEncoder()
    
    train_data['region'] = encoder.fit_transform(train_data['region'])
    test_data['region'] = encoder.transform(test_data['region'])
    
    return train_data, test_data


def preprocess_data(train_data, test_data):
    train_data = train_data.drop(columns=['Surface Area'])
    test_data = test_data.drop(columns=[ 'Surface Area'])

    train_data['GDP per Capita'] = train_data['GDP per Capita'].fillna(train_data['GDP per Capita'].mean())
    train_data = train_data.dropna().reset_index(drop=True)

    return train_data, test_data

def get_x_y(train_data, test_data):
    X_train = train_data.drop(columns=['region']) 
    y_train = train_data['region']

    X_test = test_data.drop(columns=['region']) 
    y_test = test_data['region']

    return X_train, X_test, y_train, y_test


def get_score(X_train, y_train, X_test, y_test):
    gmm = GaussianMixture(n_components=4, random_state=31, covariance_type='tied', init_params='random_from_data')
    gmm.fit(X_train)
    y_pred = gmm.predict(X_test)
    
    return v_measure_score(y_test, y_pred)

if __name__ == '__main__':
    _, train_dataset_path, test_dataset_path = sys.argv[0], sys.argv[1], sys.argv[2]
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)
    
    train_data, test_data = label_encode_data(train_data, test_data)
    train_data, test_data = preprocess_data(train_data, test_data)
    X_train, X_test, y_train, y_test = get_x_y(train_data, test_data)

    score = get_score(X_train, y_train, X_test, y_test)
    print(score)


