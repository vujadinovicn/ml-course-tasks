import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
import sys


def label_encode_data(train_data, test_data):
    encoder = LabelEncoder()
    
    train_data['Gaming_Platform'] = encoder.fit_transform(train_data['Gaming_Platform'])
    test_data['Gaming_Platform'] = encoder.transform(test_data['Gaming_Platform'])
    
    return train_data, test_data

def get_x_y(train_data, test_data):
    X_train = train_data.drop('Genre', axis=1) 
    y_train = train_data['Genre']

    X_test = test_data.drop('Genre', axis=1) 
    y_test = test_data['Genre']

    return X_train, X_test, y_train, y_test


def fix_missing_values(X_train):
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    return X_train

def get_score(X_train, y_train, X_test, y_test):
    rf1 = RandomForestClassifier(max_depth = 15, min_samples_leaf=3,  n_estimators=50, random_state=0)
    rf2 = RandomForestClassifier(max_depth = 15, min_samples_leaf=3, n_estimators=50, random_state=20)
    rf3 = RandomForestClassifier(max_depth = 15, min_samples_leaf=3, n_estimators=50, random_state=40)
    
    voting = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)], 
        voting='soft'
    )
    voting.fit(X_train, y_train)

    y_pred = voting.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    return score

if __name__ == "__main__":
    _, train_dataset_path, test_dataset_path = sys.argv[0], sys.argv[1], sys.argv[2]
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)
    
    train_data, test_data = label_encode_data(train_data, test_data)
    X_train, X_test, y_train, y_test = get_x_y(train_data, test_data)
    X_train = fix_missing_values(X_train)
    X_test = X_test.values

    score = get_score(X_train, y_train, X_test, y_test)
    print(score)