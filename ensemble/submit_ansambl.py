import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
import sys

def one_hot_encode_data(train_data, test_data):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    encoded_platforms = encoder.fit_transform(train_data[['Gaming_Platform']])
    encoded_df = pd.DataFrame(encoded_platforms, columns=encoder.get_feature_names_out(['Gaming_Platform']))
    train_data = pd.concat([train_data.drop('Gaming_Platform', axis=1), encoded_df], axis=1)

    encoded_platforms = encoder.transform(test_data[['Gaming_Platform']])
    encoded_df = pd.DataFrame(encoded_platforms, columns=encoder.get_feature_names_out(['Gaming_Platform']))
    test_data = pd.concat([test_data.drop('Gaming_Platform', axis=1), encoded_df], axis=1)

    return train_data, test_data

def get_x_y(train_data, test_data):
    X_train = train_data.drop('Genre', axis=1) 
    y_train = train_data['Genre']

    X_test = test_data.drop('Genre', axis=1) 
    y_test = test_data['Genre']

    return X_train, X_test, y_train, y_test

def fix_missing_values(X_train):
    imputer = SimpleImputer(strategy='most_frequent')
    X_train = imputer.fit_transform(X_train)
    return X_train

def get_score(X_train, y_train, X_test, y_test):
    gbc = GradientBoostingClassifier(n_estimators=105, learning_rate=0.7, max_depth=3, random_state=0)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')

    return score

if __name__ == "__main__":
    _, train_dataset_path, test_dataset_path = sys.argv[0], sys.argv[1], sys.argv[2]
    train_data = pd.read_csv(train_dataset_path)
    test_data = pd.read_csv(test_dataset_path)

    train_data, test_data = one_hot_encode_data(train_data, test_data)
    X_train, X_test, y_train, y_test = get_x_y(train_data, test_data)
    X_train = fix_missing_values(X_train)
    X_test = X_test.values

    score = get_score(X_train, y_train, X_test, y_test)
    print(score)