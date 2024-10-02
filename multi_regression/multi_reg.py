import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scikit_reg import regression, elastic_net
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def regression(X_train, y_train, X_test, y_test, type):
    best_i = -1
    best_score = 1000000
    alpha = 0
    for i in range(100):
        alpha += 0.02
        if type == "Ridge":
            clf = Ridge(alpha=alpha)
        if type == "Lasso":
            clf = Lasso(alpha=alpha)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        # print(np.sqrt(mse), i)
        if np.sqrt(mse) < best_score:
            best_score = np.sqrt(mse)
            best_i = i
    print(f"RMSE {type} scikit learn for {best_i}:", best_score)

def elastic_net(X_train, y_train, X_test, y_test):
    best_i = -1
    best_j = -1
    best_score = 1000000
    alpha = 0
    for i in range(100):
        alpha += 0.02
        l1_ratio = 0
        for j in range(49):
            l1_ratio += 0.02
            clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            if np.sqrt(mse) < best_score:
                best_score = np.sqrt(mse)
                best_i = i
                best_j = j
    print(f"RMSE {type} scikit learn for {best_i} {best_j}:", best_score)


class LabelEncoder:
    def __init__(self):
        self.label_mapping = {}

    def fit(self, data):
        for col in data.columns:
            if data[col].dtype == 'object':
                unique_labels = data[col].unique()
                label_mapping = {label: i for i, label in enumerate(unique_labels)}
                self.label_mapping[col] = label_mapping

    def transform(self, data):
        transformed_data = data.copy()
        for col, mapping in self.label_mapping.items():
            transformed_data[col] = transformed_data[col].map(mapping)
        return transformed_data
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class OneHotEncoder1:
    def __init__(self):
        self.category_mapping = {}
        self.deleted_cats = {}

    def fit(self, data):
        self.categorical_columns =["Marka", "Menjac", "Karoserija", "Gorivo"]
        for col in self.categorical_columns:
            self.category_mapping[col] = data[col].unique()

        # print(self.category_mapping)

    def transform(self, data, train_set=False):
        # Preserve the original numerical columns
        original_numerical_data = data.select_dtypes(exclude=['object'])

        data_transformed = pd.DataFrame(index=data.index)
        for col in self.categorical_columns:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            all_categories = [f"{col}_{category}" for category in self.category_mapping[col]]

            if train_set:
                self.deleted_cats[col] = set(all_categories) - set(dummies.columns)
            # Create a new DataFrame for the missing columns initialized to zero
            missing_cols = {category: [0] * len(data) for category in all_categories if category not in dummies and category not in self.deleted_cats[col]}

            # print(missing_cols)

            missing_df = pd.DataFrame(missing_cols, index=data.index)
            # Combine the existing dummies and the missing columns DataFrame
            dummies = pd.concat([dummies, missing_df], axis=1)
            # Select and reorder the columns based on the training categories
            dummies = dummies[[col_new for col_new in all_categories if col_new not in self.deleted_cats[col]]]
            dummies = dummies.astype(int)
            data_transformed = pd.concat([data_transformed, dummies], axis=1)

        # Concatenate the numerical columns with the transformed categorical data
        complete_transformed_data = pd.concat([original_numerical_data, data_transformed], axis=1)

        return complete_transformed_data


    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
class BinaryEncoder:
    def __init__(self):
        pass

    def fit(self, data):
        self.data = data
        self.categorical_columns = data.select_dtypes(include=['object']).columns
        self.binary_columns = {}
        for column in self.categorical_columns:
            unique_categories = data[column].unique()
            no_of_unique = len(unique_categories)
            no_of_bits = len(bin(no_of_unique)[2:])
            self.binary_columns[column] = no_of_bits

    def transform(self, data):
        data_encoded = data.copy()
        for column, no_of_bits in self.binary_columns.items():
            category_dict = {val: idx + 1 for idx, val in enumerate(self.data[column].unique())} 
            data_encoded[column] = self.data[column].map(category_dict)
            binary_reprs = data_encoded[column].apply(lambda x: bin(x)[2:].zfill(no_of_bits))
            for i in range(no_of_bits):
                data_encoded[f'{column}_{i}'] = binary_reprs.apply(lambda x: int(x[i]))
            del data_encoded[column]
        return data_encoded
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
    
class CustomTargetEncoder:
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


# ne treba pozivati ove funkcije vec se encoderi pozivaju kao dole sto je napravljeno posle
def get_one_hot_encoded_data(data):
    ohe = OneHotEncoder()
    ohe.fit(data)
    data_encoded = ohe.transform()
    return data_encoded

def get_binary_encoded_data(data):
    be = BinaryEncoder()
    be.fit(data)
    data_encoded = be.transform()
    return data_encoded

def get_target_encoded_data(data):
    te = CustomTargetEncoder(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka'])
    te.fit(data, data['Cena'])
    data_encoded = te.transform()
    return data_encoded

def show_correlation_matrix(data):
    # data = data.drop(columns=['Cena'])
    correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap for Non-Categorical Columns")
    plt.show()

def plot_scatter_for_two_columns(data, column1_name, column2_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[column1_name], data[column2_name], alpha=0.5)
    plt.title(f'Scatter Plot of {column2_name} for {column1_name}')
    plt.xlabel(column1_name)
    plt.ylabel(column2_name)
    plt.grid(True)
    plt.show()


def plot_boxplots(data, column):
    plt.figure(figsize=(10, 6))
    data.boxplot(column=column)
    plt.title('Boxplot of Numerical Columns')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.show()

def drop_too_old_cars(data, year = 1970):
    indices = data[data['Godina proizvodnje'] <= year].index
    return data.drop(indices, inplace=False)

def drop_older_pricey_cars(data, year = 1990, price = 20000):
    indices = data[(data['Godina proizvodnje'] <= year) & (data['Cena'] >= price)].index
    return data.drop(indices, inplace=False)

def drop_enormous_price_cars(data, price = 100000):
    indices = data[data['Cena'] >= price].index
    return data.drop(indices, inplace=False)

def change_year_into_age_old(data):
    current = 2024
    data['Godina proizvodnje'] = data['Godina proizvodnje'].apply(lambda x: current - x)
    return data

def feature_z_normalization(data, columns):
    means = data[columns].mean().to_dict()
    stds = data[columns].std().to_dict()

    normalized_data = data.copy()
    for column in columns:
        normalized_data[column] = (data[column] - means[column]) / stds[column]
    
    stats_dict = {column: {'mean': means[column], 'std': stds[column]} for column in columns}
    
    return normalized_data, stats_dict


def feature_min_max_normalization(data, columns):
    mins = data[columns].min().to_dict()
    maxs = data[columns].max().to_dict()

    normalized_data = data.copy()
    for column in columns:
        normalized_data[column] = (data[column] - mins[column]) / (maxs[column]-mins[column])
    
    stats_dict = {column: {'min': mins[column], 'max': maxs[column]} for column in columns}
    
    return normalized_data, stats_dict

def min_max_normalize_test_set(data, coeffs):
    normalized_data = data.copy()
    # cols = data.select_dtypes(exclude='object')
    # print(cols)
    # ['Marka', 'Grad', 'Godina proizvodnje', 'Karoserija', 'Gorivo',
    #         'Kilometraza', 'Konjske snage', 'Menjac']
    # ['Godina proizvodnje',
            # 'Kilometraza', 'Konjske snage']
    for column in ['Godina proizvodnje', 'Kilometraza', 'Konjske snage']:
        if column == "Cena":
            continue
        normalized_data[column] = (data[column] - coeffs[column]['min']) / (coeffs[column]['max']-coeffs[column]['min'])
    return normalized_data


def plot_histogram_of_column(data, column_name):
    plt.hist(data[column_name], bins=20, color='blue', alpha=0.7)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of ' + column_name)
    plt.grid(True)
    plt.show()

# ovo su funkcije koje se koriste kad ucitas celokupni train file da bismo izbacili nepotrebne stvari
# drop_too_old_cars(data)
# drop_older_pricey_cars(data)
# drop_enormous_price_cars(data)
# change_year_into_age_old(data)
# data = pd.read_csv('changed_train.tsv', delimiter='\t')
# indices = data[data['Cena'] >= 100000].index


def predict(X, theta):
    return np.dot(X,theta.T)

def compute_cost(X, y, theta):
    m = len(X)
    y_pred = predict(X, theta)
    err_sq = (y_pred-y.T)**2
    # reg_term = 0.5 * np.abs(theta).sum()  # L1 regularization term
    reg_term = 0.5 * np.sum(theta ** 2)
    reg_term = 0
    return (err_sq.sum() + reg_term) / (m)

def gradient_descent(X, y, alpha, num_iters):
    m = len(y)
    error = []
    # treba promeniti ako koristimo OHE, onda je 259, ako je TargetEncoding onda je 10
    theta = np.zeros((10))

    for _ in range(num_iters):
        cost = compute_cost(X, y, theta)
        error.append(cost)
        y_pred = predict(X,theta)
        delta = (np.dot((y_pred-y.T),X))
        reg_term = (0.5 / m) * theta 
        reg_term = 0

        # reg_term = 0.5 * np.sign(theta)  # Derivative of L1 regularization term
        deltasum = ((np.sum(delta, axis=0) / m) + reg_term) * alpha
        theta -= deltasum

    return theta, error


class NonLinearEquation:
    def __init__(self, alpha=1.0, use_ridge = True):
        self.alpha = alpha
        self.use_ridge = use_ridge
    
    def fit(self, X, y):
        x0 = np.ones((len(X), 1))
        X = np.append(x0, X, axis=1)
        regularization_term = self.alpha * np.identity(X.shape[1]) if self.use_ridge else 0
        inv = np.linalg.inv(X.T @ X + regularization_term)
        # X = X.astype(np.float64)
        # y = y.astype(np.float64)
        # inv = np.linalg.inv(X.T @ X)
        multi = X.T @ y
        self.theta = inv @ multi
    
    def predict(self, X):
        x0 = np.ones((len(X), 1))
        X = np.append(x0, X, axis=1)
        # X = X.astype(np.float64)
        return np.dot(X, self.theta)
    
    def calc_rmse_error(self, y, y_pred):
        error = y_pred - y
        mse = np.average(error**2)
        return np.sqrt(mse)


class LassoRegression():
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
 
    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.iterations):
            self.update_weights()
        return self
 
    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = np.zeros(self.n)
        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) +
                         self.l1_penalty) / self.m
            else:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - Y_pred) -
                         self.l1_penalty) / self.m
 
        db = -2 * np.sum(self.Y - Y_pred) / self.m
 
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self
 
    def predict(self, X):
        return X.dot(self.W) + self.b
    

class KNNRegressor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = np.zeros(len(X_test))
        for i, x_test in enumerate(X_test):
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            y_pred[i] = np.mean(k_nearest_labels)
        return y_pred
    
# TODO: Kernel ridge regression


def prediction_with_onehot_encoding(train_data, test_data, k = 1):
    from sklearn.model_selection import KFold
    X = train_data
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    results = {"linreg": 0, "NLE": 0, "knnn": 0, "knn": 0, "kernel": 0}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        ohe = OneHotEncoder1()
        train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        print(train_data)
        y_train = train_data['Cena']

        y_test = test_data['Cena']
        train_data = train_data.drop(columns=['Zapremina motora', 'Grad', 'Cena'])
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.drop(columns=['Zapremina motora', 'Grad', 'Cena'])
        test_data = test_data.reset_index(drop=True)

        print(train_data)
        ohe.fit(train_data)
        train_data_encoded = ohe.transform(train_data, train_set=True)
        train_data_encoded.to_csv("oheeee.csv", index=False)
        # X_train = train_data_encoded.drop(columns=['Cena', 'Zapremina motora', 'Grad'])
        
        # CCCC = np.array(X_train.columns)

        # X_train, train_min_max_coeffs = feature_min_max_normalization(train_data_encoded, ['Godina proizvodnje',
        #     'Kilometraza', 'Konjske snage'])
        X_train = train_data_encoded
        
        print(X_train)
        # print(X_test)
        X_train = X_train.astype(np.float64)
        print(X_train)
        X_train = X_train.values

        minn = np.min(y_train)
        maxx = np.max(y_train)
        y_train = (y_train - minn) / (maxx - minn)

        test_data_encoded = ohe.transform(test_data)
        X_test = test_data_encoded
        # X_test = test_data_encoded.drop(columns=['Cena', 'Zapremina motora', 'Grad'])
        # print(X_test.columns)
        # X_test = min_max_normalize_test_set(test_data_encoded, train_min_max_coeffs)
        X_test = X_test.values
        X_test = X_test.astype(np.float64)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = (y_pred) * (maxx - minn) + minn
        mse = mean_squared_error(y_test, y_pred)
        print("RMSE regression scikit:", np.sqrt(mse))
        
        # from sklearn.preprocessing import OneHotEncoder
        # ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]
        # train_data = train_data.reset_index(drop=True)
        # test_data = test_data.reset_index(drop=True)
        # # print("tuuuuuuuuuu")
        # # print(train_data)
        # # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # ohe.fit(train_data[['Karoserija', 'Menjac', 'Gorivo', 'Marka']])
        # train_data_encoded = ohe.transform(train_data[['Karoserija', 'Menjac', 'Gorivo', 'Marka']])
        # train_data_encoded = pd.DataFrame(train_data_encoded, columns=ohe.get_feature_names_out(['Karoserija', 'Menjac', 'Gorivo', 'Marka']))
        # # print(len(encoded_df))
        # # print((data['Godina proizvodnje']))

        # train_data_encoded['Godina proizvodnje'] = train_data['Godina proizvodnje']
        # train_data_encoded['Zapremina motora'] = train_data['Zapremina motora']
        # train_data_encoded['Kilometraza'] = train_data['Kilometraza']
        # train_data_encoded['Konjske snage'] = train_data['Konjske snage']
        # train_data_encoded['Cena'] =  train_data['Cena']
        # # te = CustomTargetEncoder(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka'])
        # # te.fit(train_data, train_data['Cena'])

        # # train_data = te.transform(train_data)
        # X_train = train_data_encoded.drop(columns=['Cena', 'Zapremina motora'])
        # # X_train = X_train.values
        # y_train = train_data_encoded['Cena']
        # minn = np.min(y_train)
        # maxx = np.max(y_train)
        # y_train = (y_train - minn) / (maxx - minn)
        # # print(y_train)
        # # (data[column] - coeffs[column]['min']) / (coeffs[column]['max']-coeffs[column]['min'])

        # X_train, train_min_max_coeffs = feature_min_max_normalization(X_train, ['Godina proizvodnje',
        #     'Kilometraza', 'Konjske snage'])
        # print(X_train)
        # X_train = X_train.values
        # X_train = X_train.astype(np.float64)

        # test_data_encoded = ohe.transform(test_data[['Karoserija', 'Menjac', 'Gorivo', 'Marka']])
        # test_data_encoded = pd.DataFrame(test_data_encoded, columns=ohe.get_feature_names_out(['Karoserija', 'Menjac', 'Gorivo', 'Marka']))
        # # print(len(encoded_df))
        # # print((data['Godina proizvodnje']))

        # test_data_encoded['Godina proizvodnje'] = test_data['Godina proizvodnje']
        # test_data_encoded['Zapremina motora'] = test_data['Zapremina motora']
        # test_data_encoded['Kilometraza'] = test_data['Kilometraza']
        # test_data_encoded['Konjske snage'] = test_data['Konjske snage']
        # test_data_encoded['Cena'] =  test_data['Cena']
        # # te = CustomTargetEncoder(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka'])
        # # te.fit(train_data, train_data['Cena'])

        # # train_data = te.transform(train_data)
        # X_test = test_data_encoded.drop(columns=['Cena', 'Zapremina motora'])
        # X_test = min_max_normalize_test_set(X_test, train_min_max_coeffs)
        # print(X_test)
        # X_test = X_test.values
        # X_test = X_test.astype(np.float64)

        # # X_train = X_train.values
        # y_test = test_data_encoded['Cena']
        
        # # y_test = (y_test) * (maxx - minn) + minn

        # print(X_train.shape)
        # print(X_test.shape)

        # model = LinearRegression()
        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # # y_pred = (y_pred) * (maxx - minn) + minn
        # mse = mean_squared_error(y_test, y_pred)
        # print("RMSE regression scikit:", np.sqrt(mse))
        # results['linreg'] += np.sqrt(mse)
        # # # nas non linear equation
        model = NonLinearEquation(alpha=0.6, use_ridge=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_non_eq = model.calc_rmse_error(y_test, y_pred)
        print("RMSE non lin nas:", rmse_non_eq)

        # # gradient descent ridge L2 nas
        # # iteration = 1000
        # # alpha = 0.001
        # # X0 = np.ones((len(X_train),1))
        # # X = np.append(X0,X_train,axis = 1)
        # # theta, error = gradient_descent(X,y_train,alpha,iteration)
        # # X0 = np.ones((len(X_test),1))
        # # ypred = predict(np.append(X0,X_test,axis = 1),theta)
        # # print(np.sqrt(mean_squared_error(y_test,ypred)))

        # # gradient descent ridge L2 nas
        regression(X_train, y_train, X_test, y_test, type="Lasso")
        regression(X_train, y_train, X_test, y_test, type="Ridge")
        # elastic_net(X_train, y_train, X_test, y_test)
        
        # # nas KNN
        # knn = KNNRegressor(k=k)
        # knn.fit(X_train, y_train)
        # y_pred = knn.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        # print(f"RMSE KNN NAS:", rmse)

        # # scikitov KNN
        # from sklearn.neighbors import KNeighborsRegressor
        # knn_regressor = KNeighborsRegressor(n_neighbors=k)
        # knn_regressor.fit(X_train, y_train)
        # y_pred = knn_regressor.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        # print(f"RMSE KNN scikit:", rmse)

        # # scikitov kernel ridge
        # from sklearn.kernel_ridge import KernelRidge
        # kernel_reg = KernelRidge(kernel='rbf') 
        # kernel_reg.fit(X_train, y_train)
        # y_pred = kernel_reg.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        # print(f"RMSE Kernel ridge scikit:", rmse)
        break

    return results

def test_custom_one_hot_encoding(train, test):
    ohe = OneHotEncoder1()

    data_train = pd.read_csv(train, delimiter='\t')
    # data_train = cleanup_data(data_train)

    data_test = pd.read_csv(test, delimiter='\t')

    print(data_train.shape)

    data_train.drop(columns=["Grad", "Zapremina motora"], inplace=True)
    data_train = data_train.reset_index(drop=True)
    data_test.drop(columns=["Grad", "Zapremina motora"], inplace=True)

    print(data_train.shape)

    ohe.fit(data_train)
    encoded_df_train = ohe.transform(data_train, True)
    encoded_df_test = ohe.transform(data_test)

    encoded_df_train.to_csv(f'ohe_train.csv', index=False)
    encoded_df_test.to_csv(f'ohe_test.csv', index=False)

    X_train = encoded_df_test.drop(columns=["Cena"])
    

    model = LinearRegression()
        # y_train = np.log(y_train)
    model.fit(X_train, y_train)
    # X_test = min_max_normalize_test_set(X_test, train_min_max_coeffs).values
    y_pred = model.predict(X_test)
    # y_pred = np.exp(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # print("RMSE regression scikit:", np.sqrt(mse))
    results['linreg'] += np.sqrt(mse)

    print(f"{encoded_df_train.shape}   {encoded_df_test.shape}")
        

def prediction_with_target_encoding(train_data, test_data, k = 1, random_state=42):
    from sklearn.model_selection import KFold
    # data = 
    # X = train_data.drop(columns=['Cena'])
    # y = train_data['Cena']
    X = train_data

    # # Initialize k-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)

    # # Perform k-fold cross-validation
    results = {"linreg": 0, "NLE": 0, "knnn": 0, "knn": 0, "kernel": 0}
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        # print(train_idx)
        # print(test_idx)
        train_data, test_data = X.iloc[train_idx], X.iloc[test_idx]
        # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        te = CustomTargetEncoder(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka'])
        te.fit(train_data, train_data['Cena'])

        train_data = te.transform(train_data)
        X_train = train_data.drop(columns=['Cena', 'Zapremina motora', 'Grad'])
        y_train = train_data['Cena']

        test_data = te.transform(test_data)
        X_test = test_data.drop(columns=['Cena', 'Zapremina motora', 'Grad'])
        y_test = test_data['Cena']

        # print(X_test)
        # te.fit(train_data, train_data['Cena'])

        # train_data = te.transform(train_data)
        # X_train = train_data.drop(columns=['Cena', 'Zapremina motora'])
        # y_train = train_data['Cena']

        # test_data = te.transform(test_data)
        # X_test = test_data.drop(columns=['Cena', 'Zapremina motora'])
        # y_test = test_data['Cena']

        X_train, train_min_max_coeffs = feature_min_max_normalization(X_train, ['Marka', 'Godina proizvodnje', 'Karoserija', 'Gorivo',
            'Kilometraza', 'Konjske snage', 'Menjac'])
        X_train['Godina proizvodnje'] = np.exp(X_train['Godina proizvodnje']+0.0000001)
        # X_train['Kilometraza'] = np.exp(-X_train['Kilometraza']+0.0000001)
        # X_train['Konjske snage'] = np.exp(X_train['Konjske snage']+0.0000001)
        X_train, y_train = X_train.values, y_train.values

        X_test = min_max_normalize_test_set(X_test, train_min_max_coeffs)
        X_test['Godina proizvodnje'] = np.exp(X_test['Godina proizvodnje']+0.0000001)
        # X_test['Kilometraza'] = np.exp(-X_test['Kilometraza']+0.0000001)
        # X_test['Konjske snage'] = np.exp(X_test['Konjske snage']+0.0000001)
        X_test = X_test.values
        model = LinearRegression()
        # y_train = np.log(y_train)
        model.fit(X_train, y_train)
        # X_test = min_max_normalize_test_set(X_test, train_min_max_coeffs).values
        y_pred = model.predict(X_test)
        # y_pred = np.exp(y_pred)
        mse = mean_squared_error(y_test, y_pred)
        # print("RMSE regression scikit:", np.sqrt(mse))
        results['linreg'] += np.sqrt(mse)


        # for d in range(2, 4):
        #     X_train = np.concatenate((X_train, X_train**d), axis=1)
        #     X_test= np.concatenate((X_test, X_test**d), axis=1)
        model = NonLinearEquation(alpha=0.6, use_ridge=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse_non_eq = model.calc_rmse_error(y_test, y_pred)
        # print("RMSE non lin nas:", rmse_non_eq)
        results['NLE'] += rmse_non_eq

        # # gradient descent ridge L2 nas
        # iteration = 1000
        # alpha = 0.001
        # X0 = np.ones((len(X_train),1))
        # X = np.append(X0,X_train,axis = 1)
        # theta, error = gradient_descent(X,y_train,alpha,iteration)
        # X0 = np.ones((len(X_test),1))
        # ypred = predict(np.append(X0,X_test,axis = 1),theta)
        # print("RMSE grad desc R2 nas:", np.sqrt(mean_squared_error(y_test,ypred)))

        # ovo su sve iz scikita
        # regression(X_train, y_train, X_test, y_test, type="Lasso")
        # regression(X_train, y_train, X_test, y_test, type="Ridge")
        # elastic_net(X_train, y_train, X_test, y_test)
        
        # nas KNN
        # knn = KNNRegressor(k=k)
        # knn.fit(X_train, y_train)
        # y_pred = knn.predict(X_test)
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        # # print(f"RMSE KNN NAS:", rmse)
        # results['knnn'] += rmse

        # # scikitov KNN
        from sklearn.neighbors import KNeighborsRegressor
        knn_regressor = KNeighborsRegressor(n_neighbors=k)
        knn_regressor.fit(X_train, y_train)
        y_pred = knn_regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        # print(f"RMSE KNN scikit:", rmse)
        results['knnn'] += rmse

        # scikitov kernel ridge
        from sklearn.kernel_ridge import KernelRidge
        kernel_reg = KernelRidge(kernel='rbf') 
        kernel_reg.fit(X_train, y_train)
        y_pred = kernel_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        results['kernel'] += rmse
        # print(f"RMSE Kernel ridge scikit:", rmse)


        # print("-------")
        # break

    # for a, b in results.items():
        # print(f'For {a} we got result: {b/10}')
    return results

# ovu funkciju ne koristimo vec samo kod iz nje kad zelimo.
def custom_cross_validation(data, num_folds=10):
    np.random.seed(42)  # Set seed for reproducibility
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    fold_size = len(data) // num_folds
    for i in range(num_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(data)

        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        yield data.iloc[train_indices], data.iloc[test_indices]


def find_elbow_for_knn():
    for i in range(1, 30):
        results = prediction_with_target_encoding(data, test_data, k = i)
        stack.append(results)
        if minn > results['knn']/10:
            minn = results['knn']/10
            k = i
    print(minn, k)
    for i in range(len(stack)):
        print(i+1)
        print(stack[i]['knn']/10)

if __name__ == "__main__":
    # train_data = pd.read_csv('train_data.tsv', delimiter='\t')
    # test_data = pd.read_csv('test_data.tsv', delimiter='\t')
    
    # prediction_with_target_encoding(train_data, test_data, k = 5)
    data = pd.read_csv('train.tsv', delimiter='\t')
    data =drop_too_old_cars(data)
    data = data.reset_index(drop=True)
    data =drop_older_pricey_cars(data)
    data = data.reset_index(drop=True)
    data =drop_enormous_price_cars(data)
    data = data.reset_index(drop=True)
    data = data.drop_duplicates(inplace=False)
    data = data.reset_index(drop=True)

    # new_df = data[[
    #     'Cena',	'Godina proizvodnje', 'Kilometraza', 'Konjske snage'
    # ]]

    # # new_df['Cena'] = np.log(new_df['Cena'])
    # new_df['Godina proizvodnje'] = (new_df['Godina proizvodnje']-np.min(new_df['Godina proizvodnje'])) / (np.max(new_df['Godina proizvodnje']) - np.min(new_df['Godina proizvodnje']))

    # sns.pairplot(new_df)

    # plt.show()
    # sns.heatmap(new_df.corr(), annot = True, yticklabels = True, linewidths = 0.2)
    # plot_scatter_for_two_columns(data, 'Cena', 'Konjske snage')
    # data =change_year_into_age_old(data)
    # print(type(data.get('Godina proizvodnje')))
    # print(data)
    # print(data)
    results = prediction_with_onehot_encoding(data, data, k=0)
    print(results)

    test_custom_one_hot_encoding('train_data.tsv', 'test_data.tsv')

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder(handle_unknown = 'ignore')
    # X_num = data.select_dtypes(exclude='object')
    # X_cat = data.select_dtypes(include='object')
    # encoder = OneHotEncoder()
    # X_encoded = encoder.fit_transform(X_cat)
    # categorical_columns = [f'{col}_{cat}' for i, col in enumerate(X_cat.columns) for cat in encoder.categories_[i]]
    # one_hot_features = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka']))
    # data = X_num.join(one_hot_features)

    # X_encoded
    # data = pd.read_csv('train_data.tsv', delimiter='\t')
    # data =drop_too_old_cars(data)
    # data = data.reset_index(drop=True)
    # data =drop_older_pricey_cars(data)
    # data = data.reset_index(drop=True)
    # data =drop_enormous_price_cars(data)
    # data = data.reset_index(drop=True)
    # ohe.fit(data[['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka']])
    # encoded_df = ohe.transform(data[['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka']])
    # encoded_df = pd.DataFrame(encoded_df.toarray(), columns=ohe.get_feature_names_out(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka']))
    # # print(len(encoded_df))
    # # print((data['Godina proizvodnje']))

    # encoded_df.to_csv('ohe1.csv', index=False)
    # encoded_df['Godina proizvodnje'] = data['Godina proizvodnje']
    # encoded_df['Zapremina motora'] = data['Zapremina motora']
    # encoded_df['Kilometraza'] = data['Kilometraza']
    # encoded_df['Konjske snage'] = data['Konjske snage']
    # encoded_df['Cena'] =  data['Cena']
    # # not_encoded_df = data.select_dtypes(exclude='object')
    # # print(not_encoded_df)
    # # encoded_df.to_csv('pereca.csv', index=False)
    # print(encoded_df.shape)

    # data = pd.read_csv('test_data.tsv', delimiter='\t')
    # encoded_df = ohe.transform(data[['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka']])
    # encoded_df = pd.DataFrame(encoded_df.toarray(), columns=ohe.get_feature_names_out(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka']))
    # # print(len(encoded_df))
    # # print((data['Godina proizvodnje']))

    # encoded_df.to_csv('ohe1.csv', index=False)
    # encoded_df['Godina proizvodnje'] = data['Godina proizvodnje']
    # encoded_df['Zapremina motora'] = data['Zapremina motora']
    # encoded_df['Kilometraza'] = data['Kilometraza']
    # encoded_df['Konjske snage'] = data['Konjske snage']
    # encoded_df['Cena'] =  data['Cena']
    # # not_encoded_df = data.select_dtypes(exclude='object')
    # # print(not_encoded_df)
    # # encoded_df.to_csv('pereca.csv', index=False)
    # print(encoded_df.shape)
    # data.to_csv('ohe1.csv', index=False)
    # encoded_df = encoded_df.join(not_encoded_df)
    # encoded_df.to_csv('ohe1.csv', index=False)
    # cols = data.select_dtypes(exclude='object').columns
    # for col in cols:
    #     for i in range(len(data[col])):
    #         if not data[col][i] == encoded_df[col][i]:
    #             print("err")
    #             break
    # print(encoded_df)
    # encoded_df.to_csv('ohe1.csv', index=False)

    # te = CustomTargetEncoder(['Grad', 'Karoserija', 'Menjac', 'Gorivo', 'Marka'])
    # te.fit(data, data['Cena'])

    # train_data = te.transform(data)
    # train_data = train_data.drop(columns=['Zapremina motora'])
    # train_data['Cena'] = np.log(train_data['Cena'])
    # # plot_scatter_for_two_columns(train_data, 'Cena', 'Grad')
    # # show_correlation_matrix(train_data)
    # sns.pairplot(train_data)
    # plt.show()
    # minn = 1000000
    # k = 0
    # stack = []

    # print(data)

    # # prediction_with_onehot_encoding(train_data, test_data, k = 5)
    # prediction_with_target_encoding(train_data, test_data, k = 5)
    # import json

#     duplicates = data.duplicated()

# # Count the number of duplicate rows
#     num_duplicates = duplicates.sum()

#     # Print the number of duplicate rows
#     print("Number of duplicate rows:", num_duplicates)

#     # Print the duplicate rows themselves
#     if num_duplicates > 0:
#         print("Duplicate rows:")
#         print(data[duplicates])
#     else:
#         print("No duplicate rows found.")
    # k = 0.0
    # # # # for j in range(50):
    # minn = 100000
    # min_k = -1
    # for i in range(1, 30):
    #     k += 0.05
    #     results = prediction_with_target_encoding(data, data, k = i, random_state=39)
    #     print(results, k)
    #     if minn > results['knnn']:
    #         minn = results['knnn']
    #         min_k = k
    #     print(minn/10, min_k, i)
    # print(results)
    