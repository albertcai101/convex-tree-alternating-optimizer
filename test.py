from optimizer import BlitzOptimizer
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def generate_data(N=1000, D=200, K=20, layers=10):
    X = np.random.randn(N, D)

    # Initialize empty lists for weights and biases
    np.random.seed(42)

    weights = []
    biases = []

    # Loop to create 5 layers
    for layer in range(layers):        
        # Generate random weights and biases for each layer
        W_real = np.random.randn(D, K)
        b_real = np.random.randn(K)
        
        # Append weights and biases to the respective lists
        weights.append(W_real)
        biases.append(b_real)

    # Compute the output of each layer and select the maximum
    outputs = [X.dot(W) + b for W, b in zip(weights, biases)]
    y = np.argmax(np.maximum.reduce(outputs), axis=1)

    return X, y

def load_bace(path):
    data = pd.read_csv(path)
    # Split the DataFrame into features (X) and labels (y)
    X = data['mol'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X = np.array(X)
    y = data['Class'].values

    return X, y

def load_HIV(path):
    data = pd.read_csv(path)
    # Split the DataFrame into features (X) and labels (y)
    X = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X = np.array(X)
    y = data['HIV_active'].values

    return X, y

def load_sider(path):
    data = pd.read_csv(path)
    # Split the DataFrame into features (X) and labels (y)
    X = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()

    X = np.array(X)
    
    y = data['Eye disorders'].values

    return X, y

if __name__ == "__main__":
    # X, y = load_bace('bace_data/train.fgp2048.csv')
    # X_test, y_test = load_bace('bace_data/test.fgp2048.csv')
    # X_valid, y_valid = load_bace('bace_data/valid.fgp2048.csv')

    X, y = load_sider('sider_data/train.fgp2048.csv')
    X_test, y_test = load_sider('sider_data/test.fgp2048.csv')
    X_valid, y_valid = load_sider('sider_data/valid.fgp2048.csv')

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # try with xgboost

    model = xgb.XGBClassifier()
    model.fit(X, y)

    y_pred = model.predict(X_test)
    print(f"XGBoost Test Accuracy: {np.mean(y_pred == y_test)}")
    print(f"XGBoost Test AUC: {roc_auc_score(y_test,y_pred)}")
    y_pred = model.predict(X_valid)
    print(f"XGBoost Validation Accuracy: {np.mean(y_pred == y_valid)}")
    print(f"XGBoost Validation AUC: {roc_auc_score(y_valid, y_pred)}")

    D = X.shape[1]

    ct = BlitzOptimizer(DEPTH=10, D=X.shape[1], K=2, MAX_ITERS=3)
    ct.fit(X, y)
    print(f"Train Accuracy: {ct.accuracy(X, y)}")
    # ct.plot_training(X, y)

    y_pred = ct.batch_eval(X_test)
    print(y_pred[:10])
    print(y_test[:10])
    print(f"TAO Test Accuracy: {ct.accuracy(X_test, y_test)}")
    print(f"TAO Test AUC: {roc_auc_score(y_test,y_pred)}")

    y_pred = ct.batch_eval(X_valid)
    print(f"TAO Validation Accuracy: {ct.accuracy(X_valid, y_valid)}")
    print(f"TAO Validation AUC: {roc_auc_score(y_valid,y_pred)}")