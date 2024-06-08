import numpy as np
import pandas as pd

def random_data(N=1000, D=200, K=20, layers=10):
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

def load_bace():
    directory = 'data/bace_data/'

    # Load the training data
    data = pd.read_csv(directory + 'train.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_train = data['mol'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_train = np.array(X_train)
    y_train = data['Class'].values

    # Load the test data
    data = pd.read_csv(directory + 'test.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_test = data['mol'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_test = np.array(X_test)
    y_test = data['Class'].values

    # Load the validation data
    data = pd.read_csv(directory + 'valid.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_valid = data['mol'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_valid = np.array(X_valid)
    y_valid = data['Class'].values

    return X_train, y_train, X_test, y_test, X_valid, y_valid

def load_HIV():
    directory = 'data/HIV_data/'

    # Load the training data
    data = pd.read_csv(directory + 'train.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_train = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_train = np.array(X_train)
    y_train = data['HIV_active'].values

    # Load the test data
    data = pd.read_csv(directory + 'test.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_test = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_test = np.array(X_test)
    y_test = data['HIV_active'].values

    # Load the validation data
    data = pd.read_csv(directory + 'valid.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_valid = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_valid = np.array(X_valid)
    y_valid = data['HIV_active'].values

    return X_train, y_train, X_test, y_test, X_valid, y_valid

def load_sider():
    directory = 'data/sider_data/'

    # Load the training data
    data = pd.read_csv(directory + 'train.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_train = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_train = np.array(X_train)
    y_train = data['Eye disorders'].values


    # Load the test data
    data = pd.read_csv(directory + 'test.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_test = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_test = np.array(X_test)
    y_test = data['Eye disorders'].values

    # Load the validation data
    data = pd.read_csv(directory + 'valid.fgp2048.csv')
    # Split the DataFrame into features (X) and labels (y)
    X_valid = data['smiles'].apply(lambda x: np.array([int(char) for char in x])).tolist()
    X_valid = np.array(X_valid)
    y_valid = data['Eye disorders'].values

    return X_train, y_train, X_test, y_test, X_valid, y_valid