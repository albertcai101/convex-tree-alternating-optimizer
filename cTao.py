import numpy as np
import cvxpy as cp
from dask import delayed, compute, visualize
from cTaoTree import CTaoTree

# D: dimension of data points
# K: number of classes

class CTao():
    def __init__(self, depth, D, K):
        self.depth = depth
        self.D = D
        self.K = K
        self.tree = CTaoTree(depth, D, K)
        self.memory = []
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def accuracy(self, X, y):
        pass

    def plot_training(self):
        pass

    def print_tree(self):
        pass


if __name__ == "__main__":
    ct = CTao(depth=10, D=2, K=5)

    # generate synthetic data for classification
    N = 1000
    D = 2
    K = 5

    X = np.random.randn(N, D)

    # Initialize empty lists for weights and biases
    np.random.seed(42)

    weights = []
    biases = []

    layers = 5

    # Loop to create 5 layers
    for _ in range(layers):
        # Generate random weights and biases for each layer
        W_real = np.random.randn(D, K)
        b_real = np.random.randn(K)
        
        # Append weights and biases to the respective lists
        weights.append(W_real)
        biases.append(b_real)

    # Compute the output of each layer and select the maximum
    outputs = [X.dot(W) + b for W, b in zip(weights, biases)]
    y = np.argmax(np.maximum.reduce(outputs), axis=1)

    ct.fit(X, y)
    acc = ct.accuracy(X, y)
    print(f"Accuracy: {acc}")
    ct.plot_training()