# imports
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process
from multiprocessing import Pool # want to move off this
import tracemalloc
import time

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# custom imports
import node_ops as nops
import tree_ops as tops
import train_ops as trops
from tree import CTaoTree

# For future reference
# D: dimension of data points
# K: number of classes

class BlitzOptimizer():
    def __init__(self, DEPTH, D, K, MAX_ITERS=2, shared_memory=True, verbose=False):
        self.depth = DEPTH
        self.d = D
        self.k = K
        self.iters = MAX_ITERS

        self.tree = CTaoTree(self.depth, self.d, self.k)
        self.memory = []

        self.shared_memory = shared_memory
        self.verbose = verbose

    ''' Mutable Functinon that changes self.tree'''
    def fit(self, X, y):
        self.memory = []
        self.memory.append((self.tree.copy(), self.accuracy(X, y)))

        # N = X.shape[0]
        # shuffle = np.random.permutation(N)
        # breakpoints = np.linspace(0, N, self.iters).astype(int)
        # breakpoints = breakpoints[1:]

        # # append N-1 to the end
        # # breakpoints = np.append(breakpoints, N)
        # # TODO: this does not give desired results, need to append to X_batches

        # print(breakpoints)
        # # [ 605 1210]
        # X_batches = np.split(X[shuffle], breakpoints)
        # y_batches = np.split(y[shuffle], breakpoints)

        # X_batches = np.append(X_batches, X)
        # y_batches = np.append(y_batches, y)

        for i in range(self.iters):
            if self.verbose:
                print(f"----Training iteration {i+1}----")
            if self.verbose:
                print(f"Training tree with {X.shape[0]} data points")

            start_time = time.time()

            if self.shared_memory:
                self.tree = trops.train_tree_shared_memory(X, y, self.tree, verbose=self.verbose)
            else:
                self.tree = trops.train_tree(X, y, self.tree, verbose=self.verbose)

            end_time = time.time()

            if self.verbose:
                print(f"Accuracy: {self.accuracy(X, y)}")
                print(f"ITERATION Time taken: {end_time - start_time} seconds")
            self.memory.append((self.tree, self.accuracy(X, y)))

    def predict(self, X):
        return tops.batch_eval(X, self.tree)

    def accuracy(self, X, y):
        return tops.accuracy(X, y, self.tree)

    def plot_training(self, X, y):
        # use memory to plot
        total_plots = self.iters + 2
        colors = ['r', 'g', 'b', 'y', 'm']
        cols = min(10, total_plots)
        rows = (total_plots + 9) // 10

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axs = axs.flatten()

        axs[0].scatter(X[:, 0], X[:, 1], c=[colors[y[n]] for n in range(len(X))])
        axs[0].set_title("Ground Truth")

        for i, (tree, acc) in enumerate(self.memory):
            y_pred = tops.batch_eval(X, tree)
            axs[i+1].scatter(X[:, 0], X[:, 1], c=[colors[y_pred[n]] for n in range(len(X))])
            axs[i+1].set_title(f"Iteration {i}: Accuracy {acc:.2f}")

        for ax in axs[total_plots:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ct = BlitzOptimizer(DEPTH=10, D=2, K=5, shared_memory=True, verbose=True)

    # generate synthetic data for classification
    N = 2000
    D = 2
    K = 5

    X = np.random.randn(N, D)

    # Initialize empty lists for weights and biases
    np.random.seed(42)

    weights = []
    biases = []

    layers = 10

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
    print('plotting...')
    ct.plot_training(X, y)