# imports
from multiprocessing import Pool
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt

# custom imports
import node_ops as nops
import tree_ops as tops
from tree import CTaoTree

# For future reference
# D: dimension of data points
# K: number of classes

def find_training_data(X, y, root, node):
    S = []
    N = X.shape[0]

    # subsamble 1000 data points
    if N > 1000:
        subsample = np.random.choice(N, 1000, replace=False)
        X = X[subsample]
        y = y[subsample]
        N = X.shape[0]

    for n in range(N):
        x = X[n]
        if nops.reach_node(x, root, node):
            S.append(n)
    
    if len(S) == 0:
        return (node.is_leaf, None)

    if node.is_leaf:
        return (node.is_leaf, y[S])
    else:
        C = []
        y_bar = []
        for n in S:
            x = X[n]
            y_n = y[n]
            left_label =  nops.eval_from(x, node.left)[0]
            right_label =  nops.eval_from(x, node.right)[0]

            if left_label == y_n and right_label == y_n:
                continue

            if left_label != y_n and right_label != y_n:
                continue

            C.append(n)
            y_bar.append(1 if left_label == y_n else -1)

        return (node.is_leaf, (X[C], y_bar))

class BlitzOptimizer():
    def __init__(self, DEPTH, D, K, MAX_ITERS=2, verbose=False):
        self.depth = DEPTH
        self.d = D
        self.k = K
        self.iters = MAX_ITERS

        self.tree = CTaoTree(self.depth, self.d, self.k)
        self.memory = []

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
            print(f"----Training iteration {i+1}----")
            start_time = time.time()
            print(f"Training tree with {X.shape[0]} data points")
            self.__train_tree(X, y)
            end_time = time.time()
            print(f"Accuracy: {self.accuracy(X, y)}")
            print(f"Time taken: {end_time - start_time} seconds")
            self.memory.append((self.tree, self.accuracy(X, y)))

    def accuracy(self, X, y):
        y_pred = tops.batch_eval(X, self.tree)
        return np.mean(y_pred == y)

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

    ''' Can change global tree'''
    def __train_nodes_parallel(self, X, y, nodes):

        X_sub = X
        y_sub = y

        N = X.shape[0]
        # # subsamble 1000 data points
        # if N > 2000:
        #     subsample = np.random.choice(N, 1000, replace=False)
        #     X_sub = X[subsample]
        #     y_sub = y[subsample]
        #     N = X_sub.shape[0]

        if self.verbose:
            print(f"Training {len(nodes)} nodes in parallel...")
            print("first, finding necessary data")

        start_time = time.time()
        with Pool() as pool:
            train_data = pool.starmap(find_training_data, [(X_sub, y_sub, self.tree.root, node) for node in nodes])
        end_time = time.time()

        if self.verbose:
            print(f"Train Data Computation Time taken: {end_time - start_time} seconds")
        
        start_time = time.time()
        with Pool() as pool:
            result_data = pool.starmap(nops.find_optimal_params, train_data)
        end_time = time.time()

        if self.verbose:
            print(f"Actual Traning Time taken: {end_time - start_time} seconds")

        for i, node in enumerate(nodes):
            if result_data[i] is None:
                continue
            if node.is_leaf:
                node.label = result_data[i]
            else:
                node.w, node.b = result_data[i]

        if self.verbose:
            print(f"Finished training {len(nodes)} nodes")
            print(f"Accuracy: {self.accuracy(X, y)}")

    def __train_nodes(self, X, y, nodes):
        for node in nodes:
            train_data = find_training_data(X, y, self.tree.root, node)
            result_data = nops.find_optimal_params(*train_data)
            if result_data is None:
                continue
            if node.is_leaf:
                node.label = result_data
            else:
                node.w, node.b = result_data

    ''' Can change global tree '''
    def __train_tree(self, X, y):
        for depth in reversed(range(self.depth + 1)):
            nodes_at_depth = tops.find_nodes_at_depth(self.tree, depth)
            # print(f"ids found at depth {depth}: {[id(node) for node in nodes_at_depth]}")
            
            if self.verbose:
                print(f"Training {len(nodes_at_depth)} nodes at depth {depth}...")
            
            start_time = time.time()
            self.__train_nodes_parallel(X, y, nodes_at_depth)
            end_time = time.time()
            print(f"Parallel: {end_time - start_time} seconds")

            start_time = time.time()
            self.__train_nodes(X, y, nodes_at_depth)
            end_time = time.time()
            print(f"Sequential: {end_time - start_time} seconds")

            # with 1024 nodes:
            # Parallel: 3.916908025741577 seconds
            # Sequential: 4.654406309127808 seconds

if __name__ == "__main__":
    ct = BlitzOptimizer(DEPTH=10, D=2, K=5, verbose=True)

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

    # weights, biases, labels = ct.tree.to_numpy()
    weights, biases, leafs = tops.serialize(ct.tree)

    # ct.fit(X, y)
    # acc = ct.accuracy(X, y)
    # print('plotting...')
    # ct.plot_training(X, y)