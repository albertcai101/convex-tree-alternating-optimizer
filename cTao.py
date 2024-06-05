from dask.distributed import Client, LocalCluster
import numpy as np
import cvxpy as cp
from dask import delayed, compute, visualize
from cTaoTree import CTaoTree
import time

# D: dimension of data points
# K: number of classes

class CTao():
    def __init__(self, DEPTH, D, K, MAX_ITERS=5):
        self.depth = DEPTH
        self.d = D
        self.k = K
        self.iters = MAX_ITERS

        self.tree = CTaoTree(self.depth, self.d, self.k)
        self.memory = []

        self.cluster = LocalCluster(n_workers=4, threads_per_worker=2)  # Adjust these numbers based on your machine's cores and threads
        # cluster = LocalCluster(resources={'CPU': 1})
        self.client = Client(self.cluster)
    
    ''' Mutable Functinon that changes self.tree'''
    def fit(self, X, y):
        self.memory = []
        self.memory.append((self.tree, self.accuracy(X, y)))
        self.tree.print_tree(self.tree.root)

        for i in range(self.iters):
            self.__train_tree(X, y)
            self.tree.print_tree(self.tree.root)
            self.memory.append((self.tree, self.accuracy(X, y)))

        pass

    def predict(self, X):
        pass

    def accuracy(self, X, y):
        pass

    def plot_training(self):
        pass

    def print_tree(self):
        pass

    ''' Immutable '''
    def __train_node(self, tree, X, y, node):
        # tree = tree.copy()
        time.sleep(0.1)
        # pass

    ''' Can change global tree'''
    def __train_nodes_parallel(self, X, y, nodes):
        print(f"Training {len(nodes)} nodes in parallel...")
        new_nodes = [delayed(self.__train_node)(self.tree, X, y, node) for node in nodes]

        start_time = time.time()
        new_nodes = compute(*new_nodes)
        end_time = time.time()

        print(f"Training {len(nodes)} nodes took {end_time - start_time} seconds")

        # for node, new_node in zip(nodes, new_nodes):
        #     if new_node is not None:
        #         self.replace_node(node, new_node)

    ''' Can change global tree '''
    def __train_tree(self, X, y):
        for depth in reversed(range(self.depth+1)):
            nodes_at_depth = self.__get_nodes_at_depth(self.tree, depth)

            print(f"ids found at depth {depth}: {[id(node) for node in nodes_at_depth]}")
            self.__train_nodes_parallel(X, y, nodes_at_depth)

    ''' Immutable, not edit any class variables, parameters'''
    def __get_nodes_at_depth(self, tree, depth):
        return self.__get_subnodes_at_depth(tree.root, depth)

    def __get_subnodes_at_depth(self, node, depth):
        if node is None:
            return []
        if node.depth == depth:
            return [node]
        return self.__get_subnodes_at_depth(node.left, depth) + self.__get_subnodes_at_depth(node.right, depth)


if __name__ == "__main__":
    ct = CTao(DEPTH=2, D=2, K=5)

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