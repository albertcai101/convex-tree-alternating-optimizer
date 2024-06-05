from multiprocessing import Pool
import numpy as np
import time
from cTaoTree import CTaoTree
import cvxpy as cp
import matplotlib.pyplot as plt

# D: dimension of data points
# K: number of classes

def eval(x, node):
    if node.is_leaf:
        return node.label, node
    else:
        decision = np.dot(x, node.w) + node.b
        if decision > 0:
            return eval(x, node.left)
        else:
            return eval(x, node.right)

def reach_node(x, root, node):
    current_node = root
    while not current_node.is_leaf and current_node != node:
        decision = np.dot(x, current_node.w) + current_node.b
        current_node = current_node.left if decision > 0 else current_node.right
    return current_node == node


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
        if reach_node(x, root, node):
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
            left_label = eval(x, node.left)[0]
            right_label = eval(x, node.right)[0]

            if left_label == y_n and right_label == y_n:
                continue

            if left_label != y_n and right_label != y_n:
                continue

            C.append(n)
            y_bar.append(1 if left_label == y_n else -1)

        return (node.is_leaf, (X[C], y_bar))

def train_node(is_leaf, train_data):
    if train_data is None:
        return None
    if is_leaf:
        return train_leaf_node(train_data)
    else:
        return train_non_leaf_node(train_data[0], train_data[1])
    
# returns new label
def train_leaf_node(y):
    unique, counts = np.unique(y, return_counts=True)
    label = unique[np.argmax(counts)]
    return label

# returns new w and b
def train_non_leaf_node(X_C, y_bar):
    N_C, D = X_C.shape

    if N_C == 0:
        return None

    y_bar = np.array(y_bar)

    w = cp.Variable((D))
    b = cp.Variable()

    loss = cp.sum(cp.pos(1 - cp.multiply(y_bar, X_C @ w + b))) / N_C
    prob = cp.Problem(cp.Minimize(loss))
    prob.solve(solver=cp.ECOS)

    w = w.value
    b = b.value

    return w, b

class CTao():
    def __init__(self, DEPTH, D, K, MAX_ITERS=2):
        self.depth = DEPTH
        self.d = D
        self.k = K
        self.iters = MAX_ITERS

        self.tree = CTaoTree(self.depth, self.d, self.k)
        self.memory = []

    ''' Mutable Functinon that changes self.tree'''
    def fit(self, X, y):
        self.memory = []
        self.memory.append((self.tree, self.accuracy(X, y)))

        N = X.shape[0]
        shuffle = np.random.permutation(N)
        breakpoints = np.linspace(0, N, self.iters).astype(int)
        breakpoints = breakpoints[1:]

        # append N-1 to the end
        # breakpoints = np.append(breakpoints, N)
        # TODO: this does not give desired results, need to append to X_batches

        print(breakpoints)
        # [ 605 1210]
        X_batches = np.split(X[shuffle], breakpoints)
        y_batches = np.split(y[shuffle], breakpoints)

        # X_batches = np.append(X_batches, X)
        # y_batches = np.append(y_batches, y)

        for i in range(self.iters):
            print(f"----Training iteration {i+1}----")
            start_time = time.time()
            print(f"Training on {len(X_batches[i])} data points")
            self.__train_tree(X_batches[i], y_batches[i])
            end_time = time.time()
            print(f"Accuracy: {self.accuracy(X, y)}")
            print(f"Time taken: {end_time - start_time} seconds")
            self.memory.append((self.tree, self.accuracy(X, y)))

    def eval(self, x, node=None):
        if node is None:
            node = self.tree.root
        
        if node.is_leaf:
            return node.label, node
        else:
            decision = np.dot(x, node.w) + node.b
            if decision > 0:
                return self.eval(x, node.left)
            else:
                return self.eval(x, node.right)
            
    def batch_eval(self, X):
        return np.array([self.eval(x)[0] for x in X])

    def accuracy(self, X, y):
        y_pred = self.batch_eval(X)
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
            y_pred = tree.batch_eval(X)
            axs[i+1].scatter(X[:, 0], X[:, 1], c=[colors[y_pred[n]] for n in range(len(X))])
            axs[i+1].set_title(f"Iteration {i}: Accuracy {acc:.2f}")

        for ax in axs[total_plots:]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def print_tree(self):
        pass

    ''' Can change global tree'''
    def __train_nodes_parallel(self, X, y, nodes, verbose=False):

        X_sub = X
        y_sub = y

        N = X.shape[0]
        # # subsamble 1000 data points
        # if N > 2000:
        #     subsample = np.random.choice(N, 1000, replace=False)
        #     X_sub = X[subsample]
        #     y_sub = y[subsample]
        #     N = X_sub.shape[0]

        if verbose:
            print(f"Training {len(nodes)} nodes in parallel...")
            print("first, finding necessary data")

        start_time = time.time()
        with Pool() as pool:
            train_data = pool.starmap(find_training_data, [(X_sub, y_sub, self.tree.root, node) for node in nodes])
        end_time = time.time()

        if verbose:
            print(f"Train Data Computation Time taken: {end_time - start_time} seconds")
        
        start_time = time.time()
        with Pool() as pool:
            result_data = pool.starmap(train_node, train_data)
        end_time = time.time()

        if verbose:
            print(f"Actual Traning Time taken: {end_time - start_time} seconds")

        for i, node in enumerate(nodes):
            if result_data[i] is None:
                continue
            if node.is_leaf:
                node.label = result_data[i]
            else:
                node.w, node.b = result_data[i]

        if verbose:
            print(f"Finished training {len(nodes)} nodes")
            print(f"Accuracy: {self.accuracy(X, y)}")

    ''' Can change global tree '''
    def __train_tree(self, X, y):
        for depth in reversed(range(self.depth + 1)):
            nodes_at_depth = self.__get_nodes_at_depth(self.tree, depth)
            # print(f"ids found at depth {depth}: {[id(node) for node in nodes_at_depth]}")
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
    ct = CTao(DEPTH=10, D=2, K=5)

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