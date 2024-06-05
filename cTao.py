from multiprocessing import Pool
import numpy as np
import time
from cTaoTree import CTaoTree
import cvxpy as cp
import matplotlib.pyplot as plt

# D: dimension of data points
# K: number of classes

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
    prob.solve()

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

        for i in range(self.iters):
            self.__train_tree(X, y)
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

    def plot_training(self):
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
    def __train_nodes_parallel(self, X, y, nodes):
        print(f"Training {len(nodes)} nodes in parallel...")

        print("first, finding necessary data")
        start_time = time.time()
        train_data = []
        for node in nodes:
            # find all elements that reach node
            S = []
            N = X.shape[0]

            for n in range(N):
                x = X[n]
                if self.__reach_node(x, node):
                    S.append(n)
            
            if len(S) == 0:
                train_data.append((node.is_leaf, None))
                continue

            if node.is_leaf:
                train_data.append((node.is_leaf, y[S]))
            else:
                # find subset C of S that we care: changing left or right will change the label
                C = []
                y_bar = []
                for n in S:
                    x = X[n]
                    y_n = y[n]
                    left_label = self.eval(x, node.left)[0]
                    right_label = self.eval(x, node.right)[0]

                    # if both are correct, we don't care
                    if left_label == y_n and right_label == y_n:
                        continue

                    # if both are wrong, we don't care
                    if left_label != y_n and right_label != y_n:
                        continue

                    # if one is correct and the other is wrong, we care
                    C.append(n)
                    y_bar.append(1 if left_label == y_n else -1)

                train_data.append((node.is_leaf, (X[C], y_bar)))
        end_time = time.time()
        print(f"Train Data Computation Time taken: {end_time - start_time} seconds")
        
        start_time = time.time()
        with Pool() as pool:
            result_data = pool.starmap(train_node, train_data)
        end_time = time.time()

        print(f"Actual Traning Time taken: {end_time - start_time} seconds")

        for i, node in enumerate(nodes):
            if result_data[i] is None:
                continue
            if node.is_leaf:
                node.label = result_data[i]
            else:
                node.w, node.b = result_data[i]

        print(f"Finished training {len(nodes)} nodes")
        print(f"Accuracy: {self.accuracy(X, y)}")

    ''' Can change global tree '''
    def __train_tree(self, X, y):
        for depth in reversed(range(self.depth + 1)):
            nodes_at_depth = self.__get_nodes_at_depth(self.tree, depth)
            # print(f"ids found at depth {depth}: {[id(node) for node in nodes_at_depth]}")
            self.__train_nodes_parallel(X, y, nodes_at_depth)

    def __reach_node(self, x, node):
        current_node = self.tree.root
        while not current_node.is_leaf and current_node != node:
            decision = np.dot(x, current_node.w) + current_node.b
            current_node = current_node.left if decision > 0 else current_node.right
        return current_node == node

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
    ct = CTao(DEPTH=3, D=2, K=5)

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