import numpy as np
import cvxpy as cp
from dask import delayed, compute, visualize
from node import LeafNode, StandardNode

# D: dimension of data points
# K: number of classes

class CTaoTree:
    def __init__(self, depth, D, K):
        self.depth = depth
        self.D = D
        self.K = K
        self.root = self.build_random_tree(depth=0, max_depth=depth, D=D, K=K, parent=None)
        
    def build_random_tree(self, depth, max_depth, D, K, parent):
        if depth == max_depth:
            return LeafNode(depth=depth, D=D, K=K, 
                            parent=parent,
                            label=np.random.randint(0, K))
        else:
            w = np.random.randn(D)
            b = np.random.randn()
            left = self.build_random_tree(depth+1, max_depth, D, K, parent=None)
            right = self.build_random_tree(depth+1, max_depth, D, K, parent=None)
            node = StandardNode(depth=depth, D=D, K=K, 
                                parent=parent, left=left, right=right, 
                                w=w, b=b)
            left.set_parent(node)
            right.set_parent(node)
            return node
    
    def evaluate(self, x, node=None):
        if node is None:
            node = self.root
        
        if node.is_leaf:
            return node.label, node
        else:
            decision = np.dot(x, node.w) + node.b
            if decision > 0:
                return self.evaluate(x, node.left)
            else:
                return self.evaluate(x, node.right)
            
    def batch_eval(self, X):
        return np.array([self.evaluate(x)[0] for x in X])
    
    def accuracy(self, X, y):
        y_pred = self.batch_eval(X)
        return np.mean(y_pred == y)
            
    def train_node(self, X, y, node):
        # print(f"Training node at depth {node.depth}")

        if node.is_leaf:
            # if node is leaf, update the label to the most frequent class
            # print(f"Now training leaf node at depth {node.depth}")

            # find the most frequent class

            # first find all the elements that reach the node
            S = []
            N = X.shape[0]
            for n in range(N):
                x = X[n]
                if self.__reach_node(x, node):
                    S.append(n)
            
            if len(S) == 0:
                # print(f"No data points reached leaf node at depth {node.depth}")
                return
            
            # set the label to the most frequent class
            y_S = y[S]
            unique, counts = np.unique(y_S, return_counts=True)

            new_label = unique[np.argmax(counts)]

            new_node = node.copy()
            new_node.label = new_label
            return new_node

            # print(f"Trained leaf node at depth {node.depth} to label {node.label}")
        else:
            # if node is not leaf, find the best split
            # print(f"Now training non-leaf node at depth {node.depth}")

            # find subset S of data points that reach node using reach_node
            S = []
            N = X.shape[0]
            for n in range(N):
                x = X[n]
                if self.__reach_node(x, node):
                    S.append(n)

            # print(f"Reached {len(S)} data points at depth {node.depth}")
            
            # find of subset C that we care: changing left or right will change the label
            C = []
            y_bar = []
            for n in S:
                x = X[n]
                y_n = y[n]
                left_label = self.evaluate(x, node.left)[0]
                right_label = self.evaluate(x, node.right)[0]

                # if both are correct, we don't care
                if left_label == y_n and right_label == y_n:
                    continue

                # if both are wrong, we don't care
                if left_label != y_n and right_label != y_n:
                    continue

                # if one is correct and the other is wrong, we care
                C.append(n)
                y_bar.append(1 if left_label == y_n else -1)

            # print(f"Found {len(C)} data points at depth {node.depth} that we care about")

            # find the best split that minimizes the loss using cvxpy
            X_C = X[C]
            y_C = np.array(y_bar)
            N_C = len(C)

            if N_C == 0:
                # print(f"No data points to care node at depth {node.depth}, skipping training node")
                return
            
            w = cp.Variable((self.D))
            b = cp.Variable()

            loss = cp.sum(cp.pos(1 - cp.multiply(y_C, X_C @ w + b))) / N_C
            prob = cp.Problem(cp.Minimize(loss))

            def solve_passthrough(prob):
                return prob.solve()

            delayed(solve_passthrough)(prob)


            new_w = w.value
            new_b = b.value

            new_node = node.copy()
            new_node.w = new_w
            new_node.b = new_b
            return new_node

            # print(f"Trained non-leaf node at depth {node.depth} to w {node.w.flatten()} and b {node.b}")
    
    def __reach_node(self, x, node):
        current_node = self.root
        while not current_node.is_leaf and current_node != node:
            decision = np.dot(x, current_node.w) + current_node.b
            current_node = current_node.left if decision > 0 else current_node.right
        return current_node == node

    def train_nodes_parallel(self, nodes, X, y):
        print(f"Training {len(nodes)} nodes in parallel")
        new_nodes = [self.train_node(X, y, node) for node in nodes]
        new_nodes = compute(*new_nodes)

        for node, new_node in zip(nodes, new_nodes):
            if new_node is not None:
                self.replace_node(node, new_node)

    def replace_node(self, node, new_node):
        if node.parent is None:
            self.root = new_node
        elif node.parent.left == node:
            node.parent.left = new_node
        else:
            node.parent.right = new_node
        new_node.set_parent(node.parent)

    def train_iter(self, X, y):
        # Nodes at the same depth can be processed in parallel
        for depth in reversed(range(self.depth+1)):
            nodes_at_depth = self.__get_nodes_at_depth(depth)
            self.train_nodes_parallel(nodes_at_depth, X, y)
    
    def __get_nodes_at_depth(self, depth):
        return self.__get_subnodes_at_depth(self.root, depth)

    def __get_subnodes_at_depth(self, node, depth):
        # Helper method to retrieve all nodes at a specified depth
        if node is None:
            return []
        if node.depth == depth:
            return [node]
        return self.__get_subnodes_at_depth(node.left, depth) + self.__get_subnodes_at_depth(node.right, depth)

    def __prune(self, X, y):
        # dead branches
        # nodes with children that recieve no training points, replace with the other child

        # pure subtrees
        # if a subtree has all leaf nodes with the same label, replace the subtree with a leaf node with that label
        pass
        
    def print_tree(self, node, level=0):
        if node is not None:
            description = ' ' * 4 * level + f"Depth {node.depth}: {'Leaf' if node.is_leaf else 'Node'}"
            if not node.is_leaf:
                print(description + f" | w: {node.w.flatten()}, b: {node.b}")
                self.print_tree(node.left, level + 1)  # Recursively print left child
                self.print_tree(node.right, level + 1)  # Recursively print right child
            else:
                print(description + f" | label: {node.label}")



if __name__ == "__main__":
    print("Testing CTaoTree")

    np.random.seed(0)
    depth = 2
    D = 2
    K = 3

    tree = CTaoTree(depth, D, K)
    tree.print_tree(tree.root)