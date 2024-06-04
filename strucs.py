import numpy as np
import cvxpy as cp

# D: dimension of data points
# K: number of classes

class Node:
    def __init__(self, depth, D, K, is_leaf=False):
        self.depth = depth
        self.is_leaf = is_leaf
        if not is_leaf:
            self.w = np.random.randn(D, 1)  # Weight vector for the node
            self.b = np.random.randn(1)     # Bias for the node
            self.left = None
            self.right = None
        else:
            self.label = np.random.randint(0, K)  # Class label for leaf node

    def set_children(self, left, right):
        self.left = left
        self.right = right

class LeafNode(Node):
    def __init__(self, depth, D, K):
        super().__init__(depth, D, K, is_leaf=True)

class CTaoTree:
    def __init__(self, depth, D, K):
        self.depth = depth
        self.D = D
        self.K = K
        self.root = self.build_tree(0, depth, D, K)
        
    def build_tree(self, current_depth, max_depth, D, K):
        if current_depth == max_depth:
            return LeafNode(current_depth, D, K)
        else:
            node = Node(current_depth, D, K)
            node.set_children(self.build_tree(current_depth + 1, max_depth, D, K),
                              self.build_tree(current_depth + 1, max_depth, D, K))
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
            
    def train_node(self, X, y, node=None):
        # print(f"Training node at depth {node.depth}")

        if node is None:
            node = self.root

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
            node.label = unique[np.argmax(counts)]

            # print(f"Trained leaf node at depth {node.depth} to label {node.label}")
        else:
            # if node is not leaf, find the best split

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
            prob.solve()


            node.w = w.value
            node.b = b.value

            # print(f"Trained node at depth {node.depth} to w {node.w.flatten()} and b {node.b}")
    
    def __reach_node(self, x, node):
        current_node = self.root
        while not current_node.is_leaf and current_node != node:
            decision = np.dot(x, current_node.w) + current_node.b
            current_node = current_node.left if decision > 0 else current_node.right
        return current_node == node

    # def train_iter(self, X, y):
    #     # Start the recursive training, wrapping the root node training in delayed
    #     result = self.__train_iter_recursive(X, y, self.root)
    #     # Compute the final result to trigger execution
    #     result.compute()

    # def __train_iter_recursive(self, X, y, node):
    #     if node.is_leaf:
    #         # Train leaf node, delay the execution
    #         return delayed(self.train_node)(X, y, node)
        
    #     # Recursively train left and right children, delaying their execution
    #     left_task = self.__train_iter_recursive(X, y, node.left)
    #     right_task = self.__train_iter_recursive(X, y, node.right)

    #     # Bind parent node training to occur after both children are trained
    #     # Node training itself is also a delayed task
    #     return bind(delayed(self.train_node), [left_task, right_task])(X, y, node)

    def train_nodes_parallel(self, nodes, X, y):
        for node in nodes:
            self.train_node(X, y, node)

    def train_iter(self, X, y):
        # Nodes at the same depth can be processed in parallel
        for depth in reversed(range(self.depth)):
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
            print(' ' * 4 * level + f"Depth {node.depth}: {'Leaf' if node.is_leaf else 'Node'}")
            if not node.is_leaf:
                print(' ' * 4 * level + f"w: {node.w.flatten()}, b: {node.b}")
                self.print_tree(node.left, level + 1)  # Recursively print left child
                self.print_tree(node.right, level + 1)  # Recursively print right child
            else:
                print(' ' * 4 * level + f"Class Label: {node.label}")