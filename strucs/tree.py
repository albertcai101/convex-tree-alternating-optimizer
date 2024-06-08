import numpy as np
import cvxpy as cp
from dask import delayed, compute, visualize
from strucs.node import LeafNode, StandardNode

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

    def copy(self):
        new_tree = CTaoTree(self.depth, self.D, self.K)
        new_tree.root = self.__copy_tree(self.root)
        return new_tree

    def __copy_tree(self, node):
        if isinstance(node, LeafNode):
            return LeafNode(depth=node.depth, D=node.D, K=node.K, 
                            parent=None, label=node.label)
        elif isinstance(node, StandardNode):
            left_copy = self.__copy_tree(node.left)
            right_copy = self.__copy_tree(node.right)
            node_copy = StandardNode(depth=node.depth, D=node.D, K=node.K,
                                     parent=None, left=left_copy, right=right_copy,
                                     w=node.w.copy(), b=node.b)
            left_copy.set_parent(node_copy)
            right_copy.set_parent(node_copy)
            return node_copy

    def to_string(self, node, level=0, str = ''):
        if node is not None:
            description = ' ' * 4 * level + f"Depth {node.depth}: {'Leaf' if node.is_leaf else 'Node'}"
            if not node.is_leaf:
                str += description + f" | w: {node.w.flatten()}, b: {node.b}, id: {id(node)}, parent: {id(node.parent)}\n"
                str = self.to_string(node.left, level + 1, str)
                str = self.to_string(node.right, level + 1, str)
            else:
                str += description + f" | label: {node.label}, id: {id(node)}, parent: {id(node.parent)}\n"
        return str
    
    # TODO: implement a recursive tree visualizer