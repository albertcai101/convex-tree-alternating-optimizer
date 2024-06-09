import numpy as np
from node import StandardNode
import node_ops as nops
from tree import CTaoTree

def eval (x, tree: CTaoTree):
    return nops.eval_from(x, tree.root)[0]

def batch_eval (X, tree: CTaoTree):
    return np.array([ nops.eval_from(x, tree.root)[0] for x in X])

def accuracy (X, y, tree: CTaoTree):
    return np.mean(batch_eval(X, tree) == y)

def find_nodes_at_depth(tree: CTaoTree, depth: int):
    return __find_nodes_at_depth_recursive(tree.root, depth)

def compare(tree1: CTaoTree, tree2: CTaoTree):
    return __compare_recursive(tree1.root, tree2.root)

def __find_nodes_at_depth_recursive(node: StandardNode, depth):
    if node is None:
        return []
    if node.depth == depth:
        return [node]
    return __find_nodes_at_depth_recursive(node.left, depth) + __find_nodes_at_depth_recursive(node.right, depth)

def __compare_recursive(node1: StandardNode, node2: StandardNode):
    if node1 is None and node2 is None:
        return True
    
    if node1 is None or node2 is None:
        return False
    if node1.is_leaf and node2.is_leaf:
        return node1.label == node2.label
    if not np.array_equal(node1.w, node2.w) or not np.array_equal(node1.b, node2.b):
        return False
    return __compare_recursive(node1.left, node2.left) and __compare_recursive(node1.right, node2.right)