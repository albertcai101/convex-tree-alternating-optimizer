import numpy as np
from node import StandardNode
import node_ops as nops
from tree import CTaoTree

def eval (x, tree: CTaoTree):
    return nops.eval_from(x, tree.root)[0]

def batch_eval (X, tree: CTaoTree):
    return np.array([ nops.eval_from(x, tree.root)[0] for x in X])

def find_nodes_at_depth(tree, depth):
    return find_subnodes_at_depth(tree.root, depth)

def find_subnodes_at_depth(node: StandardNode, depth):
    if node is None:
        return []
    if node.depth == depth:
        return [node]
    return find_subnodes_at_depth(node.left, depth) + find_subnodes_at_depth(node.right, depth)