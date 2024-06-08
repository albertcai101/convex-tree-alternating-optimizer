import numpy as np
import cvxpy as cp
from node import StandardNode

def eval_from(x, node: StandardNode):
    if node.is_leaf:
        return node.label, node
    else:
        decision = np.dot(x, node.w) + node.b
        if decision > 0:
            return eval_from(x, node.left)
        else:
            return eval_from(x, node.right)

def reach_node(x, root: StandardNode, node: StandardNode):
    current_node = root
    while not current_node.is_leaf and current_node != node:
        decision = np.dot(x, current_node.w) + current_node.b
        current_node = current_node.left if decision > 0 else current_node.right
    return current_node == node