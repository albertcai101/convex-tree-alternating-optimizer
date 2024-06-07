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

def find_optimal_params(is_leaf, train_data):
    if train_data is None:
        return None
    if is_leaf:
        return find_optimal_leaf_params(train_data)
    else:
        return find_optimal_non_leaf_params(train_data[0], train_data[1])

def find_optimal_leaf_params(y_care):
    unique, counts = np.unique(y_care, return_counts=True)
    label = unique[np.argmax(counts)]
    return label

def find_optimal_non_leaf_params(X_care, y_bar):
    N, D = X_care.shape

    if N == 0:
        return None

    y_bar = np.array(y_bar)

    w = cp.Variable((D))
    b = cp.Variable()

    loss = cp.sum(cp.pos(1 - cp.multiply(y_bar, X_care @ w + b))) / N
    prob = cp.Problem(cp.Minimize(loss))
    prob.solve(solver=cp.ECOS)

    w = w.value
    b = b.value

    return w, b