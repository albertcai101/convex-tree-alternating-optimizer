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

# returns string
# depth 0: returns ['']
# depth 1: returns ['0', '1']
# depth 2: returns ['00', '01', '10', '11']
# ...
def find_serialized_node_paths_at_depth(depth: int):
    if depth == 0:
        return ['']
    else:
        prev = find_serialized_node_paths_at_depth(depth - 1)
        return [path + '0' for path in prev] + [path + '1' for path in prev]
    
def deserialize_node_path(node_path: str, tree: CTaoTree):
    current_node = tree.root
    for direction in node_path:
        if direction == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
    return current_node

def serialize(tree: CTaoTree):
    weights, biases, leafs =  __serialize_recursive(tree.root, [], [], [])
    weights, biases, leafs = np.array(weights), np.array(biases), np.array(leafs)

    assert weights.shape[0] == 2**(tree.depth) - 1
    assert biases.shape[0] == 2**(tree.depth) - 1
    assert leafs.shape[0] == 2**tree.depth
    return weights, biases, leafs

def deserialize(weights, biases, leafs, depth, D, K):
    tree = CTaoTree(depth, D, K)
    __deserialize_recursive(tree.root, weights, biases, leafs)
    return tree

def compare(tree1: CTaoTree, tree2: CTaoTree):
    return __compare_recursive(tree1.root, tree2.root)

def __find_nodes_at_depth_recursive(node: StandardNode, depth):
    if node is None:
        return []
    if node.depth == depth:
        return [node]
    return __find_nodes_at_depth_recursive(node.left, depth) + __find_nodes_at_depth_recursive(node.right, depth)

def __serialize_recursive(node: StandardNode, weights, biases, leafs):
    if node is None:
        return weights, biases, leafs
    
    if node.is_leaf:
        leafs.append(node.label)
    else:
        weights.append(node.w)
        biases.append(node.b)
        weights, biases, leafs = __serialize_recursive(node.left, weights, biases, leafs)
        weights, biases, leafs = __serialize_recursive(node.right, weights, biases, leafs)
    return weights, biases, leafs

def __deserialize_recursive(node: StandardNode, weights, biases, leafs):
    if node is None:
        return 0

    if node.is_leaf:
        node.label = leafs[0]
        leafs = leafs[1:]
    else:
        node.w = weights[0]
        node.b = biases[0]
        weights = weights[1:]
        biases = biases[1:]

        weights_left = weights[:len(weights)//2]
        weights_right = weights[len(weights)//2:]
        biases_left = biases[:len(biases)//2]
        biases_right = biases[len(biases)//2:]
        leafs_left = leafs[:len(leafs)//2]
        leafs_right = leafs[len(leafs)//2:]

        __deserialize_recursive(node.left, weights_left, biases_left, leafs_left)
        __deserialize_recursive(node.right, weights_right, biases_right, leafs_right)
    return 0

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

if __name__ == "__main__":
    tree = CTaoTree(depth=3, D=2, K=2)
    weights, biases, leafs = serialize(tree)
    print(tree.to_string(tree.root))
    print(weights)

    # print(find_serialized_node_paths_at_depth(2))

    # tree = CTaoTree(depth=2, D=2, K=2)
    # print(tree.to_string(tree.root))
    # print(id(deserialize_node_path('1', tree)))
