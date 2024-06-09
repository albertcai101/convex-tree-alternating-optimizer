import numpy as np

from node import StandardNode
from tree import CTaoTree

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

def get_index_from_serialized_path(node_path: str, depth:int):
    index = 0
    for i in range(len(node_path)):
        if node_path[i] == '0':
            index += 1
        elif node_path[i] == '1':
            index += 2**(depth - i)
        else:
            raise ValueError(f"Invalid node path: {node_path}")
    return index

def get_serialized_path_from_index(index: int, depth: int):
    return __get_serialized_path_from_index_recursive(index, '', depth)

def get_left_child_index(index: int, depth: int):
    return index + 1

def get_right_child_index(index: int, depth: int):
    path = get_serialized_path_from_index(index, depth)
    if len(path) >= depth:
        return None
    return get_index_from_serialized_path(path + '1', depth)

def is_leaf(index: int, depth: int):
    path = get_serialized_path_from_index(index, depth)
    return len(path) == depth

def reach_node(x, weights, biases, root_index: int, node_index: int, depth: int):
    current_index = root_index
    while not is_leaf(current_index, depth) and current_index != node_index:
        weight = get_weight_from_index(current_index, weights, depth)
        bias = get_bias_from_index(current_index, biases, depth)
        decision = np.dot(x, weight) + bias
        current_index = get_left_child_index(current_index, depth) if decision > 0 else get_right_child_index(current_index, depth)
    return current_index == node_index

def eval_from(x, weights, biases, leafs, root_index: int, depth: int):
    current_index = root_index
    while not is_leaf(current_index, depth):
        weight = get_weight_from_index(current_index, weights, depth)
        bias = get_bias_from_index(current_index, biases, depth)
        decision = np.dot(x, weight) + bias
        current_index = get_left_child_index(current_index, depth) if decision > 0 else get_right_child_index(current_index, depth)
    return get_leaf_from_index(current_index, leafs, depth)

def get_weight_from_index(index: int, weights, depth: int):
    path = get_serialized_path_from_index(index, depth)
    # print('path: ', path)
    weights_index = get_index_from_serialized_path(path, depth-1)
    # print('weights_index: ', weights_index)
    return weights[weights_index]

def get_bias_from_index(index: int, biases, depth: int):
    path = get_serialized_path_from_index(index, depth)
    biases_index = get_index_from_serialized_path(path, depth-1)
    return biases[biases_index]

def get_leaf_from_index(index: int, leafs, depth: int):
    path = get_serialized_path_from_index(index, depth)

    # convert binary path to index
    # 0000 -> 0
    # 0001 -> 1
    # 1111 -> 15

    leaf_index = 0
    for i in range(len(path)):
        leaf_index += 2**(depth - i - 1) if path[i] == '1' else 0

    return leafs[leaf_index]

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

def __get_serialized_path_from_index_recursive(index: int, path: str, subtree_depth: int):
    if index == 0:
        return path
    
    left_or_right = index // (2**subtree_depth)
    residual = index % (2**subtree_depth)

    if left_or_right == 0:
        return __get_serialized_path_from_index_recursive(index - 1, path + '0', subtree_depth - 1)
    elif left_or_right == 1:
        return __get_serialized_path_from_index_recursive(residual, path + '1', subtree_depth - 1)
    else:
        raise ValueError(f"Invalid index: {index}")

if __name__ == "__main__":
    # depth = 7
    # path = '0'

    # index = get_index_from_serialized_path(path, depth)
    # print(index)

    # path_recon = get_serialized_path_from_index(index, depth)
    # print(path_recon)

    # tree = CTaoTree(depth=2, D=2, K=2)
    # weights, biases, leafs = serialize(tree)
    # print(reach_node(np.array([0,0]), weights, biases, 0, 0, 2))



    # print(is_leaf(7, 3))
    # print(get_right_child_index(7,3))


    tree = CTaoTree(depth=4, D=2, K=2)
    print(tree.to_string(tree.root))
    weights, biases, leafs = serialize(tree)
    print(weights)
    print(biases)
    print(leafs)

    # path = '011'
    # index = get_index_from_serialized_path(path, 4)
    # weight = get_weight_from_index(index, weights, 4)
    # bias = get_bias_from_index(index, biases, 4)
    # print('path: ', path)
    # print('weight: ', weight)
    # print('bias: ',  bias)

    # path = '0011'
    # index = get_index_from_serialized_path(path, 4)
    # leaf = get_leaf_from_index(index, leafs, 4)
    # print('path: ', path)
    # print('leaf: ', leaf)

    x = np.array([0,0])
    root_index = 0
    print(eval_from(x, weights, biases, leafs, root_index, 4))