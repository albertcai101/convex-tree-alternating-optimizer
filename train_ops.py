from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import current_process, cpu_count, Process
from multiprocessing import Pool # want to move off this
import tracemalloc
import time
import warnings
import numpy as np
import cvxpy as cp
from node import StandardNode
import node_ops as nops
import tree_ops as tops
import serialize_ops as sops
from tree import CTaoTree

def train_node(X, y, tree: CTaoTree,  node: StandardNode):
    # print(f"Training node at depth {node.depth}")

    # subsample X and y if too large
    N = 1000
    if X.shape[0] > N:
        idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[idx]
        y = y[idx]

    if node is None:
        node = tree.root

    if node.is_leaf:
        # if node is leaf, update the label to the most frequent class
        # print(f"Now training leaf node at depth {node.depth}")

        # find the most frequent class

        # first find all the elements that reach the node
        S = []
        N = X.shape[0]
        for n in range(N):
            x = X[n]
            if nops.reach_node(x, tree.root, node):
                S.append(n)
        
        if len(S) == 0:
            # print(f"No data points reached leaf node at depth {node.depth}")
            return (True, None)
        
        # set the label to the most frequent class
        y_S = y[S]
        unique, counts = np.unique(y_S, return_counts=True)
        return (True, unique[np.argmax(counts)])

        # print(f"Trained leaf node at depth {node.depth} to label {node.label}")
    else:
        # if node is not leaf, find the best split

        # find subset S of data points that reach node using reach_node
        S = []
        N = X.shape[0]
        for n in range(N):
            x = X[n]
            if nops.reach_node(x, tree.root, node):
                S.append(n)

        # print(f"Reached {len(S)} data points at depth {node.depth}")
        
        # find of subset C that we care: changing left or right will change the label
        C = []
        y_bar = []
        for n in S:
            x = X[n]
            y_n = y[n]
            left_label = nops.eval_from(x, node.left)[0]
            right_label = nops.eval_from(x, node.right)[0]

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
            return (False, None, None)
        
        w = cp.Variable((X.shape[1]))
        b = cp.Variable()

        loss = cp.sum(cp.pos(1 - cp.multiply(y_C, X_C @ w + b))) / N_C
        prob = cp.Problem(cp.Minimize(loss))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob.solve()

        return (False, w.value, b.value)
    
def train_node_serialized(X, y, weights, biases, leafs, node_path, depth):
    if node_path == None:
        node_path = ''

    node_index = sops.get_index_from_serialized_path(node_path, depth)

    if len(node_path) == depth:
        # Leaf Node

        # find the most frequent class

        # first find all the elements that reach the node
        S = []
        N = X.shape[0]
        for n in range(N):
            x = X[n]
            if sops.reach_node(x, weights, biases, 0, node_index, depth):
                S.append(n)

        if len(S) == 0:
            return (True, None)
        
        # set the label to the most frequent class
        y_S = y[S]
        unique, counts = np.unique(y_S, return_counts=True)
        return (True, unique[np.argmax(counts)])
    
    else:
        # Internal Node

        # find subset S of data points that reach node using reach_node
        S = []
        N = X.shape[0]
        for n in range(N):
            x = X[n]
            if sops.reach_node(x, weights, biases, 0, node_index, depth):
                S.append(n)

        # find of subset C that we care: changing left or right will change the label
        C = []
        y_bar = []
        for n in S:
            x = X[n]
            y_n = y[n]
            left_label = sops.eval_from(x, weights, biases, leafs, sops.get_left_child_index(node_index, depth), depth)
            right_label = sops.eval_from(x, weights, biases, leafs, sops.get_right_child_index(node_index, depth), depth)

            # if both are correct, we don't care
            if left_label == y_n and right_label == y_n:
                continue

            # if both are wrong, we don't care
            if left_label != y_n and right_label != y_n:
                continue

            # if one is correct and the other is wrong, we care
            C.append(n)
            y_bar.append(1 if left_label == y_n else -1)

        # find the best split that minimizes the loss using cvxpy
        X_C = X[C]
        y_C = np.array(y_bar)
        N_C = len(C)

        if N_C == 0:
            return (False, None, None)
        
        w = cp.Variable((X.shape[1]))
        b = cp.Variable()

        loss = cp.sum(cp.pos(1 - cp.multiply(y_C, X_C @ w + b))) / N_C
        prob = cp.Problem(cp.Minimize(loss))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prob.solve()

        return (False, w.value, b.value)


def train_node_shared_memory(shm_name,
                               weights_shape, weights_dtype, weights_offset,
                               biases_shape, biases_dtype, biases_offset,
                               leafs_shape, leafs_dtype, leafs_offset,
                               X_shape, X_dtype, X_offset,
                               y_shape, y_dtype, y_offset,
                               node_path, depth, D, K, verbose=False):
    start_time = time.time()

    # if verbose:
    #     print("START TRAIN NODE SHARED MEMORY")

    shm = SharedMemory(shm_name)
    weights = np.ndarray(weights_shape, dtype=weights_dtype, buffer=shm.buf, offset=weights_offset)
    biases = np.ndarray(biases_shape, dtype=biases_dtype, buffer=shm.buf, offset=biases_offset)
    leafs = np.ndarray(leafs_shape, dtype=leafs_dtype, buffer=shm.buf, offset=leafs_offset)
    X = np.ndarray(X_shape, dtype=X_dtype, buffer=shm.buf, offset=X_offset)
    y = np.ndarray(y_shape, dtype=y_dtype, buffer=shm.buf, offset=y_offset)

    # mid_time = time.time()

    # if verbose:
    #     print(f"Time taken to get shared memory: {mid_time - start_time} seconds")
    

    tree = sops.deserialize(weights, biases, leafs, depth, D, K)
    node = sops.deserialize_node_path(node_path, tree)

    # mid_time = time.time()

    # if verbose:
    #     print(f"Time taken to deserialize tree and node: {mid_time - start_time} seconds")

    result_old = train_node(X, y, tree, node)

    # result_new = train_node_serialized(X, y, weights, biases, leafs, node_path, depth)

    # if np.any(result_old[1] != result_new[1]):
    #     print('result mismatch')
    #     print('old: ', result_old)
    #     print('new: ', result_new)

    # if verbose:
    #     print(f"Time taken to train node: {time.time() - mid_time} seconds")
    #     print("END TRAIN NODE SHARED MEMORY")

    return result_old

def train_tree(X, y, tree: CTaoTree, verbose=False):
    tree = tree.copy()

    for depth in reversed(range(tree.depth + 1)):
        nodes_at_depth = tops.find_nodes_at_depth(tree, depth)

        if verbose:
            print(f"Training {len(nodes_at_depth)} nodes at depth {depth}...")

        start_time = time.time()
        with Pool(cpu_count()) as p:
            results = p.starmap(train_node, [(X, y, tree, node) for node in nodes_at_depth])
        for node, result in zip(nodes_at_depth, results):
            is_leaf = result[0]

            if is_leaf:
                leaf_label = result[1]
                if leaf_label is not None:
                    node.label = leaf_label
            else:
                w = result[1]
                b = result[2]
                if w is not None:
                    node.w = w
                if b is not None:
                    node.b = b
        end_time = time.time()
        if verbose:
            print(f"Time taken: {end_time - start_time} seconds")

    return tree

def train_tree_shared_memory(X, y, tree: CTaoTree, verbose=False):
    tree = tree.copy()

    # subsample X and y if too large
    N = 1
    if X.shape[0] > N:
        idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[idx]
        y = y[idx]

    weights, biases, leafs = sops.serialize(tree)
    total_bytes = weights.nbytes + biases.nbytes + leafs.nbytes + X.nbytes + y.nbytes
    offset_weights = 0
    offset_biases = offset_weights + weights.nbytes
    offset_leafs = offset_biases + biases.nbytes
    offset_X = offset_leafs + leafs.nbytes
    offset_y = offset_X + X.nbytes

    with SharedMemoryManager() as smm:
        # Create a shared memory of size total_bytes
        shm = smm.SharedMemory(total_bytes)
        # Create np arrays using the buffer of shm
        shm_weights = np.ndarray(shape=weights.shape, dtype=weights.dtype, buffer=shm.buf, offset=offset_weights)
        shm_biases = np.ndarray(shape=biases.shape, dtype=biases.dtype, buffer=shm.buf, offset=offset_biases)
        shm_leafs = np.ndarray(shape=leafs.shape, dtype=leafs.dtype, buffer=shm.buf, offset=offset_leafs)
        shm_X = np.ndarray(shape=X.shape, dtype=X.dtype, buffer=shm.buf, offset=offset_X)
        shm_y = np.ndarray(shape=y.shape, dtype=y.dtype, buffer=shm.buf, offset=offset_y)
        # Copy data into shared memory
        np.copyto(shm_weights, weights)
        np.copyto(shm_biases, biases)
        np.copyto(shm_leafs, leafs)
        np.copyto(shm_X, X)
        np.copyto(shm_y, y)

        for depth in reversed(range(tree.depth + 1)):
            # nodes_at_depth = tops.find_nodes_at_depth(tree, depth)

            serialized_node_paths_at_depth = sops.find_serialized_node_paths_at_depth(depth)

            if verbose:
                print(f"Training {len(serialized_node_paths_at_depth)} nodes at depth {depth}...")

            start_time = time.time()

            with ProcessPoolExecutor(cpu_count()) as exe:
                fs = [exe.submit(train_node_shared_memory, shm.name,
                                    weights.shape, weights.dtype, offset_weights,
                                    biases.shape, biases.dtype, offset_biases,
                                    leafs.shape, leafs.dtype, offset_leafs,
                                    X.shape, X.dtype, offset_X,
                                    y.shape, y.dtype, offset_y,
                                    serialized_node_path, tree.depth, tree.D, tree.K, verbose=verbose)
                        for serialized_node_path in serialized_node_paths_at_depth]
                for _ in as_completed(fs):
                    pass

            mid_time = time.time()
            if verbose:
                print(f"Time taken to submit tasks: {mid_time - start_time} seconds")

            results = [f.result() for f in fs]
            if verbose:
                print(f"Time taken to get results: {time.time() - mid_time} seconds")

            mid_time = time.time()

            for serialized_node_path, result in zip(serialized_node_paths_at_depth, results):
                is_leaf = result[0]

                node = sops.deserialize_node_path(serialized_node_path, tree)
                if is_leaf:
                    leaf_label = result[1]
                    if leaf_label is not None:
                        node.label = leaf_label
                else:
                    w = result[1]
                    b = result[2]
                    if w is not None:
                        node.w = w
                    if b is not None:
                        node.b = b
            if verbose:
                print(f"Time taken to update tree: {time.time() - mid_time} seconds")

            end_time = time.time()

            if verbose:
                print(f"Time taken: {end_time - start_time} seconds")

            # update weights, biases, leafs
            weights, biases, leafs = sops.serialize(tree)

            # update shared memory
            np.copyto(shm_weights, weights)
            np.copyto(shm_biases, biases)
            np.copyto(shm_leafs, leafs)

    return tree