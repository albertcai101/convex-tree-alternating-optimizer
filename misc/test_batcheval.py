from strucs.tree import CTaoTree
from operations.tree_ops import batch_eval
import numpy as np
import time

# make a tree of depth 10, data is 100 features and 5 labels
tree = CTaoTree(10, 100, 5)

# generate 100,000 data points
X = np.random.randn(100000, 100)

# test how long batch_eval takes
print("Testing batch_eval...")
start = time.time()
batch_eval(X, tree)
end = time.time()
print(f"Time taken: {end - start} seconds")

# 0.497236967086792 seconds 