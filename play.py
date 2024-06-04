import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from strucs import LeafNode, Node, CTaoTree

# generate synthetic data for classification
np.random.seed(42)

# N: number of data points
# D: dimension of data points
# K: number of classes

N = 1000
D = 2
K = 5

X = np.random.randn(N, D)

# Initialize empty lists for weights and biases
weights = []
biases = []

layers = 5

# Loop to create 5 layers
for _ in range(layers):
    # Generate random weights and biases for each layer
    W_real = np.random.randn(D, K)
    b_real = np.random.randn(K)
    
    # Append weights and biases to the respective lists
    weights.append(W_real)
    biases.append(b_real)

# Compute the output of each layer and select the maximum
outputs = [X.dot(W) + b for W, b in zip(weights, biases)]
y = np.argmax(np.maximum.reduce(outputs), axis=1)

# Use TAO to solve the optimization problem
# first create a random tree structure
depth = 5

# non leaf nodes have w and b (w is a D x 1 vector, b is a scalar)
# leaf nodes is simply 0, ..., K-1

tree = CTaoTree(depth, D, K)
print(f"Tree accuracy: {tree.accuracy(X, y)}")

iters = 3

total_plots = iters + 2
colors = ['r', 'g', 'b', 'y', 'm']
cols = min(10, total_plots)
rows = (total_plots + 9) // 10

fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axs = axs.flatten()


axs[0].scatter(X[:, 0], X[:, 1], c=[colors[y[n]] for n in range(N)])
axs[0].set_title("Ground Truth")

# Plot initial state
y_pred = tree.batch_eval(X)
axs[1].scatter(X[:, 0], X[:, 1], c=[colors[y_pred[n]] for n in range(len(X))])
axs[1].set_title("Initial Tree")

# Iteration and plotting
for i in range(iters):
    tree.train_iter(X, y)
    y_pred = tree.batch_eval(X)
    print(f"Tree accuracy: {tree.accuracy(X, y)}")

    axs[i+2].scatter(X[:, 0], X[:, 1], c=[colors[y_pred[n]] for n in range(len(X))])
    axs[i+2].set_title(f"Iteration {i}: Accuracy {tree.accuracy(X, y):.2f}")

# Hide any unused axes
for ax in axs[total_plots:]:
    ax.axis('off')

plt.tight_layout()
plt.show()

# w = cp.Variable((D, 1))
# b = cp.Variable()

# loss = cp.sum([cp.maximum(0, 1 - (y[n] * (cp.matmul(X[n], w) + b))) for n in range(N)])
# objective = cp.Minimize(loss)
# constraints = []
# prob = cp.Problem(objective, constraints)
# prob.solve()

# w_cvx = w.value
# b_cvx = b.value

# # print accuracy
# y_pred = np.sign(X.dot(w_cvx) + b_cvx)
# accuracy = np.mean(y_pred == y)
# print(f"Accuracy CVX: {accuracy}")
