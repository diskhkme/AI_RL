import numpy as np
import matplotlib.pyplot as plt

count = 500
true_weight = 10
learning_rate = 1e-4
max_iter = 1000
estimated_val = 1 # initial estimation
losses = []

X = np.random.rand(count)
Y = true_weight*X + np.random.randn(count)

for i in range(max_iter+1):
    gradient = np.dot(X,X*estimated_val - Y)
    estimated_val = estimated_val - learning_rate * gradient

    loss = np.sum((Y-X*estimated_val) ** 2)/count
    losses.append(loss)
    if i % 100 == 0:
        print(f"Iter: {i} \t| Estimation: {estimated_val} \t| Loss: {loss}")

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot(X,Y,'.')
ax1.plot((0,1),(0,estimated_val))
ax2.plot(np.arange(max_iter+1), np.array(losses),'-')
plt.show()
