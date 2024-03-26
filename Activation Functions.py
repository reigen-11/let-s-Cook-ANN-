import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / 1 * np.exp(-x)


def hyperbolic_tangent(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.05):
    return np.maximum(alpha * x, x)


def soft_plus(x):
    return np.log(1 + np.exp(x))


def swish(x):
    return x * sigmoid(x)


x_values = np.linspace(-5, 5, 500)
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x_values, sigmoid(x_values))
plt.title('Sigmoid')

plt.subplot(2, 3, 2)
plt.plot(x_values, hyperbolic_tangent(x_values))
plt.title('hyperbolic tangent')

plt.subplot(2, 3, 3)
plt.plot(x_values, relu(x_values))
plt.title('ReLU')

plt.subplot(2, 3, 4)
plt.plot(x_values, leaky_relu(x_values))
plt.title('Leaky ReLU')

plt.subplot(2, 3, 5)
plt.plot(x_values, soft_plus(x_values))
plt.title('Soft-plus')

plt.subplot(2, 3, 6)
plt.plot(x_values, swish(x_values))
plt.title('swish')

plt.tight_layout()
plt.show()
