import numpy as np
from matplotlib import pyplot as plt


def map_last_digit(number):
    last = number % 10
    binary_map = np.array([1 if last == i else 0 for i in range(0, 10)])
    return binary_map


def plot_boundary(model, X, y):
    xx, yy = np.meshgrid(np.linspace(0, 9, 100), np.linspace(0, 1, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = np.array([model.predict(map_last_digit(x[0])) for x in grid]).reshape(xx.shape)

    plt.contourf(xx, yy, pred, alpha=0.8, levels=10)
    plt.scatter(X, y, c=y, edgecolors='black', marker='o', s=50)
    plt.title('Perceptron Decision Boundary')
    plt.xlabel('Input')
    plt.ylabel('Output')

    plt.show()


class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.normal(size=input_size)
        self.bias = np.random.normal()

    def predict(self, inputs):
        summation = np.dot(self.weights, inputs) + self.bias
        return 1 if summation > 0.5 else 0

    def train(self, training_inputs, labels, epochs, learning_rate):
        for epoch in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights += learning_rate * (label - prediction) * inputs
                self.bias += learning_rate * (label - prediction)


if __name__ == "__main__":
    X = np.array([i for i in range(0, 10)])
    y = np.array([1 if i % 2 == 0 else 0 for i in X])
    mapped_X = np.array([map_last_digit(x) for x in X])
    model = Perceptron(len(mapped_X[0]))
    model.train(mapped_X, y, 20, 0.1)

    z = map_last_digit(8080)
    print(model.predict(z))
