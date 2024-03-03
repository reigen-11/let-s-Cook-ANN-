import numpy as np
from matplotlib import pyplot as plt


def map_last_digit(data):
    X_train = []
    for number in data:
        last = number % 10
        last = ord(str(last))
        binary_representation = bin(last)[2:].zfill(7)
        list_of_integers = [int(digit) for digit in binary_representation]
        X_train.append(list_of_integers)
    return np.array(X_train)


class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.normal(size=input_size)
        self.bias = np.random.normal()

    def predict(self, inputs):
        summation = np.dot(self.weights, inputs.flatten()) + self.bias
        return 1 if summation > 0.5 else 0

    def train(self, training_inputs, labels, epochs, learning_rate, verbose: bool = None):
        for epoch in range(epochs):
            weight_history = []
            for inputs, label in zip(training_inputs, labels):
                pred = self.predict(inputs)
                self.weights += learning_rate * (label - pred)
                self.bias += learning_rate * (label - pred)
                weight_history.append(self.weights.copy())
                if verbose:
                    print(f"epoch-{epoch} : weights = {self.weights}, bias = {self.bias}")

            self.plot_weights(weight_history)

    def plot_weights(self, weight_history):
        epochs = len(weight_history)
        num_weights = len(weight_history[0])

        # Create subplots for each weight
        fig, axs = plt.subplots(num_weights, 1, figsize=(8, num_weights * 4))

        for i in range(num_weights):
            weights_i = [weights[i] for weights in weight_history]
            axs[i].plot(range(epochs), weights_i)
            axs[i].set_ylabel(f'Weight {i+1}')

        axs[-1].set_xlabel('Epoch')
        plt.tight_layout()
        plt.show()


if '__name__' == '__main__':
    X = np.array([i for i in range(0, 10)])
    X_train = map_last_digit(X)
    y = np.array([1 if i % 2 == 0 else 0 for i in X])
    model = Perceptron(len(X_train[0]))
    model.train(X_train, y, 20, 0.1)
    prediction = model.predict(map_last_digit([1]))
