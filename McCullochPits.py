from prettytable import PrettyTable


class McCullochPitts_Neuron:
    def __init__(self, weights: list, threshold: int):
        self.weights = weights
        self.threshold = threshold

    def step_function(self, inputs: list):

        forward_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return 1 if forward_sum >= self.threshold else 0


if __name__ == "__main__":

    weights = [1, 1]
    limit = 2

    neuron = McCullochPitts_Neuron(weights, limit)

    table = PrettyTable()
    table.field_names = ["X", "Y", "Output"]

    for i in range(4):
        x = int(input("Enter X: "))
        y = int(input("Enter Y: "))
        output = neuron.step_function([x, y])
        table.add_row([x, y, output])

    print(table)