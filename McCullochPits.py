from prettytable import PrettyTable


class McCullochPitts_Neuron_ANDNOT:
    def __init__(self, weights: list, threshold: int):
        self.weights = weights
        self.threshold = threshold

    def step_function(self, inputs: list):
        forward_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return 0 if forward_sum >= self.threshold else 1


if __name__ == "__main__":
    A_weights = [1, 1]
    A_limit = 2

    neuron_and = McCullochPitts_Neuron_ANDNOT(A_weights, A_limit)

    table_and = PrettyTable()
    table_and.field_names = ["X", "Y", "Output"]
    table_and.title = 'AND GATE'

    for i in range(4):
        x = int(input("Enter X: "))
        y = int(input("Enter Y: "))
        output_and = neuron_and.step_function([x, y])
        table_and.add_row([x, y, output_and])

    print(table_and)
