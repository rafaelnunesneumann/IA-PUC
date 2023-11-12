import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, epochs=1000):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_data, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * np.array(inputs)
                self.weights[0] += self.learning_rate * (label - prediction)

    def test(self, test_data, labels):
        predictions = [self.predict(inputs) for inputs in test_data]
        accuracy = np.mean(predictions == labels)
        return accuracy
    
    def verificaAnd(entrada):
        return all(x == 1 for x in entrada)
    
    def verificaOr(entrada):
        return any(x == 1 for x in entrada)
    
    def verificaXor(entrada):
        return sum(x == 1 for x in entrada) == 1

def main():
    # Testando a função AND com N entradas
    inputs = int(input("Digite o numero de entradas: "))
    test_cases = int(input("Digite o número de casos de teste: "))
    and_data = []
    and_labels = []
    for i in range(test_cases):
        test_input = [int(x) for x in input(f"Digite as entradas para a função AND: ").split()]
        if len(test_input) > 0 and Perceptron.verificaAnd(test_input):
            and_labels.append(1)
        else:
            and_labels.append(0)
        and_data.append(test_input)

    perceptron_and = Perceptron(num_inputs=inputs)
    perceptron_and.train(and_data, and_labels)

    print(f"Teste AND com {inputs} entradas:")
    print("Saídas previstas:", [perceptron_and.predict(inputs) for inputs in and_data])
    accuracy_and = perceptron_and.test(and_data, and_labels)
    print("Acurácia:", accuracy_and)

    # Testando a função OR com N entradas
    or_data = []
    or_labels = []
    for i in range(test_cases):
        test_input = [int(x) for x in input(f"Digite as entradas para a função OR: ").split()]
        if len(test_input) > 0 and Perceptron.verificaOr(test_input):
            or_labels.append(1)
        else:
            or_labels.append(0)
        or_data.append(test_input)

    perceptron_or = Perceptron(num_inputs=inputs)
    perceptron_or.train(or_data, or_labels)

    print(f"\nTeste OR com {inputs} entradas:")
    print("Saídas previstas:", [perceptron_or.predict(inputs) for inputs in or_data])
    accuracy_or = perceptron_or.test(or_data, or_labels)
    print("Acurácia:", accuracy_or)

    # Demonstrando que o Perceptron não resolve o XOR
    xor_data = []
    xor_labels = []
    for i in range(test_cases):
        test_input = [int(x) for x in input(f"Digite as entradas para a função XOR: ").split()]
        if len(test_input) > 0 and Perceptron.verificaXor(test_input):
            xor_labels.append(1)
        else:
            xor_labels.append(0)
        xor_data.append(test_input)

    perceptron_xor = Perceptron(num_inputs=inputs)
    perceptron_xor.train(xor_data, xor_labels)

    print(f"\nTeste XOR com {inputs} entradas:")
    print("Saídas previstas:", [perceptron_xor.predict(inputs) for inputs in xor_data])
    accuracy_xor = perceptron_xor.test(xor_data, xor_labels)
    print("Acurácia:", accuracy_xor)

if __name__ == "__main__":
    main()
