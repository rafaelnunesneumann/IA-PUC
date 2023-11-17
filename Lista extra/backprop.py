import numpy as np

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Algoritmo de Backpropagation
def backpropagation(X, y, num_epochs, learning_rate, num_inputs, hidden_size):
    input_size = num_inputs
    output_size = 1
    np.random.seed(42)

    # Inicialização de pesos
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)

    for epoch in range(num_epochs):
        # Forward pass
        hidden_input = np.dot(X, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output_input = np.dot(hidden_output, weights_hidden_output)
        predicted_output = sigmoid(output_input)

        # Backward pass
        output_error = y - predicted_output
        output_delta = output_error * sigmoid_derivative(predicted_output)

        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

        # Atualização de pesos
        weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
        weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output

# Função para testar a RNA treinada
def predict(X, weights_input_hidden, weights_hidden_output):
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, weights_hidden_output)
    predicted_output = sigmoid(output_input)

    return predicted_output

# Função principal
def main():
    # Entrada do usuário
    num_inputs = int(input("Digite o número de entradas desejado (por exemplo, 2 ou 10): "))
    logic_function = input("Digite a função lógica desejada (AND, OR, ou XOR): ").upper()

    # Geração de dados de treinamento
    X_train = np.random.randint(0, 2, size=(1000, num_inputs))

    if logic_function == "AND":
        y_train = np.all(X_train, axis=1).astype(int)
    elif logic_function == "OR":
        y_train = np.any(X_train, axis=1).astype(int)
    elif logic_function == "XOR":
        y_train = np.sum(X_train, axis=1) % 2

    # Treinamento da rede neural
    learning_rate = 0.1
    num_epochs = 10000
    hidden_size = 5  # Pode ser ajustado conforme necessário

    weights_input_hidden, weights_hidden_output = backpropagation(
        X_train, y_train.reshape(-1, 1), num_epochs, learning_rate, num_inputs, hidden_size
    )

    # Teste da rede neural
    X_test = np.array([[int(input(f"Digite a {i + 1} entrada: ")) for i in range(num_inputs)]])
    predicted_output = predict(X_test, weights_input_hidden, weights_hidden_output)

    print(f"Resultado previsto para as entradas {X_test}: {round(predicted_output[0][0])}")

if __name__ == "__main__":
    main()
