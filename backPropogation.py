import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to initialize weights
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.rand(input_size, hidden_size) * 2 - 1  # Random weights between -1 and 1
    b1 = np.random.rand(1, hidden_size) * 2 - 1
    W2 = np.random.rand(hidden_size, output_size) * 2 - 1
    b2 = np.random.rand(1, output_size) * 2 - 1
    return W1, b1, W2, b2

# Function to train the neural network
def train(X, y, W1, b1, W2, b2, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, W1) + b1
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, W2) + b2
        predicted_output = sigmoid(output_layer_input)

        # Calculate error
        error = y - predicted_output

        # Backpropagation
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(W2.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Update weights and biases
        W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden_layer) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return W1, b1, W2, b2

# Function to predict the output for new inputs
def predict(X, W1, b1, W2, b2):
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

# Main function
def main():
    # XOR input and output
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # User inputs
    input_size = 2
    hidden_size = int(input("Enter number of neurons in hidden layer: "))
    output_size = 1

    # Initial weights and biases
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

    learning_rate = float(input("Enter learning rate (e.g., 0.1): "))
    epochs = int(input("Enter number of epochs (e.g., 10000): "))

    # Train the neural network
    W1, b1, W2, b2 = train(X, y, W1, b1, W2, b2, learning_rate, epochs)

    # Testing the model
    print("\nFinal outputs after training:")
    for i in range(X.shape[0]):
        predicted_output = predict(X[i].reshape(1, -1), W1, b1, W2, b2)
        print(f"Input: {X[i]} - Predicted Output: {predicted_output} - Rounded: {np.round(predicted_output)}")

    # Allow user to test with new inputs
    while True:
        test_input = input("\nEnter test input (comma separated, e.g., 0,1) or 'exit' to quit: ")
        if test_input.lower() == 'exit':
            break
        test_input = np.array([float(i) for i in test_input.split(',')]).reshape(1, -1)
        predicted_output = predict(test_input, W1, b1, W2, b2)
        print(f"Test Input: {test_input.flatten()} - Predicted Output: {predicted_output} - Rounded: {np.round(predicted_output)}")

if __name__ == "__main__":
    main()
