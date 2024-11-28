import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=1000, initial_weights=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = initial_weights
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.random.rand(n_features)
        self.bias = np.random.rand(1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return [self.activation_function(i) for i in linear_output]

# AND Gate Input Table
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])  # AND outputs

def main():
    learning_rate = float(input("Enter learning rate (e.g., 0.1): "))
    n_iterations = int(input("Enter number of iterations (e.g., 1000): "))

    initial_weights_input = input("Enter initial weights separated by space (e.g., 0.5 0.5): ")
    initial_weights = np.array([float(w) for w in initial_weights_input.split()])

    perceptron = Perceptron(learning_rate, n_iterations, initial_weights)
    perceptron.fit(X, y)

    print("Training completed. Weights:", perceptron.weights, "Bias:", perceptron.bias)

    while True:
        test_input = input("Enter test input as two values separated by space (or 'exit' to quit): ")
        if test_input.lower() == 'exit':
            break
        test_input = test_input.replace(',', ' ')  # Replace commas with spaces
        test_input = np.array([int(x) for x in test_input.split()]).reshape(1, -1)
        prediction = perceptron.predict(test_input)
        print("Output:", prediction[0])

if __name__ == "__main__":
    main()
