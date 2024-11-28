import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=50):
        """
        Initialize the Adaline model.

        :param learning_rate: The step size for weight updates.
        :param epochs: Number of iterations over the training dataset.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.costs = []

    def fit(self, X, y):
        """
        Train the Adaline model on the given dataset.

        :param X: Input features, shape (n_samples, n_features).
        :param y: Target labels, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias
        
        for epoch in range(self.epochs):
            # Linear activation
            linear_output = np.dot(X, self.weights) + self.bias
            
            # Compute the error
            errors = y - linear_output
            
            # Update weights and bias
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * errors.sum()
            
            # Compute mean squared error (cost function)
            cost = (errors**2).mean() / 2
            self.costs.append(cost)
        
    def predict(self, X):
        """
        Predict output using the trained Adaline model.

        :param X: Input features, shape (n_samples, n_features).
        :return: Predicted labels, shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, -1)  # Convert to class labels

# Example Usage
if __name__ == "__main__":
    # Create a simple dataset (AND logic gate example)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
    y = np.array([-1, -1, -1, 1])  # Target labels
    
    # Initialize and train the Adaline model
    adaline = Adaline(learning_rate=0.01, epochs=50)
    adaline.fit(X, y)
    
    # Print weights and bias
    print("Weights:", adaline.weights)
    print("Bias:", adaline.bias)
    
    # Make predictions
    predictions = adaline.predict(X)
    print("Predictions:", predictions)
    
    # Print cost values for each epoch
    print("Cost over epochs:", adaline.costs)
