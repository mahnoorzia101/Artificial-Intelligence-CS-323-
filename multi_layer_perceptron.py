import random
import math


class Neuron():
    '''
        A conceptual Neuron hat can be trained using a
        fit and predict methodology, without any library
    '''

    def __init__(self, position_in_layer, is_output_neuron=False):
        self.weights = []
        self.inputs = []
        self.output = None

        # This is used for the backpropagation update
        self.updated_weights = []
        # This is used to know how to update the weights
        self.is_output_neuron = is_output_neuron
        # This delta is used for the update at the backpropagation
        self.delta = None
        # This is used for the backpropagation update
        self.position_in_layer = position_in_layer

    def attach_to_output(self, neurons):
        '''
            Helper function to store the reference of the other neurons
            To this particular neuron (used for backpropagation)
        '''

        self.output_neurons = neurons

    def sigmoid(self, x):
        '''
            simple sigmoid function (logistic) used for the activation
        '''
        return 1 / (1 + math.exp(-x))

    def init_weights(self, num_input):
        '''
            This is used to setup the weights when we know how many inputs there is for
            a given neuron
        '''

        # Randomly initalize the weights
        for i in range(num_input + 1):
            self.weights.append(random.uniform(0, 1))

    def predict(self, row):
        '''
            Given a row of data it will predict what the output should be for
            this given neuron. We can have many input, but only one output for a neuron
        '''

        # Reset the inputs
        self.inputs = []

        # We iterate over the weights and the features in the given row
        activation = 0
        for weight, feature in zip(self.weights, row):
            self.inputs.append(feature)
            activation = activation + weight * feature

        self.output = self.sigmoid(activation)
        return self.output

    def update_neuron(self):
        '''
            Will update a given neuron weights by replacing the current weights
            with those used during the backpropagation. This need to be done at the end of the
            backpropagation
        '''

        self.weights = []
        for new_weight in self.updated_weights:
            self.weights.append(new_weight)

    def calculate_update(self, learning_rate, target):
        '''
            This function will calculate the updated weights for this neuron. It will first calculate
            the right delta (depending if this neuron is a ouput or a hidden neuron), then it will
            calculate the right updated_weights. It will not overwrite the weights yet as they are needed
            for other update in the backpropagation algorithm.
        '''

        if self.is_output_neuron:
            # Calculate the delta for the output
            self.delta = (self.output - target) * self.output * (1 - self.output)
        else:
            # Calculate the delta
            delta_sum = 0
            # this is to know which weights this neuron is contributing in the output layer
            cur_weight_index = self.position_in_layer
            for output_neuron in self.output_neurons:
                delta_sum = delta_sum + (output_neuron.delta * output_neuron.weights[cur_weight_index])

            # Update this neuron delta
            self.delta = delta_sum * self.output * (1 - self.output)

        # Reset the update weights
        self.updated_weights = []

        # Iterate over each weight and update them
        for cur_weight, cur_input in zip(self.weights, self.inputs):
            gradient = self.delta * cur_input
            new_weight = cur_weight - learning_rate * gradient
            self.updated_weights.append(new_weight)


class Layer():
    '''
        Layer is modelizing a layer in the fully-connected-feedforward neural network architecture.
        It will play the role of connecting everything together inside and will be doing the backpropagation
        update.
    '''

    def __init__(self, num_neuron, is_output_layer=False):

        # Will create that much neurons in this layer
        self.is_output_layer = is_output_layer
        self.neurons = []
        for i in range(num_neuron):
            # Create neuron
            neuron = Neuron(i, is_output_neuron=is_output_layer)
            self.neurons.append(neuron)

    def attach(self, layer):
        '''
            This function attach the neurons from this layer to another one
            This is needed for the backpropagation algorithm
        '''
        # Iterate over the neurons in the current layer and attach
        # them to the next layer
        for in_neuron in self.neurons:
            in_neuron.attach_to_output(layer.neurons)

    def init_layer(self, num_input):
        '''
            This will initialize the weights of each neuron in the layer.
            By giving the right num_input it will spawn the right number of weights
        '''

        # Iterate over each of the neuron and initialize
        # the weights that connect with the previous layer
        for neuron in self.neurons:
            neuron.init_weights(num_input)

    def predict(self, row):
        '''
            This will calcualte the activations for the full layer given the row of data
            streaming in.
        '''
        row.append(1)  # need to add the bias
        activations = [neuron.predict(row) for neuron in self.neurons]
        return activations


class MultiLayerPerceptron():
    '''
        We will be creating the multi-layer perceptron with only two layer:
        an input layer, a perceptrons layer and a one neuron output layer which does binary classification
    '''

    def __init__(self, learning_rate=0.01, num_iteration=100):

        # Layers
        self.layers = []

        # Training parameters
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration

    def add_output_layer(self, num_neuron):
        '''
            This helper function will create a new output layer and add it to the architecture
        '''
        self.layers.insert(0, Layer(num_neuron, is_output_layer=True))

    def add_hidden_layer(self, num_neuron):
        '''
            This helper function will create a new hidden layer, add it to the architecture
            and finally attach it to the front of the architecture
        '''
        # Create an hidden layer
        hidden_layer = Layer(num_neuron)
        # Attach the last added layer to this new layer
        hidden_layer.attach(self.layers[0])
        # Add this layers to the architecture
        self.layers.insert(0, hidden_layer)

    def update_layers(self, target):
        '''
            Will update all the layers by calculating the updated weights and then updating
            the weights all at once when the new weights are found.
        '''
        # Iterate over each of the layer in reverse order
        # to calculate the updated weights
        for layer in reversed(self.layers):

            # Calculate update the hidden layer
            for neuron in layer.neurons:
                neuron.calculate_update(self.learning_rate, target)

                # Iterate over each of the layer in normal order
        # to update the weights
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_neuron()

    def fit(self, X, y):
        '''
            Main training function of the neural network algorithm. This will make use of backpropagation.
            It will use stochastic gradient descent by selecting one row at random from the dataset and
            use predict to calculate the error. The error will then be backpropagated and new weights calculated.
            Once all the new weights are calculated, the whole network weights will be updated
        '''
        num_row = len(X)
        num_feature = len(X[0])  # Here we assume that we have a rectangular matrix

        # Init the weights throughout each of the layer
        self.layers[0].init_layer(num_feature)

        for i in range(1, len(self.layers)):
            num_input = len(self.layers[i - 1].neurons)
            self.layers[i].init_layer(num_input)

        # Launch the training algorithm
        for i in range(self.num_iteration):

            # Stochastic Gradient Descent
            r_i = random.randint(0, num_row - 1)
            row = X[r_i]  # take the random sample from the dataset
            yhat = self.predict(row)
            target = y[r_i]

            # Update the layers using backpropagation
            self.update_layers(target)

            # At every 100 iteration we calculate the error
            # on the whole training set
            if i % 1000 == 0:
                total_error = 0
                for r_i in range(num_row):
                    row = X[r_i]
                    yhat = self.predict(row)
                    error = (y[r_i] - yhat)
                    total_error = total_error + error ** 2
                mean_error = total_error / num_row
                print(f"Iteration {i} with error = {mean_error}")

    def predict(self, row):
        '''
            Prediction function that will take a row of input and give back the output
            of the whole neural network.
        '''

        # Gather all the activation in the hidden layer

        activations = self.layers[0].predict(row)
        for i in range(1, len(self.layers)):
            activations = self.layers[i].predict(activations)

        outputs = []
        for activation in activations:
            # Decide if we output a 1 or 0
            if activation >= 0.5:
                outputs.append(1.0)
            else:
                outputs.append(0.0)

        # We currently have only One output allowed
        return outputs[0]


# XOR function (one or the other but not both)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Init the parameters for the network
clf = MultiLayerPerceptron(learning_rate=0.1, num_iteration=100000)
# Create the architecture backward
clf.add_output_layer(num_neuron=1)
clf.add_hidden_layer(num_neuron=3)
clf.add_hidden_layer(num_neuron=2)
# Train the network
clf.fit(X, y)
print("Expected 0.0, got: ",clf.predict([0,0]))
print("Expected 1.0, got: ",clf.predict([0,1]))
print("Expected 1.0, got: ",clf.predict([1,0]))
print("Expected 0.0, got: ",clf.predict([1,1]))