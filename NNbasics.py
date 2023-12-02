import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# A 2-D array of inputs
#X = [[1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#    [-1.5, 2.7, 3.3, -0,8]]

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    # Creating an array of random numbers of the dimension: (number of inputs, number of neurons), normalizing it 
    self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
    # Creating an array of zeroes for biases of the dimension(1, number of neuron)
    self.biases = np.zeroes((1, n_neurons))
  def forward(self, inputs):
    # Forwarding the dot product of inputs * weights + biases
    # y = mx + b
    # output = weights * inputs + biases
    self.output = np.dot(inputs, self.weight) + self.biases

# Activation Function: Rectified Linear Unit (ReLU)
class Activation_ReLU:
  def foward(self, inputs):
    # Basically outputting the input if it is >0, else outputting 0
    self.output = np.maximum(0, inputs)

# Software Activation for our output layer
# Software Activation is basically putting the values through the exponential function and then normalizing it.
class Activation_Softmax:
  def forward(self, inputs):
    # Inputs here are the ouputs of our model, hence it will be in batch form
    # exponential values is exponentiating the (inputs - max of that particular array)
    # That makes the the max value in a particular array be 0 and everything else will be less than 0.
    # Hence, mention axis=1, i.e. the max of that particular array, and keepdims=True for output in the same dimension
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    # Normalization: value/sum(values); Basically since the max value was exponentiated to 0, normalization will make it 1
    # And every other values a number between 0 to 1 but not 1
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
