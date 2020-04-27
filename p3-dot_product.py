import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

"""
layer_outputs = []      # output of current layer
for neuron_weights, neuron_biases in zip(weights, biases):
    neuron_output = 0   # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_biases
    layer_outputs.append(neuron_output)

print(layer_outputs)
"""

# zip() --> combines 2 lists into list of lists element-wise

# 1D array = vector     # 2D array = matrix
""" 3D array = tensor --> an object that can be represented as an array !!! is not just an array but 
in the context of doing deep learning - a tensor is represented as an array, will work with them in the array form """

# dot product of two vectors results in a single scaler value


output = np.dot(weights, inputs) + biases   # weights comes first martix*vector multiplication
print(output)