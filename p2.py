# 1 neuron 3 inputs
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
print(output)

# 1 neuron 4 inputs
hid_out = [1, 2, 3, 2.5]
out_weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

out_out = hid_out[0] * out_weights[0] + hid_out[1] * out_weights[1] + hid_out[2] * out_weights[2] + \
          hid_out[3] * out_weights[3] + bias
print(out_out)


# 3 neurons 4 inputs

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output3 = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
           inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
           inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

print(output3)

# figure out how best to tune out those weights and biases ---> backpropagation + gradient descent
# arrays vs tensors