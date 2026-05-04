import numpy as np
import matplotlib.pyplot as plt

import neuralnetwork
import activations
import random

np.random.seed(0)
random.seed(0)

nn = neuralnetwork.NeuralNetwork(1, [10, 10, 10, 10], 1, .2, activations.Tanh, activations.Linear)
mse_errors = []
def test_func(x):
    return np.sin(2 * np.pi * x)
data = [[i / 1000, test_func(i / 1000)] for i in range(1000 + 1)]

training_data = data * 200
random.shuffle(training_data)
for i in range(len(training_data)):
    # print("========================", i)
    nn.train(training_data[i][0], training_data[i][1])
    if i % 1000 == 0:
        if i != 0:
            nn.lr *= .995
        print(f"training: {i}, lr: {nn.lr}")
        current_error = nn.getError(data)
        mse_errors.append(np.log10(current_error / len(data)))

# print(mse_errors)
# print(nn.layer_outs)

print("Total error:" + str(nn.getError(data) / len(data)))
plt.plot(mse_errors)
plt.show()
print(nn.check([0 / 1000]))
output = [0 for i in range(1000 + 1)]
for i in range(1000 + 1):
    output[i] = nn.check([i / 1000])[0][0]

plt.plot(output)
plt.plot([test_func(i / 1000) for i in range(1000 + 1)])
plt.show()