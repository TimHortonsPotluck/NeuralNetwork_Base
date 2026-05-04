import numpy as np
import copy
import activations

class Layer:
    def __init__(self, in_nodes, out_nodes, activation, params=None):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation
        self.z = 0
        if params is None:
            self.weights = np.random.rand(out_nodes, in_nodes) * 2 - 1
            self.biases = np.random.rand(out_nodes, 1) * 2 - 1
        else:
            self.weights = params[0]
            self.biases = params[1]
    
    def feed(self, inputs):
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        elif isinstance(inputs, np.ndarray):
            pass
        else:
            return
        inputs = np.array(inputs).reshape((len(inputs), 1))
        self.z = np.matmul(self.weights, inputs) + self.biases
        return self.activation.func(self.z)
    
    def copyLayer(self):
        copyl = Layer(self.in_nodes, self.out_nodes, self.activation_func)
        copyl.weights = np.copy(self.weights)
        copyl.biases = np.copy(self.biases)
        return copyl

class NeuralNetwork:
    
    def __init__(self, Inodes, Hlayers, Onodes, lr, activation, output_activation=None, layer_params=None):
        self.Inodes = Inodes
        self.Hlayers = Hlayers # Hlayers is list where the length is number of layers, elems are # of nodes
        self.num_hiddens = len(Hlayers)
        self.num_layers = self.num_hiddens + 1
        self.Onodes = Onodes
        self.lr = lr
        self.activation = activation
        if output_activation is None:
            self.output_activation = activation
        else:
            self.output_activation = output_activation
        self.layers = []
        self.layer_outs = [0] * (self.num_hiddens + 1)
        if layer_params is None:
            self.layers.append(Layer(Inodes, Hlayers[0], activation))
            for i in range(self.num_hiddens - 1):
                self.layers.append(Layer(Hlayers[i], Hlayers[i + 1], activation))
            self.layers.append(Layer(Hlayers[-1], Onodes, activation))
        else:
            self.layers.append(Layer(Inodes, Hlayers[0], activation, params=layer_params[0]))
            for i in range(self.num_hiddens - 1):
                self.layers.append(Layer(Hlayers[i], Hlayers[i + 1], activation, params=layer_params[i + 1]))
            self.layers.append(Layer(Hlayers[-1], Onodes, output_activation, params=layer_params[-1]))
        
    def copyNN(self):
        copynn = copy.deepcopy(self)
        return copynn
    
    def saveNNToFile(self, filename):
        c = self.copyNN()
        data = []
        for i in range(c.num_layers):
            data.append(c.layers[i].weights)
            data.append(c.layers[i].biases)
        data.append([c.Inodes, c.Hlayers, c.Onodes, c.activation_func, c.lr])
        np.save(filename, data)
    
    def loadNNFromFile(filename, lr=None): #lr=None means that the lr of the new network will be taken from the saved network
        dataloaded = np.load(filename, allow_pickle=True)
        nn_params = []
        for i in range(0, len(dataloaded[:-1]), 2):
            nn_params.append((dataloaded[i], dataloaded[i + 1]))
        new = NeuralNetwork(dataloaded[-1][0], 
                            dataloaded[-1][1], 
                            dataloaded[-1][2], 
                            dataloaded[-1][3], 
                            (lr, dataloaded[-1][4])[lr is None], #this means "take the second element if lr is None, else use lr"
                            layer_params=nn_params)
        return new
    
    def setLR(self, lr):
        self.lr = lr
    
    def setOutDiffsFunction(self, output, target): # o_diffs means out_diffs
        return output - target
    
    def feedForward(self, inputs):
        self.layer_outs[0] = self.layers[0].feed(inputs)
        for i in range(self.num_hiddens - 1 + 1): # the +1 is for the output layer
            self.layer_outs[i + 1] = self.layers[i + 1].feed(self.layer_outs[i])
        return self.layer_outs
    
    def backProp(self, inputs, targets):
        inputs = np.array(inputs)
        targets = np.array(targets)
        inputs = np.reshape(inputs, (self.Inodes, 1))
        targets = np.reshape(targets, (self.Onodes, 1))
        outs = self.feedForward(inputs)
        out_diffs = outs[-1] - targets
        self.output_errors = .5 * out_diffs * out_diffs
        self.output_errors_der = out_diffs # the derivative of output_errors
        hidden_errors = [0] * self.num_hiddens
        weight_deltas = [0] * (self.num_hiddens + 1)
        bias_deltas = [0] * (self.num_hiddens + 1)
        hidden_errors[-1] = np.matmul(self.layers[-1].weights.T, self.output_errors_der) * self.layers[-2].activation.derivative(self.layers[-2].z)
        for i in range(1, self.num_hiddens):
            hidden_errors[-i - 1] = np.matmul(self.layers[-i - 1].weights.T, hidden_errors[-i]) * self.layers[-i - 2].activation.derivative(self.layers[-i - 2].z)
        weight_deltas[-1] = -self.lr * np.matmul(self.output_errors_der * self.output_activation.derivative(self.layers[-1].z), outs[-2].T)
        bias_deltas[-1] = -self.lr * self.output_errors_der * self.output_activation.derivative(self.layers[-1].z)
        
        for i in range(2, self.num_hiddens + 1):
            weight_deltas[-i] = -self.lr * np.matmul(hidden_errors[-i + 1], outs[-i - 1].T)
            bias_deltas[-i] = -self.lr * hidden_errors[-i + 1]
        weight_deltas[0] = -self.lr * np.matmul(hidden_errors[0], inputs.T)
        bias_deltas[0] = -self.lr * hidden_errors[0]
        
        return weight_deltas, bias_deltas
    
    def train(self, inputs, targets):
        
        weight_deltas, bias_deltas = self.backProp(inputs, targets)
        for i in range(self.num_hiddens + 1):
            self.layers[i].weights += weight_deltas[i]
            self.layers[i].biases += bias_deltas[i]
        
    def check(self, inputs):
        return self.feedForward(np.array(inputs))[-1]
    
    def mutate(self, sd):
        for l in self.layers:
            l.weights += sd * np.random.randn(l.out_nodes, l.in_nodes)
            l.biases += sd * np.random.randn(l.out_nodes, 1)
    
    def random_changes(self, chance, max_amt):
        for l in self.layers:
            for j in range(l.out_nodes):
                if np.random.rand() < chance:
                    l.biases[j][0] += max_amt * (np.random.rand() * 2 - 1)
                for i in range(l.in_nodes):
                    if np.random.rand() < chance:
                        l.weights[j][i] += max_amt * (np.random.rand() * 2 - 1)
    
    def getError(self, inputs_array, targets_array):
        if len(inputs_array) != len(targets_array):
            print("arrays aren't same size dumdum")
        total_error = 0
        for i in range(len(inputs_array)):
            inputs = np.array(inputs_array[i])
            targets = np.array(targets_array[i])
            inputs = np.reshape(inputs, (self.Inodes, 1))
            targets = np.reshape(targets, (self.Onodes, 1))
            outs = self.feedForward(inputs)
            out_diffs = outs[-1] - targets
            total_error += np.sum(.5 * out_diffs * out_diffs)
        return total_error
    
    def getError(self, data_array):
        total_error = 0
        for i in range(len(data_array)):
            inputs = np.array(data_array[i][0])
            targets = np.array(data_array[i][1])
            inputs = np.reshape(inputs, (self.Inodes, 1))
            targets = np.reshape(targets, (self.Onodes, 1))
            outs = self.feedForward(inputs)
            out_diffs = outs[-1] - targets
            total_error += np.sum(.5 * out_diffs * out_diffs)
        return total_error