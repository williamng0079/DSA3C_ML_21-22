import numpy as np
import random

# The tests can be run by navigating to the tests folder and executing:
#   python -m unittest discover

class Layer:
    def __init__(self, activators, layer_neurons, f): # layer constructor
       
        self.weightArray = 0.01 * np.random.randn(activators, layer_neurons) # create a 2d array (consist of random numbers) from number of layer neurons x number of activators to simulate an weight matrix
        self.biases = np.zeros((1, layer_neurons))        # initialise the bias array with values of zero for the time being
        self.functionSelector = f             

    
    def activationFunc(self, x):
        if self.functionSelector == 0:
            return x * 1                        # no function applied

        if self.functionSelector == 1:
            return x.clip(min=0)                 # ReLU function

        if self.functionSelector == 2:
            z = np.exp(-x)                      # Sigmoid function
            sigmoid = 1 / (1 + z)   
            return sigmoid
        else:
            print("no function selected")
            return x * 1

    def getMatrix(self):
        return self.weightArray

    def getBiasVector(self):
        return self.biases

    def getFunction(self):
        FunctionLists = "Functions: Sigmoid, ReLU"
        return FunctionLists

    def forward(self, inputArray):
        output = self.activationFunc(np.dot(inputArray, self.weightArray) + self.biases) # forwardfeeding process (method of traversing across layers) Matrix calculation
        return output

class NeuralNetwork:
    def __init__(self, inputArray):
        self.input = inputArray
        self.firstLayer = Layer(27, 21, 0)
        self.secondLayer = Layer(21, 21, 0)
        self.outputLayer = Layer(21, 27, 0)
    
    def getLayers(self):
        pass
        
    def propagate(self):
        self.firstOutput = self.firstLayer.forward(self.input)
        self.secondOutput = self.secondLayer.forward(self.firstOutput)
        self.output = self.outputLayer.forward(self.secondOutput)
        return self.output

class NNPlayer:
    @staticmethod
    def getSpecs():
        return (27,27)
        
    def __init__(self):#, Ms, Bs, Fs):
        
        self.allMoves = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]

    def getNN(self):
        pass

    def play(self):#, ownB, oppB, ownS, oppS, turn, gLen, pips):
        # All possible moves the NN player can choose from:
        BoardDiff = [-1, -1, -2,  0,  0, -1, -1,  0,  2, -1,  1,  0,  0,  1,  0,  2, -1,  1,  0, -1, -3,  1, -1, 3, 0,  0,  2]   #test nn calculation
        diffarray = np.asarray(BoardDiff)
        
        NNtest = NeuralNetwork(diffarray)
        
        NNoutput = NNtest.propagate()

        selectedmove = self.allMoves[NNoutput.argmax()]
        return selectedmove


test = NNPlayer()
print(test.play())