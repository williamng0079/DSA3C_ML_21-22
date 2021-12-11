import numpy as np
import random

from numpy.lib.function_base import diff

# The tests can be run by navigating to the tests folder and executing:
#   python -m unittest discover

class Layer:
    def __init__(self, n_inputs, neurons, f): # layer constructor
       
        self.weightArray = np.random.rand(neurons, n_inputs) * 2 - 1# create a matrix (consist of random numbers) from number of neurons x number of activators to simulate an weight matrix
        self.biases = np.random.rand(neurons, 1) * 2 - 1       # initialise the bias array with values of zero for the time being
        self.functionSelector = f             

    
    def activationFunc(self, x):
        if self.functionSelector == 0:
            return x                        # no function applied

        if self.functionSelector == 1:
            ReLU = x.clip(min = 0)
            return ReLU                          # ReLU function

        if self.functionSelector == 2:
            z = np.exp(-x)                      # Sigmoid function
            Sigmoid = 1 / (1 + z)   
            return Sigmoid
        else:
            return x 

    def getMatrix(self):
        return self.weightArray

    def getBiasVector(self):
        return self.biases

    def getFunction(self):
        if self.functionSelector == 1:
            currentFunc = "ReLU"
            return currentFunc
        
        elif self.functionSelector == 2:
            currentFunc = "Sigmoid"
            return currentFunc
        
        else:
            currentFunc = "no function applied"
            return currentFunc

    def forward(self, inputArray):
        outputArray = self.activationFunc(np.matmul(self.weightArray, inputArray) + self.biases) # forwardfeeding process (method of traversing across layers) Matrix calculation
        return outputArray

class NeuralNetwork:
    def __init__(self):
        
        self._weights = []
        self._biases = []
        self._functions = []
        self._layers = []
        self.firstLayer = Layer(27, 21, 1)
        self.secondLayer = Layer(21, 21, 1)
        self.outputLayer = Layer(21, 27, 1)

        self._weights.append(self.firstLayer.getMatrix())
        self._weights.append(self.secondLayer.getMatrix())
        self._weights.append(self.outputLayer.getMatrix())

        self._biases.append(self.firstLayer.getBiasVector())
        self._biases.append(self.secondLayer.getBiasVector())
        self._biases.append(self.outputLayer.getBiasVector())

        self._functions.append(self.firstLayer.getFunction())
        self._functions.append(self.secondLayer.getFunction())
        self._functions.append(self.outputLayer.getFunction())

    def getLayers(self):
        pass
        
    def propagate(self, inputArray):
        self.firstOutput = self.firstLayer.forward(inputArray)
        self.secondOutput = self.secondLayer.forward(self.firstOutput)
        self.resultOutput = self.outputLayer.forward(self.secondOutput)

        #print(self._weights)
        #print(self._biases)
        #print(self._functions)

        return self.resultOutput

class NNPlayer:
    @staticmethod
    def getSpecs():
        return (27,27)
        
    def __init__(self):#, Ms, Bs, Fs):
        # intialise all the possible choices
        self.allMoves = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
        self.NNplayer1 = NeuralNetwork()

    def getNN(self):
        pass

    def play(self, ownB, oppB, ownS, oppS, turn, gLen, pips):

        #BoardDiff = [-1, -1, -2,  0,  0, -1, -1,  0,  2, -1,  1,  0,  0,  1,  0,  2, -1,  1,  0, -1, -3,  1, -1, 3, 0,  0,  2]   #test nn calculation
        ownBoardArray = np.array(ownB).flatten() # converts to 1D array for the ease of calculation 
        oppBoardArray = np.array(oppB).flatten()
        BoardDiff = np.subtract(ownBoardArray, oppBoardArray) # (calculate the differences between red and green board and generates 27 input nodes)
        
        diffarray = np.asarray(BoardDiff)
        aligned = diffarray.reshape((-1, 1))
        
        
        
        NNoutput = self.NNplayer1.propagate(aligned)
        
        reshapedOutput = NNoutput.reshape(-1)
        selectedmove = self.allMoves[NNoutput.argmax()]
        
        print("List of decisions:", reshapedOutput)
        print("Move Selected:", selectedmove)
        return selectedmove


#test = NNPlayer()
#print(test.play())