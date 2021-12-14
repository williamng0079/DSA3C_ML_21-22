import numpy as np
import random



# The tests can be run by navigating to the tests folder and executing:
#   python -m unittest discover


class Layer:
    def __init__(self, n_inputs, neurons, f): # layer constructor
       
        self.weightArray = np.random.rand(neurons, n_inputs) * 2 - 1 # create a matrix (consist of random numbers) from number of neurons x number of activators to simulate an weight matrix
        self.biases = np.random.rand(neurons, 1) * 2 - 1     # initialise the bias array with values of zero for the time being
        self.functionSelector = f             

    
    def activationFunc(self, x):
        
        if self.functionSelector == 0:          # Softmax (only applicable in output layer)
            m = np.exp(x)
            return m/m.sum(len(x.shape)-1)
            
                                   
        elif self.functionSelector == 1:
            ReLU = x.clip(min = 0)
            return ReLU                          # ReLU 

        elif self.functionSelector == 2:         # Sigmoid
            return 1/(1+np.exp(-x))

        elif self.functionSelector == 3:            # Leaky ReLU
            return np.where(x > 0, x, x * 0.01)     
        
        else:
            return x 
          


    def getMatrix(self):
        return self.weightArray

    def getBiasVector(self):
        return self.biases

    def getFunction(self):
        if self.functionSelector == 0:
            currentFunc = "Softmax"
            return currentFunc

        elif self.functionSelector == 1:
            currentFunc = "ReLU"
            return currentFunc
        
        elif self.functionSelector == 2:
            currentFunc = "Sigmoid"
            return currentFunc

        elif self.functionSelector == 3:
            currentFunc = "Leaky ReLU"
            return currentFunc
        
        else:
            currentFunc = "no function applied"
            return currentFunc

    def forward(self, inputArray):
        outputArray = (np.matmul(self.weightArray, inputArray) + self.biases) # forwardfeeding process (method of traversing across layers) Matrix calculation
        if outputArray.shape[0] == 27:                        # check if the layer is the output layer
            softmax_activated = self.activationFunc(outputArray.reshape(-1))
            return softmax_activated
        else:
            activated_output = self.activationFunc(outputArray)
            return activated_output

class NeuralNetwork:
    def __init__(self):
        self._rewards = 0
        self._weights = []
        self._biases = []
        self._functions = []
        self._layers = []
        self.firstLayer = Layer(27, 25, 3)
        self.secondLayer = Layer(25, 25, 3)
        self.outputLayer = Layer(25, 27, 0)

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
        self._layers = [self._weights, self._biases, self._functions]
        return self._layers


    def propagate(self, inputArray):
        self.firstOutput = self.firstLayer.forward(inputArray)
        self.secondOutput = self.secondLayer.forward(self.firstOutput)
        self.resultOutput = self.outputLayer.forward(self.secondOutput)
        
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
        return self.NNplayer1.getLayers()

    def give_rewards(self, r):
        self._reward = r
        return self._reward

    def play(self, ownB, oppB, ownS, oppS, turn, gLen, pips):

        #BoardDiff = [-1, -1, -2,  0,  0, -1, -1,  0,  2, -1,  1,  0,  0,  1,  0,  2, -1,  1,  0, -1, -3,  1, -1, 3, 0,  0,  2]   #test nn calculation
        ownBoardArray = np.array(ownB).flatten() # converts to 1D array for the ease of calculation 
        oppBoardArray = np.array(oppB).flatten()
        BoardDiff = np.subtract(oppBoardArray, ownBoardArray) # (calculate the differences between red and green board and generates 27 input nodes)
        
        diffarray = np.asarray(BoardDiff)
        aligned = diffarray.reshape((-1, 1))
        
        
        
        NNoutput = self.NNplayer1.propagate(aligned)
        
    
        
        selectedmove = self.allMoves[NNoutput.argmax()]     # use the position of output neuron with highest value as index to decide the move to choose from the pool
        #print("List of decisions:", NNoutput)
        #print("Move Selected:", selectedmove)

        softmax_check = sum(NNoutput)                       # check if softmax is working as intended, sum should be 1
        #print(softmax_check)

        #print(self.getNN())

        return selectedmove


#test = NNPlayer()
#print(test.play())