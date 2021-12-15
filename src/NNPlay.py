import numpy as np
import random

### IMPORTANT! PLEASE READ ###
# This piece of code is redundant as contents of it has been merged into GA_training.py for the mutation process to work properly
# This code is kept in the src folder to demonstrate past progress and code evolution


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
        pass
     
    
    def getLayers(self):
        pass
    

    def propagate(self, inputArray):
        pass
    
    

class NNPlayer:
    @staticmethod
    def getSpecs():
        return (27,27)
        
    def __init__(self, copy = False):#, Ms, Bs, Fs):
        # intialise all the possible choices
        self.allMoves = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
        #self.NNplayer1 = NeuralNetwork()

        self._rewards = 0
        self._weights = []
        self._biases = []
        self._functions = []
        self._layers = []

        if (not copy):
            
            self.firstLayer = Layer(27, 54, 3)           
            self._weights.append(self.firstLayer.getMatrix())
            self._biases.append(self.firstLayer.getBiasVector())
            self._functions.append(self.firstLayer.getFunction())

            self.secondLayer = Layer(54, 54, 3)
            self._weights.append(self.secondLayer.getMatrix())
            self._biases.append(self.secondLayer.getBiasVector())
            self._functions.append(self.secondLayer.getFunction())

            self.outputLayer = Layer(54, 27, 0)
            self._weights.append(self.outputLayer.getMatrix())
            self._biases.append(self.outputLayer.getBiasVector())
            self._functions.append(self.outputLayer.getFunction())

    def getNN(self):
        self._layers = [self._weights, self._biases, self._functions]
        return self._layers

    
    
    def propagate(self, inputArray):
        self.firstOutput = self.firstLayer.forward(inputArray)
        self.secondOutput = self.secondLayer.forward(self.firstOutput)
        self.resultOutput = self.outputLayer.forward(self.secondOutput)
        
        return self.resultOutput
    
    
    def give_rewards(self, r):
        self._rewards = r
        return self._rewards


    def mutate(self):
        mutation_prob =  0.01               # set the probability of a mutation ocurring to 1%
        new_player = NNPlayer(copy = True) 

        # copy the parameter of existing player
        for w in self._weights:                 # iterate through the list of weights and apply mutation for the new player
            w_mutationArray = np.random.rand(w.shape[0], w.shape[1]) # generate a matrix with the same weight dimension 
            apply_mutation = np.where(w_mutationArray < mutation_prob, (np.random.rand() - 0.5)/2 , 0)  # mutation condition: when the element inside the matrix is smaller than the mutation prob, change to that element will occur, otherwise set the element to 0 if the condition is not met
            updated_w = w + apply_mutation

            new_player._weights.append(updated_w)       # add the mutated weight the new player

        for b in self._biases:                     # mutate the bias value fot the new player
            b_mutationArray = np.random.rand(b.shape[0], 1)     # generate array with same bias dimension
            apply_b_mutation = np.where(b_mutationArray < mutation_prob, (np.random.rand() - 0.5)/2, 0)
            updated_b = b + apply_b_mutation

            new_player._biases.append(updated_b)

        return new_player

    def play(self, ownB, oppB, ownS, oppS, turn, gLen, pips):

        #BoardDiff = [-1, -1, -2,  0,  0, -1, -1,  0,  2, -1,  1,  0,  0,  1,  0,  2, -1,  1,  0, -1, -3,  1, -1, 3, 0,  0,  2]   #test nn calculation
        ownBoardArray = np.array(ownB).flatten() # converts to 1D array for the ease of calculation 
        oppBoardArray = np.array(oppB).flatten()
        BoardDiff = np.subtract(oppBoardArray, ownBoardArray) # (calculate the differences between red and green board and generates 27 input nodes to feed into the NN)
        
        diffarray = np.asarray(BoardDiff)
        aligned = diffarray.reshape((-1, 1))                    # convert the shape of the array to demonstrate the NN matrix multiplication alignment
        
    
        NNoutput = self.propagate(aligned)
        
    
        selectedmove = self.allMoves[NNoutput.argmax()]     # use the position of output neuron with highest value as index to decide the move to choose from the .allmoves pool
        #print("List of decisions:", NNoutput)
        #print("Move Selected:", selectedmove)
        #print(self.getNN()[2])                               # check the randomly selected activation functions
        softmax_check = sum(NNoutput)                       # check if softmax is working as intended, sum should be 1
        #print(softmax_check)

        #print(self.getNN())

        return selectedmove


#test = NNPlayer()
#print(test.play())