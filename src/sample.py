import numpy as np

# The tests can be run by navigating to the tests folder and executing:
#   python -m unittest discover

class Layer:
    def __init__(self, M, B, f):
        pass

    def getMatrix(self):
        pass

    def getBiasVector(self):
        pass

    def getFunction(self):
        pass

    def forward(self, input):
        pass

class NeuralNetwork:
    def __init__(self, Ms, Bs, Fs):
        pass

    def getLayers(self):
        pass
        
    def propagate(self, input):
        pass

class NNPlayer:
    @staticmethod
    def getSpecs():
        return (1,1)
        
    def __init__(self, Ms, Bs, Fs):
        pass

    def getNN(self):
        pass

    def play(self, ownB, oppB, ownS, oppS, turn, gLen, pips):
        pass