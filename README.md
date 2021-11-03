# Unit tests for DSA3C coursework

Here there is information on how to structure your code and the coursework repository.

## Getting started

Go to GitHub and select the option to start a new repository. Name it DSA3C_ML_21-22 and do not add to it any of the files that GitHub suggests (i.e. do not add `README.md`, `.gitignore` and `LICENCE`). After clicking **Create repository** scroll down and find the **Import code** option. Select this and then in the url text field paste `https://github.com/mazaja/CompromiseCourseworkTests` and select **Begin import**. Now you can clone the new repository to your local machine and start working. **You should develop your code for this assignment in the `src` folder of the repository**.

## Code specifications

Your code has to be able to pass the tests in this repository. In order to do that you need the following:

### Layer

You need a class called `Layer`. This class should have the functionality of a layer of a neural network. 

Its constructor should accept a numpy matrix, which represents the weights of the layer, a numpy array, which represents the biases of the layer, and a function, which represents the activation function. The constructor should check whether the dimensions of weights and biases are compatible and raise a `ValueError` exception if not.

The constructor has to copy the matrix and the array, so that they cannot be changed externally.

The class should contain a method called `forward`, which takes the input to the layer in the form of a numpy array, and returns the output of the layer which should be a numpy array.

The class should also contain methods called `getMatrix`, `getBiasVector` and `getFunction` which accept no argument and returns the matrix, bias and function respectively.

### Neural network

You need a class called `NeuralNetwork`. This class should have the functionality of a neural network.

Its constructor should accept a list of matrices, a list of arrays and a list of functions. Then it should take consecutively one from each and create the layers of the neural network. The constructor should check that all three lists have the same length and that the output of a layer can be used as an input to the next layer and raise a `ValueError` exception if this is not the case.

The class should contain a method called `propagate`, which takes the input to the neural network, as a numpy array, and returns the output, as a numpy array.

The class should also contain a method called `getLayers`, which accepts no arguments and returns a list containing the layers of the neural network.

### Player

You need a class called `NNPlayer`. This class will be the class of your players.

Its constructor should accept the same arguments as the NeuralNetwork constructor, i.e. a list of matrices, a list of arrays and a list of functions.

The class should contain a method called `play`, which takes the state of the game (see CompromiseGame for more information) and returns a valid move, i.e. a list of length 3 whose elements are from the set `{0,1,2}`.

The class should also contain a method called `getNN`, which accepts no arguments and returns the neural network of the player.

The class should also contain a static method called `getSpecs`, that accepts no arguments and returns a 2-tuple. The first number in the tuple should be the dimension of the input to the neural network and the second number of the tuple should be the dimension of the output of the neural network. This method is essential for the tests to work.

## Running the tests

In order to run the test you should open the test files and edit any line that says `from source import Layer` etc. You need to change `source` to the name of the `.py` file that contains the appropriate class. Note that if the name of the file is `layer.py` then the above line should be `from layer import Layer`.

After that open a console, navigate to the folder that contains the tests and execute `python -m unittest discover`. Note that you will need to install the python packages `numpy` and `unittest`.



