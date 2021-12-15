import numpy as np
import matplotlib.pyplot as plt
import json

from CompromiseGame import CompromiseGame, AbstractPlayer, GreedyPlayer, SmartGreedyPlayer, DeterminedPlayer

# This code has been reorganised by merging NNPlay.py to ensure mutation and population rearranging features work properly
# Also moving code into one place makes following the code easier.

# problem encountered: 
#       During training, wins rates are often recurring decimals
#           Casue of issue: number of game played (30) as x/30 often returns with recurring num
#           Solution: change the number of games from 30 to 50 for each player for future training, reduce pop size to 200 for speed
#       DONE

# Notes:  
#       np.random.rand() ranges from 0 to 1
     

# observations: 
#       Test training observ 2: avg population win rate progress: 88 at 1489th generation (pop size = 250, 30 games each)
#       Softmax function in ouput layer often encounter overflow because values in .exp() are too large
#           Cause of issue (maybe): previous activate are by leaky RelU, the maximum ouput values are not restricted
#           Solution: no need, as its a warning message, calculation still happens              


# Todo: 
#       Make best player file and do unittests 
#       Write report    
#       Write best player genetic into a txt file           DONE
#       Restablish fitness func                             DONE: using win rate and score count to track fitness of a generation
#       Diplay more information on graph (include best and worst rate ) DONE
#       Train against greedyplayer                                      Future Objective
#       Experiment with initiating the bias value with zeros and mutate from it     # Out of time, current mutation can reliabily achieve 87+

 

class NeuralNetwork():

    def __init__(self, blank = False):
        self._rewards = 0
        self._weights = []
        self._biases = []

        if (not blank):                                              # For mutation, if its a copy (ie, for reinjecting the population), it will not generate a new sets of layers with random values
            self.first_weightArray = np.random.rand(54, 27) * 2 - 1 # Create matrix for layers (consist of random numbers) from number of neurons x number of activators to simulate an weight matrix
            self._weights.append(self.first_weightArray)            # * 2 - 1 ensures some values are negative
            
            self.first_biasArray = np.random.rand(54, 1) * 2 - 1
            self._biases.append(self.first_biasArray)

            self.second_weightArray = np.random.rand(54, 54) * 2 - 1 
            self._weights.append(self.second_weightArray)
            
            self.second_biasArray = np.random.rand(54, 1) * 2 - 1
            self._biases.append(self.second_biasArray)

            self.output_weightArray = np.random.rand(27, 54) * 2 - 1 
            self._weights.append(self.output_weightArray)
            
            self.output_biasArray = np.random.rand(27, 1) * 2 - 1
            self._biases.append(self.output_biasArray)

    def ReLU(self, x):                                          # ReLu: cuts off the negative values and set them to 0
        return x.clip(min = 0)              

    def Leaky_ReLU(self, x):
        return np.where(x > 0, x, x * 0.01)                     # Leaky RelU: 0.01 times all negative values

    def Sigmoid(self, x):
        return 1/(1+np.exp(-x))                                 # Sigmoid: limit ouput between 0 and 1, useful for binary decisions

    def Softmax(self, x):
        m = np.exp(x)
        return m/m.sum(len(x.shape)-1)                          # Softmax: similar to sigmoid but useful for multi ouptut decisions (probability of 1 between all output) in output layer 


    def forward(self, inputArray):
        aligned_input = inputArray.reshape((-1, 1))             # ensure the input array shape for the matrix calculation
        
        output = np.matmul(self._weights[0], aligned_input)     # calculate the first layer (input to hidden 1)
        output = output + self._biases[0]
        output = self.Leaky_ReLU(output)

        output = np.matmul(self._weights[1], output)            # calculate the between hidden layers
        output = output + self._biases[1]
        output = self.Leaky_ReLU(output)

        output = np.matmul(self. _weights[2], output)           # calculate the output layer
        output = output + self._biases[2]
        output = self.Softmax(output.reshape(-1))               # Apply softmax to output layer

        return output

    def give_rewards(self, r):                                  # used to track each players end reward count, used as an indicator to sort the population from best to worst later on
        self._rewards = r                   
        return self._rewards


    def mutate(self, mutate_player = True):
        mutation_prob =  0.01                                   # set the probability of a mutation ocurring to 1%
        new_player = NeuralNetwork(blank = True)                 # create new players with blank weights and biases

    
            # copy the parameter of existing player
        for w in self._weights:                                 # iterate through the list of parent weights and apply mutation to generate new player
            w_mutationArray = np.random.rand(w.shape[0], w.shape[1]) # generate a matrix with the same weight dimension as the parent
            mutation_mask = np.where(w_mutationArray < mutation_prob, (np.random.rand() - 0.5)/2 , 0)  # mutation condition: any inside the random matrix is smaller than the mutation prob (0.01), change to that value will occur, otherwise sets the values in the matrix to 0 if the condition is not met
            updated_w = w + mutation_mask                      # add the matrix of mutated value (mask) to the parent matrix

            new_player._weights.append(updated_w)               # add the mutated weights to the new player

        for b in self._biases:                                  # mutate the bias value for the new player
            b_mutationArray = np.random.rand(b.shape[0], 1)     # same process as above
            b_mutation_mask = np.where(b_mutationArray < mutation_prob, (np.random.rand() - 0.5)/2, 0)
            updated_b = b + b_mutation_mask

            new_player._biases.append(updated_b)

        return new_player
    
    def play(self, ownB, oppB, ownS, oppS, turn, gLen, pips):       
        # All possible moves a player can choose
        allMoves = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
        #BoardDiff = [-1, -1, -2,  0,  0, -1, -1,  0,  2, -1,  1,  0,  0,  1,  0,  2, -1,  1,  0, -1, -3,  1, -1, 3, 0,  0,  2]   #test nn calculation
        ownBoardArray = np.array(ownB).flatten()                # converts to 1D array for the ease of calculation 
        oppBoardArray = np.array(oppB).flatten()
        BoardDiff = np.subtract(oppBoardArray, ownBoardArray)   # (calculate the differences between red and green board and generates 27 input nodes to feed into the NN)
        
        diffarray = np.asarray(BoardDiff)
        
        NNoutput = self.forward(diffarray)
        
    
        selectedmove = allMoves[NNoutput.argmax()]              # use the position of output neuron with highest value as index to decide the move to choose from the .allmoves pool
        #print("List of decisions:", NNoutput)
        #print("Move Selected:", selectedmove)
        
        #softmax_check = sum(NNoutput)                           # check if softmax is working as intended, sum should be 1
        #print(softmax_check)

        return selectedmove

    def getNN_weights(self):
        NN = self._weights
        return NN
    
    def getNN_biases(self):
        NN = self._biases
        return NN

class Training():
    # GA Training...
    
    def __init__(self, player):
        self.player = player
        self.totalReward = 0
        self.nb_games = 50
        self.g = CompromiseGame(player, AbstractPlayer(), 30, 10)


    def game_simulation(self): # simulate 50 games for each NNplayer in the population
    
        for y in range(self.nb_games):
            self.g.resetGame()
            res = self.g.play()
            if res[0] > res[1]:
                self.totalReward += 1
    
        self.player.give_rewards(self.totalReward)          # record the reward for each player, used for selecting population to keep for later
        return self.totalReward

    # Fitness calculation of each player
    def fitness_calc(self, r):

        return (r / self.nb_games) * 100
    
    def get_fitness(self):

        fitness_value = self.fitness_calc(self.totalReward)
        return fitness_value


def restart_simulation():
    global population
    global number_of_NNplayer
    global best_score_count
    global worst_score_count
    global best_win_rate
    global worst_win_rate

    best_score_count = []
    worst_score_count = []
    best_win_rate = []
    worst_win_rate =[]


    for p in population:
        del(p)

    population = []
    for p in range(number_of_NNplayer):
        indiv = NeuralNetwork()
        population.append(indiv)


def run_simulation_with_mutation():
    global population
    global number_of_NNplayer
    global best_score_count
    global worst_score_count
    global best_win_rate
    global worst_win_rate

    rewards = []
    indiv_fitnesses = []
    
    for indiv in population:
        i = Training(indiv)
        rewards.append(i.game_simulation())
        indiv_fitnesses.append(i.get_fitness())

    population.sort(key = lambda player: player._rewards, reverse = True)               # sort the population from best to worst based on the rewards gained by each NNPlayer
    
    sorted_fitness = sorted(indiv_fitnesses)
    avg_fitness = sum(indiv_fitnesses) / number_of_NNplayer
    sorted_scores = sorted(rewards)


    #print(sorted_fitness)
    print("Average win rate of the generation:", avg_fitness)
    print("Best player of this generation achieved:", str(sorted_fitness[-1]))
    print("Worst player of this generation achieved:", str(sorted_fitness[0]))
    print("The best score was:" + str(population[0]._rewards) + "/50")
    print("The worst score was:" + str(population[-1]._rewards) + "/50")
    #plt.hist(rewards, bins = number_of_NNplayer)
    #plt.plot(rewards)
    #plt.show()

    best_score_count.append(sorted_scores[-1])
    worst_score_count.append(sorted_scores[0])
    best_win_rate.append(sorted_fitness[-1])
    worst_win_rate.append(sorted_fitness[0])

    if (avg_fitness >= 85):                                                             # write the best player genetic of a population when it reaches above a certain threshold
        
        with open("Best_NN_weights_1.txt", "w") as layer1_weightsFile:                  # Had to output each layer seperately, as by default the np.savetxt cuts out marjority of values
            np.savetxt(layer1_weightsFile, np.asarray(population[0]._weights[0], dtype=object), fmt = "%s")
            layer1_weightsFile.close()
        
        with open("Best_NN_weights_2.txt", "w") as layer2_weightsFile:
            np.savetxt(layer2_weightsFile, np.asarray(population[0]._weights[1], dtype=object), fmt = "%s")
            layer2_weightsFile.close()
        
        with open("Best_NN_weights_output.txt", "w") as output_weightsFile:
            np.savetxt(output_weightsFile, np.asarray(population[0]._weights[2], dtype=object), fmt = "%s")
            output_weightsFile.close()
        
        with open("Best_NN_biases.txt", "w") as BiasFile:
            np.savetxt(BiasFile, np.asarray(population[0]._biases, dtype=object), fmt = "%s")
            BiasFile.close()
        
            #json.dump(population[0].getNN_weights(), weightsFile)                      # json dump did not work

    #print(population[0].getNN_biases())
    return avg_fitness


def rearrange_population():
    global population
    percent_to_keep = 0.1       # we keep 10% of the best players according to their reward value from a generation

    nb_players_toKeep = int(percent_to_keep * len(population))
    new_population = population[:nb_players_toKeep]             # we shred the population count of unwanted players, and keep the first 10% (best player after sort)
    
    injected_players = []
    for i in range(len(population) - nb_players_toKeep):            # in this case 250 - 25 = 225
        parent_player = new_population[i % nb_players_toKeep]       # use the modulus operator to iterate through the list of best players index 0 to index 25 over and over and use their genetic info as parents to mutate
        child_player = parent_player.mutate()                       # produce child players with the mutated parent genetic values(weights and biases)
        injected_players.append(child_player)                       # repopulated the 90% with mutated player
    
    population = new_population + injected_players                  # combine the top 10% from previous gen and 90% of mutated players to form new population for the next generation

if __name__ == "__main__":
    
    population = []
    number_of_NNplayer = 200
    best_score_count = []
    worst_score_count = []
    
    best_win_rate = []
    avg_win_rate = []
    worst_win_rate =[]
    

    restart_simulation()
    #run_simulation_with_mutation()
    for g in range(2000):
        generation = g + 1
        print("Generation:", generation)
        avg_win_rate.append(run_simulation_with_mutation())
        rearrange_population()
        print("\n")
    
    
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("Training Progression: Pop Size: 200, No.Games: 50")
    ax1.plot(list(range(generation)), best_score_count, 'g', label = "Best Player Score")
    ax1.plot(list(range(generation)), worst_score_count, 'r', label = "Worst Player Score")
    ax1.legend(loc = 2,prop = {"size":7})
    ax1.set_ylabel("Player Score")
    ax2.plot(list(range(generation)), best_win_rate, 'g', label = "Best Player Win Rate")
    ax2.plot(list(range(generation)), avg_win_rate, 'b', label = "Average Win Rate")
    ax2.plot(list(range(generation)), worst_win_rate, 'r', label = "Worst Player Win Rate")
    ax2.legend(loc = 2,prop = {"size":7})
    ax2.set_ylabel("Win Rate %")
    plt.xlabel("Generations")
    plt.show()