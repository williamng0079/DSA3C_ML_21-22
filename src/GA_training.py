import numpy as np
import matplotlib.pyplot as plt

from NNPlay import NNPlayer
from CompromiseGame import CompromiseGame, AbstractPlayer, GreedyPlayer, SmartGreedyPlayer




class Training():

    def __init__(self, player):
        self.player = player
        self._score = [0, 0]
        self.totalReward = 0
        self.nb_games = 30
        self.g = CompromiseGame(player, GreedyPlayer(), 30, 1)

    # Fitness calculation of the player
    def fitness_calc(self, r):

        return (r/ self.nb_games) * 100


    def game_simulation(self): # simulate 30 games of 1 round senario for each NNplayer in the population
    
        for y in range(self.nb_games):
            self.g.resetGame()
            res = self.g.play()
            if res[0] > res[1]:
                self.totalReward += 1
                self._score[0] += 1
            elif res[1] > res[0]:
                self._score[1] +=1
        #print(self._score)
        #print(self.player.getNN())
        #self.player.give_rewards(self.totalReward)
        
    
        return self.totalReward

    def get_fitness(self):

        fitness_value = self.fitness_calc(self.totalReward)
        return fitness_value


if __name__ == "__main__":

    # GA Training...
    # generating population
    
    number_of_NNplayer = 250
    
    population = []
    rewards = []
    fitnesses = []
    for p in range(number_of_NNplayer):
        indiv = NNPlayer()
        population.append(indiv)

    for indiv in population:
        i = Training(indiv)
        rewards.append(i.game_simulation())
        fitnesses.append(i.get_fitness())
    
    sorted_rewards = sorted(rewards)

    avg_fitness = (sum(fitnesses))/ number_of_NNplayer
    
    print("Average fitness of the generation:", avg_fitness)
    print("The best score was:" + str(sorted_rewards[-1]) + "/30")
    print("The worst score was:" + str(sorted_rewards[0]) + "/30")
    plt.hist(rewards, bins= number_of_NNplayer)
    plt.show()
    

    