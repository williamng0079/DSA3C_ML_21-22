import numpy as np
import matplotlib.pyplot as plt

from NNPlay import NNPlayer
from CompromiseGame import CompromiseGame, AbstractPlayer, GreedyPlayer, SmartGreedyPlayer

# simulate 20 games senario for each NNplayer in the population
def game_simulation(player, nb_games = 20):
    _score = [0, 0]
    totalReward = 0
    g = CompromiseGame(player, GreedyPlayer(), 30, 1)
    for y in range(nb_games):
        g.resetGame()
        res = g.play()
        if res[0] > res[1]:
            totalReward += 1
            _score[0] += 1
        elif res[1] > res[0]:
            totalReward -= 1
            _score[1] +=1
    #print(_score)
    #print(player.getNN())
    #player.give_rewards(totalReward)
    return totalReward

if __name__ == "__main__":

    # GA Training...
    number_of_NNplayer = 250
   
    # generating population
    population = []
    rewards = []
    for p in range(number_of_NNplayer):
        indiv = NNPlayer()
        population.append(indiv)

    for indiv in population:
        rewards.append(game_simulation(indiv))
    
    plt.hist(rewards, bins=250)
    plt.show()
    

    