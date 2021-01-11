import retro
import numpy as np
import cv2
import neat
import pickle
from time import sleep



class Worker(object):
    def __init__(self,genome,config):
        self.genome = genome
        self.config = config
    def work(self):
        # create env
        self.state = 'Level1-1'
        self.env = retro.make(game='SuperMarioBros-Nes', state=self.state, record=False)
        ob = self.env.reset()

        inx, iny, _ = self.env.observation_space.shape

        inx = inx//6
        iny = iny//6
        done = False

        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
        fitness = 0
        counter = 0
        while not done:

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

            imgarray = ob.flatten()

            actions = net.activate(imgarray)

            ob, rew, done, info = self.env.step(actions)
            #self.env.render()
         #   print(rew)
            fitness = fitness + rew

            if rew == 0:
                counter += 1
            else:
                counter = 0

            # If Mario goes 250 frames without increasing fitness, he dies
            if done or counter == 250 or info['lives'] < 2:
                done = True
        print(fitness)
        return fitness



max_fitness = 0
def eval_genomes(genome,config):
    global max_fitness
    worky = Worker(genome,config)
    fitness = worky.work()
    if fitness > max_fitness:
        with open('mario'+str(fitness)+'.pkl', 'wb') as output:
            pickle.dump(genome, output, 1)
        max_fitness = fitness + 100
        print(max_fitness)
    return fitness



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-mario')

p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-23')
print(5)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(2))
pe = neat.ParallelEvaluator(6, eval_genomes)

winner = p.run(pe.evaluate)



with open('chickendinner.pkl','wb') as output:
    pickle.dump(winner,output,1)
