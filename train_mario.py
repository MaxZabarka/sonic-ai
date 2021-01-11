import retro
import numpy as np
import cv2
import neat
import pickle
from time import sleep




state = 'Level1-1'

# create env
env = retro.make(game='SuperMarioBros-Nes', state=state,record=False)
env.reset()


def eval_genomes(genomes, config):

    # this will run for amount of population
    for genome_id, genome in genomes:
        ob = env.reset()

        # width of screen, height of screen, amount of color channels
        inx, iny, inc = env.observation_space.shape
        #print(inx,iny)

        inx = inx // 6
        iny = iny // 6

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        done = False

        while not done:
            env.render()
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            imgarray = ob.flatten()

            # cv2.imshow("imgarray", ob)
            # cv2.waitKey(50)
            # sleep(50)


            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)
          #  print(info)

            fitness_current += rew

            # For every frame he goes without increasing fitness, the counter goes up
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # If Mario goes 250 frames without increasing fitness, he dies
            if done or counter == 250 or info['lives'] < 2:
                done = True
                print(genome_id,fitness_current)

            genome.fitness = fitness_current




config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-mario')

p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-20')

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(2))

winner = p.run(eval_genomes)

with open('chickendinner-'+state+'.pkl','wb') as output:
    pickle.dump(winner,output,1)
