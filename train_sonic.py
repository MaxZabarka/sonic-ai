import retro
import numpy as np
import cv2
import neat
import pickle
import time

state = 'GreenHillZone.Act1'

# create env
env = retro.make(game='SonicTheHedgehog-Genesis', state=state)


def eval_genomes(genomes, config):

    # this will run for amount of population
    for genome_id, genome in genomes:


        ob = env.reset()


        # width of screen, height of screen, amount of color channels
        inx, iny, inc = env.observation_space.shape
        #print(inx,iny)

        inx = inx // 7
        iny = iny // 7

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 80
        xpos_max = 0
        done = False

        while not done:
            env.render()
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            imgarray = ob.flatten()

            # cv2.imshow("imgarray", ob)
            # cv2.waitKey(1)

            # Put input into net (image) and get output (buttons)

            nnOutput = net.activate(imgarray)
            print(nnOutput)

            prev_xpos = xpos


            # reward is useless
            ob, rew, done, info = env.step(nnOutput)
            xpos = info['x']
            xpos_end = info['screen_x_end']



            # Every time Sonic goes further to the right he gains fitness
            if xpos > xpos_max:
                fitness_current += xpos-prev_xpos
                xpos_max = xpos

            if xpos == xpos_end and xpos > 500:
                fitness_current += 100000
                done = True

            # Every time sonic increases his fitness resets
            # For every frame he goes without increasing fitness, the counter goes up
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # If sonic goes 250 frames without increasing fitness, he dies
            if done or counter == 250:
                done = True
                print(genome_id,fitness_current,xpos)


            genome.fitness = fitness_current




config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-sonic')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('chickendinner-'+state+'.pkl','wb') as output:
    pickle.dump(winner,output,1)
