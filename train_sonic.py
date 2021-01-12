import retro
import numpy as np
import cv2
import neat
import pickle

import argparse

parser = argparse.ArgumentParser(description='Train AI to play Sonic!')
parser.add_argument('level', help="Which level to play (GreenHillZone.Act1,MarbleZone.Act2, etc)")
parser.add_argument("-r", "--render", action="store_true",
                    help="Use to enable rendering as the network trains (decreases training speed a lot)")

args = parser.parse_args()
state = args.level
render = args.render

# create env
env = retro.make(game='SonicTheHedgehog-Genesis', state=state)
print("Created Env")

def eval_genomes(genomes, config):
    # this will run for amount of population
    for genome_id, genome in genomes:
        ob = env.reset()

        # width of screen, height of screen, amount of color channels
        inx, iny, inc = env.observation_space.shape

        inx = inx // 7
        iny = iny // 7

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        x_pos = 80
        x_pos_max = 0
        done = False

        while not done:
            global render
            if render:
                env.render()
            frame += 1

            #Prepare image for input
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            img_array = ob.flatten()

            # Put input (image) into net and get output (buttons)
            nnOutput = net.activate(img_array)

            prev_x_pos = x_pos

            # reward is useless
            ob, rew, done, info = env.step(nnOutput)
            x_pos = info['x']
            x_pos_end = info['screen_x_end']


            # Every time Sonic goes further to the right he gains fitness
            if x_pos > x_pos_max:
                fitness_current += x_pos-prev_x_pos
                x_pos_max = x_pos

            # If sonic has beat the game
            if x_pos >= x_pos_end and x_pos > 500:
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
                print(f"------\nGenome Id: {genome_id}\nFitness: {fitness_current}\nX Position: {x_pos}/{x_pos_end}")

            genome.fitness = fitness_current

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-sonic')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(generation_interval=1,filename_prefix="/checkpoints/checkpoint-"))

winner = p.run(eval_genomes)
print("Winner Winner Chicken Dinner!")
with open('winner-'+state+'.pkl','wb') as output:
    pickle.dump(winner,output,1)
