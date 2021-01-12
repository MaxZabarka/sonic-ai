import retro
import numpy as np
import cv2
import neat
import pickle

import argparse

parser = argparse.ArgumentParser(description='Replay genome in .pkl file format')
parser.add_argument('file', help=".pkl file containing genome to play")
parser.add_argument('level', help="Which level to play (GreenHillZone.Act1,MarbleZone.Act2, etc)")

args = parser.parse_args()

env = retro.make('SonicTheHedgehog-Genesis', args.level,record=".")

img_array = []

x_pos_end = 0

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-sonic')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

with open(args.file, 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()

inx, iny, inc = env.observation_space.shape

inx = int(inx / 7)
iny = int(iny / 7)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
x_pos = 0
x_pos_max = 0

done = False

while not done:
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    img_array = ob.flatten()

    nnOutput = net.activate(img_array)

    ob, rew, done, info = env.step(nnOutput)
    img_array = []

    x_pos = info['x']
    x_pos_end = info['screen_x_end']

    if x_pos > x_pos_max:
        fitness_current += 1
        x_pos_max = x_pos

    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter += 1

    if counter == 250:
        done = True
print('Run "python -m retro.scripts.playback_movie <filename>.bk2" to render the generated bk2 file to a video!')