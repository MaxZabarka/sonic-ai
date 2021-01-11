import retro
import numpy as np
import cv2
import neat
import pickle

state = 'Level1-1'

# create env
env = retro.make(game='SuperMarioBros-Nes', state=state,record=False)


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward-mario')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

with open('mario2021.0.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = inx // 6
iny = iny // 6

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
imgarray = []
done = False

while not done:

    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

    imgarray = ob.flatten()

    nnOutput = net.activate(imgarray)

    ob, rew, done, info = env.step(nnOutput)

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

