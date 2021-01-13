import retro
import numpy as np
import cv2
import neat
import pickle

import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train AI to play Sonic!')
parser.add_argument('level', help="Which level to play (GreenHillZone.Act1,MarbleZone.Act2, etc)")
parser.add_argument("-r", "--render", action="store_true",
                    help="Use to enable rendering as the network trains (decreases training speed a lot)")
parser.add_argument("-t","--threads", help="Number of threads to use while training. (Greatly improves training speed)",nargs='?', const=1, type=int,default=1)

args = parser.parse_args()
state = args.level
render = args.render
threads = args.threads
class Worker:
    def __init__(self,genome,config):
        self.genome = genome
        self.config = config
    def work(self):
        self.env = retro.make(game='SonicTheHedgehog-Genesis', state=state,record=".")

        ob = self.env.reset()

        # width of screen, height of screen, amount of color channels
        inx, iny, inc = self.env.observation_space.shape

        inx = inx // 7
        iny = iny // 7
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

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
                self.env.render()
            frame += 1

            # Prepare image for input
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            img_array = ob.flatten()

            # Put input (image) into net and get output (buttons)
            nnOutput = net.activate(img_array)
            prev_x_pos = x_pos

            # reward is useless
            ob, _, done, info = self.env.step(nnOutput)
            x_pos = info['x']
            x_pos_end = info['screen_x_end']

            # Every time Sonic goes further to the right he gains fitness
            if x_pos > x_pos_max:
                difference = x_pos - prev_x_pos
                fitness_current += difference ** 4

                x_pos_max = x_pos

            # If sonic has beat the game
            if (x_pos >= x_pos_end and x_pos > 500) or x_pos > 4000:
                fitness_current += 10000000
                done = True

            # Every time sonic increases his fitness resets
            # For every frame he goes without increasing fitness, the counter goes up
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            # If sonic goes 500 frames without increasing fitness, he dies
            if done or counter == 500:
                done = True
                print(f"------\nFitness: {fitness_current}\nX Position: {x_pos}/{x_pos_end}")
                with open('winner-' + state + '.pkl', 'wb') as output:
                    pickle.dump(self.genome, output, 1)

        return fitness_current



def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()
if __name__ == "__main__":


    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-sonic')

    p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('C:\\Users\\15879\Documents\sonic-ai\checkpoints\checkpoint-118')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=1,filename_prefix="checkpoints/checkpoint-"))

    pe = neat.ParallelEvaluator(threads,eval_genomes)
    winner = p.run(pe.evaluate)

    print("Winner Winner Chicken Dinner!")
    with open('winner-'+state+'.pkl','wb') as output:
        pickle.dump(winner,output,1)
