# Sonic AI:
An AI made using [neat-python](https://github.com/CodeReclaimers/neat-python), that can beat the first few levels of Sonic the Hedgehog Classic on the Sega Genesis
## Introduction
This is my Python source code for training a Neuroevolution of Augmenting Topologies (NEAT) artifical intellegence to play Sonic The Hedghehog on Sega Genesis, using the [neat-python](https://github.com/CodeReclaimers/neat-python) library implementation.

<p align="center">
  <a href="https://youtu.be/E0s2Cp9tJD8"><img src="GreenHillZone.Act1.gif" width="600" />
  </a>
</p>
<p align="center">
  <a href="https://youtu.be/E0s2Cp9tJD8">Full Video (Click image ^)
  </a>
</p>

## Why it's great:
 * Can be applied to any level without any changes, and high success rate
 * Uses only pixel data. No in-game variables!
 * Uses uncommon NEAT approach instead of A3C
 ***
# How to use my code:

## Installation
```
pip install -r requirements.txt
python -m retro.import /roms/
```

## Requirements
* Python 3.6
* neat-python
* opencv2
* numpy
* gym-retro
* argparse

^ All found in requirements.txt
## Demo:
```
python replay.py winner-GreenHillZone.Act1.pkl GreenHillZone.Act1
```
## Train:
```
python train.py GreenHillZone.Act1 -r
```
Adding the -r flag lets you watch the AI play as it trains.

Obviously replace GreenHillZone.Act1 with any level of your choice
***
Inspired by https://www.youtube.com/watch?v=8dY3nQRcsac
