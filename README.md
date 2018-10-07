# Project 1 : Navigation
### Introduction
This repo is the project 1 on the udacity course. In this project, the agent gathers bananas in the unity environment.
If the agent collects a yellow banana, the unity env gives +1 reward. 
If the agent get a blue banana, in contrast, the env gives -1 reward.
The goal of this project is to make the agent be able to get as many yellow bananas as possible.

On the each step, the unity environment informs you the current state which space is 37 dimensions containing the agents' veloctiy, along with ray-based perception of objects
around the agent's forward direction information. After getting an arbitrary action, the env gives the next state(37-dimensions), reward and done(whether the game is finish or not).
There are 4 discrete actions, 
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

I train the agent based deep Q learning with Double and Duel DQN techniqes.
* [DQN](https://www.nature.com/articles/nature14236)
* [Double DQN](https://arxiv.org/abs/1509.06461)
* [Dueling DQN](https://arxiv.org/abs/1511.06581)

### Dependencies
- Ubuntu 16.04
- Python 3.6
- torch
- numpy
- matplotlib
- unityagents

