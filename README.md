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

### Getting Started for Unity environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the `p1_navigation/` folder, and unzip (or decompress) the file.

### Dependencies
- torch
- numpy
- matplotlib
- unityagents

