# Introduction

The following outlines the project details for the first project submission, Navigation for the Udacity Ud893 Deep Reinforcement Learning Nanodegree (DRLND).

# Getting Started

## The Environment

The goal of the project is to develop and train an agent to navigate in a large, square world (environment) and collect items - specifically, to collect yellow bananas, while avoiding blue bananas. This environment is similar to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#banana-collector).

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Install and Dependencies

The following instructions will help you set up the environment on your machine.

### Step 1 - Clone the Repository

All files required for running the project are in the main project directory. Note that a copy of the `python/` directory from the [DRLND](https://github.com/udacity/deep-reinforcement-learning#dependencies) which contains additional dependencies has also been included in the main project directory.

### Step 2 - Download the Unity Environment

Note that if your operating system is Windows (64-bit), the Unity environment is included for that OS in the main project directory and you can skip this section. If you're using a different operating system, download the file you require from one of the following links:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

Then, place the file in the main project directory folder and unzip (or decompress) the file.

## Instructions

The [Report.md](Report.md) file (also available in notebook format `Report.ipynb`) is a project summary report which includes a decription of the implementation, learning algorithm(s), hyperparameters, neural net model architectures, reward/episode plots and ideas for future work. The summary report should be read first as it explains the order in which to run the project notebook. The `Project 1.ipynb` jupyter notebook provides the code for running and training the actual agent(s).
