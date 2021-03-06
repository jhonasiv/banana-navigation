In this project, the agent learns how to navigate in a world full blue and yellow bananas. Its goal is to collect all
the yellow bananas and avoid the blue ones. The method used to teach the agent to behave in such environment was
the [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

# Table of Contents

- [Banana World Backstory](#banana-world-backstory)
- [The agent and the environment](#the-agent-and-the-environment)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
    - [Linux (Debian-based)](#linux-debian-based)
- [Demo](#demo)    
- [Running the application](#running-the-application)



# Banana World Backstory

> The night is dark and full of bananas.

In the Banana World, there was a severe lack of food, and the only fruit that was easy to obtain were bananas. The
inhabitants of this world were scientists, and as such, they did a number of experiments on the bananas. These
experiments resulted in a banana storm, flooding the world with bananas. These bananas were one of two kinds: the blue
and the yellow ones. The blue ones are not edible, while the yellow ones are.

Now, the scientists built an agent to collect the yellow bananas, so they can eat and use them to plant new ones. The
following gif is a sample of the banana world.

<p align="center">
    <img src="resources/banana.gif" alt="animated"/>
    <p align="center">Demo of random agent going through the environment</p>
</p>

---

## The agent and the environment

The agent receives a reward of _+1_ for collecting yellow bananas and _-1_ for collecting blue bananas. This is in line
with the goal previously mentioned.

The state space has _37_ dimensions, and contains the agent’s velocity, along with ray-based perception of objects
around the agent’s forward direction. The agent **MUST** learn how to best select its actions. Four discrete actions are
available, corresponding to:

- `0` - move forward
- `1` - move backward
- `2` - move left
- `3` - move right

The task is episodic, and will be considered solved if the agent reaches a score of _+13_ over _100_ consecutive
episodes.
---

# Dependencies

This project is a requirement from
the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). The environment is provided by Udacity. It depends on the following packages:

- Python 3.6
- Numpy
- PyTorch
- Unity ML-Agents Beta v0.4

---

# Getting Started

## Linux (Debian-based)

- Install python3.6 (any version above is not compatible with the unity ml-agents version needed for this environment)

``` bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6-full
```

- (Optional) Create a virtual environment for this project

```bash
cd <parent folder of venv>
python3.6 -m venv <name of the env>
source <path to venv>/bin/activate
```

- Install the python dependencies

``` bash
python3 -m pip install numpy torch
```

- Download the Unity ML-Agents [release file](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0b) for
  version Beta v0.4. Then, unzip it at folder of your choosing
- Build Unity ML-Agents

```bash
cd <path ml-agents>/python
python3 -m pip install .
```

- Clone this repository and download the environment created by Udacity and unzip it at the world folder

```bash
git clone https://github.com/jhonasiv/banana-navigation.git
cd banana-navigation
mkdir world
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
unzip Banana_Linux.zip -d world
```

---

# Demo

This is how it looks after the agent was trained until it had reached the average score of `15.0`. 
For more details on the implementation and the results, check out the [Report.md file](Report.md).

<p align="center">
    <img src="resources/banana-catcher.gif" alt="animated"/>
    <p align="center">Agent trained with the average score of 15.0</p>
</p>

---

# Running the application


- Execute the main.py file
  ```bash
  python3 src/main.py
  ```
- For more information on the available command line arguments, use:
  ```bash
  python3 src/main.py --help
  ```
    - Some notable cli arguments:
        - `--eval`: runs the application in evaluation mode, skipping training step, `model_path` must be set
        - `--model_path`: path of the model to be loaded for the agent's local network, relative to the _src_ folder
        - `--env_path`: path for the loaded Unity Enviroment
