import abc
import random
from collections import deque, namedtuple
from typing import Any, Collection, Optional

import numpy as np
import torch
from dataclasses import dataclass
from torch import optim
from torch.nn import functional as f

from epsilon_policies import EpsilonGreedyPolicy
from model import DQNModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])  #


@dataclass
class DRLAgent(abc.ABC):
    state_size: int  # size of each state in the state space
    action_size: int  # size of the action space
    lr: float  # learning rate
    epsilon_greedy_policy: EpsilonGreedyPolicy
    seed: Optional[int]  # seed
    buffer_size: int = int(1e5)  # experience memory buffer size
    batch_size: int = 64  # experience memory batch size
    update_every: int = 4  # how often the target network is updated
    gamma: float = 0.99  # discount factor
    tau: float = 1e-3  # step-size for soft updating the target network
    
    @abc.abstractmethod
    def step(self, state: Collection[float], action: int, reward: float, next_state: Collection[float],
             done: bool) -> None:
        """
        The agent registers a step it took in the environment.

        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: state that resulted from the action
        :param done: if the resulting state was terminal
        """
    
    @abc.abstractmethod
    def learn(self, experiences: Collection[experience]) -> None:
        """
        The agent learns from previous experiences.
        :param experiences: a minibatch of previous experiences with size=batch_size
        """
    
    @abc.abstractmethod
    def act(self, state: Collection[float], train: bool = True) -> int:
        """
        The agent acts following a epsilon-greedy policy.
        :param train: if training mode is active, if so the agent will follow the epsilon-greedy policy, otherwise it
        will follow the greedy policy
        :param state: current state
        :return: action selected
        """
    
    def soft_update(self) -> None:
        """
        Soft updates the target network with parameters from the local network
        """


class DQNetAgent(DRLAgent):
    """
    Agent using Deep Q-Networks to learn about the environment.
    """
    
    def __init__(self, **data: Any):
        super(DQNetAgent, self).__init__(**data)
        np.random.seed(self.seed)
        
        # Initializes the Local and the Target QNetworks.
        self.q_local = DQNModel(self.state_size, self.action_size, self.seed).to(device)
        self.q_target = DQNModel(self.state_size, self.action_size, self.seed).to(device)
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=self.lr)
        
        # Initializes the experience replay memory
        self.memory = ReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size, seed=self.seed)
        
        # Sets the initial time step to 0, this is used for updating the target network every update_every steps
        self.time_step = 0
    
    def load(self, path: str) -> None:
        """
        Load previously trained model.
        :param path: model path
        """
        self.q_local.load_state_dict(torch.load(path))
    
    def step(self, state: Collection[float], action: int, reward: float, next_state: Collection[float], done: bool) -> \
            None:
        # Store experience in memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # Check if there are enough samples in memory, if so, get a sample and learn from it
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def learn(self, experiences: Collection[experience]) -> None:
        
        # Get the informed from the experiences sample.
        states, actions, rewards, next_states, dones = experiences
        
        # Max action value for each episode in the sample
        target_values = self.q_target(next_states).cpu().max(1)[0].unsqueeze(1)
        
        # Calculate the target action-value for taking each action from each origin state in the sample. If the
        # episode is terminal, the action-value is the reward
        target_values = target_values.to(device)
        target_estimate = rewards + self.gamma * target_values * (1 - dones)
        
        # Get the estimates for the local network and gather the action-value for each action taken in the sample.
        local_estimate = self.q_local(states).gather(1, actions)
        
        loss = f.mse_loss(local_estimate, target_estimate)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network
        self.soft_update()
    
    def act(self, state: Collection[float], train: bool = True) -> int:
        epsilon = self.epsilon_greedy_policy.step(self.time_step)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get estimate action values from local network
        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state)
        self.q_local.train()
        
        # Epsilon-greedy action selection
        if train:
            if np.random.random() > epsilon:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return np.random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())
    
    def soft_update(self) -> None:
        for target_p, local_p in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_p.data.copy_(self.tau * local_p.data + (1.0 - self.tau) * target_p.data)


class ReplayBuffer:
    """
    Buffer for stores experiences and sampling them when requested.
    """
    
    def __init__(self, batch_size: int, buffer_size: int, seed: int):
        """
        ReplayBuffer constructor
        :param batch_size: number of experiences in a minibatch
        :param buffer_size: max len of memory
        :param seed: random seed
        """
        super(ReplayBuffer, self).__init__()
        random.seed(seed)
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state: Collection[float], action: int, reward: float, next_state: Collection[float], done: bool) -> \
            None:
        """
        Add a new experience to memory.
        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_state: resulting state after action
        :param done: if the resulting state is terminal
        """
        exp = experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self) -> Collection[experience]:
        """
        Randomly sample a batch of experiences from memory.
        :return: a minibatch of experiences
        """
        samples = random.sample(self.memory, k=self.batch_size)
        samples = np.array(samples, dtype=np.object)
        
        states = torch.from_numpy(np.vstack(samples[:, 0])).float().to(device)
        actions = torch.from_numpy(np.vstack(samples[:, 1])).long().to(device)
        rewards = torch.from_numpy(np.vstack(samples[:, 2])).float().to(device)
        next_states = torch.from_numpy(np.vstack(samples[:, 3])).float().to(device)
        dones = torch.from_numpy(np.vstack(samples[:, 4]).astype(np.uint8)).float().to(
                device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of internal memory"""
        return len(self.memory)
