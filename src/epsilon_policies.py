import abc
import math

from pydantic import BaseModel


class EpsilonGreedyPolicy(BaseModel):
    epsilon_min: float
    epsilon: float = 1.0
    
    # @abc.abstractmethod
    def step(self, time_step: int) -> float:
        """
        Calculate epsilon for this time step
        :param time_step: time step in episode
        :return: updated epsilon
        """
        raise NotImplementedError


class ConstantEpsilonGreedy(EpsilonGreedyPolicy):
    epsilon: float
    
    def step(self, time_step: int) -> float:
        return self.epsilon


class DecayEpsilonGreedy(EpsilonGreedyPolicy):
    """
    Epsilon greedy policy where
     
     epsilon = epsilon * epsilon_decay_rate^(time step)
    """
    epsilon_min: float
    epsilon: float = 1.0
    epsilon_decay_rate: float
    
    def step(self, time_step: int) -> float:
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon


class ExponentialEpsilonGreedy(EpsilonGreedyPolicy):
    """
    Exponentially calculate epsilon, using the following equation
    
        e^(exp_k * int((time_step / exp_b) ** 2)
    """
    exp_k: float
    exp_b: float
    epsilon_min: float
    epsilon: float = 1.0
    
    def step(self, time_step: int) -> float:
        self.epsilon = math.exp(self.exp_k * int((time_step / self.exp_b) ** 2))
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return self.epsilon
