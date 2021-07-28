import argparse
import logging
from collections import deque
from time import sleep
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from agent import DQNetAgent
from src.agent import DRLAgent
from src.epsilon_policies import DecayEpsilonGreedy


def train(env: UnityEnvironment, brain_name: str, agent: DRLAgent, model_path: str, num_eps: int = 5000) -> List[int]:
    """
    Run the application.
    :param model_path: path to save model
    :param env: unity environment
    :param brain_name: brain name of the agent
    :param agent: agent object
    :param num_eps: number of episodes to run
    :returns: list of scores
    """
    # Initialize the scores list and the scores window for the last 100 episodes
    scores = []
    scores_window = deque(maxlen=100)
    for j in range(1, num_eps + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            # Decide agent's action by following the determined policy
            action = agent.act(state)
            
            # The agent acts on the environment
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            # The agent progresses in its journey, storing the memory of the previous experience and learning after a
            # while
            agent.step(state, action, reward, next_state, done)
            
            score += reward
            state = next_state
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        print(f"\rEpisode {j}/{num_eps}\t Average Score: {np.mean(scores_window)}", end="")
        if j % 100 == 0:
            print(f"\rEpisode {j}/{num_eps}\t Average Score: {np.mean(scores_window)}")
        if np.mean(scores_window) >= 13:
            print(f"\nEnvironment solved in {j - 100:d} episodes!\t Average Score: {np.mean(scores_window):.2f}")
            torch.save(agent.q_local.state_dict(), model_path)
            break
    return scores


def plot_scores(scores: List[float]) -> None:
    """
    Plot scores from trained agent.
    :param scores: scores obtained after training
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('../resources/scores-plot.png')
    plt.show()


def render(env: UnityEnvironment, brain_name: str, agent: DRLAgent) -> None:
    """
    Render a demonstration of the agent running for some episodes in the environment.
    :param env: unity environment
    :param brain_name: name of the brain
    :param agent: agent
    """
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    while True:
        # delay so it is humanly possible to watch the agent
        sleep(0.05)
        action = agent.act(state, train=False)
        env_info = env.step(action)[brain_name]
        state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        if done:
            break


def main(batch: int, update_rate: int, dr: float, tau: float, env_path: str, lr: float, model_path: str,
         eval: bool) -> None:
    """
    Main function of the program
    """
    env = UnityEnvironment(file_name=env_path)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    
    logging.info(f"Action size: {action_size}, State size: {state_size}")
    logging.info(f"State sample:\n {env_info.vector_observations[0]}")
    
    e_policy = DecayEpsilonGreedy(epsilon=1., epsilon_min=0.05, epsilon_decay_rate=0.999)
    agent = DQNetAgent(state_size=state_size, action_size=action_size, seed=0, lr=lr, batch_size=batch, tau=tau,
                       gamma=dr, update_every=update_rate, epsilon_greedy_policy=e_policy)
    if not eval:
        scores = train(env, brain_name, agent, model_path, num_eps=2000)
        plot_scores(scores)
    else:
        agent.load(model_path)
    render(env, brain_name, agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=64, type=int, help="Batch size")
    parser.add_argument("--update_rate", default=4, type=int, help="How often the target network is updated")
    parser.add_argument("--dr", default=0.99, type=float, help="Discount rate")
    parser.add_argument("--tau", default=1e-3, type=float, help="step-size for soft updating target network")
    parser.add_argument("--env_path", default="../world/Banana_Linux/Banana.x86", type=str,
                        help="Unity environment path")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--model_path", default="../models/ckpt.pth", type=str,
                        help="Model path")
    parser.add_argument("--eval", nargs='?', const=True, help="Renders a sample of the environment")
    
    args = vars(parser.parse_args())
    print(f"Command-line arguments:\n\t{args}")
    
    main(**args)
