import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import epsilon_explore
from config import get_config
from collections import deque
from initialization import Init


def run(config):
    initialization = Init(config)
    initialization.init_seed()
    env = initialization.init_env()
    agent = initialization.init_agent()
    obs = env.reset()
    score = 0
    scores_window = deque(maxlen=100)
    frames = config.frames
    for frame in range(1, frames):
        epsilon = epsilon_explore(frame, frames)
        action = agent.get_action(obs, epsilon)
        next_obs, reward, done, _ = env.step(action)
        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        score += reward

        if done:
            scores_window.append(score)
            agent.writer.add_scalar('Average 100', np.mean(scores_window), frame)
            obs = env.reset()
            score = 0


if __name__ == '__main__':
    parser = get_config()
    config = parser.parse_args()
    run(config)
