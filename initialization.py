import os
import shutil

import gym
import torch
import numpy as np

from agent import Agent
from pathlib import Path
from gym.spaces import Box
from tensorboardX import SummaryWriter


class Init:
    def __init__(self, config):
        self.config = config
        self.state_size = 0
        self.action_size = 0
        torch.set_num_threads(self.config.num_threads)
        torch.set_default_dtype(torch.float32)

    def init_seed(self):
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def init_env(self):
        env = gym.make(self.config.env_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n
        return env #, self.state_size, self.action_size

    def init_agent(self):
        agent = Agent(self.state_size, self.action_size, hidden_size=self.config.hidden_size, config=self.config)
        return agent

    def init_results_dir(self):
        results_dir = Path('./results') / self.config.env_name / self.config.algorithm
        # Todo: 如果存在相同名字直接覆盖
        # if not model_dir.exists():
        seed_dir = f'{self.config.algorithm}_{self.config.seed}'
        logs_dir = results_dir/seed_dir
        checkpoint_dir = logs_dir / 'checkpoint'
        if logs_dir.exists():
            shutil.rmtree(logs_dir, ignore_errors=True)
        os.makedirs(checkpoint_dir)
        writer = SummaryWriter(logs_dir)
        return logs_dir, checkpoint_dir, writer


