import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import DQN
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from utils import *


class Agent:
    """与环境交互并且学习好的策略"""
    def __init__(self, state_size, action_size, hidden_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        self.q_local = DQN(state_size, action_size, hidden_size).to(self.device)
        self.q_target = DQN(state_size, action_size, hidden_size).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=config.lr)

        # ReplayBuffer
        self.buffer = ReplayBuffer(action_size, buffer_size=config.buffer_size, batch_size=config.batch_size)

        # 训练时间步骤的初始化
        self.t_step = 0
        self.q_updates = 0
        self.action_step = 4
        self.last_action = None

        self.writer = SummaryWriter('result')

    def step(self, state, action, reward, next_state, done):
        """往buffer中保存经验， 并且使用随机抽样进行学习"""
        self.buffer.add(state, action, reward, next_state, done)  # 保存经验

        # 每隔多久更新一次
        self.t_step = self.t_step + 1
        if (self.t_step) % self.config.update_every == 0:
            # 如果buffe中的数据存储的够多了，就可以学习
            if len(self.buffer) > self.config.batch_size:
                experiences = self.buffer.sample()
                loss = self.learn(experiences)
                self.q_updates += 1
                self.writer.add_scalar('q_loss', loss, self.q_updates)

    def get_action(self, state, epsilon=0.):
        # 根据当前策略返回给定状态的操作，确定性策略，画面每更新4帧多一次动作
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # 增加一个维度给batch_size
        self.q_local.eval()
        action_values = self.q_local(state).detach()
        self.q_local.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            action = np.argmax(action_values.cpu().numpy())
            self.last_action = action
            return action
        else:
            action = random.choice(np.arange(self.action_size))
            self.last_action = action
            return action

    def learn(self, experiences):
        """
        使用一个批次的经验轨迹数据来更新值网络和策略网络
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) ：这个是基于真实值的标签
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        config = self.config
        gamma = self.config.gamma
        alpha = self.config.alpha
        entropy_tau = self.config.entropy_tau
        states, actions, rewards, next_states, dones = experiences
        # 从target网络模型里得到预测next的Q值(获得一堆Q值)
        q_targets_next = self.q_target(next_states).detach()

        # 用LogSumExp计算entropy，目的是维持数值的稳定性，详情见博客
        # q_k_targets_next = q_targets_next
        v_targets_next = q_targets_next.max(1)[0].unsqueeze(-1)  # Paper为什么不用平均值
        # q_k_targets_next - v_k_targets_next q_targets_next对q_k_targets_next进行了广播
        logSum = torch.logsumexp((q_targets_next - v_targets_next) / entropy_tau, 1).unsqueeze(-1)
        # q_k_targets_next - v_k_targets_next
        tau_log_pi_next = q_targets_next - v_targets_next - entropy_tau * logSum

        # 目标策略，tau是温度系数，越小的话，不同动作的分布大小差异更大
        pi_target = F.softmax(q_targets_next / entropy_tau, dim=1)

        # q_targets的计算
        q_soft_targets = (gamma * (pi_target * (q_targets_next - tau_log_pi_next) * (1-dones)).sum(1)).unsqueeze(-1)

        # 用logSum计算munchausen的addon, 这里的q_k_targets是预测当前的states的值，而不是next_states
        # q_targets对q_k_targets进行了广播
        q_targets = self.q_target(states).detach()
        v_targets = q_targets.max(1)[0].unsqueeze(-1)
        logSum = torch.logsumexp((q_targets - v_targets) / entropy_tau, 1).unsqueeze(-1)
        tau_log_pi = q_targets - v_targets - entropy_tau * logSum
        munchausen_addon = tau_log_pi.gather(1, actions.long())

        # 计算munchausen reward
        munchausen_reward = rewards + alpha * torch.clamp(munchausen_addon, min=-1, max=0)
        q_targets = munchausen_reward + q_soft_targets

        # 用当前的状态去估计/预测Q值
        q_expected = self.q_local(states).gather(1, actions.long())

        # 计算loss,target是我们想去接近的（相当于真实值）
        loss = F.mse_loss(q_expected, q_targets)
        loss.backward()
        clip_grad_norm_(self.q_local.parameters(), max_norm=self.config.max_grad_norm)
        self.optimizer.step()

        # 软更新target
        self.soft_update(self.q_local, self.q_target, self.config.soft_update_tau)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

