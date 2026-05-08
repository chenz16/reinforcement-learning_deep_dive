"""
DDPG (Deep Deterministic Policy Gradient) on Pendulum-v1
=========================================================

SAC 的前身（Lillicrap 2015），连续动作空间的 DQN。

核心思想：
1. Actor 输出确定性动作 μ(s)（不是分布，直接输出一个值）
2. Critic 评估 Q(s, a)
3. Actor 更新：沿着 Critic 梯度方向调整动作
   loss = -Q(s, μ(s))  → 最大化 Q 值
4. 探索：在动作上加外部噪声（OU noise 或高斯噪声）
5. Off-policy：用 replay buffer

与 SAC 的区别：
  - DDPG: 确定性策略 + 外挂噪声探索 + 单 Critic
  - SAC:  随机策略 + 熵正则化探索 + 双 Critic + 自动温度

DDPG 的问题：
  - 对超参数极其敏感，容易训练崩溃
  - 单 Critic 容易高估 Q 值
  - 确定性策略探索不足
  → TD3 和 SAC 分别从不同角度解决这些问题
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import random

# ============================================================
# 超参数
# ============================================================
HIDDEN_DIM = 256
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 0.005
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 256
NUM_EPISODES = 200
MAX_STEPS = 200
WARMUP_STEPS = 1000
EXPLORATION_NOISE = 0.1    # 高斯探索噪声标准差

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0


# ============================================================
# Actor 网络：确定性策略 μ(s)
# 直接输出动作值，不是分布
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),     # 输出 [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * ACTION_SCALE   # 缩放到 [-2, 2]


# ============================================================
# Critic 网络：Q(s, a) — 只有一个（DDPG 的弱点之一）
# ============================================================
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


# ============================================================
# Replay Buffer（简单版，不用 PER）
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.float32),
            np.array(r, dtype=np.float32).reshape(-1, 1),
            np.array(s2, dtype=np.float32),
            np.array(d, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ============================================================
# 训练循环
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim).to(DEVICE)
    target_actor = Actor(state_dim, action_dim).to(DEVICE)
    target_critic = Critic(state_dim, action_dim).to(DEVICE)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    episode_rewards = []
    actor_losses = []
    critic_losses = []

    total_steps = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            total_steps += 1

            # 选动作：确定性策略 + 高斯噪声探索
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = actor(state_t).cpu().numpy()[0]
                # 加探索噪声
                noise = np.random.normal(0, EXPLORATION_NOISE * ACTION_SCALE, size=action.shape)
                action = np.clip(action + noise, -ACTION_SCALE, ACTION_SCALE)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= BATCH_SIZE and total_steps >= WARMUP_STEPS:
                s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)
                s = torch.FloatTensor(s).to(DEVICE)
                a = torch.FloatTensor(a).to(DEVICE)
                r = torch.FloatTensor(r).to(DEVICE)
                s2 = torch.FloatTensor(s2).to(DEVICE)
                d = torch.FloatTensor(d).to(DEVICE)

                # ---- Critic 更新 ----
                # target = r + γ·Q_target(s', μ_target(s'))
                with torch.no_grad():
                    target_action = target_actor(s2)
                    target_q = target_critic(s2, target_action)
                    target_value = r + GAMMA * (1.0 - d) * target_q

                current_q = critic(s, a)
                critic_loss = F.mse_loss(current_q, target_value)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # ---- Actor 更新 ----
                # 最大化 Q(s, μ(s)) → 最小化 -Q
                actor_loss = -critic(s, actor(s)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update
                soft_update(target_actor, actor, TAU)
                soft_update(target_critic, critic, TAU)

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

            if done:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1:4d} | Avg Reward (last 10): {avg:7.1f}")

    env.close()
    return actor, episode_rewards, actor_losses, critic_losses


# ============================================================
# 可视化
# ============================================================
def plot_rewards(rewards, save_path="ddpg_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 10
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DDPG on Pendulum-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_losses(actor_losses, critic_losses, save_path="ddpg_losses.png"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(actor_losses)
    axes[0].set_ylabel("Actor Loss")
    axes[0].set_title("DDPG Training Curves")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(critic_losses)
    axes[1].set_ylabel("Critic Loss")
    axes[1].set_xlabel("Update Step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved loss curves to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    actor, rewards, actor_losses, critic_losses = train()
    plot_rewards(rewards)
    plot_losses(actor_losses, critic_losses)
    print("Done.")
