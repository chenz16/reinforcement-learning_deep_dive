"""
TD3 (Twin Delayed DDPG) on Pendulum-v1
========================================

DDPG 的三个改进（Fujimoto 2018）：

1. Twin Critic（双 Q 网络）：
   - 两个独立的 Critic，取 min 计算 target
   - 减少 Q 值高估（和 SAC 一样的技巧）

2. Delayed Actor Update（延迟 Actor 更新）：
   - Critic 每步更新，Actor 每 d 步更新一次（d=2）
   - 让 Critic 先收敛到比较准的值，再指导 Actor
   - 避免 Actor 被不准确的 Critic 误导

3. Target Policy Smoothing（目标策略平滑）：
   - 计算 target 时给 target_actor 的动作加噪声
   - target_action = μ_target(s') + clip(noise, -c, c)
   - 防止 Critic 对某个精确动作过拟合，起到正则化效果

与 SAC 的对比：
  TD3: 确定性策略 + 外挂噪声 + delayed update + target smoothing
  SAC: 随机策略 + 熵正则化 + 自动温度
  两者都用双 Critic，性能相当，SAC 在探索上更优雅
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
EXPLORATION_NOISE = 0.1
POLICY_NOISE = 0.2         # Target policy smoothing 噪声
NOISE_CLIP = 0.5           # 噪声裁剪范围
ACTOR_UPDATE_FREQ = 2      # 每 N 步 Critic 更新才做一次 Actor 更新

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0


# ============================================================
# Actor：确定性策略（同 DDPG）
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state) * ACTION_SCALE


# ============================================================
# Critic：双 Q 网络（Twin Critic）
# 两个独立的 Q 网络放在一个类里
# ============================================================
class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


# ============================================================
# Replay Buffer
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
    critic = TwinCritic(state_dim, action_dim).to(DEVICE)
    target_actor = Actor(state_dim, action_dim).to(DEVICE)
    target_critic = TwinCritic(state_dim, action_dim).to(DEVICE)
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

            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = actor(state_t).cpu().numpy()[0]
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

                # ---- Critic 更新（每步都做） ----
                with torch.no_grad():
                    # 改进 3: Target Policy Smoothing
                    # 给 target actor 的输出加裁剪噪声
                    noise = torch.randn_like(a) * POLICY_NOISE
                    noise = noise.clamp(-NOISE_CLIP, NOISE_CLIP)
                    target_action = (target_actor(s2) + noise).clamp(-ACTION_SCALE, ACTION_SCALE)

                    # 改进 1: Twin Critic，取 min
                    target_q1, target_q2 = target_critic(s2, target_action)
                    target_q = torch.min(target_q1, target_q2)
                    target_value = r + GAMMA * (1.0 - d) * target_q

                current_q1, current_q2 = critic(s, a)
                critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                critic_losses.append(critic_loss.item())

                # ---- Actor 更新（改进 2: 延迟更新，每 d 步一次） ----
                if total_steps % ACTOR_UPDATE_FREQ == 0:
                    # 只用 Q1 来计算 Actor 的梯度（不需要 Q2）
                    actor_loss = -critic.q1_forward(s, actor(s)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Soft update（也延迟到 Actor 更新时才做）
                    soft_update(target_actor, actor, TAU)
                    soft_update(target_critic, critic, TAU)

                    actor_losses.append(actor_loss.item())

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
def plot_rewards(rewards, save_path="td3_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 10
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("TD3 on Pendulum-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_losses(actor_losses, critic_losses, save_path="td3_losses.png"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    axes[0].plot(actor_losses)
    axes[0].set_ylabel("Actor Loss")
    axes[0].set_title("TD3 Training Curves")
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
