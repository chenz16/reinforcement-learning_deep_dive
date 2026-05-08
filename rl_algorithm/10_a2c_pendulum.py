"""
A2C (Advantage Actor-Critic) on Pendulum-v1
=============================================

REINFORCE 的改进：加入 Critic 网络作为 baseline。

演进关系：
  REINFORCE:  loss = -G_t · log π(a|s)           ← MC 回报，高方差
  A2C:        loss = -A_t · log π(a|s)           ← 用 advantage 替代 G_t
  PPO:        loss = -min(ratio·A, clip(ratio)·A) ← 加 clipping 限制更新幅度

A2C 的核心：
1. Critic 学习 V(s)
2. Advantage A_t = r + γ·V(s') - V(s)  （TD advantage，比 MC 回报方差低很多）
3. Actor 用 advantage 加权的策略梯度更新
4. 同步版本：单个 worker 收集数据（A3C 用多个并行 worker）

与 PPO 的区别：
  - A2C 没有 clipping，ratio 可以任意大 → 策略可能跳太远
  - A2C 只用数据一次（单 epoch），PPO 多 epoch 复用
  - PPO 更稳定，A2C 更简单
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

# ============================================================
# 超参数
# ============================================================
HIDDEN_DIM = 64
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
ENTROPY_COEFF = 0.01       # 熵正则系数，鼓励探索
NUM_EPISODES = 300
MAX_STEPS = 200
ROLLOUT_STEPS = 2048       # 每次收集多少步数据

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ============================================================
# Actor 网络
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        mean = self.mean_head(h)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Normal(mean, std)

    def act(self, state: torch.Tensor):
        dist = self.forward(state)
        x = dist.sample()
        action = torch.tanh(x) * ACTION_SCALE
        log_prob = dist.log_prob(x) - torch.log(1.0 - torch.tanh(x).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, x


# ============================================================
# Critic 网络：估计 V(s)
# ============================================================
class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ============================================================
# 训练循环
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim).to(DEVICE)
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    episode_rewards = []
    actor_losses_hist = []
    critic_losses_hist = []

    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0

    while episode_count < NUM_EPISODES:
        # ---- 收集一批数据 ----
        states, log_probs, entropies = [], [], []
        rewards, dones, values = [], [], []

        for _ in range(ROLLOUT_STEPS):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                value = critic(state_t).item()

            action, log_prob, entropy, _ = actor.act(state_t)
            action_np = action.cpu().detach().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            states.append(state)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)

            episode_reward += reward
            state = next_state

            if done:
                episode_rewards.append(episode_reward)
                episode_count += 1
                episode_reward = 0.0
                state, _ = env.reset()
                if episode_count >= NUM_EPISODES:
                    break

        if not states:
            break

        # ---- 计算 TD advantage ----
        # A_t = r_t + γ·V(s_{t+1}) - V(s_t)
        # 比 REINFORCE 的 MC 回报方差低很多
        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        with torch.no_grad():
            all_values = critic(states_t).squeeze(-1)

        advantages = []
        returns = []
        for t in range(len(rewards)):
            if t + 1 < len(states):
                next_val = all_values[t + 1].item() if not dones[t] else 0.0
            else:
                next_val = 0.0
            # TD advantage
            adv = rewards[t] + GAMMA * next_val * (1.0 - dones[t]) - values[t]
            ret = rewards[t] + GAMMA * next_val * (1.0 - dones[t])
            advantages.append(adv)
            returns.append(ret)

        advantages_t = torch.FloatTensor(advantages).to(DEVICE)
        returns_t = torch.FloatTensor(returns).to(DEVICE)

        # 标准化 advantage
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ---- Actor 更新（单 epoch，无 clipping） ----
        # loss = -A_t · log π(a|s) - entropy_coeff · H(π)
        log_probs_t = torch.stack(log_probs).squeeze()
        entropies_t = torch.stack(entropies).squeeze()

        actor_loss = -(advantages_t * log_probs_t).mean() - ENTROPY_COEFF * entropies_t.mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_optimizer.step()

        # ---- Critic 更新 ----
        value_pred = critic(states_t).squeeze(-1)
        critic_loss = nn.functional.mse_loss(value_pred, returns_t)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_optimizer.step()

        actor_losses_hist.append(actor_loss.item())
        critic_losses_hist.append(critic_loss.item())

        if episode_count % 20 == 0 and episode_count > 0:
            avg = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"Episode {episode_count:4d} | Avg Reward (last 20): {avg:7.1f}")

    env.close()
    return actor, episode_rewards, actor_losses_hist, critic_losses_hist


# ============================================================
# 可视化
# ============================================================
def plot_rewards(rewards, save_path="a2c_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("A2C on Pendulum-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_losses(actor_losses, critic_losses, save_path="a2c_losses.png"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(actor_losses)
    axes[0].set_ylabel("Actor Loss")
    axes[0].set_title("A2C Training Curves")
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
