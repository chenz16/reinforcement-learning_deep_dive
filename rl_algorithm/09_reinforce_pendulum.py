"""
REINFORCE (Monte Carlo Policy Gradient) on Pendulum-v1
=======================================================

最基础的策略梯度算法（Williams 1992）。

核心思想：
1. 用当前策略跑完一整个 episode
2. 计算每一步的折扣回报 G_t = Σ γ^k · r_{t+k}
3. 策略梯度：∇J = E[G_t · ∇log π(a_t|s_t)]
   - G_t > 0 的动作：增大其概率
   - G_t < 0 的动作：减小其概率（Pendulum 的 reward 全是负的，所以看相对大小）
4. 加 baseline（减去均值）降低方差

特点：
  - 无 Critic：不学 V(s) 或 Q(s,a)，直接用 MC 回报
  - On-policy：数据用完即弃
  - 高方差：因为用完整 episode 的 MC 回报，波动大
  - 最简单的策略梯度，是所有 Actor-Critic 方法的基础

与 PPO 的关系：
  REINFORCE → 加 Critic baseline → A2C → 加 clipping → PPO
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
LR = 1e-3
GAMMA = 0.99
NUM_EPISODES = 500          # REINFORCE 收敛慢，需要更多 episode
MAX_STEPS = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ============================================================
# Policy 网络（只有 Actor，没有 Critic）
# ============================================================
class Policy(nn.Module):
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
        return action, log_prob


# ============================================================
# 训练循环
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        # ---- 跑完一整个 episode ----
        for step in range(MAX_STEPS):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action, log_prob = policy.act(state_t)
            action_np = action.cpu().detach().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action_np)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if terminated or truncated:
                break

        episode_rewards.append(sum(rewards))

        # ---- 计算折扣回报 G_t ----
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(DEVICE)

        # Baseline：减去均值，降低方差
        # 不减均值也能收敛，但方差极大，训练不稳定
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ---- REINFORCE 梯度更新 ----
        # loss = -Σ G_t · log π(a_t|s_t)
        # 含义：G_t 大的动作，增大其 log_prob（增大概率）
        #        G_t 小的动作，减小其 log_prob（减小概率）
        policy_loss = 0.0
        for log_prob, G_t in zip(log_probs, returns):
            policy_loss -= log_prob * G_t

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (episode + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1:4d} | Avg Reward (last 20): {avg:7.1f}")

    env.close()
    return policy, episode_rewards


# ============================================================
# 可视化
# ============================================================
def plot_rewards(rewards, save_path="reinforce_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("REINFORCE on Pendulum-v1 (No Critic)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    policy, rewards = train()
    plot_rewards(rewards)
    print("Done.")
