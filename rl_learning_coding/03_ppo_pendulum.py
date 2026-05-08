"""
PPO (Proximal Policy Optimization) on Pendulum-v1
==================================================

核心概念：
1. On-policy：每次用当前策略收集一批数据，训练完就扔掉，不存 replay buffer
2. 重要性采样比 r(θ) = π_new(a|s) / π_old(a|s)：用旧数据估计新策略的梯度
3. Clipping：限制 r(θ) 在 [1-ε, 1+ε] 范围内，防止策略更新过大
   - 如果 advantage > 0（好动作），r(θ) 最多到 1+ε，不会无限增大概率
   - 如果 advantage < 0（差动作），r(θ) 最多到 1-ε，不会过度惩罚
4. GAE (Generalized Advantage Estimation)：平衡 bias-variance 的 advantage 估计
5. 多 epoch 更新：同一批数据可以训练多个 epoch（因为 clipping 保护）

与 SAC 的关键区别：
  - PPO 是 on-policy：数据用完即弃，sample efficiency 低但稳定
  - SAC 是 off-policy：数据存 buffer 反复用，sample efficiency 高
  - PPO 用 clipping 限制更新幅度；SAC 用熵正则化鼓励探索
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
GAE_LAMBDA = 0.95          # GAE 的 λ 参数，1.0=MC，0.0=TD(0)
CLIP_EPS = 0.2             # PPO clipping 范围
PPO_EPOCHS = 10            # 每批数据训练多少个 epoch
BATCH_SIZE = 64
NUM_EPISODES = 300
MAX_STEPS = 200
ROLLOUT_STEPS = 2048       # 每次收集多少步数据再训练

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ============================================================
# Actor 网络：输出高斯分布的 mean 和 log_std
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
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可训练的全局 log_std

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

    def evaluate(self, state: torch.Tensor, raw_action: torch.Tensor):
        """给定 state 和 squash 前的原始动作，计算 log_prob 和 entropy"""
        dist = self.forward(state)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ============================================================
# Critic 网络：输入 state，输出 V(s)
# 注意：PPO 的 Critic 估计的是 V(s)，不是 Q(s,a)
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
# GAE (Generalized Advantage Estimation)
# A_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}
# 其中 δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
#
# λ=0: A_t = δ_t = r + γV(s') - V(s)       ← 低方差、高偏差 (TD)
# λ=1: A_t = Σ γ^l·r_{t+l} - V(s_t)        ← 高方差、低偏差 (MC)
# λ=0.95: 折中
# ============================================================
def compute_gae(rewards, values, next_values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


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
        # ---- 收集 rollout 数据 (on-policy) ----
        states, raw_actions, actions, log_probs = [], [], [], []
        rewards, dones, values, next_values = [], [], [], []

        for _ in range(ROLLOUT_STEPS):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                dist = actor(state_t)
                raw_action = dist.sample()
                action = torch.tanh(raw_action) * ACTION_SCALE
                log_prob = dist.log_prob(raw_action) - torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1)
                value = critic(state_t).item()

            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            states.append(state)
            raw_actions.append(raw_action.cpu().numpy()[0])
            actions.append(action_np)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)

            with torch.no_grad():
                next_v = critic(torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)).item()
            next_values.append(next_v)

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

        # ---- 计算 GAE advantage 和 returns ----
        advantages, returns = compute_gae(rewards, values, next_values, dones)

        # 转成 tensor
        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        raw_actions_t = torch.FloatTensor(np.array(raw_actions)).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(np.array(log_probs)).to(DEVICE)
        advantages_t = torch.FloatTensor(advantages).to(DEVICE)
        returns_t = torch.FloatTensor(returns).to(DEVICE)

        # 标准化 advantage（减均值除标准差，稳定训练）
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ---- PPO 多 epoch 更新 ----
        dataset_size = len(states)
        for epoch in range(PPO_EPOCHS):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                idx = indices[start:end]

                s_batch = states_t[idx]
                ra_batch = raw_actions_t[idx]
                old_lp_batch = old_log_probs_t[idx]
                adv_batch = advantages_t[idx]
                ret_batch = returns_t[idx]

                # 用当前策略重新计算 log_prob
                new_log_prob, entropy = actor.evaluate(s_batch, ra_batch)

                # 重要性采样比
                ratio = (new_log_prob - old_lp_batch).exp()

                # PPO clipping
                # 未裁剪的目标：ratio * advantage
                # 裁剪后的目标：clip(ratio, 1-ε, 1+ε) * advantage
                # 取两者的较小值 → 悲观估计，防止策略跳太远
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

                # Critic loss: MSE(V(s), returns)
                value_pred = critic(s_batch).squeeze(-1)
                critic_loss = nn.functional.mse_loss(value_pred, ret_batch)

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
def plot_rewards(rewards, save_path="ppo_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("PPO on Pendulum-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_losses(actor_losses, critic_losses, save_path="ppo_losses.png"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(actor_losses)
    axes[0].set_ylabel("Actor Loss")
    axes[0].set_title("PPO Training Curves")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(critic_losses)
    axes[1].set_ylabel("Critic Loss")
    axes[1].set_xlabel("Update Step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved loss curves to {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    actor, rewards, actor_losses, critic_losses = train()
    plot_rewards(rewards)
    plot_losses(actor_losses, critic_losses)
    print("Done.")
