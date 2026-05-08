"""
SAC (Soft Actor-Critic) on Pendulum-v1
=======================================

核心概念：
1. Actor-Critic：Actor 输出动作分布，Critic 评估状态-动作对的 Q 值
2. 最大熵 RL：目标不只是最大化回报，还要最大化策略的熵（鼓励探索）
     J(π) = Σ E[r + α·H(π(·|s))]
3. 双 Critic：用两个 Q 网络取较小值，减少 Q 值高估（overestimation bias）
4. 重参数化技巧：a = tanh(μ + σ·ε), ε ~ N(0,1)，使采样可微分
5. 自动温度调节：α 自动调整以维持目标熵水平
6. Soft target update (Polyak averaging)：τ 很小，target 网络缓慢跟踪

网络架构：
  - Actor:  state → [hidden×2] → (mean, log_std) → tanh squash → action ∈ [-2,2]
  - Critic: (state, action) → [hidden×2] → Q value  (×2 个独立 Critic)
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random

# ============================================================
# 超参数
# ============================================================
HIDDEN_DIM = 256
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
GAMMA = 0.99
TAU = 0.005                # Polyak averaging 系数（soft update）
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 256
NUM_EPISODES = 200
MAX_STEPS = 200
WARMUP_STEPS = 1000        # 随机探索的步数，之后才开始用 Actor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ACTION_SCALE = 2.0         # Pendulum 动作范围 [-2, 2]


# ============================================================
# Actor 网络
# 输出高斯分布的 mean 和 log_std，通过重参数化采样，再 tanh 压缩到动作范围
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        重参数化采样：
          1. 网络输出 mean, log_std
          2. 构建 Normal(mean, std) 分布
          3. 采样 x = mean + std * ε,  ε ~ N(0,1)  （这步可微分！）
          4. action = tanh(x) * scale  （squash 到动作范围）
          5. 修正 log_prob：log π(a|s) = log p(x) - Σ log(1 - tanh²(x))
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        # 重参数化：x = mean + std * ε
        x = dist.rsample()
        action = torch.tanh(x) * ACTION_SCALE

        # 修正 log probability（因为 tanh 变换改变了概率密度）
        log_prob = dist.log_prob(x) - torch.log(1.0 - torch.tanh(x).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# ============================================================
# Critic 网络 (单个 Q 函数)
# 输入 (state, action)，输出标量 Q 值
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
# SumTree：Prioritized Experience Replay 的核心数据结构
# 用完全二叉树存储优先级，支持 O(log N) 的按优先级采样
# ============================================================
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]


# ============================================================
# Prioritized Experience Replay (PER)
# 采样概率 ∝ |TD error|^α + ε，用 IS 权重修正偏差
# ============================================================
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-5


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int, beta: float = PER_BETA_START):
        indices = []
        priorities = []
        batch = []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        N = self.tree.size
        priorities = np.array(priorities, dtype=np.float64)
        sampling_probs = priorities / self.tree.total
        weights = (N * sampling_probs) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32).reshape(-1, 1),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + PER_EPSILON) ** PER_ALPHA
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size


# ============================================================
# Soft target update (Polyak averaging)
# θ_target ← τ·θ + (1-τ)·θ_target
# 为什么：比硬拷贝更平滑，训练更稳定
# ============================================================
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ============================================================
# 训练循环
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]      # 1

    # 网络初始化
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic1 = Critic(state_dim, action_dim).to(DEVICE)
    critic2 = Critic(state_dim, action_dim).to(DEVICE)
    target_critic1 = Critic(state_dim, action_dim).to(DEVICE)
    target_critic2 = Critic(state_dim, action_dim).to(DEVICE)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    # 自动温度 α：用对数空间参数化保证 α > 0
    # 目标熵 = -dim(action)，即希望策略的熵不低于这个值
    target_entropy = -float(action_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp().item()

    # 优化器
    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=LR_CRITIC)
    alpha_optimizer = optim.Adam([log_alpha], lr=LR_ALPHA)

    replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

    # 记录指标
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    alpha_values = []

    total_steps = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            total_steps += 1

            # 选动作：warmup 期间随机，之后用 Actor
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action, _ = actor.sample(state_t)
                    action = action.cpu().numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 训练
            if len(replay_buffer) >= BATCH_SIZE and total_steps >= WARMUP_STEPS:
                # β 线性退火
                beta = min(PER_BETA_END, PER_BETA_START + total_steps / (NUM_EPISODES * MAX_STEPS) * (PER_BETA_END - PER_BETA_START))
                s, a, r, s2, d, tree_indices, is_weights = replay_buffer.sample(BATCH_SIZE, beta)
                s = torch.FloatTensor(s).to(DEVICE)
                a = torch.FloatTensor(a).to(DEVICE)
                r = torch.FloatTensor(r).to(DEVICE)
                s2 = torch.FloatTensor(s2).to(DEVICE)
                d = torch.FloatTensor(d).to(DEVICE)
                is_weights = torch.FloatTensor(is_weights).to(DEVICE)

                # ---- Critic 更新 ----
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample(s2)
                    target_q1 = target_critic1(s2, next_action)
                    target_q2 = target_critic2(s2, next_action)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                    target_value = r + GAMMA * (1.0 - d) * target_q

                current_q1 = critic1(s, a)
                current_q2 = critic2(s, a)

                # 用 TD error 更新优先级（取两个 Critic 的较大 TD error）
                td1 = (current_q1 - target_value).detach().cpu().numpy().flatten()
                td2 = (current_q2 - target_value).detach().cpu().numpy().flatten()
                td_errors = np.maximum(np.abs(td1), np.abs(td2))
                replay_buffer.update_priorities(tree_indices, td_errors)

                # IS 加权 loss
                critic1_loss = (is_weights * (current_q1 - target_value) ** 2).mean()
                critic2_loss = (is_weights * (current_q2 - target_value) ** 2).mean()

                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()

                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                # ---- Actor 更新 ----
                # 最大化 Q(s, π(s)) + α·H(π) = 最小化 α·log_prob - Q
                new_action, new_log_prob = actor.sample(s)
                q1_new = critic1(s, new_action)
                q2_new = critic2(s, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * new_log_prob - q_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ---- Temperature α 更新 ----
                # 自动调节：如果熵太低（< target_entropy），增大 α 鼓励探索
                alpha_loss = -(log_alpha * (new_log_prob.detach() + target_entropy)).mean()

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp().item()

                # Soft update target networks
                soft_update(target_critic1, critic1, TAU)
                soft_update(target_critic2, critic2, TAU)

                # 记录（每个 episode 只记录最后一次 update 的值）
                _actor_loss = actor_loss.item()
                _critic_loss = (critic1_loss.item() + critic2_loss.item()) / 2

            if done:
                break

        episode_rewards.append(total_reward)

        if total_steps >= WARMUP_STEPS:
            actor_losses.append(_actor_loss)
            critic_losses.append(_critic_loss)
            alpha_values.append(alpha)
        else:
            actor_losses.append(0.0)
            critic_losses.append(0.0)
            alpha_values.append(alpha)

        if (episode + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1:4d} | Avg Reward (last 10): {avg:7.1f} | α: {alpha:.4f}")

    env.close()
    return actor, episode_rewards, actor_losses, critic_losses, alpha_values


# ============================================================
# 可视化
# ============================================================
def plot_rewards(rewards, save_path="sac_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 10
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("SAC on Pendulum-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_losses(actor_losses, critic_losses, alpha_values, save_path="sac_losses.png"):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(actor_losses)
    axes[0].set_ylabel("Actor Loss")
    axes[0].set_title("SAC Training Curves")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(critic_losses)
    axes[1].set_ylabel("Critic Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(alpha_values)
    axes[2].set_ylabel("Temperature α")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved loss curves to {save_path}")
    plt.close(fig)


def plot_policy_distribution(actor, save_path="sac_policy_dist.png"):
    """
    可视化策略在不同状态下输出的动作分布
    选几个代表性的 (θ, θ_dot) 状态，画出 Actor 输出的高斯分布
    """
    test_states = [
        (0.0, 0.0, "θ=0, ω=0 (upright)"),
        (np.pi, 0.0, "θ=π, ω=0 (down)"),
        (np.pi / 2, 0.0, "θ=π/2, ω=0"),
        (0.0, 4.0, "θ=0, ω=4"),
        (np.pi, -4.0, "θ=π, ω=-4"),
        (np.pi / 4, 2.0, "θ=π/4, ω=2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    actor.eval()

    actions_range = np.linspace(-ACTION_SCALE, ACTION_SCALE, 200)

    for ax, (theta, theta_dot, label) in zip(axes, test_states):
        state = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mean, log_std = actor(state_t)
            mean = mean.cpu().numpy()[0, 0]
            std = log_std.exp().cpu().numpy()[0, 0]

        # 未经 tanh 的高斯分布密度，经 tanh 变换后的密度
        # p(a) = p_gaussian(atanh(a/scale)) / (scale * (1 - (a/scale)²))
        a_normalized = actions_range / ACTION_SCALE
        # clip to avoid atanh of ±1
        a_clipped = np.clip(a_normalized, -0.999, 0.999)
        x = np.arctanh(a_clipped)
        gaussian_pdf = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        # Jacobian correction
        squashed_pdf = gaussian_pdf / (ACTION_SCALE * (1 - a_clipped**2) + 1e-6)

        ax.fill_between(actions_range, squashed_pdf, alpha=0.3)
        ax.plot(actions_range, squashed_pdf)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Action (torque)")
        ax.set_ylabel("Density")
        ax.set_xlim(-ACTION_SCALE * 1.1, ACTION_SCALE * 1.1)

    fig.suptitle("SAC Policy Action Distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved policy distribution to {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    actor, rewards, actor_losses, critic_losses, alpha_values = train()

    plot_rewards(rewards)
    plot_losses(actor_losses, critic_losses, alpha_values)
    plot_policy_distribution(actor)
    print("Done.")
