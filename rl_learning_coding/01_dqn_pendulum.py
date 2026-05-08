"""
DQN on Pendulum-v1 (with discretized actions)
==============================================

核心概念：
1. Q(s,a) 表示在状态 s 执行动作 a 后，按最优策略能获得的期望累计回报
2. Bellman 方程：Q(s,a) = r + γ·max_a' Q(s',a')
3. 经验回放：打破样本间的时间相关性，提高数据利用率
4. 目标网络：用一个滞后更新的网络计算 target，稳定训练
5. ε-greedy：以 ε 概率随机探索，1-ε 概率选最优动作

Pendulum-v1:
  - 观测: [cos(θ), sin(θ), θ_dot]  (3维)
  - 动作: torque ∈ [-2, 2]  (连续，我们离散化为 N 个 bin)
  - 奖励: -(θ² + 0.1·θ_dot² + 0.001·torque²)，最优约 0，最差约 -16.27
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random

# ============================================================
# 超参数
# ============================================================
NUM_DISCRETE_ACTIONS = 11       # 将 [-2, 2] 离散为 11 个 bin
HIDDEN_DIM = 128
LR = 1e-3
GAMMA = 0.99                    # 折扣因子
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 200             # 线性衰减的 episode 数
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_MODE = "soft"     # "hard": 每 N episode 整体复制; "soft": 每步 Polyak averaging
TARGET_UPDATE_FREQ = 10         # hard 模式：每 N 个 episode 更新目标网络
TARGET_TAU = 0.005              # soft 模式：Polyak 系数 θ_target ← τ·θ + (1-τ)·θ_target
NUM_EPISODES = 300
MAX_STEPS = 200                 # Pendulum-v1 默认 200 步

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 动作离散化：将连续动作空间 [-2, 2] 映射到 N 个离散 bin
ACTION_BINS = np.linspace(-2.0, 2.0, NUM_DISCRETE_ACTIONS)


def discrete_to_continuous(action_idx: int) -> float:
    """离散动作索引 → 连续 torque 值"""
    return ACTION_BINS[action_idx]


# ============================================================
# Q 网络：输入 state(3维)，输出每个离散动作的 Q 值
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# SumTree：Prioritized Experience Replay 的核心数据结构
# 用完全二叉树存储优先级，支持 O(log N) 的按优先级采样
# 叶子节点存优先级值，内部节点存子树的优先级之和
# 采样时：生成 [0, total_priority) 的随机数，沿树向下找到对应的叶子
# ============================================================
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)  # 二叉树数组
        self.data = [None] * capacity                              # 叶子节点存数据
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """更新从叶子到根的路径上所有节点的和"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """根据累积优先级 s 找到对应的叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

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
        """返回 (tree_idx, priority, data)"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ============================================================
# Prioritized Experience Replay (PER)
#
# 均匀采样的问题：所有样本被抽到的概率相同，已经学好的"简单"样本
# 和 TD error 很大的"难"样本被同等对待，浪费计算。
#
# PER 的做法：
#   - 每个 transition 的采样概率 ∝ |TD error|^α + ε
#   - α=0 退化为均匀采样，α=1 完全按优先级
#   - 非均匀采样会引入偏差，用 importance sampling weight 修正：
#       w_i = (1 / (N · P(i)))^β
#   - β 从小逐渐退火到 1，训练后期完全修正偏差
#   - 权重归一化：w_i / max(w)，避免梯度爆炸
# ============================================================
PER_ALPHA = 0.6        # 优先级指数：0=均匀，1=完全按优先级
PER_BETA_START = 0.4   # IS 修正指数起始值
PER_BETA_END = 1.0     # IS 修正指数终止值（训练结束时达到 1）
PER_EPSILON = 1e-5     # 防止优先级为 0


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_priority = 1.0  # 新样本用最大优先级（保证至少被采样一次）

    def push(self, state, action, reward, next_state, done):
        # 新样本给最大优先级，确保每条经验至少被采到一次
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int, beta: float = PER_BETA_START):
        indices = []
        priorities = []
        batch = []

        # 将总优先级均匀分成 batch_size 段，每段随机采一个
        # 这叫 stratified sampling，比纯随机更均匀地覆盖优先级分布
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        # Importance Sampling 权重
        # P(i) = priority_i / total_priority
        # w_i = (1 / (N * P(i)))^β  然后归一化
        N = self.tree.size
        priorities = np.array(priorities, dtype=np.float64)
        sampling_probs = priorities / self.tree.total
        weights = (N * sampling_probs) ** (-beta)
        weights /= weights.max()  # 归一化到 [0, 1]，避免梯度爆炸

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(indices, dtype=np.int64),
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """用新的 TD error 更新对应样本的优先级"""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + PER_EPSILON) ** PER_ALPHA
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size


# ============================================================
# 训练循环
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]  # 3

    q_net = QNetwork(state_dim, NUM_DISCRETE_ACTIONS).to(DEVICE)
    # 目标网络：结构相同，参数定期从 q_net 复制
    # 为什么需要：如果用同一个网络既选动作又计算 target，
    # target 会随训练不断移动，导致训练不稳定（追赶移动的目标）
    target_net = QNetwork(state_dim, NUM_DISCRETE_ACTIONS).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0

        # ε-greedy: ε 从 1.0 线性衰减到 0.01
        epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)
        # β 从 PER_BETA_START 线性退火到 1.0
        beta = min(PER_BETA_END, PER_BETA_START + episode / NUM_EPISODES * (PER_BETA_END - PER_BETA_START))

        for step in range(MAX_STEPS):
            # 选动作
            if random.random() < epsilon:
                action_idx = random.randrange(NUM_DISCRETE_ACTIONS)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    q_values = q_net(state_t)
                    action_idx = q_values.argmax(dim=1).item()

            continuous_action = discrete_to_continuous(action_idx)
            next_state, reward, terminated, truncated, _ = env.step([continuous_action])
            done = terminated or truncated

            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 当缓冲区够大时开始训练
            if len(replay_buffer) >= BATCH_SIZE:
                # 按优先级采样（TD error 大的样本被采到的概率更高）
                s, a, r, s2, d, tree_indices, is_weights = replay_buffer.sample(BATCH_SIZE, beta)
                s = torch.FloatTensor(s).to(DEVICE)
                a = torch.LongTensor(a).to(DEVICE)
                r = torch.FloatTensor(r).to(DEVICE)
                s2 = torch.FloatTensor(s2).to(DEVICE)
                d = torch.FloatTensor(d).to(DEVICE)
                is_weights = torch.FloatTensor(is_weights).to(DEVICE)

                # 当前 Q 值：Q(s, a)
                current_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                # 目标 Q 值：r + γ·max_a' Q_target(s', a')  (Bellman 方程)
                with torch.no_grad():
                    next_q = target_net(s2).max(dim=1)[0]
                    target_q = r + GAMMA * next_q * (1.0 - d)

                # TD error 用于更新优先级
                td_errors = (current_q - target_q).detach().cpu().numpy()
                replay_buffer.update_priorities(tree_indices, td_errors)

                # 加权 MSE loss：IS 权重修正非均匀采样带来的偏差
                loss = (is_weights * (current_q - target_q) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # soft 模式：每个训练 step 都做 Polyak averaging
                if TARGET_UPDATE_MODE == "soft":
                    for tp, sp in zip(target_net.parameters(), q_net.parameters()):
                        tp.data.copy_(TARGET_TAU * sp.data + (1.0 - TARGET_TAU) * tp.data)

            if done:
                break

        episode_rewards.append(total_reward)

        # hard 模式：每 N 个 episode 整体复制一次
        if TARGET_UPDATE_MODE == "hard" and (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (episode + 1) % 20 == 0:
            avg = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1:4d} | Avg Reward (last 20): {avg:7.1f} | ε: {epsilon:.3f}")

    env.close()
    return q_net, episode_rewards


# ============================================================
# 可视化
# ============================================================
def plot_rewards(rewards, save_path="dqn_rewards.png"):
    """Episode reward 曲线"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    # 滑动平均
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN on Pendulum-v1 (Discretized)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_q_heatmap(q_net, save_path="dqn_q_heatmap.png"):
    """
    Q 值热力图：在 (θ, θ_dot) 平面上可视化 max_a Q(s,a)
    state = [cos(θ), sin(θ), θ_dot]
    """
    thetas = np.linspace(-np.pi, np.pi, 100)
    theta_dots = np.linspace(-8.0, 8.0, 100)
    q_map = np.zeros((len(theta_dots), len(thetas)))

    q_net.eval()
    with torch.no_grad():
        for i, td in enumerate(theta_dots):
            states = np.array(
                [[np.cos(th), np.sin(th), td] for th in thetas], dtype=np.float32
            )
            q_vals = q_net(torch.FloatTensor(states).to(DEVICE))
            q_map[i, :] = q_vals.max(dim=1)[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        q_map,
        extent=[-np.pi, np.pi, -8, 8],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title("max_a Q(s, a) Heatmap")
    fig.colorbar(im, ax=ax, label="Q value")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved Q heatmap to {save_path}")
    plt.close(fig)


def plot_episode_frames(q_net, save_path="dqn_episode_frames.png", num_frames=8):
    """用训练好的策略跑一个 episode，截取 N 帧展示"""
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    state, _ = env.reset()
    frames = []

    q_net.eval()
    for step in range(MAX_STEPS):
        if step % (MAX_STEPS // num_frames) == 0:
            frames.append(env.render())

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action_idx = q_net(state_t).argmax(dim=1).item()

        continuous_action = discrete_to_continuous(action_idx)
        state, _, terminated, truncated, _ = env.step([continuous_action])
        if terminated or truncated:
            break

    env.close()

    if not frames:
        return

    fig, axes = plt.subplots(1, len(frames), figsize=(2.5 * len(frames), 2.5))
    if len(frames) == 1:
        axes = [axes]
    for ax, frame in zip(axes, frames):
        ax.imshow(frame)
        ax.axis("off")
    fig.suptitle("DQN Trained Policy — Episode Frames")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved episode frames to {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    q_net, rewards = train()

    plot_rewards(rewards)
    plot_q_heatmap(q_net)
    plot_episode_frames(q_net)
    print("Done.")
