"""
Double DQN on Pendulum-v1
==========================

DQN 的问题：高估偏差 (overestimation bias)
  - target = r + γ·max_a' Q_target(s', a')
  - 同一个 target_net 既选动作（max）又评估 Q 值
  - max 操作会系统性地选中被高估的动作 → Q 值越来越虚高

Double DQN 的解法（Van Hasselt 2016）：
  - 用 q_net（在线网络）选动作：a* = argmax_a' Q_online(s', a')
  - 用 target_net 评估该动作的 Q 值：Q_target(s', a*)
  - 解耦了"选动作"和"评估"，减少高估

代码改动：只改了一行 target 计算
  DQN:        next_q = target_net(s2).max(dim=1)[0]
  Double DQN: best_a = q_net(s2).argmax(dim=1)
              next_q = target_net(s2).gather(1, best_a.unsqueeze(1)).squeeze(1)
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
NUM_DISCRETE_ACTIONS = 11
HIDDEN_DIM = 128
LR = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 200
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_MODE = "soft"
TARGET_UPDATE_FREQ = 10
TARGET_TAU = 0.005
NUM_EPISODES = 300
MAX_STEPS = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_BINS = np.linspace(-2.0, 2.0, NUM_DISCRETE_ACTIONS)

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-5


def discrete_to_continuous(action_idx: int) -> float:
    return ACTION_BINS[action_idx]


# ============================================================
# Q 网络
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
# SumTree + PER（与 01_dqn 相同）
# ============================================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta=PER_BETA_START):
        indices, priorities, batch = [], [], []
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)
        N = self.tree.size
        priorities = np.array(priorities, dtype=np.float64)
        weights = (N * priorities / self.tree.total) ** (-beta)
        weights /= weights.max()
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
    state_dim = env.observation_space.shape[0]

    q_net = QNetwork(state_dim, NUM_DISCRETE_ACTIONS).to(DEVICE)
    target_net = QNetwork(state_dim, NUM_DISCRETE_ACTIONS).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)
        beta = min(PER_BETA_END, PER_BETA_START + episode / NUM_EPISODES * (PER_BETA_END - PER_BETA_START))

        for step in range(MAX_STEPS):
            if random.random() < epsilon:
                action_idx = random.randrange(NUM_DISCRETE_ACTIONS)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action_idx = q_net(state_t).argmax(dim=1).item()

            continuous_action = discrete_to_continuous(action_idx)
            next_state, reward, terminated, truncated, _ = env.step([continuous_action])
            done = terminated or truncated

            replay_buffer.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= BATCH_SIZE:
                s, a, r, s2, d, tree_indices, is_weights = replay_buffer.sample(BATCH_SIZE, beta)
                s = torch.FloatTensor(s).to(DEVICE)
                a = torch.LongTensor(a).to(DEVICE)
                r = torch.FloatTensor(r).to(DEVICE)
                s2 = torch.FloatTensor(s2).to(DEVICE)
                d = torch.FloatTensor(d).to(DEVICE)
                is_weights = torch.FloatTensor(is_weights).to(DEVICE)

                current_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                # ========== Double DQN 核心改动 ==========
                # DQN:        next_q = target_net(s2).max(dim=1)[0]
                # Double DQN: q_net 选动作, target_net 评估
                with torch.no_grad():
                    best_actions = q_net(s2).argmax(dim=1)    # 在线网络选动作
                    next_q = target_net(s2).gather(            # 目标网络评估
                        1, best_actions.unsqueeze(1)
                    ).squeeze(1)
                    target_q = r + GAMMA * next_q * (1.0 - d)
                # ==========================================

                td_errors = (current_q - target_q).detach().cpu().numpy()
                replay_buffer.update_priorities(tree_indices, td_errors)

                loss = (is_weights * (current_q - target_q) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if TARGET_UPDATE_MODE == "soft":
                    for tp, sp in zip(target_net.parameters(), q_net.parameters()):
                        tp.data.copy_(TARGET_TAU * sp.data + (1.0 - TARGET_TAU) * tp.data)

            if done:
                break

        episode_rewards.append(total_reward)

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
def plot_rewards(rewards, save_path="double_dqn_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Double DQN on Pendulum-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    q_net, rewards = train()
    plot_rewards(rewards)
    print("Done.")
