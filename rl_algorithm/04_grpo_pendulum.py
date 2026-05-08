"""
GRPO (Group Relative Policy Optimization) on Pendulum-v1
=========================================================

核心概念：
1. 没有 Critic 网络：不学 V(s) 或 Q(s,a)，省掉 Critic
2. 组采样：对同一个状态，用当前策略采样 G 个动作
3. 组内相对排名：用组内 reward 的排名（而非绝对值）作为 advantage
   - advantage_i = (reward_i - mean(rewards)) / std(rewards)
   - 组内最好的动作 advantage 为正，最差的为负
4. 策略更新：类似 PPO 的 clipping，但 advantage 来自组内相对比较
5. KL 惩罚：防止策略偏离参考策略太远

与 PPO 的关键区别：
  - PPO 需要 Critic 估计 baseline → GRPO 用组内均值做 baseline
  - PPO 的 advantage 依赖 value function 质量 → GRPO 的 advantage 纯粹来自比较
  - GRPO 更适合 reward 稀疏或绝对值不可靠的场景（如 LLM 评分）

在 Pendulum 上的适配：
  - 对每个 state，采样 G 个不同的动作
  - 用 env 单步 reward 作为评分（也可以用多步 rollout）
  - 组内标准化得到 advantage
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
LR = 3e-4
GAMMA = 0.99
GROUP_SIZE = 16             # 每个状态采样多少个动作/轨迹
CLIP_EPS = 0.2              # PPO-style clipping
KL_COEFF = 0.01             # KL 散度惩罚系数
NUM_ITERATIONS = 300        # 训练迭代次数
ROLLOUT_EPISODES = 4        # 每次迭代收集多少条完整轨迹
MAX_STEPS = 200
UPDATE_EPOCHS = 5

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
        return action, log_prob.sum(dim=-1), x

    def log_prob_of(self, state: torch.Tensor, raw_action: torch.Tensor):
        dist = self.forward(state)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
        return log_prob.sum(dim=-1)


# ============================================================
# 收集完整轨迹并计算折扣回报
# ============================================================
def collect_trajectory(env, policy):
    states, raw_actions, log_probs, rewards = [], [], [], []
    state, _ = env.reset()

    for _ in range(MAX_STEPS):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, log_prob, raw_action = policy.act(state_t)

        action_np = action.cpu().numpy()[0]
        next_state, reward, terminated, truncated, _ = env.step(action_np)

        states.append(state)
        raw_actions.append(raw_action.cpu().numpy()[0])
        log_probs.append(log_prob.item())
        rewards.append(reward)

        state = next_state
        if terminated or truncated:
            break

    # 计算每步的折扣回报 G_t = Σ γ^k · r_{t+k}
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    return states, raw_actions, log_probs, returns


# ============================================================
# GRPO 核心：组内相对 advantage 计算
#
# 对同一组轨迹的每一步:
#   1. 收集 G 条轨迹的 return
#   2. advantage_i = (return_i - mean) / (std + ε)
#   3. 用标准化后的 advantage 做策略梯度
# ============================================================
def compute_group_advantages(group_returns):
    """
    group_returns: list of G trajectories, each is a list of step returns
    对每条轨迹计算总回报，然后在组内标准化
    返回每条轨迹的标准化 advantage（标量，应用到轨迹的每一步）
    """
    total_returns = [sum(r for r in traj_returns) / len(traj_returns) for traj_returns in group_returns]
    total_returns = np.array(total_returns, dtype=np.float32)
    mean = total_returns.mean()
    std = total_returns.std() + 1e-8
    advantages = (total_returns - mean) / std
    return advantages


# ============================================================
# 训练循环
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim).to(DEVICE)
    ref_policy = Policy(state_dim, action_dim).to(DEVICE)
    ref_policy.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=LR)

    all_episode_rewards = []
    policy_losses_hist = []

    for iteration in range(NUM_ITERATIONS):
        # ---- 收集一组轨迹 ----
        group_data = []  # list of (states, raw_actions, old_log_probs, returns)
        group_returns = []

        for _ in range(GROUP_SIZE):
            states, raw_actions, log_probs, returns = collect_trajectory(env, policy)
            group_data.append((states, raw_actions, log_probs, returns))
            group_returns.append(returns)

        # 记录每条轨迹的总 reward
        for traj_returns in group_returns:
            all_episode_rewards.append(sum(r for r in traj_returns))

        # ---- 组内相对 advantage ----
        group_advantages = compute_group_advantages(group_returns)

        # ---- 展开所有轨迹为 flat batch ----
        all_states, all_raw_actions, all_old_log_probs, all_advantages = [], [], [], []

        for traj_idx, (states, raw_actions, log_probs, returns) in enumerate(group_data):
            adv = group_advantages[traj_idx]
            for s, ra, lp in zip(states, raw_actions, log_probs):
                all_states.append(s)
                all_raw_actions.append(ra)
                all_old_log_probs.append(lp)
                all_advantages.append(adv)

        states_t = torch.FloatTensor(np.array(all_states)).to(DEVICE)
        raw_actions_t = torch.FloatTensor(np.array(all_raw_actions)).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(np.array(all_old_log_probs)).to(DEVICE)
        advantages_t = torch.FloatTensor(np.array(all_advantages)).to(DEVICE)

        # ---- 策略更新（多 epoch） ----
        for epoch in range(UPDATE_EPOCHS):
            new_log_probs = policy.log_prob_of(states_t, raw_actions_t)

            # 重要性采样比 + clipping（同 PPO）
            ratio = (new_log_probs - old_log_probs_t).exp()
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL 惩罚：防止偏离参考策略太远
            with torch.no_grad():
                ref_log_probs = ref_policy.log_prob_of(states_t, raw_actions_t)
            kl = (new_log_probs - ref_log_probs).mean()
            total_loss = policy_loss + KL_COEFF * kl

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        policy_losses_hist.append(policy_loss.item())

        # 定期更新参考策略
        if (iteration + 1) % 20 == 0:
            ref_policy.load_state_dict(policy.state_dict())

        if (iteration + 1) % 10 == 0:
            recent = all_episode_rewards[-GROUP_SIZE * 10:]
            avg = np.mean(recent) if recent else 0
            print(f"Iter {iteration+1:4d} | Avg Reward (recent): {avg:7.1f}")

    env.close()
    return policy, all_episode_rewards, policy_losses_hist


# ============================================================
# 可视化
# ============================================================
def plot_rewards(rewards, save_path="grpo_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw (per trajectory)")
    window = GROUP_SIZE * 5
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Trajectory")
    ax.set_ylabel("Total Reward")
    ax.set_title("GRPO on Pendulum-v1 (No Critic)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_losses(losses, save_path="grpo_losses.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy Loss")
    ax.set_title("GRPO Policy Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved loss curve to {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    policy, rewards, losses = train()
    plot_rewards(rewards)
    plot_losses(losses)
    print("Done.")
