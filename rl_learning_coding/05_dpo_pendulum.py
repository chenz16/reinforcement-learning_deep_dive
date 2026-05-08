"""
DPO (Direct Preference Optimization) on Pendulum-v1
=====================================================

核心概念：
1. 不需要显式的 Reward Model：直接从偏好数据学策略
2. 偏好数据：(s, a_win, a_lose) 三元组
   - 对同一个 state，采样两条轨迹，总 reward 高的是 winner
3. DPO Loss:
   L = -log σ(β · (log π(a_w|s)/π_ref(a_w|s) - log π(a_l|s)/π_ref(a_l|s)))
   含义：让策略相对于参考策略，更偏好 winner 动作
4. β 控制偏好的锐度：β 大 → 对偏好更敏感

DPO 的本质：
  - RLHF 的简化版：把 "训练 RM → 用 RM 做 RL" 两步合成一步
  - 隐含了一个 reward model：r(s,a) = β · log(π(a|s)/π_ref(a|s))
  - 在 LLM 对齐中很成功，因为不需要单独训练 RM

在 Pendulum 上的适配：
  - Pendulum 已有明确的 reward 函数，用 DPO 显得多此一举
  - 但教学价值：展示如何从比较数据中隐式学到 reward 信息
  - 流程：先用随机策略跑很多轨迹 → 两两配对 → 用 DPO loss 训练
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
LR = 1e-4
BETA = 0.1                  # DPO 温度参数
NUM_PREFERENCE_PAIRS = 50000 # 偏好对数量
BATCH_SIZE = 256
NUM_TRAIN_EPOCHS = 30
ROLLOUT_EPISODES = 2000     # 用于生成偏好数据的轨迹数量
MAX_STEPS = 200
EVAL_EPISODES = 20          # 每次评估的 episode 数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ============================================================
# Policy 网络
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
        return action, x

    def log_prob_of(self, state: torch.Tensor, raw_action: torch.Tensor):
        dist = self.forward(state)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
        return log_prob.sum(dim=-1)


# ============================================================
# Phase 1：收集轨迹数据
# 用一个随机 + 部分训练的策略跑很多 episode，记录每步的 (s, a, reward)
# ============================================================
def collect_trajectories(env, policy, num_episodes):
    """收集轨迹，返回 step 级别的数据"""
    all_data = []  # list of (trajectory_data, total_reward)

    for _ in range(num_episodes):
        state, _ = env.reset()
        traj_steps = []
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, raw_action = policy.act(state_t)

            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)

            traj_steps.append({
                "state": state.copy(),
                "raw_action": raw_action.cpu().numpy()[0].copy(),
                "reward": reward,
            })
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        all_data.append((traj_steps, total_reward))

    return all_data


# ============================================================
# Phase 2：构造偏好对
# 随机选两条轨迹，总 reward 高的是 winner
# 取同一时间步的 (state, action) 作为 step-level 偏好对
# ============================================================
def build_preference_pairs(trajectories, num_pairs):
    """
    从轨迹中构造偏好对
    返回: (states, winning_raw_actions, losing_raw_actions)
    """
    states = []
    win_actions = []
    lose_actions = []

    n = len(trajectories)
    for _ in range(num_pairs):
        # 随机选两条轨迹
        i, j = np.random.randint(0, n, size=2)
        while i == j:
            j = np.random.randint(0, n)

        traj_i, reward_i = trajectories[i]
        traj_j, reward_j = trajectories[j]

        # reward 高的是 winner
        if reward_i >= reward_j:
            win_traj, lose_traj = traj_i, traj_j
        else:
            win_traj, lose_traj = traj_j, traj_i

        # 取两条轨迹中较短的长度，随机选一个时间步
        min_len = min(len(win_traj), len(lose_traj))
        t = np.random.randint(0, min_len)

        states.append(win_traj[t]["state"])
        win_actions.append(win_traj[t]["raw_action"])
        lose_actions.append(lose_traj[t]["raw_action"])

    return (
        np.array(states, dtype=np.float32),
        np.array(win_actions, dtype=np.float32),
        np.array(lose_actions, dtype=np.float32),
    )


# ============================================================
# Phase 3：DPO 训练
#
# DPO Loss:
#   L = -log σ(β · (log π_θ(a_w|s) - log π_ref(a_w|s)
#                   - log π_θ(a_l|s) + log π_ref(a_l|s)))
#
# 直觉：
#   - 让 π_θ 相对于 π_ref，更喜欢 a_w（winner），更不喜欢 a_l（loser）
#   - β 控制偏好强度：β 大 → 强烈偏好 winner
#   - 隐含 reward: r(s,a) = β · log(π_θ(a|s) / π_ref(a|s)) + const
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 参考策略（固定不动）
    ref_policy = Policy(state_dim, action_dim).to(DEVICE)

    # 收集数据阶段：用多种水平的策略生成轨迹
    print("Phase 1: Collecting trajectories with diverse policies...")
    all_trajectories = []

    # 用随机策略收集
    random_policy = Policy(state_dim, action_dim).to(DEVICE)
    all_trajectories.extend(collect_trajectories(env, random_policy, ROLLOUT_EPISODES // 2))

    # 用稍微训练过的策略收集（增加数据多样性）
    semi_policy = Policy(state_dim, action_dim).to(DEVICE)
    # 简单的 behavior cloning 几步让它不完全随机
    all_trajectories.extend(collect_trajectories(env, semi_policy, ROLLOUT_EPISODES // 2))

    traj_rewards = [r for _, r in all_trajectories]
    print(f"  Collected {len(all_trajectories)} trajectories, "
          f"reward range: [{min(traj_rewards):.0f}, {max(traj_rewards):.0f}]")

    # 构造偏好对
    print("Phase 2: Building preference pairs...")
    states, win_actions, lose_actions = build_preference_pairs(all_trajectories, NUM_PREFERENCE_PAIRS)
    print(f"  Built {len(states)} preference pairs")

    # DPO 训练
    print("Phase 3: DPO training...")
    policy = Policy(state_dim, action_dim).to(DEVICE)
    policy.load_state_dict(ref_policy.state_dict())  # 从 ref 策略初始化
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    states_t = torch.FloatTensor(states).to(DEVICE)
    win_actions_t = torch.FloatTensor(win_actions).to(DEVICE)
    lose_actions_t = torch.FloatTensor(lose_actions).to(DEVICE)

    # 预计算 ref policy 的 log probs（固定不变）
    with torch.no_grad():
        ref_log_prob_win = ref_policy.log_prob_of(states_t, win_actions_t)
        ref_log_prob_lose = ref_policy.log_prob_of(states_t, lose_actions_t)

    losses_hist = []
    eval_rewards_hist = []
    dataset_size = len(states)

    for epoch in range(NUM_TRAIN_EPOCHS):
        epoch_losses = []
        indices = np.random.permutation(dataset_size)

        for start in range(0, dataset_size, BATCH_SIZE):
            end = min(start + BATCH_SIZE, dataset_size)
            idx = indices[start:end]

            s = states_t[idx]
            a_w = win_actions_t[idx]
            a_l = lose_actions_t[idx]
            ref_lp_w = ref_log_prob_win[idx]
            ref_lp_l = ref_log_prob_lose[idx]

            # 当前策略的 log prob
            pi_lp_w = policy.log_prob_of(s, a_w)
            pi_lp_l = policy.log_prob_of(s, a_l)

            # DPO loss
            # log_ratio_w = log π_θ(a_w|s) - log π_ref(a_w|s)
            # log_ratio_l = log π_θ(a_l|s) - log π_ref(a_l|s)
            # loss = -log σ(β · (log_ratio_w - log_ratio_l))
            log_ratio_w = pi_lp_w - ref_lp_w
            log_ratio_l = pi_lp_l - ref_lp_l
            logits = BETA * (log_ratio_w - log_ratio_l)
            loss = -torch.nn.functional.logsigmoid(logits).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses_hist.append(avg_loss)

        # 评估
        eval_reward = evaluate_policy(env, policy, EVAL_EPISODES)
        eval_rewards_hist.append(eval_reward)
        print(f"  Epoch {epoch+1:3d} | DPO Loss: {avg_loss:.4f} | Eval Reward: {eval_reward:.1f}")

    env.close()
    return policy, losses_hist, eval_rewards_hist


def evaluate_policy(env, policy, num_episodes):
    """评估策略的平均 reward"""
    total = 0.0
    policy.eval()
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        for _ in range(MAX_STEPS):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _ = policy.act(state_t)
            state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            ep_reward += reward
            if terminated or truncated:
                break
        total += ep_reward
    policy.train()
    return total / num_episodes


# ============================================================
# 可视化
# ============================================================
def plot_results(losses, eval_rewards, save_path="dpo_results.png"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(losses)
    axes[0].set_ylabel("DPO Loss")
    axes[0].set_title("DPO on Pendulum-v1")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_rewards)
    axes[1].set_ylabel("Eval Reward")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved results to {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    policy, losses, eval_rewards = train()
    plot_results(losses, eval_rewards)
    print("Done.")
