"""
ProRL (Progressive Reinforcement Learning) on Pendulum-v1
==========================================================

核心概念：
1. 渐进式训练：把困难任务分解成多个阶段，从简单到难逐步推进
2. 多目标动态权重：不同训练阶段，reward 各分量的权重动态调整
3. 课程学习 (Curriculum Learning)：逐步增加任务难度

在 Pendulum 上的适配：
  Stage 1 - 基础控制：只关心角度（把摆锤摆上去），忽略能量消耗
            reward_1 = -θ²
  Stage 2 - 稳定控制：加入角速度惩罚（摆上去后要稳住）
            reward_2 = -θ² - 0.1·θ̇²
  Stage 3 - 高效控制：加入能量惩罚（用最小力稳住）
            reward_3 = -θ² - 0.1·θ̇² - 0.001·torque²

与标准 RL 的区别：
  - 标准 RL 从头到尾用同一个 reward 函数
  - ProRL 动态调整 reward 权重，降低学习难度
  - 类似人类学骑车：先学平衡 → 再学转弯 → 最后优化效率

实现方式：PPO 作为基础算法 + 渐进式 reward shaping
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
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 10
BATCH_SIZE = 64
MAX_STEPS = 200
ROLLOUT_STEPS = 2048

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

# ============================================================
# 渐进式训练阶段配置
# 每个阶段定义 reward 权重和持续的 episode 数
# ============================================================
STAGES = [
    {
        "name": "Stage 1: Angle Only",
        "episodes": 100,
        "w_angle": 1.0,       # -θ² 的权重
        "w_velocity": 0.0,    # -θ̇² 的权重
        "w_torque": 0.0,      # -torque² 的权重
    },
    {
        "name": "Stage 2: Angle + Velocity",
        "episodes": 100,
        "w_angle": 1.0,
        "w_velocity": 0.1,
        "w_torque": 0.0,
    },
    {
        "name": "Stage 3: Full Objective",
        "episodes": 100,
        "w_angle": 1.0,
        "w_velocity": 0.1,
        "w_torque": 0.001,
    },
]


# ============================================================
# 自定义 reward 函数
# Pendulum 的 state = [cos(θ), sin(θ), θ̇]
# 原始 env reward = -(θ² + 0.1·θ̇² + 0.001·u²)
# ProRL: 用可配置权重的版本
# ============================================================
def compute_shaped_reward(state, action, w_angle, w_velocity, w_torque):
    """
    state: [cos(θ), sin(θ), θ̇]
    action: torque 值
    """
    cos_theta, sin_theta, theta_dot = state[0], state[1], state[2]
    # 从 cos/sin 恢复 θ（atan2 范围 [-π, π]）
    theta = np.arctan2(sin_theta, cos_theta)

    reward = 0.0
    reward -= w_angle * (theta ** 2)
    reward -= w_velocity * (theta_dot ** 2)
    reward -= w_torque * (float(action) ** 2)
    return reward


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
        return action, log_prob.sum(dim=-1), x

    def evaluate(self, state: torch.Tensor, raw_action: torch.Tensor):
        dist = self.forward(state)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ============================================================
# Critic 网络
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
# GAE
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

    # 记录指标
    all_episode_rewards = []     # 用 env 原始 reward 评估（公平对比）
    all_shaped_rewards = []      # 用 shaped reward 记录
    stage_boundaries = []        # 记录阶段切换点
    actor_losses_hist = []
    critic_losses_hist = []

    total_episodes = 0

    for stage_idx, stage in enumerate(STAGES):
        print(f"\n{'='*60}")
        print(f"{stage['name']}")
        print(f"  Weights: angle={stage['w_angle']}, velocity={stage['w_velocity']}, torque={stage['w_torque']}")
        print(f"{'='*60}")

        stage_boundaries.append(total_episodes)

        # 阶段切换时重置 Critic（因为 reward 函数变了，旧的 V(s) 估计不准了）
        if stage_idx > 0:
            critic = Critic(state_dim).to(DEVICE)
            critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

        stage_episode_count = 0
        state, _ = env.reset()
        episode_reward_env = 0.0      # env 原始 reward
        episode_reward_shaped = 0.0   # shaped reward

        while stage_episode_count < stage["episodes"]:
            # ---- 收集 rollout ----
            states, raw_actions, log_probs = [], [], []
            shaped_rewards, dones, values, next_values = [], [], [], []
            env_rewards_step = []

            for _ in range(ROLLOUT_STEPS):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action, log_prob, raw_action = actor.act(state_t)
                    value = critic(state_t).item()

                action_np = action.cpu().numpy()[0]
                next_state, env_reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                # 用 shaped reward 训练
                shaped_r = compute_shaped_reward(
                    state, action_np,
                    stage["w_angle"], stage["w_velocity"], stage["w_torque"],
                )

                states.append(state)
                raw_actions.append(raw_action.cpu().numpy()[0])
                log_probs.append(log_prob.item())
                shaped_rewards.append(shaped_r)
                env_rewards_step.append(env_reward)
                dones.append(float(done))
                values.append(value)

                with torch.no_grad():
                    next_v = critic(torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)).item()
                next_values.append(next_v)

                episode_reward_env += env_reward
                episode_reward_shaped += shaped_r
                state = next_state

                if done:
                    all_episode_rewards.append(episode_reward_env)
                    all_shaped_rewards.append(episode_reward_shaped)
                    total_episodes += 1
                    stage_episode_count += 1
                    episode_reward_env = 0.0
                    episode_reward_shaped = 0.0
                    state, _ = env.reset()
                    if stage_episode_count >= stage["episodes"]:
                        break

            if not states:
                break

            # ---- GAE + PPO 更新 ----
            advantages, returns = compute_gae(shaped_rewards, values, next_values, dones)

            states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
            raw_actions_t = torch.FloatTensor(np.array(raw_actions)).to(DEVICE)
            old_log_probs_t = torch.FloatTensor(np.array(log_probs)).to(DEVICE)
            advantages_t = torch.FloatTensor(advantages).to(DEVICE)
            returns_t = torch.FloatTensor(returns).to(DEVICE)

            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

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

                    new_log_prob, entropy = actor.evaluate(s_batch, ra_batch)
                    ratio = (new_log_prob - old_lp_batch).exp()
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_batch
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optimizer.step()

                    value_pred = critic(s_batch).squeeze(-1)
                    critic_loss = nn.functional.mse_loss(value_pred, ret_batch)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    critic_optimizer.step()

            actor_losses_hist.append(actor_loss.item())
            critic_losses_hist.append(critic_loss.item())

        # 阶段结束，打印统计
        stage_rewards = all_episode_rewards[-stage["episodes"]:]
        print(f"  Stage done | Avg Env Reward (last {stage['episodes']}): "
              f"{np.mean(stage_rewards):.1f}")

    env.close()
    return (actor, all_episode_rewards, all_shaped_rewards,
            stage_boundaries, actor_losses_hist, critic_losses_hist)


# ============================================================
# 可视化
# ============================================================
def plot_rewards(env_rewards, shaped_rewards, stage_boundaries, save_path="prorl_rewards.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Env reward（公平对比标准）
    axes[0].plot(env_rewards, alpha=0.3, label="Raw")
    window = 20
    if len(env_rewards) >= window:
        smoothed = np.convolve(env_rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window - 1, len(env_rewards)), smoothed, label=f"MA-{window}")
    for b in stage_boundaries:
        axes[0].axvline(x=b, color="red", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Env Reward")
    axes[0].set_title("ProRL on Pendulum-v1 — Progressive Reward Shaping")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Shaped reward
    axes[1].plot(shaped_rewards, alpha=0.3, color="orange", label="Shaped Raw")
    if len(shaped_rewards) >= window:
        smoothed = np.convolve(shaped_rewards, np.ones(window) / window, mode="valid")
        axes[1].plot(range(window - 1, len(shaped_rewards)), smoothed, color="red", label=f"MA-{window}")
    for i, b in enumerate(stage_boundaries):
        axes[1].axvline(x=b, color="red", linestyle="--", alpha=0.5)
        if i < len(STAGES):
            axes[1].text(b + 2, axes[1].get_ylim()[1] if axes[1].get_ylim()[1] != 0 else -100,
                        STAGES[i]["name"], fontsize=8, rotation=0, va="top")
    axes[1].set_ylabel("Shaped Reward")
    axes[1].set_xlabel("Episode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curves to {save_path}")
    plt.close(fig)


def plot_losses(actor_losses, critic_losses, save_path="prorl_losses.png"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(actor_losses)
    axes[0].set_ylabel("Actor Loss")
    axes[0].set_title("ProRL Training Curves")
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
    actor, env_rewards, shaped_rewards, boundaries, a_losses, c_losses = train()
    plot_rewards(env_rewards, shaped_rewards, boundaries)
    plot_losses(a_losses, c_losses)
    print("Done.")
