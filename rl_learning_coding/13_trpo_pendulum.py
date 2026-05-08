"""
TRPO (Trust Region Policy Optimization) on Pendulum-v1
========================================================

PPO 的理论基础（Schulman 2015）。

核心思想：
  每次更新策略时，限制新旧策略的 KL 散度不超过 δ：
    max  E[ratio · advantage]
    s.t. KL(π_old || π_new) ≤ δ

  PPO 用 clipping 近似这个约束（简单但不精确）
  TRPO 用共轭梯度 + 线搜索精确求解（复杂但理论保证）

TRPO 的求解步骤：
  1. 计算策略梯度 g = ∇ E[ratio · advantage]
  2. 计算 Fisher 信息矩阵 F = E[∇log π · (∇log π)^T]
     F 表示参数空间中 KL 散度的曲率
  3. 用共轭梯度法求解 F·x = g → x = F^{-1}·g
     （不直接求逆 F，用矩阵-向量乘积迭代求解）
  4. 计算最大步长 β = sqrt(2δ / (x^T · F · x))
  5. 线搜索：从 β 开始缩小步长，直到 KL 约束满足

与 PPO 的对比：
  TRPO: 精确约束 KL ≤ δ，理论保证单调改进，但计算复杂
  PPO:  近似约束（clipping），无理论保证但实践效果好且简单
  一般推荐用 PPO，除非需要严格的理论保证
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
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_KL = 0.01              # KL 散度约束 δ
CG_ITERS = 10              # 共轭梯度迭代次数
CG_DAMPING = 0.1           # Fisher 矩阵阻尼（防止数值不稳定）
LINE_SEARCH_STEPS = 10     # 线搜索最大步数
LINE_SEARCH_DECAY = 0.8    # 线搜索步长衰减因子
NUM_EPISODES = 300
MAX_STEPS = 200
ROLLOUT_STEPS = 2048

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_SCALE = 2.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ============================================================
# Actor 网络
# ============================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        h = self.trunk(state)
        mean = self.mean_head(h)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        return Normal(mean, std)

    def act(self, state):
        dist = self.forward(state)
        x = dist.sample()
        action = torch.tanh(x) * ACTION_SCALE
        log_prob = dist.log_prob(x) - torch.log(1.0 - torch.tanh(x).pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1), x

    def log_prob_of(self, state, raw_action):
        dist = self.forward(state)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - torch.tanh(raw_action).pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def get_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_params(self, flat_params):
        idx = 0
        for p in self.parameters():
            size = p.data.numel()
            p.data.copy_(flat_params[idx:idx + size].view(p.shape))
            idx += size


# ============================================================
# Critic 网络
# ============================================================
class Critic(nn.Module):
    def __init__(self, state_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
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
# 共轭梯度法 (Conjugate Gradient)
# 求解 Ax = b，其中 A = Fisher 信息矩阵（通过函数 fvp 计算 A·v）
# 不需要显式构造 A，只需要矩阵-向量乘积
# ============================================================
def conjugate_gradient(fvp_fn, b, iters=CG_ITERS, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)

    for _ in range(iters):
        Ap = fvp_fn(p)
        alpha = rdotr / (p.dot(Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr

    return x


# ============================================================
# Fisher-Vector Product (FVP)
# 计算 F·v，其中 F 是 KL 散度的 Hessian
# 用两次自动微分实现，不需要显式构造 F 矩阵
# ============================================================
def fisher_vector_product(actor, states, v, damping=CG_DAMPING):
    dist = actor(states)
    mean = dist.loc
    std = dist.scale

    # KL(π_old || π_new) 对参数的梯度
    # 用当前策略自身的 KL（= 0），但它的 Hessian 就是 Fisher 矩阵
    kl = 0.5 * (std.pow(2) + mean.pow(2) - 1 - 2 * std.log()).sum(dim=-1).mean()

    grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])

    # 梯度和 v 的内积
    kl_v = flat_grads.dot(v)

    # 对内积再求梯度 → Hessian · v = Fisher · v
    grads2 = torch.autograd.grad(kl_v, actor.parameters())
    flat_fvp = torch.cat([g.contiguous().view(-1) for g in grads2])

    # 加阻尼防止数值不稳定
    return flat_fvp + damping * v


# ============================================================
# 训练循环
# ============================================================
def train():
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim).to(DEVICE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    episode_rewards = []
    kl_values = []

    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0

    while episode_count < NUM_EPISODES:
        # ---- 收集 rollout ----
        states, raw_actions, log_probs = [], [], []
        rewards, dones, values, next_values = [], [], [], []

        for _ in range(ROLLOUT_STEPS):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                value = critic(state_t).item()
                action, log_prob, raw_action = actor.act(state_t)

            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            states.append(state)
            raw_actions.append(raw_action.cpu().numpy()[0])
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

        # ---- GAE ----
        advantages, returns = compute_gae(rewards, values, next_values, dones)

        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        raw_actions_t = torch.FloatTensor(np.array(raw_actions)).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(np.array(log_probs)).to(DEVICE)
        advantages_t = torch.FloatTensor(advantages).to(DEVICE)
        returns_t = torch.FloatTensor(returns).to(DEVICE)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ---- TRPO Actor 更新 ----
        # Step 1: 计算策略梯度 g
        new_log_probs = actor.log_prob_of(states_t, raw_actions_t)
        ratio = (new_log_probs - old_log_probs_t).exp()
        surrogate = (ratio * advantages_t).mean()

        grads = torch.autograd.grad(surrogate, actor.parameters())
        policy_gradient = torch.cat([g.contiguous().view(-1) for g in grads]).detach()

        # Step 2: 共轭梯度求 F^{-1} · g
        def fvp_fn(v):
            return fisher_vector_product(actor, states_t, v)

        step_dir = conjugate_gradient(fvp_fn, policy_gradient)

        # Step 3: 计算最大步长
        shs = step_dir.dot(fvp_fn(step_dir))
        max_step = torch.sqrt(2 * MAX_KL / (shs + 1e-8))
        full_step = max_step * step_dir

        # Step 4: 线搜索（确保 KL 约束满足且目标改进）
        old_params = actor.get_params()
        expected_improve = policy_gradient.dot(full_step)

        success = False
        for k in range(LINE_SEARCH_STEPS):
            scale = LINE_SEARCH_DECAY ** k
            new_params = old_params + scale * full_step
            actor.set_params(new_params)

            with torch.no_grad():
                new_lp = actor.log_prob_of(states_t, raw_actions_t)
                new_ratio = (new_lp - old_log_probs_t).exp()
                new_surrogate = (new_ratio * advantages_t).mean()

            improve = new_surrogate - surrogate.detach()

            # 计算实际 KL
            with torch.no_grad():
                dist_new = actor(states_t)
                dist_old_mean = dist_new.loc.detach()  # 近似：用新策略参数评估
                kl_approx = 0.5 * ((new_lp - old_log_probs_t).pow(2)).mean()

            if improve > 0 and kl_approx < MAX_KL * 1.5:
                success = True
                break

        if not success:
            actor.set_params(old_params)
            kl_values.append(0.0)
        else:
            kl_values.append(kl_approx.item())

        # ---- Critic 更新（多步梯度下降） ----
        for _ in range(5):
            value_pred = critic(states_t).squeeze(-1)
            critic_loss = nn.functional.mse_loss(value_pred, returns_t)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        if episode_count % 20 == 0 and episode_count > 0:
            avg = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            print(f"Episode {episode_count:4d} | Avg Reward (last 20): {avg:7.1f}")

    env.close()
    return actor, episode_rewards, kl_values


# ============================================================
# 可视化
# ============================================================
def plot_rewards(rewards, save_path="trpo_rewards.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="Raw")
    window = 20
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("TRPO on Pendulum-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved reward curve to {save_path}")
    plt.close(fig)


def plot_kl(kl_values, save_path="trpo_kl.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(kl_values)
    ax.axhline(y=MAX_KL, color="red", linestyle="--", label=f"δ={MAX_KL}")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("KL Divergence")
    ax.set_title("TRPO KL Divergence per Update")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved KL curve to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    actor, rewards, kl_values = train()
    plot_rewards(rewards)
    plot_kl(kl_values)
    print("Done.")
