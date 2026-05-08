# RL 算法发展史

## 三大分支演进路线

```
离散 Q-learning 系（主要 DeepMind）
├── 1989  Q-Learning          Watkins          表格型 Q 学习
├── 2013  DQN                 Mnih, DeepMind   神经网络+replay+target net，Atari
├── 2015  PER                 Schaul, DeepMind 优先经验回放
├── 2016  Double DQN          Van Hasselt, DeepMind  解耦选动作和评估
└── 2016  Dueling DQN         Wang, DeepMind   V+A 分流结构

策略梯度 on-policy 系（Berkeley → OpenAI）
├── 1992  REINFORCE            Williams         最早的策略梯度
├── 2015  TRPO                 Schulman, Berkeley (Abbeel 组)  KL 约束+共轭梯度
├── 2015  GAE                  Schulman, Berkeley  advantage 估计方法
├── 2016  A3C/A2C              Mnih, DeepMind   并行 Actor-Critic
└── 2017  PPO                  Schulman, OpenAI  用 clipping 简化 TRPO

连续动作 off-policy 系（DeepMind + Berkeley）
├── 2014  DPG                  Silver, DeepMind  确定性策略梯度理论
├── 2015  DDPG                 Lillicrap, DeepMind  DPG + 深度网络
├── 2018  TD3                  Fujimoto, McGill/Mila  DDPG 三项改进
└── 2018  SAC                  Haarnoja, Berkeley (Levine 组)  最大熵框架

LLM 对齐方向
├── 2022  PPO for RLHF         Ouyang, OpenAI   InstructGPT
├── 2023  DPO                  Rafailov, Stanford  去掉 reward model
└── 2024  GRPO                 Shao, DeepSeek   去掉 Critic，组内排名
```

## 关键人物

- **Watkins** — 1989 年提出 Q-Learning，奠定了 value-based RL 的基础
- **Williams** — 1992 年提出 REINFORCE，奠定了 policy gradient 的基础
- **Mnih (DeepMind)** — 2013 DQN 在 Atari 上超越人类，引爆深度 RL；2016 A3C 并行训练
- **Silver (DeepMind)** — 2014 确定性策略梯度 (DPG) 理论，后来主导 AlphaGo/AlphaZero
- **Schulman (Berkeley → OpenAI)** — 一个人搞了 TRPO (2015)、GAE (2015)、PPO (2017)，定义了 on-policy 这条线
- **Abbeel (Berkeley)** — Schulman 的导师，Berkeley Robot Learning Lab，培养了大量 RL 人才
- **Lillicrap (DeepMind)** — 2015 DDPG，把 DQN 的思路推广到连续动作空间
- **Fujimoto (McGill/Mila)** — 2018 TD3，系统性解决 DDPG 的高估和不稳定问题
- **Haarnoja & Levine (Berkeley)** — 2018 SAC，最大熵 RL 框架，目前连续控制最主流的算法
- **Ouyang (OpenAI)** — 2022 InstructGPT，PPO + RLHF 对齐 LLM，开启 LLM 对齐时代
- **Rafailov (Stanford)** — 2023 DPO，证明可以不需要 reward model 直接从偏好数据学
- **Shao (DeepSeek)** — 2024 GRPO，去掉 Critic 用组内排名，用于 DeepSeek-R1 数学推理

## 各算法核心对比

| 算法 | 动作空间 | On/Off-Policy | Actor | Critic | 探索方式 | 本项目文件 |
|------|----------|---------------|-------|--------|----------|-----------|
| DQN | 离散 | Off | 无 (argmax Q) | Q(s,a) ×1 | ε-greedy | `01_dqn` |
| Double DQN | 离散 | Off | 无 | Q(s,a) ×1 | ε-greedy | `07_double_dqn` |
| Dueling DQN | 离散 | Off | 无 | V(s)+A(s,a) | ε-greedy | `08_dueling_dqn` |
| REINFORCE | 连续 | On | 高斯分布 | 无 | 策略本身随机 | `09_reinforce` |
| A2C | 连续 | On | 高斯分布 | V(s) | 策略随机+熵正则 | `10_a2c` |
| PPO | 连续 | On | 高斯分布 | V(s) | 策略随机+熵正则 | `03_ppo` |
| TRPO | 连续 | On | 高斯分布 | V(s) | 策略随机 | `13_trpo` |
| DDPG | 连续 | Off | 确定性 μ(s) | Q(s,a) ×1 | 外挂高斯噪声 | `11_ddpg` |
| TD3 | 连续 | Off | 确定性 μ(s) | Q(s,a) ×2 | 外挂噪声+target smoothing | `12_td3` |
| SAC | 连续 | Off | 高斯分布 | Q(s,a) ×2 | 最大熵自动探索 | `02_sac` |
| GRPO | 连续 | On | 高斯分布 | 无 (组内排名) | 策略随机 | `04_grpo` |
| DPO | 连续 | 离线 | 高斯分布 | 无 | 无 (离线数据) | `05_dpo` |
| ProRL | 连续 | On | 高斯分布 | V(s) | 策略随机 | `06_prorl` |

## 算法演进的核心问题与解法

### 1. Q 值高估
- **问题**: max 操作系统性选中被高估的动作
- **解法演进**: DQN (无解) → Double DQN (解耦选择和评估) → TD3/SAC (双 Critic 取 min)

### 2. 连续动作空间
- **问题**: DQN 需要枚举所有动作取 argmax，连续空间做不到
- **解法演进**: 离散化 (DQN) → 确定性策略 (DDPG) → 随机策略 (SAC)

### 3. 探索与利用
- **解法演进**: ε-greedy (DQN) → 外挂噪声 (DDPG) → 熵正则化 (SAC，自动平衡)

### 4. 训练稳定性
- **解法演进**: 无约束 (REINFORCE) → KL 约束 (TRPO) → Clipping 近似 (PPO) → 最大熵 (SAC)

### 5. 数据效率
- **On-policy** (PPO/A2C): 数据用完即弃，效率低但稳定
- **Off-policy** (SAC/TD3): Replay buffer 反复利用，效率高
- **PER**: 按 TD error 优先采样，进一步提高效率

### 6. LLM 对齐
- **问题**: 把 RL 用于语言模型对齐
- **解法演进**: PPO+RM (RLHF) → DPO (去掉 RM) → GRPO (去掉 Critic)
