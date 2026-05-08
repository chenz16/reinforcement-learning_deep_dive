# 强化学习与其他模型的关联

> **定位说明**：本文讨论 RL 不是孤立存在的——它可以从零训练，也可以在 SL 基座模型上微调，还可以与 Diffusion/Flow Matching 模型结合，甚至在 Sim/Real 之间迁移。本文从模型关联的角度，梳理 RL 在不同场景下如何与其他范式衔接。

---

## 1. 从零训练的纯 RL

最经典的 RL 场景：没有预训练基座，模型从随机初始化开始，纯靠与环境交互来学习。

这里有两条路线：

### 1.1 学习价值偏好（Value-based）

模型学习的是：每个 state-action 对的长期价值是多少。

```text
Q(s,a) → 对未来回报的估计
```

代表算法：Q-learning、DQN、Double DQN、Dueling DQN

决策方式：argmax Q(s,a)

本质：**先学估值，再从估值中提取行为**。模型压缩的是 value landscape。

### 1.2 学习动作偏好（Policy-based）

模型学习的是：在每个 state 下，应该以什么概率做什么 action。

```text
π(a|s) → 行为偏好分布
```

代表算法：REINFORCE、PPO、SAC

决策方式：从 π(a|s) 采样

本质：**直接学偏好分布**。模型压缩的是 behavior preference。

### 1.3 两者融合（Actor-Critic）

实际上大多数现代 RL 算法都是 Actor-Critic：

```text
Critic：学价值（V 或 Q），提供偏好信号
Actor：学行为分布，接收 Critic 的指导
```

代表算法：A2C、DDPG、TD3、SAC、PPO（V-critic 版本）

从零训练的 RL 适用于：游戏 AI（Atari、围棋）、机器人控制、自动驾驶的低层控制。

### 1.4 Reward 的来源：必须在训练前准备好

RL 的训练循环需要 reward 信号，但 **reward 不是训练出来的——它必须在 RL 训练之前就准备好**。Reward 的来源有三种：

```text
来源 1：环境自带 reward（最干净）
  游戏分数、物理仿真中的距离/碰撞/能耗
  → 环境直接返回标量 reward
  → 不需要额外准备

来源 2：基于规则的 reward（工程设计）
  数学题的答案正确性、代码编译通过率、格式合规检查
  → 人工定义规则函数 R(output) → scalar
  → GRPO 常用这种方式

来源 3：从偏好数据学习 Reward Model（统计近似）
  收集人类偏好对：(chosen, rejected)
  → 训练一个 Reward Model 来近似人类偏好
  → RM 本质是一个 SL 模型：输入 (prompt, response)，输出标量分数
  → PPO-RLHF 的标准做法
```

关键理解：

```text
Reward Model 不是 RL 的一部分——它是 RL 训练的前置条件。
它用 SL 方式从偏好数据中学习，然后在 RL 训练过程中作为"评分器"冻结使用。
```

三种来源的对比：

| Reward 来源 | 准备方式 | 优点 | 缺点 | 典型应用 |
|---|---|---|---|---|
| 环境自带 | 无需准备 | 干净、无偏 | 只适用于有明确数值反馈的环境 | 游戏、仿真 |
| 基于规则 | 人工编写规则 | 精确、可解释 | 难以覆盖复杂偏好 | 数学/代码验证、GRPO |
| Reward Model | 收集偏好 + SL 训练 | 能捕捉模糊偏好 | RM 本身有偏差和噪声 | RLHF、InstructGPT |

### 1.5 模型架构图：训练 vs 推理用了哪些模块

纯 RL 的模型架构可以分三种典型模式。关键区别在于：**训练时需要的模块比推理时多——Critic 只在训练时存在，推理时丢弃。**

#### 模式 A：Value-based（DQN）

```text
┌─────────────────────────────────────────────────────────┐
│                    训练 + 推理 都用                       │
│                                                         │
│   Observation ──→ ┌──────────────┐ ──→ ┌─────────────┐  │
│   (图像/状态)      │   Backbone   │     │  Q Head     │  │
│                    │  (CNN/MLP)   │     │ 输出每个     │  │
│                    │              │     │ action 的 Q  │  │
│                    └──────────────┘     └──────┬──────┘  │
│                                               │         │
│                                          argmax Q       │
│                                               │         │
│                                          选出 action    │
└─────────────────────────────────────────────────────────┘

推理时：整个网络都要用（Backbone + Q Head + argmax）
训练时：同一个网络 + target network（冻结副本）
没有独立的 policy 网络——policy 隐含在 argmax Q 中
```

#### 模式 B：Actor-Critic 共享 Backbone（PPO / A2C 常见）

```text
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    ┌──────────────┐                          │
│   Observation ──→  │   共享       │                          │
│   (图像/状态)      │   Backbone   │                          │
│                    │  (CNN/MLP)   │                          │
│                    └──────┬───────┘                          │
│                           │                                 │
│                     ┌─────┴──────┐                          │
│                     │            │                          │
│               ┌─────▼─────┐  ┌──▼──────────┐               │
│               │ Policy    │  │ Value       │               │
│               │ Head      │  │ Head        │               │
│               │           │  │             │               │
│               │ 输出 π(a|s)│  │ 输出 V(s)   │               │
│               │ (softmax/ │  │ (标量)      │               │
│               │  Gaussian)│  │             │               │
│               └─────┬─────┘  └──────┬──────┘               │
│                     │               │                      │
│                 ┌───┴───┐     ┌─────┴──────┐               │
│   推理时用 ✓    │采样    │     │ 计算       │  仅训练时用 ✗  │
│                 │action  │     │ advantage  │               │
│                 └────────┘     └────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

推理时：只用 Backbone + Policy Head → 输出 action
训练时：Backbone + Policy Head + Value Head → advantage 指导 policy 更新
推理部署时 Value Head 直接丢弃
```

#### 模式 C：Actor-Critic 独立网络（DDPG / TD3 / SAC 常见）

```text
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ┌─── Actor Network（推理时用 ✓）───────────────────────┐       │
│   │                                                      │       │
│   │  Observation ──→ ┌────────────┐ ──→ ┌─────────────┐  │       │
│   │                  │  Actor     │     │ Policy Head  │  │       │
│   │                  │  Backbone  │     │ 输出 action  │  │       │
│   │                  │  (MLP)     │     │ (μ(s) 或     │  │       │
│   │                  │            │     │  π(a|s))     │  │       │
│   │                  └────────────┘     └──────────────┘  │       │
│   └──────────────────────────────────────────────────────┘       │
│                                                                  │
│   ┌─── Critic Network（仅训练时用 ✗）────────────────────┐       │
│   │                                                      │       │
│   │  (s, a) ──→ ┌────────────┐ ──→ ┌────────────────┐   │       │
│   │              │  Critic    │     │ Q Head         │   │       │
│   │              │  Backbone  │     │ 输出 Q(s,a)    │   │       │
│   │              │  (MLP)     │     │ (标量)         │   │       │
│   │              └────────────┘     └────────────────┘   │       │
│   │                                                      │       │
│   │  TD3/SAC 通常有两个 Critic（double Q）                │       │
│   └──────────────────────────────────────────────────────┘       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

推理时：只用 Actor Network → 输出 action
训练时：Actor + Critic → Critic 的 Q 梯度指导 Actor 更新
推理部署时整个 Critic Network 直接丢弃
```

#### 总结：训练 vs 推理的模块对比

| 架构 | 训练时用 | 推理时用 | 推理时丢弃 |
|---|---|---|---|
| **DQN** | Backbone + Q Head + Target Network | Backbone + Q Head | Target Network |
| **PPO（共享）** | 共享 Backbone + Policy Head + Value Head | 共享 Backbone + Policy Head | Value Head |
| **SAC/TD3（独立）** | Actor Net + Critic Net(s) | Actor Net | 整个 Critic Net |

核心规律：

```text
Critic / Value Head 是训练的"脚手架"——
它在训练时提供梯度信号（advantage / Q gradient），
但推理部署时完全不需要，直接丢弃。

唯一部署到生产环境的是 Policy / Actor 网络。
```

#### 训练时 Value 和 Policy 是两个独立的迭代过程

在 Actor-Critic 架构中，虽然 Critic 和 Actor 在同一个训练循环里更新，但它们是**两个独立的优化过程**，各自有各自的 loss：

```text
同一个训练 step 中：

  Step 1: Critic 更新（独立）
    loss_critic = (V(s) - V_target)^2     或  (Q(s,a) - Q_target)^2
    只更新 Critic 的参数
    目标：让 value 估计更准

  Step 2: Actor 更新（独立）
    loss_actor = - log π(a|s) · A(s,a)    或  - Q(s, μ(s))
    只更新 Actor 的参数
    目标：让 policy 更好

  两个 loss 各自反向传播，各自更新各自的参数。
  它们共享同一批数据，但优化过程是隔离的。
```

这个隔离很重要，因为：

```text
1. Critic 的目标是估值准确（回归问题）
2. Actor 的目标是行为更优（偏好优化问题）
3. 两者的 loss 性质完全不同，混在一起会互相干扰
4. TD3 甚至刻意让 Actor 更新频率低于 Critic（delayed policy update）
   → 让 Critic 先稳定，再指导 Actor
```

即使在共享 Backbone 的架构（PPO/A2C）中，Policy Head 和 Value Head 的梯度也是分别计算、分别应用的。共享 Backbone 的参数会同时接收两个方向的梯度，但两个 Head 的更新仍然是独立的 loss 驱动的。

#### Q/V 和 Policy 的耦合与解耦：采样耦合，梯度解耦

这是一个非常容易混淆的点。Q/V 和 Policy 之间既有耦合也有解耦，但发生在不同层面：

**采样层面：耦合的**

Q/V 的训练数据来自 policy 的 rollout。policy 决定了 agent 走什么轨迹、看到什么 state、做什么 action、拿到什么 reward。所以：

```text
policy 变了 → 采样分布变了 → Q/V 的训练数据变了 → Q/V 也会变

Q/V 和 policy 通过采样过程间接耦合。
```

这就是 RL 闭环问题的本质——前面第 12 章讨论的"算法改变自己的数据分布"。

**梯度层面：解耦的**

但在任意一个训练 step 的梯度计算中，Q/V 对 policy gradient 来说是一个**常数**，不参与梯度计算：

```text
Policy gradient:
  ∇_θ J = E[ ∇_θ log π_θ(a|s) · A(s,a) ]
                  ↑                  ↑
            对 θ 求导           A 是常数，不对 θ 求导
                               （A 来自 Q/V，但 Q/V 的参数不参与这个梯度）

换句话说：
  - log π_θ(a|s) 对 policy 参数 θ 求导 ✓
  - A(s,a) 来自 Critic，但在这个梯度中被当作固定权重 ✗（stop gradient）
```

具体到不同算法：

```text
PPO:
  loss = - clip(ratio, 1-ε, 1+ε) · A(s,a)
  A(s,a) = r + γV(s') - V(s)
  → 反向传播时只对 π_θ 求导，V 的输出被 detach / stop_gradient

DDPG/TD3（max Q loss 例外）:
  loss_actor = - Q(s, μ_θ(s))
  → 这里梯度确实流过 Q 网络到达 action，再到达 actor
  → 但 Q 网络的参数仍然不更新（Q 的参数被冻结/detach）
  → 只有 actor 参数 θ 在更新

SAC:
  loss_actor = E[ α·log π(a|s) - Q(s,a) ]
  → Q 的输出参与前向计算，但 Q 的参数不接收梯度
```

所以完整的理解是：

```text
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  采样阶段：  Policy ←──耦合──→ Q/V                          │
│             （policy 决定数据，数据影响 Q/V 学习）            │
│                                                            │
│  梯度阶段：  Policy ←──解耦──→ Q/V                          │
│             （更新 policy 时，Q/V 的输出是常数）              │
│             （更新 Q/V 时，policy 的输出也是常数）            │
│                                                            │
│  两者通过"交替迭代"间接协作：                                 │
│    1. 用当前 policy 采样数据                                 │
│    2. 用数据更新 Q/V（policy 参数冻结）                      │
│    3. 用更新后的 Q/V 更新 policy（Q/V 参数冻结）             │
│    4. 回到 1                                                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

这也解释了为什么 Actor-Critic 有时不稳定——两个独立优化器在交替迭代中追赶彼此：Critic 在追 policy 变化的数据分布，Actor 在追 Critic 变化的 value landscape。如果其中一个跑太快，另一个会被带偏。

---

## 2. 在 SL 基座模型上做 RL 偏好微调

这是 LLM 对齐的核心范式：先用监督学习训练一个强大的基座模型，再用 RL 方法做偏好微调。

### 2.1 整体流程

```text
阶段 1：预训练（SL）
  大规模语料 → 自回归语言建模 → 基座模型 (pretrained LLM)

阶段 2：监督微调 SFT（SL）
  指令-回答对 → 监督学习 → SFT 模型

阶段 3：偏好微调（RL）
  人类偏好数据 → RL 方法 → 对齐后的模型
```

阶段 3 的 RL 方法有多种实现方式，核心区别在于需要哪些组件以及如何更新参数。

### 2.2 PPO 方案（InstructGPT 路线）

PPO 是最经典的 LLM RL 微调方案。它需要四个模型同时在内存中：

```text
┌─────────────────────────────────────────────────────┐
│  1. Policy Model（正在训练的 LLM）                    │
│  2. Reference Model（冻结的 SFT 模型副本，用于 KL 约束）│
│  3. Reward Model（从偏好数据训练，冻结）                │
│  4. Value Head / Critic（估计 V(s)，提供 baseline）    │
│     通常从 Reward Model 初始化（不是从零开始）           │
└─────────────────────────────────────────────────────┘
```

训练循环：

```text
1. Policy 生成回答
2. Reward Model 打分
3. Critic 估计 baseline → 计算 advantage
4. PPO loss 更新 Policy（clipped ratio × advantage）
5. KL penalty 防止 Policy 偏离 Reference 太远
```

**Reward Model** 几乎总是一个独立模型。通常从 SFT checkpoint 初始化，把 language modeling head 换成一个标量 reward head，在人类偏好数据上训练后冻结。

**Critic（Value Function）** 通常从 Reward Model 初始化——因为 Reward Model 已经学到了质量表征，从零初始化会导致早期训练不稳定。

### 2.3 GRPO 方案（DeepSeek 路线）

GRPO 消除了 Critic，大幅简化了架构：

```text
┌──────────────────────────────────────────────────────┐
│  1. Policy Model（正在训练的 LLM）                     │
│  2. Reference Model（冻结的 SFT 模型副本，用于 KL 约束） │
│  3. Reward 来源（Reward Model 或 基于规则的验证器）       │
│     不需要 Critic / Value Network                      │
└──────────────────────────────────────────────────────┘
```

核心机制：

```text
1. 对每个 prompt，Policy 生成一组 G 个回答
2. Reward 对每个回答打分
3. 组内相对排序作为 baseline（不需要 Critic）
4. Policy gradient 更新 Policy
```

DeepSeek-R1 使用 GRPO 时采用**全量微调**（非 LoRA），训练所有参数。

GRPO 的 reward 来源可以是：
- 训练好的 Reward Model
- 基于规则的验证器（如数学题的答案正确性检查、代码的编译/测试通过率）

### 2.4 DPO 方案（Stanford 路线）

DPO 更进一步，连 Reward Model 都不需要：

```text
┌──────────────────────────────────────────────────────┐
│  1. Policy Model（正在训练的 LLM）                     │
│  2. Reference Model（冻结的 SFT 模型副本）              │
│     不需要 Reward Model，不需要 Critic                  │
│     直接在偏好对上训练                                   │
└──────────────────────────────────────────────────────┘
```

DPO 直接在 (chosen, rejected) 偏好对上优化 policy，用一个分类式的 loss：

```text
L_DPO = -log σ(β · (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
```

### 2.5 参数更新方式：全量 vs LoRA vs Adapter

不管是 PPO、GRPO 还是 DPO，Policy Model 的参数更新方式都有选择：

| 方式 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **全量微调** | 更新所有参数 | 表达能力最强 | 内存消耗巨大，4x 模型 |
| **LoRA** | 冻结基座，只训练低秩矩阵 A·B | 内存大幅降低；Reference Model 免费（即基座本身） | 表达能力受限于 rank |
| **QLoRA** | 基座量化(4-bit) + LoRA | 消费级 GPU 可跑 RLHF | 量化引入精度损失 |
| **Prefix Tuning** | 只训练前缀 token embedding | 参数极少 | 效果通常不如 LoRA |

**LoRA 在 RL 微调中的关键优势**：Reference Model 不需要单独存储。因为基座参数冻结，不加 LoRA adapter 的前向传播就是 Reference Model 的输出。这省掉了 PPO 四模型中最大的一块内存开销。

实践中的典型选择：

```text
PPO:   全量微调（InstructGPT）或 LoRA（资源受限时）
GRPO:  全量微调（DeepSeek-R1）
DPO:   LoRA 最常见（DPO + LoRA 是最省内存的对齐方案）
```

### 2.6 各方案对比总览

| 方案 | 需要的模型 | Bellman 方程 | 偏好来源 | 复杂度 |
|---|---|---|---|---|
| PPO | Policy + Ref + RM + Critic (4个) | Critic 内部仍在用 | Reward Model | 高 |
| GRPO | Policy + Ref + Reward源 (2-3个) | 完全不用 | RM 或规则验证器 | 中 |
| DPO | Policy + Ref (2个) | 完全不用 | 偏好对数据 | 低 |

---

## 3. RL 与 Diffusion / Flow Matching 模型

Diffusion 和 Flow Matching 模型本质上是生成模型——它们通过一个多步去噪/流动过程，从噪声生成样本。当把它们作为 policy 使用时，RL 可以对它们进行微调。

### 3.1 Diffusion 模型作为 Policy

**核心思路**：把 diffusion 的多步去噪过程看成一个 policy，每一步去噪就是一个 action。

**Diffuser**（Janner et al., 2022）：开创性地用 diffusion 模型做轨迹规划。模型生成完整的 state-action 轨迹，在推理时用 reward guidance 引导生成高回报轨迹。

**Diffusion Policy**（Chi et al., 2023）：用 DDPM 表示视觉运动策略。输出的是 action chunk（一段连续动作序列），diffusion 的多模态建模能力让它比 Gaussian policy 更好地处理多样化示教数据。

**Decision Diffuser**（Ajay et al., 2023）：用 classifier-free guidance 条件化生成，把 return、约束、技能作为条件。

### 3.2 对 Diffusion 模型做 RL 微调

关键挑战：reward 只在最终生成结果上可用，但 diffusion 的 "policy" 跨越 T 步去噪。

**DDPO — Denoising Diffusion Policy Optimization**（Black et al., 2023）：

```text
核心思路：把去噪过程建模为 MDP
- State: 当前噪声图像 x_t
- Action: 去噪一步的输出
- Episode: 一个完整的去噪链 x_T → x_0
- Reward: 只在最终 x_0 上给出

用 REINFORCE / PPO 估计每一步的 policy gradient
不需要对整个去噪链做反向传播
```

DDPO 成功用于：用人类偏好 / 美学评分微调 text-to-image diffusion 模型。

**DRaFT — Differentiable Reward Fine-Tuning**（Clark et al., 2023）：直接通过（截断的）去噪链反向传播 reward 梯度。比 DDPO 更 sample-efficient，但需要可微的 reward model。

**DPPO — Diffusion Policy Policy Optimization**（Ren et al., 2025）：专门为连续控制中的 diffusion policy 适配 PPO。处理跨去噪步的 clipping、高维 action space 等问题。

### 3.3 Flow Matching 模型与 RL

Flow Matching（Lipman et al., 2023）用确定性 ODE 代替随机 SDE，推理速度更快（5-10 步 vs 50-100 步），对实时控制至关重要。

在 RL 中的应用：

```text
- Flow matching policy：用 ODE 积分从噪声生成 action
- RL 微调方式类似 DDPO：把 ODE 离散化的每一步看成 MDP 中的 action
- Action Flow Matching：用条件 flow matching 学习 action 生成
```

### 3.4 关键技术挑战

| 挑战 | 说明 | 常见解决方案 |
|---|---|---|
| 跨步 credit assignment | Reward 只在最终输出，需要分配到 T 步 | 把去噪链建模为 episodic MDP |
| 梯度方差 | 长去噪链导致 REINFORCE 方差大 | 减少步数 / baseline / 截断反向传播 |
| Reward 可微性 | 直接反向传播需要可微 reward | Policy gradient 方法避开此限制 |
| KL 正则化 | 防止微调后模型崩坏 | 类似 RLHF 的 KL penalty |
| Action 一致性 | Diffusion policy 输出 action chunk，需要时序连贯 | 重叠窗口设计 |

### 3.5 时间线

```text
Diffuser (2022) → Decision Diffuser, Diffusion Policy, DDPO, DRaFT (2023) → DPPO, Flow Matching Policy (2024-2025)
```

趋势：把去噪过程当作 multi-step MDP 做 policy gradient；flow matching 作为更快推理的替代方案正在兴起。

---

## 4. 环境的本质 与 Agent RL vs One-Round RL

### 4.0 先理解"环境"：RL 里最容易被忽视的角色

RL 的所有讨论都围绕 policy、Q、loss，但**环境（environment）才是 RL 的地基**。没有环境，就没有 state、没有 reward、没有 rollout。

环境不是一个抽象概念——它是一个具体的系统，agent 的 action 作用于它，它返回 next state 和 reward。不同场景下，"环境"的物理形态完全不同：

```text
┌───────────────────────────────────────────────────────────────┐
│                    "环境"在不同场景下的实体                      │
├──────────────┬────────────────────────────────────────────────┤
│  场景        │  环境是什么                                      │
├──────────────┼────────────────────────────────────────────────┤
│  游戏/仿真   │  仿真器（Atari, MuJoCo, Isaac Gym）             │
│              │  完全可控、可重置、可并行                          │
│              │  reward 由游戏规则直接给出                        │
│              │                                                │
│  Agentic AI  │  计算机系统本身（"harness"）                     │
│              │  文件系统、终端、网络、API、数据库                  │
│              │  agent 的 action = 执行命令/写文件/调用 API       │
│              │  环境的 state = 文件内容/程序输出/网页内容          │
│              │  reward = 测试通过率/任务完成度/人类反馈            │
│              │                                                │
│  自动驾驶    │  物理世界                                        │
│              │  道路、车辆、行人、天气、交通信号                   │
│              │  agent 的 action = 油门/刹车/转向                 │
│              │  环境的 state = 传感器观测（相机/LiDAR/雷达）      │
│              │  reward = 安全/效率/舒适度的工程评分                │
│              │                                                │
│  机器人      │  物理世界 + 机器人本体                            │
│              │  关节、力矩、接触面、重力                          │
│              │  agent 的 action = 关节力矩/目标位置               │
│              │  环境的 state = 关节角度/力传感器/视觉             │
│              │  reward = 任务完成/能耗/碰撞惩罚                  │
└──────────────┴────────────────────────────────────────────────┘
```

关键理解：

```text
仿真器环境：便宜、可控、可并行 → 适合大规模 RL 训练
Agentic AI 环境（harness）：中等成本、可重置 → 适合 code/tool agent 训练
物理世界环境：昂贵、不可逆、有安全风险 → 需要 sim2real 或极其谨慎的在线学习
```

环境的性质决定了 RL 的训练策略：
- 环境便宜 → 可以大量 rollout → on-policy 方法可行（PPO）
- 环境昂贵 → 必须复用数据 → off-policy 方法更实际（SAC）
- 环境危险 → 先在仿真中训练 → sim2real 迁移

### 4.1 传统 RL：单轮时序交互

传统 RL 的交互是一个连续的时间序列 rollout：

```text
s_0 → a_0 → r_0 → s_1 → a_1 → r_1 → ... → s_T
```

每个 action 直接作用于环境，立即产生下一个 state 和 reward。整个 rollout 是一个连续的、单轮的 episode。

典型场景：
- **自动驾驶**：每一时刻感知环境 → 输出控制指令（油门、刹车、转向）→ 车辆移动 → 新的环境状态
- **机器人控制**：每个 action chunk 是一段连续动作序列，agent 持续与物理环境交互
- **Atari 游戏**：每帧观测 → 选择动作 → 游戏状态更新

### 4.2 Agentic RL：多轮对话式交互

Agentic RL 的交互是多轮的、嵌套的：

```text
Episode = 多轮对话/交互

Round 1: agent 观察 → 思考 → 调用工具/输出 → 环境反馈
Round 2: agent 观察反馈 → 思考 → 调用工具/输出 → 环境反馈
...
Round N: agent 观察反馈 → 最终输出 → episode 结束
```

典型场景：
- **Agentic AI**：与 coding harness 多轮交互（读代码 → 写代码 → 跑测试 → 看结果 → 修改 → 重跑）
- **对话系统**：多轮对话中根据用户反馈调整策略
- **工具使用 Agent**：搜索 → 阅读 → 总结 → 再搜索 → 最终回答

### 4.3 两者的本质联系

看似不同，但从 RL 的 MDP 框架看，两者是统一的：

```text
传统 RL:
  state = 物理环境状态
  action = 控制指令
  reward = 即时/延迟奖励
  episode = 一条连续轨迹

Agentic RL:
  state = 对话历史 + 环境状态
  action = agent 的一次完整输出（可能包含工具调用）
  reward = 每轮反馈 / 最终任务完成度
  episode = 一次完整的多轮任务
```

**关键洞察：自动驾驶本质上就是一个 agentic RL 过程。**

在自动驾驶中：
- 每个 action chunk（比如未来 2 秒的控制序列）相当于 agentic AI 中的一个"对话回合"
- 每个 action chunk 执行后，agent 观察新的环境状态，做新的决策
- 整个驾驶 rollout 是多个 action chunk 串起来的多轮交互

对比：

| 维度 | 自动驾驶 | Agentic AI |
|---|---|---|
| 一个 "回合" | 一个 action chunk（~100ms-2s 的控制序列） | 一次 agent 输出（工具调用 / 文本生成） |
| 环境反馈 | 新的传感器观测 | 工具执行结果 / 用户回复 |
| Episode | 一段完整驾驶（几分钟到几小时） | 一个完整任务（代码修改 / 问题解答） |
| State 空间 | 连续、高维（图像 + 点云 + 车辆状态） | 离散、变长（文本 + 工具状态） |
| Action 空间 | 连续（油门、刹车、转向） | 离散/混合（token 生成 + 工具选择） |

### 4.4 训练方式的差异

虽然 MDP 框架统一，但训练方式有重要差异：

**传统 RL（自动驾驶/机器人）**：

```text
- 通常使用 continuous action policy（Gaussian / Diffusion / Flow）
- Actor-Critic 架构常见（SAC / TD3 / PPO）
- 数据来自仿真器的高频交互
- Reward 通常是工程设计的（距离、碰撞、舒适度）
```

**Agentic RL（LLM Agent）**：

```text
- 通常使用 autoregressive token policy（LLM）
- PPO / GRPO / DPO
- 数据来自 agent 与环境的多轮交互
- Reward 来自任务完成度 / 人类偏好 / 验证器
- 每个 "action" 的计算成本远高于传统 RL
```

### 4.5 核心共性

不管是自动驾驶还是 agentic AI，RL 的核心结构都是：

```text
agent 与环境交互 → 获得反馈 → 更新策略 → 改变未来的交互方式
```

区别只是：
- action 的粒度不同（毫秒级控制 vs 秒级/分钟级推理）
- 环境的性质不同（物理世界 vs 数字/文本世界）
- 交互的频率不同（高频连续 vs 低频离散）

---

## 5. Sim & Real：RL 在仿真与现实之间的迁移

RL 训练通常需要大量交互，但真实环境中的交互昂贵、危险、缓慢。因此 Sim2Real 和 Real2Sim 成为关键技术。

### 5.1 Sim2Real：从仿真到现实

核心挑战是 **reality gap**——在仿真中表现完美的 policy，部署到真实环境往往失败，因为物理引擎、视觉渲染、动力学参数都与现实有偏差。

主要方法：

**Domain Randomization（域随机化）**

```text
训练时随机化仿真器参数：
- 物理参数：摩擦力、质量、延迟、阻尼
- 视觉参数：光照、纹理、颜色、相机位置
- 动力学：传动误差、传感器噪声

→ 真实世界变成"只是另一组随机参数"
→ Policy 学会对参数变化 robust
```

里程碑：OpenAI 用 Automatic Domain Randomization（ADR）训练机械手解魔方（2019），动态扩展数千个参数的随机化范围。

**System Identification（系统辨识）**

```text
测量真实世界物理参数 → 校准仿真器
可与 Domain Randomization 结合：在校准值附近随机化，而非全范围均匀随机
```

**Domain Adaptation（域适应）**

```text
用 CycleGAN / 对抗训练 对齐仿真和真实的视觉分布
不需要配对数据即可缩小视觉差距
```

**Progressive Transfer**

```text
先在仿真中训练，再在真实环境中用少量数据微调
保留仿真学到的特征（Rusu et al., 2017 Progressive Nets）
```

### 5.2 Real2Sim：从现实到仿真

反方向——用真实世界数据来构建或改进仿真器，使仿真训练更有效。

**Neural Scene Reconstruction（神经场景重建）**

```text
用 NeRF / 3D Gaussian Splatting 从真实传感器数据重建场景
→ 可编辑的、照片级逼真的仿真环境
例：MARS (2023)、UniSim (2023) 重建驾驶场景用于闭环仿真
```

**Learned World Models（学习的世界模型）**

```text
从真实数据训练生成模型，作为隐式仿真器
- GAIA-1 (Wayve, 2023)：从驾驶视频学习世界模型，生成新场景
- DayDreamer (Hafner et al., 2022)：从真实机器人交互学习世界模型，完全在模型内训练 policy
```

**Digital Twins（数字孪生）**

```text
真实环境/设备的高精度数字副本
持续用传感器数据更新
广泛应用于制造业和自动驾驶验证
```

**Sim Parameter Calibration（仿真参数校准）**

```text
用真实轨迹数据优化仿真器参数（摩擦力、阻力、延迟）
方法：贝叶斯优化、可微仿真
目标：最小化 sim-real 差距
```

### 5.3 自动驾驶中的 Sim-Real 循环

自动驾驶是 Sim2Real / Real2Sim 循环的最典型应用：

```text
┌─────────────────────────────────────────────┐
│                                             │
│   Real2Sim:                                 │
│   真实驾驶日志 → 重建 3D 场景 → 仿真环境      │
│   (Waymo SurfelGAN, NVIDIA DRIVE Sim,       │
│    Tesla fleet data → simulation)           │
│                                             │
│          ↓                                  │
│                                             │
│   Sim Training:                             │
│   在重建/合成环境中训练 policy               │
│   域随机化天气、交通、传感器噪声              │
│   测试危险/罕见场景（edge cases）             │
│                                             │
│          ↓                                  │
│                                             │
│   Sim2Real:                                 │
│   部署到真实车辆                              │
│   收集新的真实数据                            │
│                                             │
│          ↓                                  │
│                                             │
│   (循环：新数据改进仿真 → 更好的训练 → ...)   │
│                                             │
└─────────────────────────────────────────────┘
```

趋势：**持续循环**——真实数据改进仿真（Real2Sim），更好的仿真训练改进真实表现（Sim2Real），然后收集新数据，循环往复。

### 5.4 Sim-Real 方法对比

| 方向 | 方法 | 核心思路 | 典型应用 |
|---|---|---|---|
| Sim2Real | Domain Randomization | 随机化仿真参数，让 policy 对变化 robust | 机器人灵巧操作 |
| Sim2Real | System Identification | 校准仿真器参数匹配真实 | 工业控制 |
| Sim2Real | Domain Adaptation | 对齐 sim/real 视觉分布 | 视觉抓取 |
| Sim2Real | Progressive Transfer | 先仿真训练，再真实微调 | 移动机器人 |
| Real2Sim | Neural Reconstruction | 从传感器数据重建场景 | 自动驾驶仿真 |
| Real2Sim | Learned World Model | 从真实数据学隐式仿真器 | 驾驶/机器人 |
| Real2Sim | Digital Twin | 高精度数字副本 | 制造业/驾驶验证 |
| Real2Sim | Parameter Calibration | 用真实轨迹校准仿真参数 | 物理仿真 |

---

## 6. 总览：RL 在不同场景下的角色

```text
┌──────────────────────────────────────────────────────────┐
│                        RL 的角色                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  从零训练          从 SL 基座微调        与生成模型结合     │
│  ┌─────────┐      ┌──────────┐        ┌──────────┐      │
│  │ Q-learn │      │ PPO+RM   │        │ DDPO     │      │
│  │ DQN     │      │ GRPO     │        │ DPPO     │      │
│  │ PPO     │      │ DPO      │        │ DRaFT    │      │
│  │ SAC     │      │ +LoRA    │        │ Flow RL  │      │
│  └─────────┘      └──────────┘        └──────────┘      │
│       ↕                 ↕                   ↕            │
│  纯 RL 环境        LLM 对齐            图像/控制生成       │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Sim ←→ Real              Agent RL vs One-Round RL       │
│  ┌──────────┐             ┌──────────────────────┐      │
│  │ Domain   │             │ 自动驾驶 ≈ Agentic    │      │
│  │ Random.  │             │ (action chunk = 回合) │      │
│  │ Real2Sim │             │ LLM Agent = 多轮交互  │      │
│  │ World    │             │ 本质都是闭环 MDP      │      │
│  │ Models   │             │                      │      │
│  └──────────┘             └──────────────────────┘      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```
