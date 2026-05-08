# 从几个维度理解强化学习：从 Bellman、Argmax 到 Preference Optimization

> **定位说明**：本文是对 RL 的**微观机制分析**——从 loss 设计、数据利用、探索策略、bias 管理、压缩表示、action 建模等多个维度，拆解每个技术点在做什么、为什么这么做。它与《从 Bellman 到 Preference Learning》形成互补：后者是宏观演化史（为什么 RL 会从 DP 走向偏好学习），本文是微观工程视角（每个机制的具体原理和设计动机）。

---

## 0. 总纲：RL 不是一个干净的监督学习问题

监督学习通常有一个相对清晰的结构：固定数据分布、明确标签、明确 loss。强化学习不是这样。

在 RL 里，policy 决定采样，采样决定数据分布，数据分布决定 Q / advantage 的估计，而 Q / advantage 又反过来更新 policy。policy 一变，未来采样分布也会变。

所以 RL 的核心不是"给定数据以后优化一个干净 loss"，而是在一个动态闭环里持续处理采样偏差、估计偏差和优化稳定性。

可以把 RL 理解成：

```text
在不断变化的数据分布上，用带偏的采样数据，估计长期回报，并优化下一轮采样策略。
```

因此，TD、Bellman、Q max、argmax、entropy、importance sampling、clipping、replay buffer、target network、double Q、actor-critic、PPO、SAC 这些东西不是孤立技巧。它们都在处理同一个问题：

```text
如何在 biased sampling 下，稳定地优化 biased objective。
```

---

## 1. 从 loss 组织角度理解：MSE → max Q → 熵增 → 加权对数似然

### 1.1 Value-based 方法：把 RL 组织成 MSE 回归

DQN / Q-learning 这类方法，本质上是把 RL 变成一个回归问题。目标是让当前 Q 逼近 Bellman target：

```text
Q(s, a) ≈ r + γ max_a' Q_target(s', a')
```

所以 loss 可以写成：

```text
L = (Q(s,a) - y)^2

y = r + γ max_a' Q_target(s', a')
```

这看起来像监督学习，但关键区别是：这里的 label `y` 不是外部世界直接给的，而是模型自己通过 Bellman bootstrap 和 max Q 算出来的。

也就是说：

```text
模型自己造 label，然后再拟合这个 label。
```

这就是 TD learning 的核心特点。它不是普通 MSE，而是带 bootstrap 的自举回归。

这个结构会带来两个问题：

1. target 本身有噪声。
2. max / argmax 会放大噪声。

所以 value-based RL 的 MSE loss 并不等价于普通监督学习里的 MSE。它的 label 是动态的、模型相关的、带偏的。

### 1.2 Q max 为什么容易 overestimate

关键问题出在：

```text
max_a Q(s, a)
```

如果每个 action 的 Q 估计都有噪声：

```text
Q_hat(a) = Q_true(a) + noise(a)
```

那么取 max 的时候，更容易选中那些被噪声高估的 action。因此一般会有：

```text
E[max Q_hat] > max E[Q_hat]
```

这就是 Q overestimation 的来源。

它不是 DQN 特有的问题。只要算法依赖 max Q、argmax Q、advantage ranking，都会面对类似问题。区别只是不同算法用不同方式缓解它。

### 1.3 Double DQN / TD3：稳定 MSE target

Double DQN 的核心思路是把 action selection 和 value evaluation 分开。

原始 DQN：

```text
y = r + γ max_a Q_target(s', a)
```

Double DQN：

```text
a* = argmax_a Q_online(s', a)

y = r + γ Q_target(s', a*)
```

它不是完全消除 bias，而是减少"同一个 noisy Q 同时负责选择和估值"带来的高估。

TD3 在连续动作空间里也做了类似事情。它使用两个 critic，并取较小值作为 target：

```text
y = r + γ min(Q1, Q2)
```

这是一种保守估计，用来对抗 actor 对 critic 错误的过度利用。

### 1.4 Max Q Loss：Actor 的 loss 就是直接最大化 Q

在 value-based 方法里，loss 是 MSE——让 Q 逼近 Bellman target。但当 action 空间是连续的，argmax 不能直接算，于是引入 actor 网络。

Actor 的 loss 不再是 MSE，而是一个全新的形态——**直接最大化 Q**：

```text
L_actor = - Q(s, μ_θ(s))
```

也就是说，actor 调整参数 θ，让 critic 给它打的分尽可能高。

这是 DDPG 的核心思路。它可以理解为 argmax 的连续化：

```text
离散 action：  a* = argmax_a Q(s,a)          → 枚举所有 action，选最大
连续 action：  θ* = argmax_θ Q(s, μ_θ(s))    → 用梯度上升让 actor 输出高 Q 的 action
```

所以 max Q loss 的本质是：

```text
把 argmax 这个离散操作，变成了一个可微的梯度优化问题。
```

Actor 通过 critic 的梯度信号 ∇_a Q(s,a) 来调整自己的输出方向。

但 max Q loss 有一个根本风险：**actor 会 exploit critic 的估计误差**。

Critic 的 Q 不是真值，它有噪声和高估。如果 actor 不断朝着 Q 最高的方向走，它可能走到的不是真正好的 action，而是 critic 估计误差最大的 action。

这就是为什么 TD3 要加那么多保护措施：

```text
double critic + min Q：压制高估
delayed policy update：让 critic 先稳定，再更新 actor
target policy smoothing：防止 actor exploit Q 的尖锐峰值
```

从 loss 演化的角度看，max Q loss 处于 MSE 和 policy gradient 之间：

```text
MSE loss：    让 Q 逼近 target            → 被动拟合
Max Q loss：  让 actor 输出高 Q 的 action   → 主动利用 Q landscape
PG loss：     让好 action 概率变大          → 直接塑造分布
```

MSE 是"我估值"；max Q 是"我利用估值做决策"；PG 是"我直接优化决策"。

### 1.5 Policy Gradient：从 max Q 走向加权对数似然

Policy gradient 不再先学一个 Q，然后通过 argmax 得到 action，而是直接优化 policy。

目标可以写成：

```text
J(θ) = E[return]
```

通过 policy gradient theorem，梯度可以写成：

```text
∇J(θ) = E[∇ log πθ(a|s) · A(s,a)]
```

因此 loss 常被组织成：

```text
L = - log πθ(a|s) · A(s,a)
```

这非常像加权最大似然：

```text
好的 action：提高概率。
差的 action：降低概率。
```

所以从 loss 组织角度看，policy gradient 是从 Q 的 MSE 回归，转向 action probability 的加权 MLE。

这是一个重要的哲学变化。

Q-learning 的思路是：

```text
我先估计一个 Q 世界，然后通过 argmax 选 action。
```

Policy gradient 的思路是：

```text
我直接调整采样分布，让好的 action 更容易被采样。
```

这就是从 value-first 到 policy-first 的转变。

#### 为什么 value-based 讲 loss，而 policy gradient 讲 gradient？

初学 RL 时一个常见的困惑是：DQN 这类方法直接写 loss function，而 policy gradient 推导时先讲 gradient，最后才提到 loss。为什么？

原因是两者的推导起点不同。

**Value-based 方法**的起点是一个自然的回归问题：

```text
我有一个 target y（Bellman target），我希望 Q(s,a) 逼近它。
所以 loss = (Q - y)^2。梯度从 loss 推出来。
```

这和监督学习完全同构：先有 loss，再求梯度。

**Policy gradient** 的起点不是一个 loss，而是一个目标函数的梯度：

```text
我想最大化 J(θ) = E[return]。
通过 policy gradient theorem，得到：
∇J(θ) = E[∇ log πθ(a|s) · A(s,a)]
```

注意这个 gradient 是从 J(θ) 的数学推导里出来的，不是先写一个 loss 再求导。它的推导用到了对轨迹概率的微分（likelihood ratio trick），直接得到了梯度的形式。

那 loss 是什么？其实 loss 是**事后构造的**。为了能用 PyTorch / TensorFlow 的自动微分，我们需要构造一个 loss，使得 autograd 对它求导之后，恰好等于上面的 policy gradient。

这个 loss 就是：

```text
L = - log πθ(a|s) · A(s,a)
```

验证：对 θ 求导：

```text
∇L = - ∇ log πθ(a|s) · A(s,a)
```

加上负号（因为 optimizer 做 minimize，而我们要 maximize J），就恢复了 policy gradient 的形式。

所以完整的逻辑是：

```text
Value-based: loss 是起点 → 梯度是从 loss 推出来的。
Policy gradient: 梯度是起点 → loss 是为了配合 autograd 事后构造的。
```

但最终在代码里，两者的训练循环完全一样：

```text
loss = compute_loss(batch)
loss.backward()
optimizer.step()
```

这就是为什么说 RL 和 SL 的 optimization framework 基本相同——不管 loss 是先有还是后构造的，最终都落入了同一套 loss → backward → update 的流程。

### 1.6 Policy Gradient 和最大似然：从无权重 MLE 到加权 MLE

最大似然估计本质上是：使用统计行为来估计参数。

在普通监督学习 / 行为克隆里，如果数据是：

```text
(s_i, a_i)
```

最大似然就是最大化：

```text
Σ log πθ(a_i | s_i)
```

它的意思是：

```text
让模型更容易生成数据里出现过的 action。
```

在这个最普通的 MLE 里，每条统计样本默认权重是一样的。只要数据里出现了某个 action，模型就倾向于提高它的概率。

所以普通 MLE / behavior cloning 的问题是：

```text
它只知道"人/历史策略这么做过"，但不知道"这么做好不好"。
```

Policy gradient 在这个基础上多了一个权重。它不是简单最大化：

```text
log πθ(a|s)
```

而是最大化带权重的 log likelihood：

```text
A(s,a) · log πθ(a|s)
```

或者从 loss 角度写成：

```text
L = - A(s,a) · log πθ(a|s)
```

这里的 advantage / return 就是权重。

因此 policy gradient 可以理解成：

```text
加权最大 log 似然。
```

普通最大似然是：

```text
这个 action 在数据里出现了，所以提高它的概率。
```

Policy gradient 是：

```text
这个 action 在数据里出现了，而且它带来了更高回报，所以更大幅度提高它的概率。
```

可以压缩成一句话：

```text
MLE 是对行为频率建模；Policy Gradient 是对带偏好权重的行为频率建模。
```

### 1.7 Entropy：不是装饰项，而是采样分布控制项

PPO / SAC 里经常会出现 entropy bonus。它的作用不是简单"鼓励随机"，而是控制 policy 不要太快塌缩。

从分布角度看，entropy 最大的时候，policy 更接近均匀分布。entropy 小的时候，policy 更尖锐，更接近确定性选择。

所以熵增项本质上是在鼓励：

```text
不要让 policy 太快变成一个极端偏好的分布。
```

如果没有 entropy，policy gradient 会倾向于快速把概率集中到当前 advantage 高的 action 上。这样会出现：

```text
policy 太快集中到少数 action
→ 采样变窄
→ 数据多样性下降
→ Q / advantage 估计更偏
→ 训练更容易陷入局部最优
```

所以 entropy 是一种采样分布正则化项。

更准确地说，policy 优化里其实有两个力量：

```text
偏好项：让高 advantage 的 action 概率变大。
熵增项：让分布不要太尖锐，向更均匀的方向拉。
```

因此最终学到的 policy 不是纯粹的均匀分布，也不是完全的偏好塌缩，而是在二者之间取得平衡：

```text
最终 policy = 偏好分布 和 均匀分布 之间的均衡。
```

SAC 更进一步，直接把目标写成：

```text
maximize reward + entropy
```

这意味着 exploration 不再是外部附加技巧，而是目标函数的一部分。

从 loss 组织角度，可以这样理解：

```text
MSE：拟合 Bellman target。
加权 log π：用 reward / advantage 加权行为似然。
entropy：把 policy 从过度偏好拉回到更均匀的分布。
```

这三个东西对应 RL 的三个层面：

```text
value estimation
preference-weighted behavior modeling
sampling diversity control
```

---

## 2. 从第一性原理判断 on-policy / off-policy / near on-policy

### 2.1 不要先背算法名，要先看"数据是怎么来的"

判断一个算法是 on-policy 还是 off-policy，第一步不是看它叫 PPO、DQN 还是 SAC，而是看：

```text
当前这次参数更新，使用的数据，是不是由当前正在优化的 policy 生成的？
```

更具体地说，要看两条链路是否一致：

```text
采样链路：数据里的 action 是哪个 policy 选出来的？
优化链路：当前更新时，优化的是哪个 policy / 哪个 Q target？
```

如果采样 action 的 policy 和当前被优化的 policy 是同一个，或者近似同一个，那就是 on-policy / near on-policy。

如果采样 action 来自旧 policy、历史 replay buffer、人工数据、其他控制器，或者任意 behavior policy，而当前算法仍然可以用这些数据更新自己的 Q / policy，那就是 off-policy。

核心判断点：

```text
不只是"有没有历史数据"，而是"历史数据里的 action 分布是否必须等于当前 policy 的 action 分布"。
```

### 2.2 第一性原理判据

```text
如果 loss / target 要求 a ~ π_current(a|s)，就是 on-policy。

如果 loss / target 可以接受 a ~ μ(a|s)，其中 μ 是任意行为策略，就是 off-policy。

如果 loss 可以接受 a ~ π_old(a|s)，但要求 π_old 和 π_current 不能差太远，就是 near on-policy。
```

这里：

```text
π_current = 当前正在优化的 policy
π_old     = 最近采样时使用的旧 policy
μ         = 任意 behavior policy，可以是历史 policy、随机策略、人类策略、规则控制器、replay buffer 来源
```

真正的判断不是"有没有 replay buffer"，而是：

```text
这个更新公式是否要求数据 action 来自当前 policy？
```

### 2.3 为什么 Policy Gradient 原始形式是 on-policy

Policy gradient 的核心形式是：

```text
∇J(θ) = E_{s,a ~ πθ}[∇ log πθ(a|s) · A^{πθ}(s,a)]
```

注意这个期望下面的采样分布是：

```text
s,a ~ πθ
```

也就是说，这个梯度公式本身要求 action 是当前 policy πθ 采样出来的。

因此原始 policy gradient 是 on-policy。

如果这些 action 不是当前 policy 采样的，而是很久以前的 policy 或别的策略采样的，那么：

```text
log πθ(a|s) · A(s,a)
```

这个加权最大似然就不再是原始 policy gradient 的无偏估计。

所以原始 PG / A2C / A3C 这类算法是 on-policy，不是因为名字，而是因为它的梯度推导要求：

```text
a 必须来自当前 πθ 的采样分布。
```

### 2.4 为什么 Q-learning / DQN 是 off-policy

Q-learning 的核心 target 是：

```text
y = r + γ max_a' Q(s', a')
```

注意这里并不要求 replay buffer 里的 action 是当前 policy 选出来的。

Q-learning 的目标不是直接模仿 behavior policy，而是学习：

```text
在状态 s 执行动作 a 之后，如果未来都按 greedy policy 走，长期回报是多少。
```

数据里的 action `a` 只负责告诉我们：

```text
我曾经在 s 做过 a，看到了 r 和 s'。
```

而 target 里的未来动作不跟随历史 behavior policy，而是直接使用：

```text
max_a' Q(s', a')
```

因此 Q-learning 可以脱离采样 policy 学习目标 policy，这就是 off-policy 的根本原因。

### 2.5 为什么 PPO 是 near on-policy

PPO 的数据通常来自最近一轮旧 policy：

```text
a ~ π_old(a|s)
```

但更新的是当前 policy：

```text
πθ(a|s)
```

PPO 又不是完整 off-policy，因为它不允许旧数据和当前 policy 差太远。它用 importance ratio：

```text
r(θ) = πθ(a|s) / π_old(a|s)
```

然后 clip：

```text
clip(r, 1-ε, 1+ε)
```

这表示：

```text
我可以用最近旧 policy 的数据。
但如果当前 policy 和旧 policy 差太远，这条数据的更新权重就要被限制。
```

因此 PPO 的准确理解是：

```text
它不是纯 on-policy，而是 near on-policy。
它允许有限历史数据复用，但通过 ratio clipping 限制 distribution shift。
```

### 2.6 为什么 DDPG / TD3 / SAC 是 off-policy

DDPG、TD3、SAC 都通常使用 replay buffer。

但更根本的原因是：它们的 critic 可以用历史 transition 学习 Bellman target。

例如 DDPG / TD3 的 critic target 类似：

```text
y = r + γ Q_target(s', μ_target(s'))
```

当前更新时，并不要求这个 `a` 是当前 actor μθ 生成的。

SAC 也是类似。SAC 的 critic 可以从 replay buffer 里学习 soft Bellman target，actor 再优化带 entropy 的目标。它不要求 replay buffer 里的 action 来自当前 policy，所以也是 off-policy。

### 2.7 最简判决表

| 算法 | 数据里的 action 来自哪里 | 更新是否要求 action 来自当前 policy | 判定 |
|---|---|---|---|
| REINFORCE | 当前 policy | 是 | On-policy |
| A2C / A3C | 当前 policy | 是 | On-policy |
| PPO | 最近旧 policy | 近似要求，需要 ratio/clip 控制 | Near on-policy |
| TRPO | 最近旧 policy | 近似要求，需要 trust region 控制 | Near on-policy |
| Q-learning | 任意 behavior policy | 否 | Off-policy |
| DQN | replay buffer / 旧 policy / ε-greedy | 否 | Off-policy |
| Double DQN | replay buffer / 旧 policy / ε-greedy | 否 | Off-policy |
| DDPG | replay buffer / 旧 actor + noise | 否 | Off-policy |
| TD3 | replay buffer / 旧 actor + noise | 否 | Off-policy |
| SAC | replay buffer / historical stochastic policy | 否 | Off-policy |

### 2.8 最压缩的判断逻辑

```text
如果更新公式要求样本 action 是当前 policy 采出来的，就是 on-policy。
如果更新公式可以使用任意历史 action，只要有 (s,a,r,s')，就是 off-policy。
如果只能用最近旧 policy 的 action，并且要限制新旧 policy 差异，就是 near on-policy。
```

核心：

```text
历史数据里的 action 分布，是否可以和当前正在优化的 policy 分布不一致。
```

---

## 3. 从 exploration / exploitation 理解：为什么 argmax 要加随机

### 3.1 Greedy 的根本问题

如果直接使用：

```text
a = argmax_a Q(s,a)
```

那么 policy 会很快变成确定性策略。

问题是，Q 本身在早期还没有估准，但 argmax 已经开始强行选择。这样会把估计噪声直接转化为采样偏差。

典型过程是：

```text
早期 Q 有噪声
→ argmax 选中了被噪声高估的 action
→ policy 过早偏向它
→ 其他 action 没机会被采样
→ 数据分布变窄
→ Q 更难纠正
```

所以 greedy 的问题不是简单"太贪心"，而是：

```text
greedy 会把估计噪声转化成采样偏差。
```

### 3.2 ε-greedy：给 argmax 加随机性

DQN 里常用 ε-greedy：

```text
以 1-ε 的概率选 argmax Q。
以 ε 的概率随机选 action。
```

这相当于大部分时间 exploitation，少部分时间 exploration。

它解决的问题是：不要让早期 noisy argmax 彻底控制数据分布。

### 3.3 为什么有人会用均匀分布目标

在一些探索设计中，会让 action 采样更接近均匀分布，或者让 policy 不要太尖锐。

原因是：

```text
如果 policy 太早集中到少数 action，其他 action 的 Q 永远估不准。
```

从统计角度看：

```text
没有采样，就没有估计。
没有覆盖，就没有泛化保证。
```

所以均匀探索的意义不是"随机而随机"，而是先保证 action space 被足够覆盖，然后再谈优化。

### 3.4 Entropy 比 ε-greedy 更软

ε-greedy 是硬规则：要么 greedy，要么 random。

Entropy 是软约束：让 policy 自己保持一定随机性。

SAC 更进一步，直接把目标改成：

```text
maximize reward + entropy
```

这意味着 exploration 不再是外部附加技巧，而是目标函数的一部分。

采样多样性本身就是优化目标的一部分。

---

## 4. 从数据利用率理解：理论利用率 vs 工程利用率

### 4.1 理论利用率

理论上，off-policy 的数据利用率更高，因为 replay buffer 里的数据可以反复使用。

```text
一条 transition 可以训练很多次。
```

所以 DQN、DDPG、TD3、SAC 这类方法的样本利用率通常比纯 on-policy 方法更高。

On-policy 的问题是数据和当前 policy 绑定太强。policy 一更新，旧数据很快就过期。因此理论样本利用率低。

### 4.2 工程利用率不等于理论利用率

工程上，数据利用率还取决于很多东西：

```text
训练是否稳定。
调参是否困难。
是否容易并行采样。
是否容易复现。
是否对超参数敏感。
是否能在真实系统中安全采样。
```

例如，PPO 的理论样本利用率不如 SAC，但 PPO 工程上很稳定，容易调参，适合大规模并行。因此在机器人、RLHF、simulation 系统中经常被使用。

SAC 理论上样本利用率高，但工程上也有代价，例如 critic 训练复杂、temperature 调节、Q bias、replay 分布控制等。

一个重要区分是：

```text
理论利用率 = 每条样本能被优化器使用多少次。
工程利用率 = 花同样工程成本，最终能不能稳定得到好 policy。
```

有时候，理论利用率高并不等于工程效率高。如果 off-policy 的 Q 学崩了，replay buffer 里的样本被重复使用很多次也没有意义。

---

## 5. 从 greedy 到稳定 greedy，再到抛弃 argmax

### 5.1 最早：直接 greedy

Q-learning 的核心动作选择是：

```text
a = argmax Q(s,a)
```

这是最直接的 exploitation。

但 argmax 是一个很强的非线性函数。它会放大 Q 的估计误差。只要 Q 有一点噪声，argmax 就可能选错。而一旦选错，后续采样分布也会被影响。

### 5.2 稳定 greedy 的方法

为了让 greedy 不那么容易崩，后续算法引入了很多稳定技巧。

**DQN** 使用三类关键手段：

```text
replay：打散相关性，提高数据复用。
target network：稳定 bootstrap target。
ε-greedy：避免过早采样塌缩。
```

**Double DQN** 把 action selection 和 value evaluation 分开，减少 max over noisy Q 的高估。

**Dueling DQN** 把 Q 拆成：

```text
Q(s,a) = V(s) + A(s,a)
```

让网络结构更容易表达"这个 state 本身好不好"和"这个 action 相对其他 action 好多少"。

**TD3** 针对连续动作空间下的 greedy actor，引入：

```text
double critic
delayed policy update
target policy smoothing
```

这些方法的本质都是避免 actor 过度 exploit critic 的错误。

### 5.3 为什么很多方法最后抛弃 hard argmax

Policy gradient、PPO、SAC 的方向是：

```text
不再显式用 hard argmax 选 action，而是直接优化 policy distribution。
```

也就是从：

```text
先学 Q，再 argmax。
```

变成：

```text
直接学 π(a|s)。
```

这样可以避免 argmax 这个强非线性操作直接支配训练。

PPO 用 advantage 加权 log π。SAC 同时最大化 reward 和 entropy。它们的共同点是把 action 选择从 hard argmax 变成 soft distribution optimization。

这就是从 greedy 到 soft policy 的演化。

---

## 6. 从 bias optimization 角度理解

### 6.1 RL 不是追求完全无偏，而是在管理 bias

RL 里很难得到真正无偏的优化过程，因为采样、估计、目标构造都带偏。

主要 bias 来源包括：

```text
sampling bias
bootstrapping bias
max / argmax bias
function approximation bias
replay buffer distribution bias
reward design bias
policy-induced data bias
```

所以 RL 不是在做"找到真实 objective，然后无偏优化"，而是在做：

```text
在一堆偏差里，选择一个工程上更稳定、更可控的偏差结构。
```

### 6.2 Q-learning 的 bias

Q-learning 的 bias 主要来自：

```text
max Q target
bootstrap
off-policy replay
```

它的优点是数据利用率高，缺点是容易 overestimate，容易被错误 Q 引导。

所以后续方法不断修正它：

```text
target network
Double Q
clipped double Q
conservative Q
```

这些都属于 bias control。

### 6.3 Policy gradient 的 bias

Policy gradient 避开了 hard argmax，但它有自己的问题：variance 大、sample inefficient、advantage 估计有噪声。

所以它用 baseline、GAE、critic、advantage normalization、entropy、clipping 等方法来控制 variance 和 update bias。

PPO 的 clipping 本质上就是主动引入 bias。它不让某些样本的 importance ratio 太大，这会让梯度不完全等于真实 policy gradient，但工程上更稳定。

所以 PPO 是典型的：

```text
用 bias 换 stability。
```

### 6.4 SAC 的 bias 结构

SAC 更优雅的地方在于，它承认 exploration 不是额外技巧，而应该进入目标函数。

它优化的是：

```text
reward + entropy
```

这本身就是一个 biased objective。它不是原始 reward 最大化，而是最大化带熵正则的 reward。

但是这个 bias 是有意设计的。好处是 policy 不容易塌缩，Q 学习更平滑，exploration 更自然。

所以 SAC 的 bias 可以理解为：

```text
用 soft objective 换稳定探索和更好的数据覆盖。
```

---

## 7. 从压缩角度理解：Q 是价值压缩，Policy 是偏好压缩

### 7.1 Q function：对稀疏 reward / 长期价值的压缩

Q function 可以理解成一种压缩器。它把复杂的未来 rollout、稀疏 reward、延迟反馈，压缩成一个标量：

```text
Q(s,a) = 在状态 s 执行动作 a 后，未来长期回报的压缩表达。
```

Q 不是简单的"即时好坏判断"，而是对未来价值的压缩：

```text
稀疏 reward
→ 多步 rollout
→ 折扣累计回报
→ Q(s,a)
```

### 7.2 Policy：对偏好的压缩

Policy 也是一种压缩器，但它压缩的不是价值，而是偏好。

如果是随机 policy：

```text
π(a|s) = action preference distribution
```

如果是确定性 policy：

```text
a = μ(s) = 把偏好压缩成一个最倾向的 action
```

### 7.3 Q 和 Policy 的压缩对象不同

```text
Q 压缩的是：这个 action 未来值多少钱。     → value compression
Policy 压缩的是：我在这个状态下偏好怎么选。  → preference compression
```

Q 更接近一种评估器；Policy 更接近一种决策器。

### 7.4 为什么 Q-based 方法需要 argmax

如果只有 Q，那么它只是价值压缩，还不是最终行为。要从 Q 变成 action，通常需要再做一步：

```text
a = argmax_a Q(s,a)
```

也就是说，Q-based 方法是：先压缩价值，再通过 argmax 提取偏好。

但问题也在这里：argmax 是一个非常硬的偏好提取器。它会把 Q 里的微小误差放大成行为选择的巨大差异。

### 7.5 为什么 policy-based 方法直接优化偏好

Policy gradient / PPO / SAC 不一定先要求学出一个完整 Q landscape，再通过 argmax 选 action，而是直接优化 policy distribution。

```text
直接压缩偏好，而不是先压缩价值再提取偏好。
```

### 7.6 Actor-Critic：价值压缩和偏好压缩协同

Actor-Critic 结构可以理解成同时维护两个压缩器：

```text
Critic：压缩 value / advantage。  →  这个 action 好不好？好多少？
Actor：压缩 preference / behavior。  →  基于好坏信号，我以后应该更倾向于怎么选？
```

所以 Actor-Critic 不是简单地"一个网络选动作，一个网络打分"，而是：价值压缩器指导偏好压缩器。

### 7.7 这个视角下重新理解几个算法

| 算法 | 压缩对象 | 决策方式 | 核心风险 |
|---|---|---|---|
| Q-learning / DQN | 压缩长期价值 Q | argmax 提取偏好 | Q 噪声被 argmax 放大 |
| Double DQN / TD3 | 更稳地压缩价值 | 更保守地提取偏好 | 低估或学习变慢 |
| Policy Gradient | 直接压缩偏好 π | 从 policy 采样 | variance 大、样本利用率低 |
| PPO | 稳定压缩偏好 π | 限制偏好更新幅度 | 可能过于保守 |
| SAC | 压缩 soft preference | reward + entropy | 目标被 entropy bias 改写 |
| Actor-Critic | Q/A 压缩价值，Actor 压缩偏好 | critic 指导 actor | 两个压缩器相互污染 |

### 7.8 最精炼表达

```text
Q 是对稀疏 reward 和长期价值的压缩。
Policy 是对行为偏好和采样倾向的压缩。
```

Value-based 方法：先压缩价值，再从价值里提取偏好。

Policy-based 方法：直接压缩偏好，再通过采样和反馈不断修正偏好。

Actor-Critic：用价值压缩器指导偏好压缩器。

这也解释了为什么 RL 会从 hard argmax 逐渐走向 soft policy：

```text
因为 hard argmax 是从价值压缩到偏好压缩的脆弱转换。
而 soft policy 直接把偏好作为可学习对象。
```

### 7.9 Actor-Critic 的纠缠迭代：交替固定，交替优化

Actor-Critic 的训练过程有一个容易被忽视但极其重要的结构：**Actor 和 Critic 是纠缠在一起的，但每次只优化一个，把另一个当常数。**

这跟求解一个耦合方程非常像：

```text
类比：求解 f(x, y) = 0

x 和 y 是耦合的——x 的最优值取决于 y，y 的最优值取决于 x。
直接联立求解很难（或不可能）。

工程做法：坐标下降（coordinate descent）
  Step 1: 固定 y = y_current，优化 x → 得到 x_new
  Step 2: 固定 x = x_new，  优化 y → 得到 y_new
  Step 3: 回到 Step 1，循环

每一步只解一个变量，另一个当常数。
虽然每一步都不是全局最优，但交替迭代可以逼近。
```

Actor-Critic 完全是这个结构：

```text
耦合关系：
  Q 的最优值取决于 π（不同 policy 下的 Q 不同）
  π 的最优值取决于 Q（policy 要朝 Q 高的方向调整）

训练做法：交替固定，交替优化

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Step 1: 更新 Critic（固定 Actor）                           │
  │                                                             │
  │    Actor 的参数冻结，视为常数                                  │
  │    用 Actor 当前的 policy 采的数据，拟合 Q / V                 │
  │                                                             │
  │    loss_critic = (Q(s,a) - target)²                         │
  │    只对 Critic 参数求导，Actor 参数不参与梯度                   │
  │                                                             │
  │         ┌──────────┐                                        │
  │         │  Actor   │ ← 冻结（当常数）                        │
  │         └──────────┘                                        │
  │         ┌──────────┐                                        │
  │         │  Critic  │ ← 更新                                 │
  │         └──────────┘                                        │
  │                                                             │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  Step 2: 更新 Actor（固定 Critic）                           │
  │                                                             │
  │    Critic 的参数冻结，视为常数                                  │
  │    用 Critic 的 Q 值 / advantage 指导 Actor 调整              │
  │                                                             │
  │    loss_actor = -Q(s, π(s))  或  -log π(a|s) · A(s,a)      │
  │    只对 Actor 参数求导，Critic 参数不参与梯度                   │
  │                                                             │
  │         ┌──────────┐                                        │
  │         │  Actor   │ ← 更新                                 │
  │         └──────────┘                                        │
  │         ┌──────────┐                                        │
  │         │  Critic  │ ← 冻结（当常数）                        │
  │         └──────────┘                                        │
  │                                                             │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  Step 3: 回到 Step 1，循环                                   │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

用数学语言写得更精确：

```text
Critic 更新时：
  min_φ  L(φ) = (Q_φ(s,a) - y)²
  其中 y = r + γ·Q_target(s', π_θ(s'))
  θ（Actor 参数）不参与 ∂L/∂φ 的计算

Actor 更新时：
  min_θ  L(θ) = -Q_φ(s, π_θ(s))     （DDPG/SAC 风格）
  或
  min_θ  L(θ) = -log π_θ(a|s) · A    （PPO 风格）
  φ（Critic 参数）不参与 ∂L/∂θ 的计算
  （即使 DDPG 中梯度流过 Q 网络，Q 的参数也被 stop_gradient）
```

这个"纠缠迭代"结构解释了几个重要现象：

**1. 为什么 Actor-Critic 可能不稳定**

```text
坐标下降在凸问题上保证收敛，但 Actor-Critic 不是凸问题。

两个优化器在互相追赶：
  Critic 在追 Actor 变化带来的数据分布变化
  Actor 在追 Critic 变化带来的 value landscape 变化

如果一个跑太快，另一个跟不上，系统就会震荡甚至发散。
```

**2. 为什么 TD3 要 delayed policy update**

```text
TD3 让 Critic 更新频率 > Actor 更新频率（比如 Critic 更新 2 次，Actor 才更新 1 次）。

原因：让 Critic 先稳定下来（y 先收敛），再让 Actor 去追（x 再优化）。
如果两个同频更新，Critic 还没稳定 Actor 就开始追，追的方向可能是错的。
```

**3. 为什么 target network 有帮助**

```text
Target network 进一步减慢了 Critic target 的变化速度。
相当于在坐标下降中，不直接用 y_new，而是用 y_target = 0.995·y_old + 0.005·y_new。
让"常数"真的更像常数，减少交替迭代的震荡。
```

**4. 对比纯 value-based 方法（DQN）**

```text
DQN 没有这个纠缠——它只有一个 Q 网络，policy 是 argmax Q 的副产品。
不存在"两个网络交替优化"的问题。
这也是 DQN 在某些场景下更稳定的原因之一。
代价是：argmax 只能用于离散 action，无法处理连续控制。
```

最精炼的表达：

```text
Actor-Critic = 耦合方程的坐标下降

Q 和 π 互相依赖，但每一步只优化一个，把另一个当常数。
这是一种工程上的妥协：没有全局联立求解，只有交替逼近。
稳定性取决于两个优化器的步调是否协调。
```

---

## 8. 从 action 建模角度理解：离散 action、连续 action、确定性与随机性

### 8.1 离散 action：可以显式枚举，所以容易使用 argmax

离散 action 空间里，action 是有限集合：

```text
A = {a1, a2, a3, ..., ak}
```

这时候 Q-learning 很自然，因为可以为每个 action 估计一个 Q 值，决策时直接 argmax。

DQN 就是这个结构。神经网络输入 state，输出每个离散 action 的 Q 值：

```text
state → network → [Q(s,a1), Q(s,a2), ..., Q(s,ak)]
```

离散 action 的核心便利是 argmax 可以直接算。缺点是如果 action 数量很大或连续，argmax 就不再简单。

### 8.2 连续 action：不能枚举，所以需要 actor

连续 action 空间里，action 不是有限集合，而是连续变量：

```text
a ∈ R^n
```

这时候 argmax 变成一个连续优化问题，每次决策都要在连续 action 空间里找最大 Q 的 action，通常不可行。

所以连续控制里经常引入 actor：

```text
a = μθ(s)
```

actor 直接输出 action，相当于用一个网络近似 argmax_a Q(s,a)。

DDPG 就是这个思路：Critic 学 Q(s,a)，Actor 学 μ(s)，让 Q(s, μ(s)) 尽可能大。

```text
连续 action 下，actor 是 argmax 的函数近似。
```

### 8.3 离散 action 的 policy 建模

对于离散 action，policy 可以建模成 categorical distribution：

```text
state → network → logits → softmax → action probabilities
```

采样时从 categorical distribution 里抽一个 action。

### 8.4 连续 action 的 policy 建模

连续 action 通常建模成参数化分布，比如 Gaussian：

```text
π(a|s) = Normal(μθ(s), σθ(s))
```

网络输出均值和方差：

```text
state → network → mean, std → sample action
```

mean 表示偏好的中心，std 表示探索范围。SAC / PPO 在连续控制里常见的形式就是 Gaussian policy。

### 8.5 Deterministic policy vs Stochastic policy

**确定性 policy**：`a = μθ(s)`

* 优点：执行稳定，推理简单，适合连续控制。
* 缺点：探索能力弱，容易过早收敛，容易 exploit critic 的错误。
* 代表：DDPG / TD3。通常需要额外加 exploration noise。

**随机 policy**：`a ~ πθ(a|s)`

* 优点：天然包含探索，可以用 log probability 做 policy gradient，可以用 entropy 控制分布形状。
* 缺点：方差更大，训练更依赖采样。
* 代表：PPO / SAC。

### 8.6 从 deterministic 到 stochastic 的演化逻辑

早期控制问题常常喜欢 deterministic——给定状态，输出一个最优动作。

但是 RL 里，policy 同时承担两个角色：

```text
1. 决策器：当前该做什么。
2. 采样器：未来能看到什么数据。
```

如果 policy 是 deterministic，它作为决策器很清楚，但作为采样器很贫乏。它会让数据分布快速变窄。

所以 RL 逐渐走向 stochastic policy，是因为：

```text
RL 不只是要选当前最优动作，还要维持未来学习所需的数据覆盖。
```

从压缩角度看，stochastic policy 是更完整的 preference compression——偏好本来就不一定是一个点，而可以是一个分布：

```text
stochastic policy = soft preference representation
deterministic policy = hard preference representation
```

---

## 9. 从 DP 到 RL：逐步 relax 假设，逐步改变优化方式

### 9.1 DP 的上帝视角假设

Dynamic Programming 假设你知道完整 MDP：

```text
状态集合 S
动作集合 A
状态转移概率 P(s'|s,a)
奖励函数 R(s,a)
折扣因子 γ
```

DP 有一个近似"上帝视角"——知道环境怎么转移，知道每个动作会带来什么 reward，可以系统性地对所有状态做 Bellman backup。

### 9.2 第一次 relax：不知道完整环境，只能采样

现实中通常不知道 P(s'|s,a) 和 R(s,a)，只能通过和环境交互得到样本 (s, a, r, s')。

Bellman backup 从精确期望变成采样估计。这就是 TD learning 的来源。

### 9.3 第二次 relax：不能遍历所有状态，只能用函数近似

经典 DP 可以在表格状态空间里更新每个 state / action。现实状态空间巨大，甚至连续，不能维护完整表格。

于是引入函数近似：

```text
V(s) ≈ Vθ(s)
Q(s,a) ≈ Qθ(s,a)
π(a|s) ≈ πθ(a|s)
```

代价是泛化带来效率，也带来 function approximation bias。

### 9.4 第三次 relax：不能保证数据来自目标策略

DP 里没有"数据分布"问题，因为它直接操作模型和全状态空间。

RL 里必须采样，而采样由 behavior policy 决定。数据分布由 policy 生成，而 policy 又由数据训练——这个闭环是 RL 的本质困难之一。

### 9.5 第四次 relax：从 exact greedy 到 approximate greedy

DP 里如果知道完整 Q，可以直接 argmax。但 RL 里的 Q 是估计出来的，有噪声、有偏差、不完整。

exact greedy 变成 approximate greedy，引入 max bias、overestimation、exploration collapse。于是需要 ε-greedy、Double Q、target network、entropy regularization、soft policy 等方法。

本质：既然 Q 不是上帝视角的真实 Q，就不要完全相信 hard argmax。

### 9.6 第五次 relax：从最优控制转向偏好优化

DP / optimal control 的思路是：有环境模型 → 求最优 value → 导出最优 policy。

RL 后来的 policy gradient / PPO / SAC 更像是：没有完整模型 → 通过采样得到行为反馈 → 直接优化行为分布。

这就是从"求解最优控制问题"逐步转向"偏好分布优化"。

### 9.7 从 DP 到 RL 的演化链条

```text
DP:
已知模型 + 精确 Bellman backup + 全状态遍历 + exact greedy

↓ relax 环境模型

Tabular RL:
未知模型 + 采样 Bellman backup + 表格 Q/V

↓ relax 状态空间规模

Deep Q Learning:
函数近似 Q + replay + target network + approximate greedy

↓ relax action 空间

Actor-Critic:
actor 近似 argmax / policy，critic 估计 value

↓ relax hard greedy

Policy Gradient / PPO / SAC:
直接优化 stochastic policy，用偏好分布替代 hard argmax
```

### 9.8 每一步 relax 改变了什么 assumption

| 阶段 | 原始假设 | Relax 之后 | 新问题 |
|---|---|---|---|
| DP | 已知 P/R 模型 | 只能采样 | 采样噪声 |
| Tabular RL | 可遍历状态表 | 状态巨大/连续 | 函数近似 bias |
| Q-learning | 可稳定估计 Q | Q 有噪声 | max overestimation |
| DQN | 离散 action 可枚举 | 连续 action 不可枚举 | 需要 actor |
| DDPG/TD3 | deterministic actor 足够 | 探索不足 | 需要噪声/entropy |
| PPO/SAC | hard greedy 可行 | hard greedy 太脆弱 | soft policy / preference optimization |

---

## 10. 统一视角：Optimal Control 和 RL 本质上在做同一件事

Optimal Control 和 RL 经常被当成两个领域来教。但从三个核心维度——**rollout（想象未来）、optimization（做决策）、data（数据需求）**——来看，它们是同一个问题的不同工程权衡。

### 10.1 维度一：Rollout / Imagination —— 怎么"想象"未来

做决策的前提是预判未来。Optimal Control 和 RL 用不同方式做这件事：

**Optimal Control：基于模型的在线 rollout**

```text
有一个显式的动力学模型：
  s_{t+1} = f(s_t, a_t)     （确定性）
  s_{t+1} ~ P(·|s_t, a_t)   （随机性，但通常做简单假设）

Rollout 方式：
  用这个模型向前推演多步，预测不同 action 序列的后果
  → MPC (Model Predictive Control) 就是典型做法

环境的 uncertainty 处理：
  通常做简化假设（线性、高斯、小扰动）
  因为是实时在线的，假设可以比较粗糙——
  每一步都能从环境拿到新的状态反馈来修正
```

本质上，Optimal Control 也是 agent-环境交互的模型：

```text
离线准备：标定好动力学模型的有限参数（质量、摩擦系数、传动比等）
在线执行：用模型做 rollout + 用实时传感器数据修正

它假设了一个 dynamics 模型，外加实时的环境参数反馈。
```

**RL：基于 model-free 的离线 rollout**

```text
没有显式动力学模型。
Rollout 通过直接与环境（或仿真器）交互来完成。

输入要求更宽松：
  Optimal Control 需要精确的低维状态（位置、速度、角度）
  RL 可以接受更模糊的高维输入（图像、点云、原始传感器数据）
  → 因为模型本身学会了从高维输入提取有用信息

模型是离线学好的：
  通过大量的 environment interaction 训练 Q / V / Policy
  把"怎么从当前状态做好决策"压缩进了网络参数里
  → rollout 的知识被隐式编码在网络中，而不是显式动力学方程
```

对比：

```text
Optimal Control:
  显式模型 + 在线 rollout + 精确低维输入 + 实时修正
  → "我知道世界怎么运转，每一步都重新推演"

RL:
  隐式模型（编码在网络中）+ 离线训练 + 高维输入 + 部署时直接前向推理
  → "我训练时已经见过足够多场景，部署时直接反应"
```

### 10.2 维度二：Optimization —— 什么时候做优化

**Optimal Control：在线优化**

```text
每个时刻都在线求解一个优化问题：
  a* = argmin_a J(a, s_current)

MPC 在每个 step 都：
  1. 拿到当前状态
  2. 用模型向前 rollout 多步
  3. 在线求解最优 action 序列
  4. 执行第一个 action
  5. 下一步重复

也可以离线做查表（lookup table）：
  对输入的可能范围预计算最优解
  在线时直接查表
  → 适用于低维、可枚举的场景
```

**RL：离线优化**

```text
优化（训练）在离线阶段完成：
  通过大量 environment interaction 学习 Q / V / Policy

在线部署时不做优化，只做前向推理：
  a = π_θ(s)  或  a = argmax Q_θ(s,a)
  → 一次前向传播，直接输出 action
  → 计算量远小于在线优化
```

对比：

| 维度 | Optimal Control | RL |
|---|---|---|
| 优化发生在 | 在线（每一步都求解） | 离线（训练阶段） |
| 部署时计算量 | 大（每步求解优化问题） | 小（一次前向推理） |
| 适应新场景 | 强（实时重新优化） | 弱（需要重新训练或泛化） |
| 对模型的要求 | 需要显式 dynamics | 不需要（model-free） |

### 10.3 维度三：Data —— 数据从哪来，怎么用

**Optimal Control：依赖精确参数，不需要大量数据**

```text
需要的"数据"是环境模型的参数：
  质量、惯性、摩擦系数、传动特性...
  → 通常通过物理测量或系统辨识获得
  → 参数数量有限（几十到几百个）

在线数据 = 实时传感器反馈
  → 用于修正模型预测的偏差
  → 不用于训练，用于在线修正
```

**RL：依赖大量交互数据，压缩为网络参数**

```text
需要大量的 (s, a, r, s') 交互数据
  → 来自仿真器或真实环境的 rollout
  → 数据量可能是百万到数十亿条 transition

这些数据在训练过程中被"压缩"进了网络参数：
  离线信息 → 训练 → 网络参数 → 在线推理

RL 的训练过程本质上就是一种信息压缩：
  把"在各种状态下应该怎么做"的海量经验
  压缩成一个能快速推理的网络
```

这种压缩的意义在于：

```text
在线部署时，不需要重新与环境交互
网络已经把离线的交互经验编码进去了
一次前向传播就输出 action

→ RL 模型 = 离线经验的压缩表示，用于在线快速决策
```

### 10.4 Optimal Control 的隐含强约束

Optimal Control 看起来优雅，但它能工作是因为对问题做了很强的假设。这些假设在复杂场景下很容易不成立：

**约束 1：需要显式的 reference 和可解析的约束**

```text
MPC / LQR 等方法通常需要：
  - 一条显式的 reference trajectory（参考轨迹）
  - 约束条件表达为线性或凸约束：
      A·x ≤ b   （线性不等式约束）
      x_min ≤ x ≤ x_max   （状态/输入边界）
  - cost function 通常是二次型：
      J = Σ (x - x_ref)^T Q (x - x_ref) + u^T R u

如果约束是非线性的、非凸的、或者 reference 不好定义
→ 问题变得非常难解，甚至不可解
```

RL 不需要这些：没有显式 reference，没有约束形式要求，reward 可以是任意可计算的标量。

而且 reference 的需求会层层向上堆积：

```text
Controller 需要 reference trajectory
  → 谁提供 reference？→ Planner

Planner 需要自己的输入：
  → 地图、目标点、障碍物、交通规则、语义信息...
  → Planner 自己也可能需要 reference（如车道中心线、全局路径）

Planner 的输入又依赖 Perception：
  → 检测、跟踪、预测、语义分割...

于是形成一个级联依赖链：
  Perception → Planner → Controller
  每一层都有自己的显式输入要求
  每一层的误差都会向下游传播
  系统复杂度是各层复杂度的乘积，不是加法
```

这就是传统自动驾驶"模块化堆叠"架构的根本困境——每个模块都对上游有强依赖，任何一层出错都会级联放大。

RL（尤其是 end-to-end RL）的思路是跳过这个级联：

```text
传统：Perception → Planner → Controller（每层都有显式接口要求）
E2E RL：raw input ──→ neural network ──→ action（一个网络端到端学）
```

代价是需要更多数据和训练，但避免了层层堆积的显式接口问题。

**约束 2：自由度不能太高**

```text
Optimal Control 的在线求解复杂度随自由度急剧增长：
  - 状态维度 n，控制维度 m，预测步数 T
  - MPC 的优化变量数 ≈ (n + m) × T
  - 二次规划 QP 复杂度 ≈ O(((n+m)T)^3)

自由度高的场景（高维机器人、多 agent、像素级控制）：
  → 在线求解根本跑不动
  → 必须大幅简化模型或降维

RL 不受这个限制：
  离线训练时可以花任意多时间
  在线推理只是一次前向传播，跟维度关系是线性的
```

**约束 3：不好做并行计算**

```text
Optimal Control 的在线求解通常是串行的：
  - 每一步依赖上一步的解
  - MPC 的迭代求解器（QP solver、ADMM）本质是顺序迭代
  - 很难利用 GPU 的大规模并行能力

RL 的训练天然适合并行：
  - 数据采集：多个环境并行 rollout（vectorized env）
  - 梯度计算：batch 内的样本天然并行（GPU 矩阵运算）
  - 推理：batch inference，可以同时处理多个输入
```

总结：

```text
Optimal Control 的代价：
  ✗ 需要显式 reference + 线性/凸约束
  ✗ 自由度高时求解爆炸
  ✗ 不好并行，难以利用 GPU
  ✓ 但在低维、模型精确、约束规整的场景下非常高效

RL 的代价：
  ✗ 需要大量离线训练数据
  ✗ 训练不稳定，调参困难
  ✗ 不容易做实时在线适应
  ✓ 但对输入维度、约束形式、模型精度没有硬要求
```

### 10.5 统一视角：同一个问题的不同工程权衡

```text
┌──────────────────────────────────────────────────────────────┐
│                   核心问题都是一样的：                          │
│                                                              │
│        给定当前状态，选择最优（或足够好的）action               │
│                                                              │
├──────────────┬───────────────────┬───────────────────────────┤
│   维度       │ Optimal Control   │ RL                        │
├──────────────┼───────────────────┼───────────────────────────┤
│ Rollout      │ 显式模型在线推演   │ 隐式模型（编码在网络中）    │
│ 输入要求     │ 精确低维状态       │ 可接受高维模糊输入         │
│ Optimization │ 在线求解          │ 离线训练，在线推理          │
│ 数据需求     │ 少量精确参数       │ 大量交互数据              │
│ 部署计算量   │ 大（在线优化）     │ 小（前向推理）            │
│ 环境适应     │ 实时修正          │ 泛化能力                  │
│ 适用场景     │ 模型精确、低维     │ 模型未知、高维            │
└──────────────┴───────────────────┴───────────────────────────┘
```

所以 Optimal Control 和 RL 不是两个对立的东西，而是：

```text
同一个决策问题在不同约束下的工程选择：

知道 dynamics + 低维输入 + 需要实时适应 → Optimal Control
不知道 dynamics + 高维输入 + 可以离线训练 → RL

现实中两者经常融合：
  - Model-based RL（如 Dreamer、MBPO）：学一个 dynamics 模型，用它做 rollout
  - 自动驾驶的层级架构：高层用 RL 做规划，低层用 MPC 做控制
  - Sim2Real：在仿真中用 RL 离线训练，部署到真实环境用传感器在线修正
```

---

## 11. 总表：几条主线合在一起

| 维度 | 早期做法 | 问题 | 后续修正 | 更高级方向 |
|---|---|---|---|---|
| Loss | TD MSE | target 自举、有噪声 | target network / double Q | policy log-likelihood |
| Action selection | argmax Q | 放大 Q 噪声 | ε-greedy / double Q | soft policy |
| Exploration | random trick | 不系统 | entropy bonus | max entropy RL |
| Data reuse | on-policy 少量复用 | 样本浪费 | replay buffer | off-policy actor-critic |
| Stability | hard greedy | 容易崩 | PPO clip / TD3 tricks | bias-controlled optimization |
| Bias | 追求 Bellman target | overestimate | clipped / double / conservative | 接受 bias，换稳定性 |

---

## 12. 一句话总结典型算法

**Q-learning / DQN**：用 MSE 学 Q，用 argmax 做决策，用 replay 提高数据利用率。问题是 max Q 会放大估计误差。

**Double DQN / TD3**：承认 Q max 会高估，于是把选择和估值分开，或者用多个 critic 做保守估计。

**DDPG**：把 Q-learning 推到连续动作空间，用 actor 近似 argmax Q。问题是 actor 会 exploit critic 的错误。

**PPO**：直接优化 policy，但限制 policy 不要离采样数据太远。本质是 near on-policy 的稳定 policy gradient。

**SAC**：把 entropy 放进目标函数，用 soft policy 避免过早 greedy。本质是用最大熵目标统一 exploitation 和 exploration。

---

## 13. RL vs SL：看似相同的优化框架，本质不同的采样结构

### 12.1 相同点：最终都是在最小化一个 loss

从 loss 和优化器的角度看，RL 和 SL 其实非常接近。

SL 的典型结构是：

```text
loss = L(f_θ(x), y)
θ ← θ - α ∇L
```

RL 最终也是在优化一个 loss：

```text
Value-based:  loss = (Q_θ(s,a) - y)^2
Policy-based: loss = - log π_θ(a|s) · A(s,a)
```

优化器都是 SGD / Adam。梯度都是对参数 θ 求导。反向传播也一样。

所以从"写代码训练模型"的层面，RL 和 SL 的 optimization framework 基本相同：都是 forward → compute loss → backward → update。

### 12.2 不同点一：RL 的数据有时序依赖，标签来自环境反馈

SL 的数据通常是 i.i.d. 的：

```text
(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)
```

每条数据独立同分布，标签 y 是外部世界预先给定的。

RL 的数据有本质不同：

```text
s_0 → a_0 → r_0, s_1 → a_1 → r_1, s_2 → ...
```

它是一条时间序列。当前 state 依赖上一步的 action，reward 依赖当前 state-action 对，下一个 state 由环境转移概率决定。

也就是说：

```text
SL: 标签是预先给定的，不依赖模型。
RL: "标签"（reward / advantage / Q target）来自环境反馈，依赖 agent 的行为。
```

这意味着 RL 的 label 不是静态的，而是 agent 和环境交互的产物。你做不同的 action，看到不同的 reward 和 state。

### 12.3 不同点二（核心）：RL 的算法会改变自己的采样分布

这是 RL 和 SL 最根本的区别。

在 SL 里：

```text
数据集 D 是固定的。
训练过程中，优化器更新 θ，但 D 不变。
模型变了，数据不变。
```

在 RL 里：

```text
数据来自 policy 的采样。
训练过程中，优化器更新 θ → policy 变了 → 采样分布变了 → 未来数据变了。
模型变了，数据也变了。
```

可以画成这样：

```text
SL 的结构（开环）：

  固定数据 D → loss → 更新 θ → 更好的模型
                ↑                    |
                |____________________|
                （但 D 不变）


RL 的结构（闭环）：

  policy π_θ → 采样数据 → loss → 更新 θ → 新 policy π_θ'
      ↑                                        |
      |________________________________________|
      （policy 变了 → 采样分布也变了 → 数据分布变了）
```

这个闭环就是 RL 所有复杂性的根源。

因为 policy 既是"被优化的对象"，又是"数据的生成器"。优化器每走一步，不只是模型变了，连训练数据的分布都跟着变了。

这导致了 SL 里不存在的问题：

```text
1. 分布漂移：旧数据是旧 policy 生成的，用来训练新 policy 会有偏差。
2. 采样塌缩：如果 policy 过早确定化，未来只能采到很窄的数据。
3. 自举放大：Q target 自己造 label，label 反过来影响未来采样。
4. 探索-利用矛盾：想学好必须采到多样数据，但好的 policy 倾向于只做"好的"action。
```

### 12.4 一个类比

可以这样理解：

```text
SL 像是一个学生，课本已经印好了，每次复习都是同一本书。
   学生怎么学，不影响课本内容。

RL 像是一个学生，边学边写自己的课本。
   他学到什么影响他下次会写什么练习题，练习题又影响他接下来学到什么。
   如果他过早形成偏见，后面的课本就只有偏见的内容。
```

### 12.5 这个区别如何解释 RL 的各种技术

RL 里很多看起来"额外"的技术，都是在处理这个闭环问题：

| 技术 | 解决什么闭环问题 |
|---|---|
| Replay Buffer | 打破时序相关性，让旧数据可以重复使用，缓解分布漂移 |
| Target Network | 稳定 bootstrap target，避免"自己追自己" |
| ε-greedy / Entropy | 防止采样塌缩，保证数据覆盖 |
| Importance Sampling | 修正旧 policy 和新 policy 的分布差异 |
| PPO Clipping | 限制新旧 policy 差距，防止分布突变 |
| Off-policy 方法 | 允许使用非当前 policy 的数据，提高数据利用率 |
| On-policy 方法 | 避免分布不匹配，但代价是数据用完即弃 |

这些技术在 SL 里要么不存在，要么不必要——因为 SL 的数据分布不会被模型改变。

### 12.6 最压缩的表达

```text
SL：固定数据上优化模型。           算法不影响数据分布。
RL：动态数据上优化模型和数据分布。    算法本身就是数据分布的生成器。
```

所以 RL 的真正难度不在 loss 设计——loss 和 SL 差不多——而在于：

```text
你在优化一个目标的同时，这个目标的测量方式（采样分布）也在被你改变。
```

这就是为什么 RL 需要那么多看似 adhoc 的稳定技巧：不是因为优化本身更难，而是因为优化的同时数据地基在移动。

---

## 14. 最终理解

RL 不是简单优化一个干净 loss。RL 是在采样、估值、决策三者互相影响的闭环里做优化。

早期方法使用 MSE + Bellman + max Q。后来发现 max / argmax 太硬、Q 估计有噪声、off-policy replay 有分布偏差、探索不足会恶化估计。

于是引入 random exploration、target network、double Q、entropy、importance sampling、clipping、soft policy、actor-critic。

最后 RL 的思想从：

```text
找到最优 Q，然后 greedy。
```

逐渐变成：

```text
在带偏采样下，控制数据分布、估计偏差和优化步长，让 policy 稳定变好。
```

最精炼的版本是：

```text
RL 的本质不是 argmax。
RL 的本质是 biased sampling 下的 preference optimization。
```

或者换一种更统计派的说法：

```text
没有绝对上帝视角。
你对环境的理解，取决于你采样到的数据、你的采样偏好，以及你愿意接受哪种 bias。
```
