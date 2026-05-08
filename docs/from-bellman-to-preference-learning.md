# 从 Bellman 到 Preference Learning：对强化学习本质的一条重新理解

这篇文档整理的是一条更接近"研究视角"的强化学习理解路径。它不是按教材顺序罗列算法，而是试图回答一个更根本的问题：**为什么强化学习从经典动态规划出发，后来会走向 Q-learning、DQN、Policy Gradient、Actor-Critic、PPO、SAC、RLHF / GRPO 这些看起来越来越"经验化""统计化""偏好化"的方法？**

核心观点可以先压成一句话：

**经典 DP / Bellman 理论是在已知模型下求全局最优；现实 RL 是在有限采样、噪声、偏差和分布不确定下，学习一个足够好、足够稳定、符合偏好的策略。**

这两者的目标性不同。前者更像上帝视角的解析优化，后者更像统计视角下的偏好学习。

---

## 1. Bellman 原理本身没错，但它描述的是理想极限

经典动态规划的世界里，环境模型是已知的。也就是说，我们知道：

$$
P(s'|s,a)
$$

也知道 reward：

$$
r(s,a)
$$

在这个世界里，Bellman equation 是非常自然的。对于某个 policy $\pi$，有：

$$
V^{\pi}(s)=\mathbb{E}_{a\sim\pi, s'\sim P}\left[r(s,a)+\gamma V^{\pi}(s')\right]
$$

对于最优控制，则有：

$$
V^{*}(s)=\max_a \mathbb{E}_{s'\sim P}\left[r(s,a)+\gamma V^{*}(s')\right]
$$

或者写成 Q 的形式：

$$
Q^{*}(s,a)=\mathbb{E}_{s'\sim P}\left[r(s,a)+\gamma \max_{a'}Q^{*}(s',a')\right]
$$

这些数学上都是严格成立的。问题不在于 Bellman 原理错了，而在于：**这个公式描述的是一个理想 fixed point，而训练过程不是 fixed point。**

最终静态解上，$\max$ 是一个正确结论；但在学习过程中，你手里的不是 $Q^{*}$，而是一个带噪声、带偏差、不断变化的估计：

$$
\hat Q = Q^{*} + \epsilon
$$

于是训练过程实际变成：

$$
Q_{t+1}=r+\gamma \max(\hat Q_t)
$$

这已经不是一个干净的解析优化问题，而是一个**带噪声的非线性递归迭代系统**。

---

## 2. V 和 Q 的关系：Q 只是多记了一层 action 条件，但意义巨大

从定义看，V 和 Q 都是在描述未来累计回报。

$$
V^{\pi}(s)=\mathbb{E}_\pi[G_t|s_t=s]
$$

$$
Q^{\pi}(s,a)=\mathbb{E}_\pi[G_t|s_t=s,a_t=a]
$$

所以 Q 可以理解为 V 多记了一个 action 维度。更准确地说：**Q 是把 action condition 显式展开了。**

V 只记录：

$$
s \rightarrow value
$$

Q 记录：

$$
(s,a) \rightarrow value
$$

这看似只是"多记了一维账"，但对 control 问题意义巨大。因为控制问题的核心不是只判断 state 好不好，而是要判断：**在这个 state 下做哪个 action 更好。**

对于 V 来说，action 被藏在 Bellman operator 里面：

$$
V^{*}(s)=\max_a \left(r(s,a)+\gamma\mathbb{E}[V^{*}(s')]\right)
$$

这里 reward 和 transition 都依赖 action，所以如果没有环境模型，就不能直接从 V 推出哪个 action 最好。

而 Q 把 action 固定之后：

$$
Q(s,a)=r(s,a)+\gamma\mathbb{E}[V(s')]
$$

然后 policy improvement 变得非常直接：

$$
a^{*}=\arg\max_a Q(s,a)
$$

这就是 Q-function 的巨大价值：**它把 action optimization 显式化了。**

但代价也很明显。Q 比 V 更细，估计方差更大，更容易出现局部 noisy peak。再叠加 $\max$，就容易把估计误差放大。

---

## 3. 为什么早期人们先用 V：不是不知道 Qmax，而是 V 更稳

现在回头看，似乎 Qmax 更适合控制：既然最后要选动作，为什么不一开始就学 $Q(s,a)$ 并做 $\arg\max$？

原因是早期动态规划和控制理论的世界更偏向于：

* 模型已知；
* expectation 可精确计算；
* 目标是求一个全局最优 fixed point；
* 稳定性和可证明性非常重要。

在这个世界里，V-function 很自然，因为它是对 action 之后结果的平均化表达。对于某个 policy，V 的 Bellman evaluation 是：

$$
V^{\pi}(s)=\sum_a \pi(a|s)\sum_{s'}P(s'|s,a)\left[r(s,a)+\gamma V^{\pi}(s')\right]
$$

这个 operator 本质上是 averaging。Averaging 的好处是平滑、稳定、容易证明 contraction。

而 $\max$ 是 winner-take-all。它不平滑，对噪声敏感，会偏向选择被高估的 action。

所以历史上不是"人们没想到 Qmax"，而是：**Qmax 太 aggressive，需要足够多理论、数据、计算和工程手段之后才比较可控。**

---

## 4. Qmax 是一匹脱缰野马：数据效率高，但会放大噪声

Q-learning 的核心是：

$$
Q(s,a)\leftarrow r+\gamma\max_{a'}Q(s',a')
$$

这个式子非常有力量。它的意义是：即使真实 rollout 没有走到最优 action，更新时仍然假设未来会选择最优 action。

这就是 Q-learning 数据效率高的原因：**一个高 value 信号可以被快速向前传播。**

但是问题也在这里。训练时 Q 是估计值，不是真值：

$$
\hat Q=Q+\epsilon
$$

于是：

$$
\max(\hat Q)=\max(Q+\epsilon)
$$

$\max$ 会 preferentially 选择正噪声大的那个 action。也就是说，它会把偶然高估、lucky sample、network hallucination 当成真实偏好。

这会形成一个反馈循环：

1. 某个 action 因为噪声被高估；
2. $\max$ 选中它；
3. 它进入 Bellman target；
4. target 被写回 Q-network；
5. 未来更新继续传播这个高估；
6. 系统可能越来越偏，甚至发散。

这就是所谓 overestimation 的本质。不是 Bellman fixed point 错，而是 **noisy max iteration** 会出问题。

可以把它理解成：

**Bellman optimality 在静态 fixed point 上是正确的；但学习过程是多轮带噪声的强非线性 max 迭代，系统很容易自激发不稳定。**

这和普通 ODE 或普通 gradient descent 很不一样。普通 averaging system 会把噪声平均掉，而 max-bootstrap system 会主动挑噪声最大的方向并递归传播。

---

## 5. 平均太保守，max 太猛：RL 的核心张力

V / policy evaluation 类方法更像 averaging：

$$
V(s)=\sum_a\pi(a|s)Q(s,a)
$$

它稳定，因为不同 action 的噪声会相互抵消。但它也保守，因为它不会主动放大最好的 action。

Qmax 则相反：

$$
\max_a Q(s,a)
$$

它数据效率高、传播快、能快速利用 sparse reward，但也会放大误差。

所以 RL 的核心张力可以写成：

| 方法倾向 | 优点 | 缺点 |
| --- | --- | --- |
| averaging / expectation | 稳定、平滑、低方差 | 学得慢，优化弱 |
| max / greedy | 数据效率高，偏好传播快 | overestimate，不稳定，容易崩 |

这也是为什么现代 RL 不是简单选择一边，而是不断寻找中间状态：

**既要保留 greedy 的优化能力，又不能让 greedy operator 把系统搞飞。**

---

## 6. DQN 的意义：不是发明 Q-learning，而是重新驯化 Qmax

DQN 本质上是 deep Q-learning。它没有发明 Q-learning，也没有改变核心 Bellman target：

$$
r+\gamma\max_{a'}Q(s',a')
$$

DQN 的真正意义是：**第一次比较稳定地把 deep neural network 和 off-policy Q-learning 结合起来。**

它靠几个关键工程和算法设计来驯化 Qmax：

1. **CNN encoder**：把高维图像状态映射到 feature；
2. **replay buffer**：打破样本相关性，提高数据复用；
3. **target network**：避免 target network 自己追自己，降低 bootstrap instability；
4. **off-policy learning**：允许使用过去数据训练当前 value function。

但 DQN 仍然没有根治 $\max$ bias。因为同一个 Q-network 既负责选 action，又负责给 action 估值，容易出现"自己选冠军，自己给冠军打分"的高估问题。

于是后面出现了 Double DQN：

* online network 负责 action selection；
* target network 负责 action evaluation。

本质是减少 max operator 对噪声的放大。

再后来还有 Dueling DQN，把：

$$
Q(s,a)=V(s)+A(s,a)
$$

拆开。这里不是放弃 Q，而是希望用 V 表示 state 的整体价值，用 advantage 表示 action 相对差异，从而让估计更稳定。

所以 DQN 之后的一大条线可以理解为：

**保留 Qmax 的 aggressive optimization，但给它加缰绳。**

---

## 7. RL 和 DP 的本质区别：不是公式不同，而是世界不同

DP 的世界里，expectation 是可以精确计算的：

$$
\mathbb{E}[\cdot]
$$

是真 expectation。

RL 的世界里，expectation 只能靠采样估计：

$$
\hat{\mathbb{E}}[\cdot]
$$

这就带来了根本变化。

DP 是：

**已知模型下的确定性 fixed-point 求解。**

RL 是：

**未知模型下的带噪声、带偏差、带分布不确定性的递归优化。**

这不是小差别，而是目标性质的改变。

在 DP 中，Bellman optimality 是全局最优解的数学表达。在 RL 中，Bellman optimality 更像一个 guiding principle。因为现实里：

* 数据有限；
* rollout 昂贵；
* exploration 不充分；
* reward sparse；
* environment stochastic；
* function approximation 有误差；
* replay buffer 分布和当前 policy 分布不一致。

于是，RL 的首要目标往往不是"求绝对最优"，而是：

**在有限采样下，尽快形成一个稳定、有用、统计意义上较优的行为偏好。**

这就是本质：

**RL 是带噪音和数据分布不确定下的 bias optimization。**

更完整地说：

**RL 是在噪声、偏差和分布漂移下进行递归偏好优化。**

---

## 8. 从上帝视角到统计视角

经典控制 / DP 更像上帝视角。它默认世界有真实模型，有真实转移概率，有真实 reward function，也有真实最优 $Q^{*}$、$V^{*}$。算法的任务是把这个真值算出来。

统计学视角完全不同。统计学里没有直接可见的真相。你只有：

* samples；
* assumptions；
* likelihood；
* prior；
* estimator。

最大似然估计不是在求宇宙真理，而是在问：

**什么参数最能解释当前观测数据？**

RL 越往后发展，越接近这个统计世界观。尤其是 policy gradient、RLHF、GRPO 这类方法，它们并不强求一个全局一致、Bellman-consistent 的 $Q^{*}$。它们更关心：

**哪些行为在采样统计中带来了更好的 reward / preference，于是这些行为未来应该更容易发生。**

这就是从"求真值"转向"塑造行为分布"。

---

## 9. Policy Gradient：RL 从 value-centric 转向 behavior-centric 的分水岭

Q-learning / DQN 的核心是 value-centric：

$$
Q^{*}(s,a)
$$

它相信存在一个客观最优 action-value function，然后通过 Bellman consistency 去逼近它。

Policy Gradient 则完全不同。它直接把 policy 作为优化对象：

$$
\pi_\theta(a|s)
$$

目标是：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

这里的关键变化是：

**action 不再是通过 $\arg\max Q$ 选出来的，而是从 policy distribution 中采样出来的。**

policy update 形式是：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log\pi_\theta(a|s) A(s,a)\right]
$$

它的含义很直观：

* 如果某个 action 的 advantage 是正的，就提高它未来出现的概率；
* 如果 advantage 是负的，就降低它未来出现的概率。

这不是 hard max，而是 probability reshaping。

Q-learning 像是在说：

> 当前估值最高的 action 就应该被选中。

Policy Gradient 像是在说：

> 统计上表现更好的行为，未来应该更容易发生。

这就是一个哲学转向：

**从上帝视角的最优值求解，转向基于采样统计的行为偏好塑造。**

---

## 10. Policy Gradient 的"两层概率"

Policy Gradient 的统计本质可以用"两层概率"来理解。

第一层是 rollout occurrence probability：

$$
a\sim\pi_\theta(a|s)
$$

也就是说，哪些行为被采样到，取决于当前 policy 的概率分布。

第二层是 update weighting：

$$
\nabla\log\pi(a|s)\cdot A(s,a)
$$

采样到了某个 action，并不代表它就应该被强化。它还要被 advantage 加权。

所以 Policy Gradient 不是简单地复制采样行为，而是：

**在真实采样分布中，用 reward / advantage 对行为进行偏好重加权。**

这就非常接近统计学中的估计思想：你没有上帝视角的真值，只能通过样本和假设来调整参数。

---

## 11. Actor-Critic：policy 和 value 的分工

Policy Gradient 直接优化 policy，但它仍然需要估计 advantage。于是就出现 Actor-Critic。

Actor 是：

$$
\pi_\theta(a|s)
$$

负责决策。

Critic 可以有不同形式：

| Critic 类型 | 学什么 | 典型算法 |
| --- | --- | --- |
| V-critic | $V(s)$ | A2C, A3C, PPO |
| Q-critic | $Q(s,a)$ | DDPG, TD3, SAC |
| Advantage critic | $A(s,a)$ | 一些直接 advantage 方法 |

PPO / A2C 这类通常用 V-critic。它们估计：

$$
A_t \approx r_t+\gamma V(s_{t+1})-V(s_t)
$$

这里 critic 更像是一个 statistical estimator 或 control variate，用来降低 policy gradient 的方差。它不是要严格求一个全局最优 $Q^{*}$。

DDPG / TD3 / SAC 则保留 Q-critic：

$$
Q(s,a)
$$

actor 通过 critic 来优化 action。尤其连续控制中，actor 可以通过：

$$
\nabla_a Q(s,a)
$$

来调整 action direction。

所以 Actor-Critic 不是"不要 Q 了"，而是把：

* policy optimization；
* value estimation；

拆开。Critic 可以是 V，也可以是 Q。

---

## 12. V-only actor-critic 是对 Qmax 的 relaxation

PPO 这类方法通常不显式维护 Q-network，而是用 V-network 估计 baseline，再用 rollout return 或 GAE 估计 advantage。

这可以理解为对硬 $Qmax$ 的 relaxation。

Q-learning 是：

$$
\arg\max_a Q(s,a)
$$

PPO 是：

$$
\nabla_\theta\log\pi_\theta(a|s)A(s,a)
$$

前者是硬优化，后者是软偏好调整。

当然，代价也很明显：

* 更依赖采样；
* 数据效率通常低；
* 如果好 action 很少被采样到，policy gradient 很难发现；
* advantage estimate 有方差和偏差。

所以 PPO 更稳，但不是更"全局最优"。它追求的是：

**有限采样下稳定地把 policy 往更好的方向推。**

---

## 13. 为什么说这不是"退化"，而是目标函数变了

从 Qmax 到 policy gradient，不只是算法形式改变，而是优化目标改变。

Q-learning 的目标像是：

**逼近一个上帝视角的最优 Q-value。**

Policy Gradient 的目标像是：

**让采样中 reward / preference 更好的行为未来更容易发生。**

这两种 loss 的精髓不一样。

Q-learning 的 Bellman loss 是 consistency loss：

$$
\left(Q(s,a)-\left(r+\gamma\max_{a'}Q(s',a')\right)\right)^2
$$

它追求的是 value function 自洽。

Policy Gradient 的 loss 更像 behavior distribution shaping：

$$
-\log\pi(a|s)A(s,a)
$$

它追求的是提高好行为的概率、降低坏行为的概率。

所以一个是 value truth fitting，一个是 preference-weighted behavior update。

---

## 14. Argmax 作为结论 vs argmax 作为训练过程

这是一个非常关键的区分。

在 Bellman optimality 里，$\arg\max$ 是最终结论：

> 如果你已经知道真实 $Q^{*}$，那么选 $\arg\max Q^{*}$ 是对的。

但在训练过程中，$\arg\max$ 变成了每一步的优化动作：

> 即使你现在的 Q 还带噪声，也强行选当前最大值。

这两者完全不同。

可以用一个类比：

* Bellman optimality 的 $\arg\max$：像最终健康状态下，知道每天跑步对身体好；
* 训练过程中的 hard greedy：像身体状态很差时还每天极限跑步，结果反而 overshoot、受伤、崩溃。

所以现代 RL 的很多方法，本质上就是让优化目标"软一点"：

* PPO 用 clipping / trust region，避免 policy 一步变太多；
* SAC 用 entropy，让 policy 不要过早 collapse；
* Double Q / TD3 用保守估计压制 overestimation；
* RLHF 用 KL regularization 保持接近 reference policy。

这不是否定最优性，而是承认：

**有限数据下，不能把理论最优条件当成每一步训练的硬执行规则。**

---

## 15. 现代 RL：带刹车的 greedy optimization

现代 RL 的很多方法都可以理解为：

**greedy 是必须的，但必须带刹车。**

为什么 greedy 必须存在？因为数据稀缺。如果没有 greedy amplification，高 reward 信号传播太慢，模型可能根本学不到东西。

为什么必须带刹车？因为 greedy 会放大噪声、偏差和 function approximation error。

于是现代算法做的是各种折中：

| 算法 / 技术 | 本质作用 |
| --- | --- |
| Target Network | 减少 bootstrap target drift |
| Replay Buffer | 提高数据效率，降低样本相关性 |
| Double Q | 降低 max overestimation |
| Dueling Network | 稳定 Q 的结构分解 |
| PPO Clip | 限制 policy update 过猛 |
| TRPO / KL constraint | trust region，防止策略突变 |
| Entropy Regularization | 防止过早 deterministic collapse |
| TD3 clipped double Q | 用保守估计压制 overestimation |
| SAC | soft greedy + entropy + Q critic |
| CQL | offline 下主动压低 OOD action 的 Q |

这些方法的共同点是：

**不放弃优化，但限制 optimizer exploit estimation noise。**

---

## 16. RLHF / GRPO：adhoc preference 不是 bug，而是人类偏好的真实形态

到了 LLM RLHF / GRPO 这类方法，事情变得更明显。

人类 preference 本来就不是一个干净的、全局一致的、Bellman-consistent reward function。

人类偏好往往是：

* noisy；
* contextual；
* local；
* heuristic；
* sometimes contradictory；
* dependent on prompt, culture, task, evaluator state。

所以在这种场景下，强行追求一个全局最优 $Q^{*}$ 反而可能是假的。

GRPO / RLHF 这类方法更像：

**从采样回答中，根据人类偏好或 reward model 进行相对偏好重加权。**

它不是在求一个绝对最优控制律，而是在做：

**preference distribution shaping。**

这也是为什么很多细节看起来 adhoc：

* reference model；
* KL penalty；
* reward normalization；
* group relative advantage；
* clipping；
* rejection sampling；
* pairwise ranking。

但 adhoc 不一定是缺陷，因为人类偏好本身就是 adhoc 的。它不是解析数学对象，而是统计采样对象。

所以，GRPO / RLHF 的"adhoc"反而符合 preference sampling 的本质。

---

## 17. 从 DP 到 RLHF：一条大历史

可以把整个演化压成下面这条线：

### 阶段 1：DP / V-function

目标是已知模型下的稳定 evaluation 和全局最优求解。

核心特点：

* expectation 精确；
* V-function 稳定；
* Bellman fixed point 干净；
* 但 action optimization 不够直接。

### 阶段 2：Q-learning / Qmax

开始显式 action optimization。

核心特点：

* 数据效率高；
* off-policy；
* greedy propagation 快；
* 但 overestimation 和不稳定严重。

### 阶段 3：DQN 系列

用 deep network 扩展 Q-learning，并通过 replay / target network / Double Q 驯化 Qmax。

核心特点：

* scalable；
* 仍然 value-centric；
* 仍然围绕 Bellman consistency；
* 但开始大量工程稳定技巧。

### 阶段 4：Policy Gradient / Actor-Critic

直接优化 policy distribution。

核心特点：

* 从 value-centric 转向 behavior-centric；
* 不再强求完美 Q；
* 通过 advantage 调整行为概率；
* 更统计化、更软、更稳定，但数据效率较低。

### 阶段 5：PPO / SAC / TD3

融合 value 和 policy，用各种约束和 regularization 平衡 greedy 与稳定性。

核心特点：

* PPO：稳定 policy update；
* TD3：保守 Q critic；
* SAC：soft Q + stochastic policy + entropy；
* 都是在解决 greedy 过猛和采样噪声之间的矛盾。

### 阶段 6：RLHF / GRPO

强化学习进一步变成偏好分布塑造。

核心特点：

* reward 来自 human preference 或 reward model；
* 偏好本身不一定全局一致；
* 算法更像统计偏好学习；
* 目标不是绝对最优，而是让行为分布落在人类偏好区域内。

---

## 18. 最终总结：RL 的精髓是什么

如果用一句话概括：

**RL 的精髓不是 Bellman equation 本身，而是在未知环境、有限采样、噪声偏差和分布不确定下，如何稳定地放大有用偏好，并避免 optimizer exploit 自己的估计误差。**

更具体地说：

1. DP 是已知模型下的精确优化；RL 是未知模型下的采样优化。
2. Bellman optimality 是理想 fixed point；训练过程是 noisy nonlinear recursive iteration。
3. V 稳定但保守；Q 显式 action optimization，但更容易放大误差。
4. Qmax 数据效率高，因为它快速传播高 value；但它也会 preferentially 选择正噪声。
5. 现代 RL 的核心不是简单追求全局最优，而是在有限数据下学到稳定、有用、统计上较优的策略。
6. Policy Gradient 是重要分水岭：从求最优 value function，转向直接塑造 behavior distribution。
7. Actor-Critic 是 value estimation 和 policy optimization 的融合，不是简单"要 Q"或"不要 Q"。
8. PPO / SAC / TD3 等方法本质上都是带刹车的 greedy optimization。
9. RLHF / GRPO 进一步说明：很多现实偏好本来就是 adhoc、统计化、上下文相关的，不适合强行套一个全局 Bellman-consistent 真值函数。
10. 所以现代 RL 更像统计偏好学习，而不是传统意义上的解析最优控制。

最终可以把这条线压成一个非常强的表述：

**DP 追求的是全局最优解；RL 追求的是在有限、有噪声、有偏差的数据中，演化出足够好的行为偏好。**

而这也解释了为什么 RL 的历史一直在摇摆：

* 太 average，学不动；
* 太 greedy，会崩；
* 最后的答案不是二选一，而是：

**用 greedy 放大有效信号，用统计约束和正则化防止系统被噪声带飞。**
