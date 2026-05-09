# Understanding Reinforcement Learning from Multiple Dimensions: From Bellman, Argmax to Preference Optimization

> **Positioning note**: This essay provides a **micro-mechanism analysis** of RL -- dissecting what each technique does and why, across multiple dimensions including loss design, data utilization, exploration strategy, bias management, compressed representation, and action modeling. It complements "From Bellman to Preference Learning": that essay traces the macro-evolution (why RL evolved from DP toward preference learning), while this one offers a micro-engineering perspective (the specific principles and design motivations behind each mechanism).

---

## 0. Overview: RL Is Not a Clean Supervised Learning Problem

Supervised learning typically has a relatively clear structure: a fixed data distribution, explicit labels, and an explicit loss. Reinforcement learning is not like that.

In RL, the policy determines sampling, sampling determines the data distribution, the data distribution determines the estimation of Q / advantage, and Q / advantage in turn updates the policy. Once the policy changes, the future sampling distribution also changes.

Therefore, the core of RL is not "optimizing a clean loss given fixed data," but rather continuously handling sampling bias, estimation bias, and optimization stability within a dynamic closed loop.

RL can be understood as:

```text
On a constantly changing data distribution, using biased sampled data, estimate long-term returns, and optimize the next round's sampling strategy.
```

Consequently, TD, Bellman, Q max, argmax, entropy, importance sampling, clipping, replay buffer, target network, double Q, actor-critic, PPO, SAC -- these are not isolated tricks. They all address the same problem:

```text
How to stably optimize a biased objective under biased sampling.
```

---

## 1. Understanding from the Loss Organization Perspective: MSE -> Max Q -> Entropy -> Weighted Log-Likelihood

### 1.1 Value-based Methods: Organizing RL as MSE Regression

DQN / Q-learning methods essentially turn RL into a regression problem. The goal is to make the current Q approximate the Bellman target:

```text
Q(s, a) ≈ r + γ max_a' Q_target(s', a')
```

So the loss can be written as:

```text
L = (Q(s,a) - y)^2

y = r + γ max_a' Q_target(s', a')
```

This looks like supervised learning, but the key difference is: the label `y` here is not directly given by the external world, but computed by the model itself through Bellman bootstrap and max Q.

In other words:

```text
The model creates its own labels, then fits those labels.
```

This is the core characteristic of TD learning. It is not ordinary MSE, but self-bootstrapping regression.

This structure introduces two problems:

1. The target itself is noisy.
2. max / argmax amplifies noise.

Therefore, the MSE loss in value-based RL is not equivalent to MSE in ordinary supervised learning. Its labels are dynamic, model-dependent, and biased.

### 1.2 Why Q Max Tends to Overestimate

The key issue lies in:

```text
max_a Q(s, a)
```

If the Q estimate for each action has noise:

```text
Q_hat(a) = Q_true(a) + noise(a)
```

Then when taking the max, actions whose values are overestimated by noise are more likely to be selected. Therefore, in general:

```text
E[max Q_hat] > max E[Q_hat]
```

This is the source of Q overestimation.

It is not a problem unique to DQN. As long as an algorithm relies on max Q, argmax Q, or advantage ranking, it faces a similar issue. The difference is merely that different algorithms mitigate it in different ways.

### 1.3 Double DQN / TD3: Stabilizing the MSE Target

The core idea of Double DQN is to separate action selection from value evaluation.

Original DQN:

```text
y = r + γ max_a Q_target(s', a)
```

Double DQN:

```text
a* = argmax_a Q_online(s', a)

y = r + γ Q_target(s', a*)
```

It does not completely eliminate bias, but reduces the overestimation caused by "the same noisy Q being responsible for both selection and evaluation."

TD3 does something similar in continuous action spaces. It uses two critics and takes the smaller value as the target:

```text
y = r + γ min(Q1, Q2)
```

This is a conservative estimate, designed to counteract the actor's excessive exploitation of critic errors.

### 1.4 Max Q Loss: The Actor's Loss Is Directly Maximizing Q

In value-based methods, the loss is MSE -- making Q approximate the Bellman target. But when the action space is continuous, argmax can't be computed directly, so an actor network is introduced.

The actor's loss is no longer MSE, but an entirely new form -- **directly maximizing Q**:

```text
L_actor = - Q(s, μ_θ(s))
```

The actor adjusts parameters θ to make the critic give it the highest possible score.

This is the core idea of DDPG. It can be understood as the continuous version of argmax:

```text
Discrete action:    a* = argmax_a Q(s,a)          → enumerate all actions, pick the max
Continuous action:  θ* = argmax_θ Q(s, μ_θ(s))    → use gradient ascent so the actor outputs high-Q actions
```

So the essence of max Q loss is:

```text
Turning argmax (a discrete operation) into a differentiable gradient optimization problem.
```

The actor uses the critic's gradient signal ∇_a Q(s,a) to adjust its output direction.

But max Q loss has a fundamental risk: **the actor will exploit the critic's estimation errors.**

The critic's Q is not the true value -- it has noise and overestimation. If the actor keeps moving toward where Q is highest, it may end up not at truly good actions, but at actions where the critic's estimation error is largest.

This is why TD3 needs so many protective measures:

```text
double critic + min Q:      suppress overestimation
delayed policy update:      let critic stabilize first, then update actor
target policy smoothing:    prevent actor from exploiting sharp peaks in Q
```

From the loss evolution perspective, max Q loss sits between MSE and policy gradient:

```text
MSE loss:      Make Q approximate a target         → passive fitting
Max Q loss:    Make actor output high-Q actions     → actively exploiting the Q landscape
PG loss:       Make good actions more probable      → directly shaping the distribution
```

MSE is "I estimate value"; max Q is "I exploit my estimates for decisions"; PG is "I directly optimize decisions."

### 1.5 Policy Gradient: From Max Q to Weighted Log-Likelihood

Policy gradient no longer first learns a Q and then derives actions via argmax, but instead directly optimizes the policy.

The objective can be written as:

```text
J(θ) = E[return]
```

Through the policy gradient theorem, the gradient can be written as:

```text
∇J(θ) = E[∇ log πθ(a|s) · A(s,a)]
```

Therefore the loss is often organized as:

```text
L = - log πθ(a|s) · A(s,a)
```

This closely resembles weighted maximum likelihood:

```text
Good actions: increase probability.
Bad actions: decrease probability.
```

So from the loss organization perspective, policy gradient shifts from MSE regression of Q to weighted MLE of action probability.

This is an important philosophical change.

The Q-learning approach is:

```text
I first estimate a Q landscape, then select actions via argmax.
```

The policy gradient approach is:

```text
I directly adjust the sampling distribution to make good actions more likely to be sampled.
```

This is the shift from value-first to policy-first.

#### Why Value-Based Methods Talk About "Loss" While Policy Gradient Talks About "Gradient"

A common confusion when learning RL: DQN-style methods directly write a loss function, while policy gradient derivations start with the gradient and only mention loss at the end. Why?

The reason is that the two have different derivation starting points.

**Value-based methods** start from a natural regression problem:

```text
I have a target y (Bellman target). I want Q(s,a) to approximate it.
So loss = (Q - y)^2. The gradient is derived from the loss.
```

This is completely isomorphic to supervised learning: loss comes first, then gradient.

**Policy gradient** doesn't start from a loss, but from the gradient of an objective function:

```text
I want to maximize J(θ) = E[return].
Through the policy gradient theorem:
∇J(θ) = E[∇ log πθ(a|s) · A(s,a)]
```

This gradient comes from mathematical derivation of J(θ), not from writing a loss first and differentiating. The derivation uses differentiation of trajectory probabilities (the likelihood ratio trick) to directly obtain the gradient form.

So what is the loss? The loss is actually **constructed after the fact**. To use PyTorch / TensorFlow's automatic differentiation, we need to construct a loss such that autograd's derivative of it exactly equals the policy gradient above.

This loss is:

```text
L = - log πθ(a|s) · A(s,a)
```

Verification -- differentiating with respect to θ:

```text
∇L = - ∇ log πθ(a|s) · A(s,a)
```

With the negative sign (because the optimizer minimizes, but we want to maximize J), this recovers the policy gradient form.

So the complete logic is:

```text
Value-based: loss is the starting point → gradient is derived from the loss.
Policy gradient: gradient is the starting point → loss is constructed after the fact to work with autograd.
```

But ultimately in code, both have exactly the same training loop:

```text
loss = compute_loss(batch)
loss.backward()
optimizer.step()
```

This is why RL and SL share essentially the same optimization framework -- regardless of whether the loss came first or was constructed afterward, everything ultimately falls into the same loss → backward → update pipeline.

### 1.6 Policy Gradient and Maximum Likelihood: From Unweighted MLE to Weighted MLE

Maximum likelihood estimation is essentially: using statistical behavior to estimate parameters.

In ordinary supervised learning / behavior cloning, if the data is:

```text
(s_i, a_i)
```

Maximum likelihood is maximizing:

```text
Σ log πθ(a_i | s_i)
```

Its meaning is:

```text
Make the model more likely to generate the actions that appeared in the data.
```

In this most basic MLE, each statistical sample has equal default weight. As long as an action appeared in the data, the model tends to increase its probability.

So the problem with ordinary MLE / behavior cloning is:

```text
It only knows "a human/historical policy did this," but not "whether doing this is good or bad."
```

Policy gradient adds a weight on top of this. It does not simply maximize:

```text
log πθ(a|s)
```

But rather maximizes a weighted log likelihood:

```text
A(s,a) · log πθ(a|s)
```

Or from the loss perspective:

```text
L = - A(s,a) · log πθ(a|s)
```

Here the advantage / return serves as the weight.

Therefore policy gradient can be understood as:

```text
Weighted maximum log likelihood.
```

Ordinary maximum likelihood is:

```text
This action appeared in the data, so increase its probability.
```

Policy gradient is:

```text
This action appeared in the data, and it brought higher returns, so increase its probability by a larger amount.
```

This can be compressed into one sentence:

```text
MLE models behavioral frequency; Policy Gradient models behavioral frequency with preference weights.
```

### 1.7 Entropy: Not a Decorative Term, but a Sampling Distribution Control Term

PPO / SAC frequently include an entropy bonus. Its role is not simply "encouraging randomness," but controlling the policy from collapsing too quickly.

From a distributional perspective, when entropy is maximized, the policy is closer to a uniform distribution. When entropy is small, the policy is sharper and closer to a deterministic choice.

So the entropy term essentially encourages:

```text
Do not let the policy too quickly become an extremely biased distribution.
```

Without entropy, policy gradient would tend to quickly concentrate probability on actions with currently high advantage. This would lead to:

```text
Policy concentrates too quickly on a few actions
-> Sampling narrows
-> Data diversity decreases
-> Q / advantage estimates become more biased
-> Training more easily gets trapped in local optima
```

So entropy is a form of sampling distribution regularization.

More precisely, policy optimization actually involves two forces:

```text
Preference term: increase probability of high-advantage actions.
Entropy term: keep the distribution from becoming too sharp, pulling toward uniformity.
```

Therefore, the final learned policy is neither a purely uniform distribution nor a complete preference collapse, but strikes a balance between the two:

```text
Final policy = equilibrium between preference distribution and uniform distribution.
```

SAC goes further by directly writing the objective as:

```text
maximize reward + entropy
```

This means exploration is no longer an externally appended trick, but part of the objective function.

From the loss organization perspective, it can be understood as:

```text
MSE: fit the Bellman target.
Weighted log π: weight behavioral likelihood by reward / advantage.
Entropy: pull the policy back from excessive preference toward a more uniform distribution.
```

These three correspond to three layers of RL:

```text
value estimation
preference-weighted behavior modeling
sampling diversity control
```

---

## 2. Determining On-Policy / Off-Policy / Near On-Policy from First Principles

### 2.1 Don't Start by Memorizing Algorithm Names -- First Look at "Where Does the Data Come From"

To determine whether an algorithm is on-policy or off-policy, the first step is not to check whether it is called PPO, DQN, or SAC, but to ask:

```text
For this parameter update, was the data generated by the policy currently being optimized?
```

More specifically, check whether two chains are consistent:

```text
Sampling chain: which policy selected the actions in the data?
Optimization chain: which policy / Q target is being optimized during this update?
```

If the policy that sampled the actions and the policy currently being optimized are the same, or approximately the same, then it is on-policy / near on-policy.

If the sampled actions come from an old policy, historical replay buffer, human data, other controllers, or any behavior policy, and the current algorithm can still use this data to update its own Q / policy, then it is off-policy.

Core criterion:

```text
It is not just about "whether there is historical data," but "whether the action distribution in the historical data must equal the current policy's action distribution."
```

### 2.2 First-Principles Criterion

```text
If the loss / target requires a ~ π_current(a|s), it is on-policy.

If the loss / target can accept a ~ μ(a|s), where μ is any behavior policy, it is off-policy.

If the loss can accept a ~ π_old(a|s), but requires π_old and π_current to not differ too much, it is near on-policy.
```

Here:

```text
π_current = the policy currently being optimized
π_old     = the old policy used during the most recent sampling
μ         = any behavior policy, which can be a historical policy, random policy, human policy, rule-based controller, or replay buffer source
```

The real criterion is not "whether there is a replay buffer," but:

```text
Does the update formula require the data actions to come from the current policy?
```

### 2.3 Why the Original Form of Policy Gradient Is On-Policy

The core form of policy gradient is:

```text
∇J(θ) = E_{s,a ~ πθ}[∇ log πθ(a|s) · A^{πθ}(s,a)]
```

Note the sampling distribution under the expectation:

```text
s,a ~ πθ
```

This means the gradient formula itself requires the actions to be sampled by the current policy πθ.

Therefore, the original policy gradient is on-policy.

If those actions were not sampled by the current policy, but by a much older policy or a different strategy, then:

```text
log πθ(a|s) · A(s,a)
```

This weighted maximum likelihood is no longer an unbiased estimate of the original policy gradient.

So original PG / A2C / A3C are on-policy, not because of their names, but because their gradient derivation requires:

```text
a must come from the current πθ's sampling distribution.
```

### 2.4 Why Q-learning / DQN Is Off-Policy

The core target of Q-learning is:

```text
y = r + γ max_a' Q(s', a')
```

Note that this does not require the actions in the replay buffer to be selected by the current policy.

The goal of Q-learning is not to directly imitate the behavior policy, but to learn:

```text
After executing action a in state s, if the future always follows a greedy policy, what is the long-term return.
```

The action `a` in the data only tells us:

```text
I once performed a in s, and observed r and s'.
```

And the future actions in the target do not follow the historical behavior policy, but directly use:

```text
max_a' Q(s', a')
```

Therefore Q-learning can learn the target policy independently of the sampling policy -- this is the fundamental reason it is off-policy.

### 2.5 Why PPO Is Near On-Policy

PPO's data typically comes from the most recent old policy:

```text
a ~ π_old(a|s)
```

But the update targets the current policy:

```text
πθ(a|s)
```

Yet PPO is not fully off-policy, because it does not allow old data and the current policy to diverge too much. It uses the importance ratio:

```text
r(θ) = πθ(a|s) / π_old(a|s)
```

Then clips:

```text
clip(r, 1-ε, 1+ε)
```

This means:

```text
I can use data from the most recent old policy.
But if the current policy and old policy diverge too much, the update weight for that data point will be limited.
```

Therefore, the precise understanding of PPO is:

```text
It is not purely on-policy, but near on-policy.
It allows limited historical data reuse, but constrains distribution shift through ratio clipping.
```

### 2.6 Why DDPG / TD3 / SAC Are Off-Policy

DDPG, TD3, and SAC all typically use a replay buffer.

But the more fundamental reason is: their critics can learn Bellman targets from historical transitions.

For example, the critic target of DDPG / TD3 is similar to:

```text
y = r + γ Q_target(s', μ_target(s'))
```

During the current update, this does not require `a` to be generated by the current actor μθ.

SAC is similar. SAC's critic can learn the soft Bellman target from the replay buffer, and the actor then optimizes the entropy-augmented objective. It does not require the actions in the replay buffer to come from the current policy, so it is also off-policy.

### 2.7 Quick Reference Table

| Algorithm | Where do the actions in the data come from | Does the update require actions from the current policy | Verdict |
|---|---|---|---|
| REINFORCE | Current policy | Yes | On-policy |
| A2C / A3C | Current policy | Yes | On-policy |
| PPO | Most recent old policy | Approximately required, needs ratio/clip control | Near on-policy |
| TRPO | Most recent old policy | Approximately required, needs trust region control | Near on-policy |
| Q-learning | Any behavior policy | No | Off-policy |
| DQN | Replay buffer / old policy / ε-greedy | No | Off-policy |
| Double DQN | Replay buffer / old policy / ε-greedy | No | Off-policy |
| DDPG | Replay buffer / old actor + noise | No | Off-policy |
| TD3 | Replay buffer / old actor + noise | No | Off-policy |
| SAC | Replay buffer / historical stochastic policy | No | Off-policy |

### 2.8 The Most Compressed Decision Logic

```text
If the update formula requires sample actions to be drawn from the current policy, it is on-policy.
If the update formula can use any historical actions, as long as there is (s,a,r,s'), it is off-policy.
If it can only use actions from the most recent old policy, and must constrain the difference between old and new policies, it is near on-policy.
```

Core:

```text
Whether the action distribution in historical data is allowed to be inconsistent with the policy distribution currently being optimized.
```

---

## 3. Understanding from Exploration / Exploitation: Why Argmax Needs Randomness

### 3.1 The Fundamental Problem with Greedy

If you directly use:

```text
a = argmax_a Q(s,a)
```

Then the policy quickly becomes a deterministic strategy.

The problem is that Q itself has not been accurately estimated in the early stages, but argmax is already forcefully selecting. This converts estimation noise directly into sampling bias.

The typical process is:

```text
Early Q has noise
-> argmax selects an action overestimated by noise
-> policy prematurely biases toward it
-> other actions lose the chance to be sampled
-> data distribution narrows
-> Q becomes harder to correct
```

So the problem with greedy is not simply "too greedy," but:

```text
Greedy converts estimation noise into sampling bias.
```

### 3.2 ε-greedy: Adding Randomness to Argmax

DQN commonly uses ε-greedy:

```text
With probability 1-ε, select argmax Q.
With probability ε, select a random action.
```

This amounts to exploitation most of the time, exploration a small fraction of the time.

The problem it solves is: do not let early noisy argmax completely control the data distribution.

### 3.3 Why Some Approaches Use a Uniform Distribution Target

In some exploration designs, action sampling is pushed closer to a uniform distribution, or the policy is prevented from becoming too sharp.

The reason is:

```text
If the policy concentrates on a few actions too early, the Q values of other actions can never be accurately estimated.
```

From a statistical perspective:

```text
Without sampling, there is no estimation.
Without coverage, there is no generalization guarantee.
```

So the significance of uniform exploration is not "being random for the sake of randomness," but first ensuring sufficient coverage of the action space, and only then talking about optimization.

### 3.4 Entropy Is Softer Than ε-greedy

ε-greedy is a hard rule: either greedy or random.

Entropy is a soft constraint: letting the policy maintain a certain level of randomness on its own.

SAC goes further by directly changing the objective to:

```text
maximize reward + entropy
```

This means exploration is no longer an externally appended trick, but part of the objective function.

Sampling diversity itself is part of the optimization objective.

---

## 4. Understanding from Data Utilization: Theoretical vs. Engineering Utilization

### 4.1 Theoretical Utilization

Theoretically, off-policy methods have higher data utilization because data in the replay buffer can be reused repeatedly.

```text
A single transition can be trained on many times.
```

Therefore, DQN, DDPG, TD3, SAC and similar methods typically have higher sample efficiency than purely on-policy methods.

The problem with on-policy is that data is too tightly bound to the current policy. Once the policy is updated, old data quickly becomes stale. Hence, theoretical sample utilization is low.

### 4.2 Engineering Utilization Does Not Equal Theoretical Utilization

In engineering practice, data utilization also depends on many factors:

```text
Whether training is stable.
Whether hyperparameter tuning is difficult.
Whether parallel sampling is easy.
Whether results are reproducible.
Whether the method is sensitive to hyperparameters.
Whether safe sampling in real systems is possible.
```

For example, PPO's theoretical sample efficiency is lower than SAC's, but PPO is very stable in engineering, easy to tune, and suitable for large-scale parallelism. Therefore, it is frequently used in robotics, RLHF, and simulation systems.

SAC theoretically has high sample efficiency, but it also has engineering costs, such as complex critic training, temperature tuning, Q bias, and replay distribution control.

An important distinction is:

```text
Theoretical utilization = how many times each sample can be used by the optimizer.
Engineering utilization = given the same engineering effort, whether you can stably obtain a good policy.
```

Sometimes, high theoretical utilization does not mean high engineering efficiency. If the off-policy Q diverges, samples in the replay buffer being reused many times is meaningless.

---

## 5. From Greedy to Stable Greedy, Then to Abandoning Argmax

### 5.1 Initially: Direct Greedy

The core action selection in Q-learning is:

```text
a = argmax Q(s,a)
```

This is the most direct form of exploitation.

But argmax is a very strong nonlinear function. It amplifies Q estimation errors. As long as Q has the slightest noise, argmax may select incorrectly. And once it selects incorrectly, the subsequent sampling distribution is also affected.

### 5.2 Methods for Stabilizing Greedy

To prevent greedy from collapsing so easily, subsequent algorithms introduced many stabilization techniques.

**DQN** uses three key types of mechanisms:

```text
Replay: break correlations, improve data reuse.
Target network: stabilize the bootstrap target.
ε-greedy: avoid premature sampling collapse.
```

**Double DQN** separates action selection from value evaluation, reducing overestimation from max over noisy Q.

**Dueling DQN** decomposes Q into:

```text
Q(s,a) = V(s) + A(s,a)
```

This makes the network architecture better at expressing "how good is this state itself" and "how much better is this action relative to other actions."

**TD3** addresses greedy actors in continuous action spaces by introducing:

```text
double critic
delayed policy update
target policy smoothing
```

The essence of all these methods is to prevent the actor from excessively exploiting the critic's errors.

### 5.3 Why Many Methods Eventually Abandon Hard Argmax

The direction of Policy Gradient, PPO, and SAC is:

```text
No longer explicitly using hard argmax to select actions, but instead directly optimizing the policy distribution.
```

That is, moving from:

```text
First learn Q, then argmax.
```

To:

```text
Directly learn π(a|s).
```

This avoids having the strong nonlinear argmax operation directly dominate training.

PPO uses advantage-weighted log π. SAC simultaneously maximizes reward and entropy. Their common thread is transforming action selection from hard argmax into soft distribution optimization.

This is the evolution from greedy to soft policy.

---

## 6. Understanding from a Bias Optimization Perspective

### 6.1 RL Does Not Pursue Complete Unbiasedness -- It Manages Bias

It is very difficult to achieve a truly unbiased optimization process in RL, because sampling, estimation, and target construction are all biased.

The main sources of bias include:

```text
sampling bias
bootstrapping bias
max / argmax bias
function approximation bias
replay buffer distribution bias
reward design bias
policy-induced data bias
```

So RL is not doing "find the true objective, then optimize without bias," but rather:

```text
Among a bunch of biases, choose a bias structure that is more stable and controllable in engineering.
```

### 6.2 Q-learning's Bias

Q-learning's bias mainly comes from:

```text
max Q target
bootstrap
off-policy replay
```

Its advantage is high data utilization; its disadvantage is susceptibility to overestimation and being misled by erroneous Q.

So subsequent methods continuously correct it:

```text
target network
Double Q
clipped double Q
conservative Q
```

These all belong to bias control.

### 6.3 Policy Gradient's Bias

Policy gradient avoids hard argmax, but it has its own problems: high variance, sample inefficiency, and noisy advantage estimates.

So it uses baseline, GAE, critic, advantage normalization, entropy, clipping, and other methods to control variance and update bias.

PPO's clipping is essentially intentionally introducing bias. It prevents the importance ratio of certain samples from becoming too large, which means the gradient does not exactly equal the true policy gradient, but it is more stable in engineering.

So PPO is a typical case of:

```text
Trading bias for stability.
```

### 6.4 SAC's Bias Structure

What makes SAC more elegant is that it acknowledges exploration is not an extra trick, but should be part of the objective function.

It optimizes:

```text
reward + entropy
```

This itself is a biased objective. It is not raw reward maximization, but maximization of entropy-regularized reward.

But this bias is intentionally designed. The benefit is that the policy is less likely to collapse, Q learning is smoother, and exploration is more natural.

So SAC's bias can be understood as:

```text
Trading a soft objective for stable exploration and better data coverage.
```

---

## 7. Understanding from a Compression Perspective: Q Is Value Compression, Policy Is Preference Compression

### 7.1 Q Function: Compression of Sparse Rewards / Long-term Value

The Q function can be understood as a compressor. It compresses complex future rollouts, sparse rewards, and delayed feedback into a scalar:

```text
Q(s,a) = compressed representation of long-term returns after executing action a in state s.
```

Q is not a simple "immediate good/bad judgment," but a compression of future value:

```text
Sparse rewards
-> Multi-step rollouts
-> Discounted cumulative returns
-> Q(s,a)
```

### 7.2 Policy: Compression of Preferences

Policy is also a compressor, but it compresses preferences rather than values.

If it is a stochastic policy:

```text
π(a|s) = action preference distribution
```

If it is a deterministic policy:

```text
a = μ(s) = compressing preferences into the single most preferred action
```

### 7.3 Q and Policy Compress Different Objects

```text
Q compresses: how much is this action worth in the future.     -> value compression
Policy compresses: how I prefer to choose in this state.       -> preference compression
```

Q is more like an evaluator; Policy is more like a decision-maker.

### 7.4 Why Q-based Methods Need Argmax

If you only have Q, it is just value compression, not yet final behavior. To go from Q to action, you typically need one more step:

```text
a = argmax_a Q(s,a)
```

In other words, Q-based methods: first compress value, then extract preferences from value via argmax.

But the problem lies precisely here: argmax is an extremely hard preference extractor. It amplifies small errors in Q into large differences in behavioral choices.

### 7.5 Why Policy-based Methods Directly Optimize Preferences

Policy gradient / PPO / SAC do not necessarily require first learning a complete Q landscape and then selecting actions via argmax; instead, they directly optimize the policy distribution.

```text
Directly compress preferences, rather than first compressing value and then extracting preferences.
```

### 7.6 Actor-Critic: Value Compression and Preference Compression Working Together

The Actor-Critic architecture can be understood as simultaneously maintaining two compressors:

```text
Critic: compresses value / advantage.  ->  Is this action good? How much better?
Actor: compresses preference / behavior.  ->  Based on the good/bad signal, how should I tend to choose in the future?
```

So Actor-Critic is not simply "one network selects actions, another network scores them," but rather: the value compressor guides the preference compressor.

### 7.7 Re-understanding Several Algorithms from This Perspective

| Algorithm | Compression target | Decision method | Core risk |
|---|---|---|---|
| Q-learning / DQN | Compress long-term value Q | Extract preferences via argmax | Q noise amplified by argmax |
| Double DQN / TD3 | More stably compress value | More conservatively extract preferences | Underestimation or slower learning |
| Policy Gradient | Directly compress preferences π | Sample from policy | High variance, low sample efficiency |
| PPO | Stably compress preferences π | Limit the magnitude of preference updates | May be overly conservative |
| SAC | Compress soft preferences | reward + entropy | Objective rewritten by entropy bias |
| Actor-Critic | Q/A compresses value, Actor compresses preferences | Critic guides actor | Two compressors can contaminate each other |

### 7.8 The Most Refined Expression

```text
Q is the compression of sparse rewards and long-term value.
Policy is the compression of behavioral preferences and sampling tendencies.
```

Value-based methods: first compress value, then extract preferences from value.

Policy-based methods: directly compress preferences, then continuously refine preferences through sampling and feedback.

Actor-Critic: use the value compressor to guide the preference compressor.

This also explains why RL has gradually moved from hard argmax toward soft policy:

```text
Because hard argmax is a fragile conversion from value compression to preference compression.
While soft policy directly treats preferences as a learnable object.
```

### 7.9 Actor-Critic's Entangled Iteration: Alternating Freeze, Alternating Optimize

Actor-Critic training has an easily overlooked but critically important structure: **Actor and Critic are entangled, but each step only optimizes one while treating the other as a constant.**

This is exactly like solving a coupled equation:

```text
Analogy: solving f(x, y) = 0

x and y are coupled — x's optimal value depends on y, y's optimal depends on x.
Solving them jointly is hard (or impossible).

Engineering approach: coordinate descent
  Step 1: Fix y = y_current, optimize x → get x_new
  Step 2: Fix x = x_new,     optimize y → get y_new
  Step 3: Go back to Step 1, repeat

Each step only solves for one variable, treating the other as constant.
Not globally optimal per step, but alternating iteration can converge.
```

Actor-Critic has exactly this structure:

```text
Coupling:
  Q's optimal value depends on π (different policies yield different Q)
  π's optimal value depends on Q (policy should move toward high Q)

Training: alternating freeze, alternating optimize

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Step 1: Update Critic (freeze Actor)                       │
  │                                                             │
  │    Actor's parameters frozen, treated as constant            │
  │    Fit Q / V using data sampled by Actor's current policy    │
  │                                                             │
  │    loss_critic = (Q(s,a) - target)²                         │
  │    Gradient only w.r.t. Critic params; Actor params excluded │
  │                                                             │
  │         ┌──────────┐                                        │
  │         │  Actor   │ ← frozen (constant)                    │
  │         └──────────┘                                        │
  │         ┌──────────┐                                        │
  │         │  Critic  │ ← updating                             │
  │         └──────────┘                                        │
  │                                                             │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  Step 2: Update Actor (freeze Critic)                       │
  │                                                             │
  │    Critic's parameters frozen, treated as constant           │
  │    Use Critic's Q / advantage to guide Actor adjustment      │
  │                                                             │
  │    loss_actor = -Q(s, π(s))  or  -log π(a|s) · A(s,a)      │
  │    Gradient only w.r.t. Actor params; Critic params excluded │
  │                                                             │
  │         ┌──────────┐                                        │
  │         │  Actor   │ ← updating                             │
  │         └──────────┘                                        │
  │         ┌──────────┐                                        │
  │         │  Critic  │ ← frozen (constant)                    │
  │         └──────────┘                                        │
  │                                                             │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  Step 3: Go back to Step 1, repeat                          │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

In mathematical terms:

```text
Critic update:
  min_φ  L(φ) = (Q_φ(s,a) - y)²
  where y = r + γ·Q_target(s', π_θ(s'))
  θ (Actor params) does not participate in ∂L/∂φ

Actor update:
  min_θ  L(θ) = -Q_φ(s, π_θ(s))     (DDPG/SAC style)
  or
  min_θ  L(θ) = -log π_θ(a|s) · A    (PPO style)
  φ (Critic params) does not participate in ∂L/∂θ
  (even in DDPG where gradients flow through Q, Q's params are stop_gradient)
```

This "entangled iteration" structure explains several important phenomena:

**1. Why Actor-Critic can be unstable**

```text
Coordinate descent converges on convex problems, but Actor-Critic is non-convex.

Two optimizers are chasing each other:
  Critic chases the data distribution shift caused by Actor changes
  Actor chases the value landscape shift caused by Critic changes

If one runs too fast and the other can't keep up, the system oscillates or diverges.
```

**2. Why TD3 uses delayed policy update**

```text
TD3 makes Critic update more frequently than Actor (e.g., Critic updates 2x, Actor 1x).

Reason: let Critic stabilize first (y converges), then let Actor follow (x optimizes).
If both update at the same rate, Critic hasn't stabilized but Actor is already chasing —
the chase direction may be wrong.
```

**3. Why target networks help**

```text
Target networks further slow down Critic target changes.
Like using y_target = 0.995·y_old + 0.005·y_new instead of y_new directly.
Makes the "constant" more truly constant, reducing oscillation in alternating iteration.
```

**4. Contrast with pure value-based methods (DQN)**

```text
DQN has no entanglement — only one Q network, policy is a byproduct of argmax Q.
No "two networks alternating optimization" problem.
This is one reason DQN can be more stable in some scenarios.
Cost: argmax only works for discrete actions, can't handle continuous control.
```

Most compressed expression:

```text
Actor-Critic = coordinate descent on a coupled equation

Q and π depend on each other, but each step only optimizes one,
treating the other as constant.
An engineering compromise: no global joint solve, only alternating approximation.
Stability depends on whether the two optimizers' paces are coordinated.
```

### 7.10 Action's Different Roles in Q Training vs. Policy Training

Section 7.9 covered how Actor and Critic alternate. Here we zoom in on an easily confused detail: **when training the Q network, where does the action a come from? Does it participate in gradients?**

#### Q training: action is input data, not an optimization variable

Q learns Q_φ(s, a), so training Q requires (s, a) as input. This a comes from two sources:

**Source A: From replay buffer (historical data)**

```text
Replay buffer stores historical transitions: (s, a, r, s')

This a was produced during past environment interaction, possibly from:
  - Historical Actor + exploration noise
  - Random exploration
  - Human data / offline dataset
  - Old policy

When entering the Critic loss, it is just a data field — a pure constant.

  Replay Buffer
  (s, a, r, s')
       │
       ├── s, a  → Critic Q_φ(s, a)  ← gradient only w.r.t. φ
       │
       └── r, s' → construct TD target y

Actor does not participate in Q(s,a)'s forward computation at all.
```

**Source B: Generated by current/target policy (but detached)**

```text
Critic's TD target often needs "next action" a':

DDPG/TD3:
  a' = μ_{θ_target}(s')           ← target actor generates
  y  = r + γ · Q_{φ_target}(s', a')

SAC:
  a' ~ π_θ(·|s')                  ← current actor samples
  y  = r + γ · [min Q_{φ_target}(s', a') - α · log π(a'|s')]

In code:

  with torch.no_grad():                    ← entire block has no gradients
      a_next, logp = actor(s_next)         ← policy forward, but detached
      target_q = target_critic(s_next, a_next)
      y = r + gamma * (target_q - alpha * logp)

  q = critic(s, a_from_buffer)             ← a from buffer
  critic_loss = mse(q, y)
  critic_loss.backward()                   ← only updates critic params φ
  critic_optimizer.step()

Policy participates in generating the target, but wrapped in no_grad.
Equivalent to: treating policy as a temporary "constant generator."
```

#### Comparing the two sources

```text
┌───────────────────────────────────────────────────────────────────┐
│            Action's role during Q training                        │
├──────────────────────┬────────────────────────────────────────────┤
│ Source A (buffer)    │ Source B (policy-generated then detached)   │
├──────────────────────┼────────────────────────────────────────────┤
│ a from historical    │ a' from current/target policy forward pass │
│ data                 │                                            │
│ Naturally a constant │ Detached/no_grad after generation =        │
│                      │ equivalent to constant                     │
│ Used for Q(s,a)      │ Used for TD target y computation           │
│ input                │                                            │
│ Actor not involved   │ Actor involved in forward, not in gradient │
│ at all               │                                            │
├──────────────────────┴────────────────────────────────────────────┤
│ Common: during Q training, action is always "input data,"         │
│ never an "optimization variable." Only Critic params φ updated.   │
└───────────────────────────────────────────────────────────────────┘
```

#### Actor training: action becomes the optimization variable

Only during Actor update does action connect to gradients:

```text
Actor update (DDPG/SAC style):

  a = π_θ(s)                       ← Actor forward, participates in gradient!
  J = Q_φ(s, a)                    ← Critic forward, but φ frozen
  actor_loss = -J                  ← maximize Q
  actor_loss.backward()            ← gradient path: loss → Q → ∂Q/∂a → a → ∂a/∂θ → θ

  Gradient flow:
    Critic Q_φ    ──(∂Q/∂a)──→    action a    ──(∂a/∂θ)──→    Actor θ
    (params frozen)               (intermediate)               (params updated)
```

```text
Actor update (PPO style):

  loss = -log π_θ(a|s) · A(s,a)
  A comes from Critic but is detached → A is a constant weight
  Gradient only w.r.t. π_θ's parameters θ
```

#### One-sentence summary

```text
Q training:     action is "input data"           → constant, regardless of source
Actor training: action is "optimization variable" → gradients flow through action back to Actor params

The same action plays different roles in different phases —
this is the concrete implementation of "freeze one, optimize the other" in alternating iteration.
```

### 7.11 Advantage: Causal Inversion — Estimate V First, Construct Q On-the-Fly

This is an often underestimated but critically important conceptual shift.

#### Traditional causality: Q is primary, V is derived

In Q-learning / DQN, Q is the core:

```text
Primary:  Q(s,a)              ← explicitly learned, one value per (s,a) pair
Derived:  V(s) = max_a Q(s,a) ← computed from Q
Decision: a* = argmax_a Q(s,a) ← extracted from Q

Causal chain: Q first → derive V from Q → select action from Q
```

This seems natural. But the problem: **Q maintains a separate estimate for every action, high-dimensional, high variance, easily amplified by max.**

#### Advantage inverts the causality: V is primary, Q is constructed on-the-fly

Advantage reverses the causal direction:

```text
Primary:     V(s)                          ← depends only on state, not action, more stable
Constructed: A(s,a) = Q(s,a) - V(s)        ← "how much better is this action than average"
              ≈ r + γV(s') - V(s)           ← computed via TD on-the-fly, no Q network needed

Causal chain: V first → construct A on-the-fly → A guides policy update
```

Why does this inversion matter?

```text
1. V(s) is easier to learn than Q(s,a)
   - V has only state dimensions (n-dim), Q has state×action dimensions (n+m dim)
   - V averages over all actions, lower variance
   - V doesn't explicitly depend on the policy's specific choices

2. Advantage only cares about relative differences between actions
   - A(s,a) = "how much better is this action than average"
   - No need for absolute values, only differences
   - Relative values are easier to estimate accurately

3. No Q network needed
   - PPO / A2C don't need a Q network at all
   - Only V-critic + TD advantage
   - One fewer network simplifies training
```

#### Historical timeline

Advantage wasn't invented all at once — it emerged in stages:

```text
1993  Baird "Advantage Updating"
      First proposed Q = V + A decomposition
      Motivation: in continuous-time problems, Q differences are too small to learn
      Directly learning A (the residual) is easier than learning full Q

1999  Sutton et al. Policy Gradient Theorem
      Proved policy gradient can be expressed using Q(s,a)
      Subtracting baseline V(s) doesn't change expectation but reduces variance
      Q(s,a) - V(s) = A(s,a) ← advantage appears naturally

2016  Schulman et al. GAE (Generalized Advantage Estimation)
      Provided bias-variance tradeoff for advantage estimation
      A^GAE = Σ (γλ)^t δ_t
      λ=1 → high-variance unbiased (Monte Carlo advantage)
      λ=0 → low-variance biased (single-step TD advantage)
      Analogous to TD(λ)

2016  Wang et al. Dueling DQN
      Applied Q = V + A decomposition in Q-learning architecture
      Network splits into two streams: one for V(s), one for A(s,a)
      Advantage used within value-based methods
```

#### Why did advantage flourish in policy gradient but not in Q-learning?

```text
In policy gradient methods (PPO / A2C):
  - Already need a baseline to reduce variance
  - V(s) is the natural baseline
  - A = r + γV(s') - V(s) is directly the advantage
  - V-first causality is the most natural fit
  → advantage became standard

In Q-learning methods (DQN):
  - Q itself is the primary quantity; learning Q directly is more natural
  - Q = V + A decomposition has an identifiability problem:
    constants can shift freely between V and A
    (Dueling DQN patches this with A - mean(A), but it's a heuristic)
  - In discrete action spaces, max_a Q implicitly recovers V, so decomposition
    doesn't have a clear advantage
  → advantage only appears as network architecture (Dueling DQN), not mainstream
```

#### Comparison overview

```text
┌──────────────────────────────────────────────────────────────┐
│           Q-first vs V-first: Causal Comparison              │
├──────────────────────┬───────────────────────────────────────┤
│ Q-first (traditional)│ V-first (advantage)                   │
├──────────────────────┼───────────────────────────────────────┤
│ Primary: Q(s,a)      │ Primary: V(s)                         │
│ High-dim (state×act) │ Low-dim (state only)                  │
│ High variance        │ Low variance                          │
│ max amplifies noise  │ Not affected by max                   │
│ V = max Q (derived)  │ A = r+γV(s')-V(s) (constructed)      │
│ Needs Q network      │ No Q network needed                   │
│ Repr: DQN, DDPG      │ Repr: PPO, A2C                       │
│ Suits: off-policy    │ Suits: on/near on-policy              │
└──────────────────────┴───────────────────────────────────────┘
```

#### One-sentence summary

```text
Advantage is not just a "trick" —
it is a causal inversion:
from "learn Q first, derive V" to "learn V first, construct advantage on-the-fly."

V is more stable, lower-dimensional, action-independent.
Advantage only cares about relative action quality, not absolute values.

This inversion lets PPO / A2C operate without any Q network at all —
just one V-critic is enough to train the policy.
```

---

## 8. Understanding from the Action Modeling Perspective: Discrete Actions, Continuous Actions, Deterministic vs. Stochastic

### 8.1 Discrete Actions: Can Be Explicitly Enumerated, So Argmax Is Easy to Use

In discrete action spaces, actions form a finite set:

```text
A = {a1, a2, a3, ..., ak}
```

In this case, Q-learning is natural because you can estimate a Q value for each action and directly argmax at decision time.

DQN uses exactly this structure. The neural network takes state as input and outputs the Q value for each discrete action:

```text
state → network → [Q(s,a1), Q(s,a2), ..., Q(s,ak)]
```

The key convenience of discrete actions is that argmax can be computed directly. The downside is that if the number of actions is very large or continuous, argmax is no longer straightforward.

### 8.2 Continuous Actions: Cannot Be Enumerated, So an Actor Is Needed

In continuous action spaces, actions are not a finite set but continuous variables:

```text
a ∈ R^n
```

Now argmax becomes a continuous optimization problem -- finding the action that maximizes Q in a continuous action space at each decision step -- which is typically infeasible.

So in continuous control, an actor is often introduced:

```text
a = μθ(s)
```

The actor directly outputs an action, effectively using a network to approximate argmax_a Q(s,a).

DDPG follows this idea: the Critic learns Q(s,a), the Actor learns μ(s), making Q(s, μ(s)) as large as possible.

```text
In continuous action spaces, the actor is a function approximation of argmax.
```

### 8.3 Policy Modeling for Discrete Actions

For discrete actions, the policy can be modeled as a categorical distribution:

```text
state → network → logits → softmax → action probabilities
```

At sampling time, an action is drawn from the categorical distribution.

### 8.4 Policy Modeling for Continuous Actions

Continuous actions are typically modeled as parameterized distributions, such as Gaussian:

```text
π(a|s) = Normal(μθ(s), σθ(s))
```

The network outputs the mean and variance:

```text
state → network → mean, std → sample action
```

The mean represents the center of preference; std represents the exploration range. Gaussian policy is a common form in SAC / PPO for continuous control.

### 8.5 Deterministic Policy vs Stochastic Policy

**Deterministic policy**: `a = μθ(s)`

* Pros: stable execution, simple inference, suitable for continuous control.
* Cons: weak exploration capability, prone to premature convergence, prone to exploiting critic errors.
* Representatives: DDPG / TD3. Usually requires additional exploration noise.

**Stochastic policy**: `a ~ πθ(a|s)`

* Pros: naturally includes exploration, can use log probability for policy gradient, can use entropy to control distribution shape.
* Cons: higher variance, training more dependent on sampling.
* Representatives: PPO / SAC.

### 8.6 The Evolutionary Logic from Deterministic to Stochastic

Early control problems often favored deterministic approaches -- given a state, output a single optimal action.

But in RL, the policy simultaneously serves two roles:

```text
1. Decision-maker: what to do right now.
2. Sampler: what data to see in the future.
```

If the policy is deterministic, it is clear as a decision-maker but impoverished as a sampler. It causes the data distribution to narrow quickly.

So RL's gradual shift toward stochastic policy is because:

```text
RL is not just about selecting the current optimal action, but also about maintaining the data coverage needed for future learning.
```

From the compression perspective, a stochastic policy is a more complete preference compression -- preferences are not necessarily a single point, but can be a distribution:

```text
stochastic policy = soft preference representation
deterministic policy = hard preference representation
```

---

## 9. From DP to RL: Progressively Relaxing Assumptions, Progressively Changing Optimization Methods

### 9.1 DP's God's-Eye View Assumption

Dynamic Programming assumes you know the complete MDP:

```text
State set S
Action set A
State transition probability P(s'|s,a)
Reward function R(s,a)
Discount factor γ
```

DP has a near "god's-eye view" -- knowing how the environment transitions, knowing what reward each action brings, and being able to systematically perform Bellman backups over all states.

### 9.2 First Relaxation: Not Knowing the Full Environment, Only Able to Sample

In practice, P(s'|s,a) and R(s,a) are usually unknown; you can only obtain samples (s, a, r, s') through interaction with the environment.

The Bellman backup changes from an exact expectation to a sampled estimate. This is the origin of TD learning.

### 9.3 Second Relaxation: Cannot Traverse All States, Must Use Function Approximation

Classical DP can update every state / action in a tabular state space. Real state spaces are enormous, even continuous, and cannot maintain a complete table.

So function approximation is introduced:

```text
V(s) ≈ Vθ(s)
Q(s,a) ≈ Qθ(s,a)
π(a|s) ≈ πθ(a|s)
```

The cost is that generalization brings efficiency, but also function approximation bias.

### 9.4 Third Relaxation: Cannot Guarantee Data Comes from the Target Policy

In DP, there is no "data distribution" problem because it directly operates on the model and the full state space.

In RL, sampling is necessary, and sampling is determined by the behavior policy. The data distribution is generated by the policy, and the policy is in turn trained by the data -- this closed loop is one of the essential difficulties of RL.

### 9.5 Fourth Relaxation: From Exact Greedy to Approximate Greedy

In DP, if you know the complete Q, you can directly argmax. But in RL, Q is estimated, with noise, bias, and incompleteness.

Exact greedy becomes approximate greedy, introducing max bias, overestimation, and exploration collapse. This necessitates ε-greedy, Double Q, target network, entropy regularization, soft policy, and other methods.

The essence: since Q is not the true Q from a god's-eye view, you should not fully trust hard argmax.

### 9.6 Fifth Relaxation: From Optimal Control to Preference Optimization

The DP / optimal control approach is: have an environment model -> find the optimal value -> derive the optimal policy.

Later RL approaches like policy gradient / PPO / SAC are more like: no complete model -> obtain behavioral feedback through sampling -> directly optimize the behavior distribution.

This is the shift from "solving an optimal control problem" toward "preference distribution optimization."

### 9.7 The Evolution Chain from DP to RL

```text
DP:
Known model + exact Bellman backup + full state traversal + exact greedy

↓ relax environment model

Tabular RL:
Unknown model + sampled Bellman backup + tabular Q/V

↓ relax state space scale

Deep Q Learning:
Function-approximated Q + replay + target network + approximate greedy

↓ relax action space

Actor-Critic:
Actor approximates argmax / policy, critic estimates value

↓ relax hard greedy

Policy Gradient / PPO / SAC:
Directly optimize stochastic policy, replace hard argmax with preference distribution
```

### 9.8 What Assumption Each Relaxation Step Changes

| Stage | Original assumption | After relaxation | New problem |
|---|---|---|---|
| DP | Known P/R model | Can only sample | Sampling noise |
| Tabular RL | Can traverse state table | State space huge/continuous | Function approximation bias |
| Q-learning | Can stably estimate Q | Q has noise | Max overestimation |
| DQN | Discrete actions enumerable | Continuous actions not enumerable | Need actor |
| DDPG/TD3 | Deterministic actor suffices | Insufficient exploration | Need noise/entropy |
| PPO/SAC | Hard greedy is feasible | Hard greedy too fragile | Soft policy / preference optimization |

---

## 10. Unifying Perspective: Optimal Control and RL Are Fundamentally Doing the Same Thing

Optimal Control and RL are often taught as separate fields. But viewed through three core dimensions -- **rollout (imagining the future), optimization (making decisions), and data (requirements)** -- they are different engineering trade-offs for the same problem.

### 10.1 Dimension One: Rollout / Imagination -- How to "Imagine" the Future

Making decisions requires predicting the future. Optimal Control and RL do this differently:

**Optimal Control: Model-based online rollout**

```text
Has an explicit dynamics model:
  s_{t+1} = f(s_t, a_t)       (deterministic)
  s_{t+1} ~ P(·|s_t, a_t)     (stochastic, but usually with simple assumptions)

Rollout method:
  Use this model to simulate multiple steps forward,
  predict consequences of different action sequences
  → MPC (Model Predictive Control) is the typical approach

Uncertainty handling:
  Usually simplified assumptions (linear, Gaussian, small perturbations)
  Because it runs in real-time, assumptions can be rough --
  each step gets fresh state feedback from the environment to correct

```

Essentially, Optimal Control is also an agent-environment interaction model:

```text
Offline preparation: calibrate a few dynamics parameters (mass, friction, gear ratios, etc.)
Online execution:    use the model for rollout + use real-time sensor data to correct

It assumes a dynamics model, plus real-time environment parameter feedback.
```

**RL: Model-free offline rollout**

```text
No explicit dynamics model.
Rollout through direct interaction with the environment (or simulator).

Input requirements are more relaxed:
  Optimal Control needs precise low-dimensional states (position, velocity, angle)
  RL can accept fuzzier high-dimensional inputs (images, point clouds, raw sensor data)
  → Because the model itself learns to extract useful information from high-dimensional inputs

The model is learned offline:
  Through massive environment interactions, train Q / V / Policy
  Compress "how to make good decisions from the current state" into network parameters
  → Rollout knowledge is implicitly encoded in the network, not in explicit dynamics equations
```

Comparison:

```text
Optimal Control:
  Explicit model + online rollout + precise low-dim input + real-time correction
  → "I know how the world works; I re-simulate at every step"

RL:
  Implicit model (encoded in network) + offline training + high-dim input + forward inference at deployment
  → "I've seen enough scenarios during training; I react directly at deployment"
```

### 10.2 Dimension Two: Optimization -- When Does Optimization Happen

**Optimal Control: Online optimization**

```text
Solves an optimization problem online at every timestep:
  a* = argmin_a J(a, s_current)

MPC at each step:
  1. Get current state
  2. Roll out multiple steps using the model
  3. Solve for optimal action sequence online
  4. Execute the first action
  5. Repeat next step

Can also precompute offline lookup tables:
  Pre-solve for the range of possible inputs
  Look up online
  → Suitable for low-dimensional, enumerable scenarios
```

**RL: Offline optimization**

```text
Optimization (training) is completed offline:
  Learn Q / V / Policy through massive environment interaction

Online deployment does no optimization, only forward inference:
  a = π_θ(s)  or  a = argmax Q_θ(s,a)
  → One forward pass, directly outputs action
  → Far less computation than online optimization
```

Comparison:

| Dimension | Optimal Control | RL |
|---|---|---|
| Optimization happens | Online (solve at every step) | Offline (training phase) |
| Deployment compute | High (solve optimization each step) | Low (one forward pass) |
| Adapting to new scenarios | Strong (re-optimize in real-time) | Weak (needs retraining or generalization) |
| Model requirement | Needs explicit dynamics | Not needed (model-free) |

### 10.3 Dimension Three: Data -- Where Does It Come From, How Is It Used

**Optimal Control: Relies on precise parameters, doesn't need large datasets**

```text
The "data" needed is environment model parameters:
  Mass, inertia, friction coefficients, transmission characteristics...
  → Usually obtained through physical measurement or system identification
  → Limited number of parameters (tens to hundreds)

Online data = real-time sensor feedback
  → Used to correct model prediction errors
  → Not used for training, used for online correction
```

**RL: Relies on massive interaction data, compressed into network parameters**

```text
Needs large volumes of (s, a, r, s') interaction data
  → From simulator or real-environment rollouts
  → Data volume can be millions to billions of transitions

This data is "compressed" into network parameters during training:
  Offline information → training → network parameters → online inference

RL's training process is essentially information compression:
  Compress "what to do in various states" from massive experience
  into a network that can do fast inference
```

The significance of this compression:

```text
At online deployment, no need to re-interact with the environment
The network has already encoded offline interaction experience
One forward pass outputs the action

→ RL model = compressed representation of offline experience, for fast online decisions
```

### 10.4 The Hidden Strong Constraints of Optimal Control

Optimal Control looks elegant, but it works because it makes very strong assumptions about the problem. These assumptions easily break down in complex scenarios:

**Constraint 1: Requires explicit reference and analytically tractable constraints**

```text
MPC / LQR etc. typically require:
  - An explicit reference trajectory
  - Constraints expressed as linear or convex constraints:
      A·x ≤ b         (linear inequality constraints)
      x_min ≤ x ≤ x_max   (state/input bounds)
  - Cost function is usually quadratic:
      J = Σ (x - x_ref)^T Q (x - x_ref) + u^T R u

If constraints are nonlinear, non-convex, or reference is hard to define
→ The problem becomes very hard to solve, or even intractable
```

RL needs none of this: no explicit reference, no constraint form requirements, reward can be any computable scalar.

Moreover, the reference requirement cascades upward layer by layer:

```text
Controller needs a reference trajectory
  → Who provides the reference? → Planner

Planner needs its own inputs:
  → Maps, goal points, obstacles, traffic rules, semantic information...
  → Planner itself may also need a reference (lane centerlines, global path)

Planner's inputs depend on Perception:
  → Detection, tracking, prediction, semantic segmentation...

This forms a cascading dependency chain:
  Perception → Planner → Controller
  Each layer has its own explicit input requirements
  Each layer's errors propagate downstream
  System complexity is the PRODUCT of each layer's complexity, not the sum
```

This is the fundamental dilemma of traditional autonomous driving's "modular stack" architecture -- every module has strong dependencies on upstream modules, and errors at any layer cascade and amplify downstream.

RL (especially end-to-end RL) aims to skip this cascade:

```text
Traditional: Perception → Planner → Controller (each layer has explicit interface requirements)
E2E RL:      raw input ──→ neural network ──→ action (one network learns end-to-end)
```

The cost is more data and training, but it avoids the layer-upon-layer explicit interface problem.

**Constraint 2: Degrees of freedom cannot be too high**

```text
Optimal Control's online solving complexity grows steeply with DOF:
  - State dimension n, control dimension m, prediction horizon T
  - MPC optimization variables ≈ (n + m) × T
  - QP complexity ≈ O(((n+m)T)^3)

High-DOF scenarios (high-dim robots, multi-agent, pixel-level control):
  → Online solving simply can't keep up
  → Must drastically simplify models or reduce dimensions

RL is not bound by this:
  Offline training can take as much time as needed
  Online inference is just one forward pass, scales linearly with dimension
```

**Constraint 3: Poor parallelizability**

```text
Optimal Control's online solving is typically serial:
  - Each step depends on the previous step's solution
  - MPC iterative solvers (QP solver, ADMM) are inherently sequential
  - Very hard to leverage GPU's massive parallelism

RL training is naturally parallel:
  - Data collection: multiple environments roll out in parallel (vectorized env)
  - Gradient computation: samples within a batch are naturally parallel (GPU matrix ops)
  - Inference: batch inference, can process multiple inputs simultaneously
```

Summary:

```text
Optimal Control's costs:
  ✗ Needs explicit reference + linear/convex constraints
  ✗ Solving explodes at high DOF
  ✗ Hard to parallelize, can't leverage GPUs well
  ✓ But very efficient when low-dim, model is precise, constraints are well-structured

RL's costs:
  ✗ Needs massive offline training data
  ✗ Training is unstable, hyperparameter tuning is hard
  ✗ Not easy to do real-time online adaptation
  ✓ But has no hard requirements on input dimension, constraint form, or model precision
```

### 10.5 Unified View: Different Engineering Trade-offs for the Same Problem

```text
┌──────────────────────────────────────────────────────────────┐
│               The core problem is the same:                  │
│                                                              │
│    Given the current state, choose the optimal               │
│    (or good-enough) action                                   │
│                                                              │
├──────────────┬───────────────────┬───────────────────────────┤
│   Dimension  │ Optimal Control   │ RL                        │
├──────────────┼───────────────────┼───────────────────────────┤
│ Rollout      │ Explicit model,   │ Implicit model (encoded   │
│              │ online simulation │ in network)               │
│ Input req.   │ Precise low-dim   │ Accepts high-dim fuzzy    │
│              │ states            │ inputs                    │
│ Optimization │ Online solving    │ Offline training,         │
│              │                   │ online inference          │
│ Data needs   │ Few precise       │ Massive interaction data  │
│              │ parameters        │                           │
│ Deploy cost  │ High (online opt) │ Low (forward pass)        │
│ Env. adapt.  │ Real-time correct.│ Generalization ability    │
│ Best for     │ Known model,      │ Unknown model,            │
│              │ low-dim           │ high-dim                  │
└──────────────┴───────────────────┴───────────────────────────┘
```

So Optimal Control and RL are not opposites, but rather:

```text
Different engineering choices for the same decision problem under different constraints:

Know dynamics + low-dim input + need real-time adaptation → Optimal Control
Don't know dynamics + high-dim input + can train offline  → RL

In practice, the two often fuse:
  - Model-based RL (Dreamer, MBPO): learn a dynamics model, use it for rollout
  - Autonomous driving hierarchies: high-level RL for planning, low-level MPC for control
  - Sim2Real: train with RL offline in simulation, deploy with sensor correction online
```

---

## 11. Summary Table: Multiple Threads Combined

| Dimension | Early approach | Problem | Subsequent fix | More advanced direction |
|---|---|---|---|---|
| Loss | TD MSE | Self-bootstrapping target, noisy | Target network / double Q | Policy log-likelihood |
| Action selection | argmax Q | Amplifies Q noise | ε-greedy / double Q | Soft policy |
| Exploration | Random tricks | Unsystematic | Entropy bonus | Max entropy RL |
| Data reuse | On-policy, little reuse | Sample waste | Replay buffer | Off-policy actor-critic |
| Stability | Hard greedy | Prone to collapse | PPO clip / TD3 tricks | Bias-controlled optimization |
| Bias | Pursuing Bellman target | Overestimate | Clipped / double / conservative | Accept bias, trade for stability |

---

## 12. One-Sentence Summaries of Typical Algorithms

**Q-learning / DQN**: Learn Q via MSE, make decisions via argmax, improve data utilization via replay. The problem is that max Q amplifies estimation errors.

**Double DQN / TD3**: Acknowledge that Q max overestimates, so separate selection from evaluation, or use multiple critics for conservative estimation.

**DDPG**: Extend Q-learning to continuous action spaces, using an actor to approximate argmax Q. The problem is the actor will exploit the critic's errors.

**PPO**: Directly optimize the policy, but constrain the policy from deviating too far from the sampling data. Essentially, it is near on-policy stable policy gradient.

**SAC**: Incorporate entropy into the objective function, using a soft policy to avoid premature greediness. Essentially, it uses a maximum entropy objective to unify exploitation and exploration.

---

## 13. RL vs SL: Similar Optimization Frameworks, Fundamentally Different Sampling Structures

### 12.1 Similarity: Both Ultimately Minimize a Loss

From the perspective of loss and optimizer, RL and SL are remarkably similar.

SL's typical structure:

```text
loss = L(f_θ(x), y)
θ ← θ - α ∇L
```

RL also ultimately optimizes a loss:

```text
Value-based:  loss = (Q_θ(s,a) - y)^2
Policy-based: loss = - log π_θ(a|s) · A(s,a)
```

Both use SGD / Adam. Gradients are taken with respect to parameters θ. Backpropagation is the same.

So at the "writing code to train a model" level, RL and SL share essentially the same optimization framework: forward → compute loss → backward → update.

### 12.2 Difference One: RL Data Has Temporal Dependencies; Labels Come From Environment Feedback

SL data is typically i.i.d.:

```text
(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)
```

Each data point is independently and identically distributed. Labels y are given by the external world in advance.

RL data is fundamentally different:

```text
s_0 → a_0 → r_0, s_1 → a_1 → r_1, s_2 → ...
```

It is a time series. The current state depends on the previous action, reward depends on the current state-action pair, and the next state is determined by the environment's transition probability.

That is:

```text
SL: Labels are pre-given, independent of the model.
RL: "Labels" (reward / advantage / Q target) come from environment feedback, dependent on the agent's behavior.
```

This means RL's labels are not static -- they are products of agent-environment interaction. Different actions yield different rewards and states.

### 12.3 Difference Two (The Core): RL's Algorithm Changes Its Own Sampling Distribution

This is the most fundamental difference between RL and SL.

In SL:

```text
Dataset D is fixed.
During training, the optimizer updates θ, but D doesn't change.
The model changes; the data doesn't.
```

In RL:

```text
Data comes from the policy's sampling.
During training, the optimizer updates θ → policy changes → sampling distribution changes → future data changes.
The model changes; the data changes too.
```

Illustrated:

```text
SL structure (open loop):

  Fixed data D → loss → update θ → better model
                  ↑                      |
                  |______________________|
                  (but D stays the same)


RL structure (closed loop):

  policy π_θ → sample data → loss → update θ → new policy π_θ'
      ↑                                            |
      |____________________________________________|
      (policy changed → sampling distribution changed → data distribution changed)
```

This closed loop is the root of all RL complexity.

Because the policy is both "the object being optimized" and "the generator of data." Every step the optimizer takes changes not just the model, but the training data distribution itself.

This creates problems that don't exist in SL:

```text
1. Distribution drift: Old data was generated by old policy; using it to train a new policy introduces bias.
2. Sampling collapse: If the policy becomes deterministic too early, future data becomes very narrow.
3. Bootstrap amplification: Q targets create their own labels, which in turn affect future sampling.
4. Exploration-exploitation tension: Learning well requires diverse data, but a good policy tends to only take "good" actions.
```

### 12.4 An Analogy

Think of it this way:

```text
SL is like a student with a pre-printed textbook. Every study session uses the same book.
   How the student studies doesn't change the textbook.

RL is like a student who writes their own textbook as they learn.
   What they learn affects what practice problems they write next,
   and those problems affect what they learn after that.
   If they form premature biases, the rest of the textbook only contains biased content.
```

### 12.5 How This Difference Explains RL's Many Techniques

Many seemingly "extra" techniques in RL exist to handle this closed-loop problem:

| Technique | What Closed-Loop Problem It Solves |
|---|---|
| Replay Buffer | Break temporal correlation, allow old data reuse, mitigate distribution drift |
| Target Network | Stabilize bootstrap targets, prevent "chasing itself" |
| ε-greedy / Entropy | Prevent sampling collapse, ensure data coverage |
| Importance Sampling | Correct distribution mismatch between old and new policies |
| PPO Clipping | Limit old-new policy divergence, prevent distribution jumps |
| Off-policy methods | Allow using data from non-current policies, improve data efficiency |
| On-policy methods | Avoid distribution mismatch, but data is discarded after use |

These techniques are either nonexistent or unnecessary in SL -- because SL's data distribution is not changed by the model.

### 12.6 Most Compressed Expression

```text
SL: Optimize a model on fixed data.              The algorithm doesn't affect the data distribution.
RL: Optimize a model and data distribution jointly.  The algorithm itself IS the data distribution generator.
```

So RL's real difficulty isn't in loss design -- the loss is similar to SL -- but in:

```text
While optimizing an objective, the measurement method for that objective (the sampling distribution) is also being changed by you.
```

This is why RL needs so many seemingly ad-hoc stabilization tricks: not because the optimization itself is harder, but because the data ground is shifting while you optimize.

---

## 14. RL's Contribution Back to SL/SSL: Bootstrap Learning

We have been using the SL perspective to understand RL. But the reverse is also true: RL made a deep contribution to SL and self-supervised learning (SSL): **bootstrap target construction -- training targets come not from external labels, but from the model's own estimates.**

### 14.1 Comparing Label Sources: SL, RL, SSL

```text
SL: label comes from outside
  Data: (x, y)
  y is human-annotated, training does not change y
  Model passively fits fixed labels

RL: target comes from self-estimation (bootstrap)
  Q(s,a) ← r + γ max_a' Q(s',a')
  Target is not externally given — model constructs it from its own future estimate
  Model creates its own label, then fits that label

SSL: target comes from data's own structure
  Masked LM: mask a token, predict the masked token (label = original token)
  Contrastive: two augmentations of the same image should have similar repr (label = self)
  BYOL/DINO: student chases teacher/momentum model (label = slow version of self)
```

### 14.2 How RL's Bootstrap Idea Was Absorbed by SSL

RL proved something early on:

```text
Even without complete external labels,
as long as you have "current estimate + environment feedback + bootstrapped target,"
you can form an effective learning loop.
```

This idea was widely absorbed by SSL -- only the object of bootstrapping differs:

| Domain | Bootstrap Object | Target Source | What Is Learned |
|---|---|---|---|
| RL / TD | Q / V value | reward + next-state value estimate | behavioral value |
| DQN | Q network | TD target from target network | action value |
| BYOL / DINO | student representation | teacher / momentum encoder | semantic representation |
| Masked LM (BERT) | missing token | original data itself | language structure |
| MAE | missing image patch | original pixels | visual structure |
| Diffusion Model | clean data | predict noise/clean from noisy data | data distribution |

### 14.3 The Shared Underlying Philosophy

```text
SL's worldview: there are god-given labels; the model's job is to fit them.
RL's worldview: no god-given labels; only sampling, feedback, and bootstrap.
SSL's worldview: no human labels; only data's own structure and model-generated targets.

RL and SSL share a core belief:
  A learning system need not depend entirely on external labels.
  It can create its own supervision signal through sampling, perturbation,
  prediction, and bootstrapping.
```

This also aligns perfectly with the RL evolution arc from earlier:

```text
From "god's-eye precise optimization" (DP, known model, known reward)
to "no god's-eye statistical learning" (sampling, preferences, bootstrap, iterative correction)

This path was not walked by RL alone — all of modern ML walked it:
  SL (needs labels) → SSL (no labels, self-constructed targets) → generative models (learn the whole distribution)
```

### 14.4 One-Sentence Summary

```text
RL's deep contribution to modern ML is not just reward and policy,
but that it pushed learning from "fixed-label supervision"
to "bootstrap target construction" very early on.

TD learning's bootstrap idea → SSL's self-constructed targets
Same philosophy, different domains.
```

---

## 15. Final Understanding

RL is not simply optimizing a clean loss. RL is optimizing within a closed loop where sampling, value estimation, and decision-making mutually influence each other.

Early methods used MSE + Bellman + max Q. Later it was discovered that max / argmax is too hard, Q estimates are noisy, off-policy replay has distributional bias, and insufficient exploration worsens estimation.

This led to the introduction of random exploration, target network, double Q, entropy, importance sampling, clipping, soft policy, and actor-critic.

In the end, RL's philosophy shifted from:

```text
Find the optimal Q, then be greedy.
```

Gradually toward:

```text
Under biased sampling, control the data distribution, estimation bias, and optimization step size, so that the policy stably improves.
```

The most refined version is:

```text
The essence of RL is not argmax.
The essence of RL is preference optimization under biased sampling.
```

Or in a more statistical phrasing:

```text
There is no absolute god's-eye view.
Your understanding of the environment depends on the data you sampled, your sampling preferences, and which biases you are willing to accept.
```
