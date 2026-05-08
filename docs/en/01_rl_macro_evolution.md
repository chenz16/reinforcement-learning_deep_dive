# From Bellman to Preference Learning: Rethinking the Essence of Reinforcement Learning

This document traces a research-oriented path through reinforcement learning. Rather than listing algorithms in textbook order, it attempts to answer a more fundamental question: **Why did reinforcement learning, starting from classical dynamic programming, evolve toward Q-learning, DQN, Policy Gradient, Actor-Critic, PPO, SAC, RLHF, and GRPO -- methods that look increasingly "empirical," "statistical," and "preference-driven"?**

The core thesis in one sentence:

**Classical DP / Bellman theory solves for the global optimum under a known model; real-world RL learns a good-enough, stable, preference-aligned policy under limited sampling, noise, bias, and distributional uncertainty.**

These two objectives differ in nature. The former resembles god's-eye analytical optimization; the latter resembles statistical preference learning.

---

## 1. The Bellman Principle Is Correct -- But It Describes an Ideal Limit

In the world of classical dynamic programming, the environment model is known. That is, we know:

$$
P(s'|s,a)
$$

and the reward:

$$
r(s,a)
$$

In this world, the Bellman equation is perfectly natural. For a given policy $\pi$:

$$
V^{\pi}(s)=\mathbb{E}_{a\sim\pi, s'\sim P}\left[r(s,a)+\gamma V^{\pi}(s')\right]
$$

For optimal control:

$$
V^{\ast}(s)=\max_a \mathbb{E}_{s'\sim P}\left[r(s,a)+\gamma V^{\ast}(s')\right]
$$

Or in Q-function form:

$$
Q^{\ast}(s,a)=\mathbb{E}_{s'\sim P}\left[r(s,a)+\gamma \max_{a'}Q^{\ast}(s',a')\right]
$$

These are all mathematically rigorous. The problem is not that the Bellman principle is wrong, but that: **this equation describes an ideal fixed point, while the training process is not a fixed point.**

At the final static solution, $\max$ is a correct conclusion; but during learning, what you have is not $Q^{\ast}$, but a noisy, biased, constantly changing estimate:

$$
\hat Q = Q^{\ast} + \epsilon
$$

So the training process actually becomes:

$$
Q_{t+1}=r+\gamma \max(\hat Q_t)
$$

This is no longer a clean analytical optimization problem -- it is a **noisy nonlinear recursive iteration system**.

---

## 2. The Relationship Between V and Q: Q Simply Adds an Action Condition, But the Implications Are Enormous

By definition, V and Q both describe future cumulative returns.

$$
V^{\pi}(s)=\mathbb{E}_\pi[G_t|s_t=s]
$$

$$
Q^{\pi}(s,a)=\mathbb{E}_\pi[G_t|s_t=s,a_t=a]
$$

So Q can be understood as V with an additional action dimension. More precisely: **Q makes the action condition explicit.**

V records:

$$
s \rightarrow value
$$

Q records:

$$
(s,a) \rightarrow value
$$

This seems like merely "one extra dimension of bookkeeping," but it is enormously significant for control problems. The core of a control problem is not just judging whether a state is good, but determining: **which action is better in this state.**

For V, the action is hidden inside the Bellman operator:

$$
V^{\ast}(s)=\max_a \left(r(s,a)+\gamma\mathbb{E}[V^{\ast}(s')]\right)
$$

Here both reward and transition depend on the action, so without an environment model, you cannot directly infer from V which action is best.

But Q fixes the action:

$$
Q(s,a)=r(s,a)+\gamma\mathbb{E}[V(s')]
$$

Then policy improvement becomes straightforward:

$$
a^{\ast}=\arg\max_a Q(s,a)
$$

This is the enormous value of Q-functions: **they make action optimization explicit.**

But the cost is also clear. Q is finer-grained than V, has higher estimation variance, and is more prone to local noisy peaks. Layer on $\max$, and estimation errors get amplified.

---

## 3. Why V Came First: Not Because People Didn't Know About Qmax, But Because V Is More Stable

In hindsight, Qmax seems more natural for control: since you ultimately need to choose actions, why not learn $Q(s,a)$ and do $\arg\max$ from the start?

The reason is that early dynamic programming and control theory favored:

* Known models;
* Exactly computable expectations;
* Finding a globally optimal fixed point;
* Stability and provability as paramount concerns.

In this world, V-functions are natural because they average over action outcomes. For a given policy, V's Bellman evaluation is:

$$
V^{\pi}(s)=\sum_a \pi(a|s)\sum_{s'}P(s'|s,a)\left[r(s,a)+\gamma V^{\pi}(s')\right]
$$

This operator is fundamentally averaging. Averaging is smooth, stable, and easy to prove contraction for.

$\max$ is winner-take-all. It is not smooth, is noise-sensitive, and biases toward overestimated actions.

So historically, it was not that "people didn't think of Qmax," but rather: **Qmax is too aggressive -- it requires sufficient theory, data, computation, and engineering before it becomes controllable.**

---

## 4. Qmax Is an Unbridled Horse: Data-Efficient, But Amplifies Noise

The core of Q-learning is:

$$
Q(s,a)\leftarrow r+\gamma\max_{a'}Q(s',a')
$$

This is a powerful formula. It means: even if the actual rollout didn't take the optimal action, the update still assumes the future will choose optimally.

This is why Q-learning is data-efficient: **a high-value signal can be rapidly propagated forward.**

But this is also where the problem lies. During training, Q is an estimate, not the true value:

$$
\hat Q=Q+\epsilon
$$

Therefore:

$$
\max(\hat Q)=\max(Q+\epsilon)
$$

$\max$ preferentially selects the action with the largest positive noise. It treats accidental overestimation, lucky samples, and network hallucinations as genuine preferences.

This creates a feedback loop:

1. An action gets overestimated due to noise;
2. $\max$ selects it;
3. It enters the Bellman target;
4. The target is written back to the Q-network;
5. Future updates continue propagating this overestimation;
6. The system may become increasingly biased, even divergent.

This is the essence of overestimation. The Bellman fixed point isn't wrong -- **noisy max iteration** is the problem.

Think of it this way:

**Bellman optimality is correct at the static fixed point; but the learning process is multi-round noisy strongly-nonlinear max iteration, and the system easily self-excites into instability.**

This is very different from ordinary ODEs or gradient descent. An averaging system smooths out noise; a max-bootstrap system actively picks the noisiest direction and recursively propagates it.

---

## 5. Averaging Is Too Conservative, Max Is Too Aggressive: The Core Tension of RL

V / policy evaluation methods are more like averaging:

$$
V(s)=\sum_a\pi(a|s)Q(s,a)
$$

Stable, because noise across different actions cancels out. But also conservative, because it doesn't actively amplify the best action.

Qmax is the opposite:

$$
\max_a Q(s,a)
$$

Data-efficient, fast-propagating, able to quickly exploit sparse rewards -- but also amplifies errors.

So RL's core tension can be written as:

| Approach | Strengths | Weaknesses |
| --- | --- | --- |
| averaging / expectation | Stable, smooth, low variance | Slow learning, weak optimization |
| max / greedy | Data-efficient, fast preference propagation | Overestimation, instability, prone to collapse |

This is why modern RL doesn't simply pick one side, but continuously seeks a middle ground:

**Retain greedy's optimization power without letting the greedy operator blow up the system.**

---

## 6. DQN's Significance: Not Inventing Q-Learning, But Taming Qmax

DQN is essentially deep Q-learning. It didn't invent Q-learning, nor did it change the core Bellman target:

$$
r+\gamma\max_{a'}Q(s',a')
$$

DQN's true significance is: **the first reasonably stable combination of deep neural networks with off-policy Q-learning.**

It tamed Qmax through several key engineering and algorithmic designs:

1. **CNN encoder**: mapping high-dimensional image states to features;
2. **Replay buffer**: breaking sample correlation, improving data reuse;
3. **Target network**: preventing the target from chasing itself, reducing bootstrap instability;
4. **Off-policy learning**: allowing past data to train the current value function.

But DQN still didn't cure $\max$ bias. The same Q-network selects actions and evaluates them -- "picking the champion and scoring the champion" -- leading to overestimation.

Then came Double DQN:

* Online network handles action selection;
* Target network handles action evaluation.

The essence is reducing the max operator's amplification of noise.

Later came Dueling DQN, decomposing:

$$
Q(s,a)=V(s)+A(s,a)
$$

This isn't abandoning Q, but using V to represent overall state value and advantage to represent relative action differences, making estimates more stable.

So the major thread after DQN can be understood as:

**Retaining Qmax's aggressive optimization while putting reins on it.**

---

## 7. The Essential Difference Between RL and DP: Not Different Equations, But Different Worlds

In DP's world, expectations are exactly computable:

$$
\mathbb{E}[\cdot]
$$

is a true expectation.

In RL's world, expectations can only be estimated through sampling:

$$
\hat{\mathbb{E}}[\cdot]
$$

This brings a fundamental change.

DP is:

**Deterministic fixed-point solving under a known model.**

RL is:

**Recursive optimization under unknown models with noise, bias, and distributional uncertainty.**

This is not a minor difference -- it changes the nature of the objective.

In DP, Bellman optimality is the mathematical expression of the global optimal solution. In RL, Bellman optimality is more of a guiding principle. Because in reality:

* Data is limited;
* Rollouts are expensive;
* Exploration is insufficient;
* Rewards are sparse;
* Environments are stochastic;
* Function approximation introduces errors;
* Replay buffer distributions don't match the current policy distribution.

Therefore, RL's primary goal is often not "find the absolute optimum," but rather:

**Under limited sampling, quickly form a stable, useful, statistically superior behavioral preference.**

This is the essence:

**RL is bias optimization under noise and distributional uncertainty.**

More completely:

**RL is recursive preference optimization under noise, bias, and distribution drift.**

---

## 8. From God's-Eye View to Statistical View

Classical control / DP is more like a god's-eye view. It assumes the world has a true model, true transition probabilities, a true reward function, and true optimal $Q^{\ast}$, $V^{\ast}$. The algorithm's task is to compute these true values.

The statistical perspective is entirely different. In statistics, there is no directly observable truth. You only have:

* samples;
* assumptions;
* likelihood;
* prior;
* estimator.

Maximum likelihood estimation isn't seeking cosmic truth -- it asks:

**What parameters best explain the current observed data?**

As RL has evolved, it has increasingly approached this statistical worldview. Policy gradient, RLHF, and GRPO methods don't insist on a globally consistent, Bellman-consistent $Q^{\ast}$. They care more about:

**Which behaviors brought better rewards / preferences in sampling statistics, and therefore should occur more frequently in the future.**

This is the shift from "finding true values" to "shaping behavioral distributions."

---

## 9. Policy Gradient: The Watershed From Value-Centric to Behavior-Centric RL

Q-learning / DQN is fundamentally value-centric:

$$
Q^{\ast}(s,a)
$$

It believes there exists an objectively optimal action-value function, and approaches it through Bellman consistency.

Policy Gradient is entirely different. It directly optimizes the policy:

$$
\pi_\theta(a|s)
$$

The objective is:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

The key change here is:

**Actions are no longer selected via $\arg\max Q$, but sampled from the policy distribution.**

The policy update takes the form:

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log\pi_\theta(a|s) A(s,a)\right]
$$

The meaning is intuitive:

* If an action's advantage is positive, increase its future probability;
* If the advantage is negative, decrease its future probability.

This is not hard max, but probability reshaping.

Q-learning says:

> The action with the highest current estimated value should be selected.

Policy Gradient says:

> Behaviors that perform statistically better should occur more frequently in the future.

This is a philosophical turn:

**From god's-eye optimal value solving to sampling-statistics-based behavioral preference shaping.**

---

## 10. The "Two Layers of Probability" in Policy Gradient

The statistical essence of Policy Gradient can be understood through "two layers of probability."

The first layer is rollout occurrence probability:

$$
a\sim\pi_\theta(a|s)
$$

Which behaviors get sampled depends on the current policy's probability distribution.

The second layer is update weighting:

$$
\nabla\log\pi(a|s)\cdot A(s,a)
$$

Sampling an action doesn't mean it should be reinforced. It must also be weighted by advantage.

So Policy Gradient doesn't simply copy sampled behaviors, but rather:

**Within the actual sampling distribution, re-weights behaviors by reward / advantage as preferences.**

This closely mirrors statistical estimation: without a god's-eye true value, you can only adjust parameters through samples and assumptions.

---

## 11. Actor-Critic: The Division of Labor Between Policy and Value

Policy Gradient directly optimizes the policy, but still needs to estimate advantage. Hence Actor-Critic.

The Actor is:

$$
\pi_\theta(a|s)
$$

Responsible for decision-making.

The Critic can take different forms:

| Critic Type | What It Learns | Typical Algorithms |
| --- | --- | --- |
| V-critic | $V(s)$ | A2C, A3C, PPO |
| Q-critic | $Q(s,a)$ | DDPG, TD3, SAC |
| Advantage critic | $A(s,a)$ | Some direct advantage methods |

PPO / A2C typically use a V-critic. They estimate:

$$
A_t \approx r_t+\gamma V(s_{t+1})-V(s_t)
$$

Here the critic is more of a statistical estimator or control variate to reduce policy gradient variance. It's not trying to rigorously find a globally optimal $Q^{\ast}$.

DDPG / TD3 / SAC retain a Q-critic:

$$
Q(s,a)
$$

The actor optimizes actions through the critic. Especially in continuous control, the actor can use:

$$
\nabla_a Q(s,a)
$$

to adjust action direction.

So Actor-Critic isn't "abandoning Q," but separating:

* policy optimization;
* value estimation.

The Critic can be V or Q.

---

## 12. V-Only Actor-Critic Is a Relaxation of Qmax

PPO-style methods typically don't maintain an explicit Q-network, but use a V-network to estimate baselines and rollout returns or GAE to estimate advantage.

This can be understood as a relaxation of hard $Qmax$.

Q-learning is:

$$
\arg\max_a Q(s,a)
$$

PPO is:

$$
\nabla_\theta\log\pi_\theta(a|s)A(s,a)
$$

The former is hard optimization; the latter is soft preference adjustment.

The costs are clear:

* More dependent on sampling;
* Typically lower data efficiency;
* If good actions are rarely sampled, policy gradient struggles to discover them;
* Advantage estimates have variance and bias.

So PPO is more stable, but not more "globally optimal." What it pursues is:

**Stably pushing the policy toward better directions under limited sampling.**

---

## 13. This Isn't "Regression" -- The Objective Function Has Changed

From Qmax to policy gradient, it's not just the algorithmic form that changes -- the optimization objective changes.

Q-learning's objective is:

**Approximate a god's-eye optimal Q-value.**

Policy Gradient's objective is:

**Make behaviors with better reward / preference in sampling occur more frequently in the future.**

The two losses have different essences.

Q-learning's Bellman loss is a consistency loss:

$$
\left(Q(s,a)-\left(r+\gamma\max_{a'}Q(s',a')\right)\right)^2
$$

It pursues value function self-consistency.

Policy Gradient's loss is more like behavior distribution shaping:

$$
-\log\pi(a|s)A(s,a)
$$

It pursues increasing good behavior probabilities and decreasing bad ones.

One is value truth fitting; the other is preference-weighted behavior update.

---

## 14. Argmax as Conclusion vs. Argmax as Training Process

This is a critical distinction.

In Bellman optimality, $\arg\max$ is the final conclusion:

> If you already know the true $Q^{\ast}$, then choosing $\arg\max Q^{\ast}$ is correct.

But during training, $\arg\max$ becomes the optimization action at every step:

> Even though your current Q is noisy, you forcefully select the current maximum.

These two are completely different.

An analogy:

* Bellman optimality's $\arg\max$: knowing that daily running is good for health when you're already fit;
* Hard greedy during training: doing extreme running every day when your body is in poor condition -- overshooting, getting injured, collapsing.

So many modern RL methods essentially make the optimization target "softer":

* PPO uses clipping / trust region to prevent policy from changing too much in one step;
* SAC uses entropy to prevent premature policy collapse;
* Double Q / TD3 uses conservative estimates to suppress overestimation;
* RLHF uses KL regularization to stay close to the reference policy.

This isn't denying optimality, but acknowledging:

**Under limited data, you cannot treat theoretical optimality conditions as hard execution rules at every training step.**

---

## 15. Modern RL: Greedy Optimization with Brakes

Many modern RL methods can be understood as:

**Greedy is necessary, but must come with brakes.**

Why must greedy exist? Because data is scarce. Without greedy amplification, high-reward signals propagate too slowly, and the model may learn nothing at all.

Why must there be brakes? Because greedy amplifies noise, bias, and function approximation error.

So modern algorithms make various trade-offs:

| Algorithm / Technique | Essential Function |
| --- | --- |
| Target Network | Reduce bootstrap target drift |
| Replay Buffer | Improve data efficiency, reduce sample correlation |
| Double Q | Reduce max overestimation |
| Dueling Network | Stabilize Q's structural decomposition |
| PPO Clip | Limit overly aggressive policy updates |
| TRPO / KL constraint | Trust region, prevent policy jumps |
| Entropy Regularization | Prevent premature deterministic collapse |
| TD3 clipped double Q | Use conservative estimates to suppress overestimation |
| SAC | Soft greedy + entropy + Q critic |
| CQL | Actively suppress Q for OOD actions in offline settings |

What these methods share:

**Don't abandon optimization, but prevent the optimizer from exploiting estimation noise.**

---

## 16. RLHF / GRPO: Ad-Hoc Preferences Aren't a Bug -- They're the True Form of Human Preferences

With LLM RLHF / GRPO methods, this becomes even more apparent.

Human preferences are inherently not a clean, globally consistent, Bellman-consistent reward function.

Human preferences are often:

* noisy;
* contextual;
* local;
* heuristic;
* sometimes contradictory;
* dependent on prompt, culture, task, and evaluator state.

So in this setting, forcing a globally optimal $Q^{\ast}$ may itself be a fiction.

GRPO / RLHF methods are more like:

**From sampled responses, performing relative preference re-weighting based on human preferences or a reward model.**

They're not seeking an absolute optimal control law, but doing:

**Preference distribution shaping.**

This is why many details appear ad-hoc:

* reference model;
* KL penalty;
* reward normalization;
* group relative advantage;
* clipping;
* rejection sampling;
* pairwise ranking.

But ad-hoc isn't necessarily a flaw, because human preferences are themselves ad-hoc. They're not analytical mathematical objects, but statistical sampling objects.

So GRPO / RLHF's "ad-hoc" nature actually fits the essence of preference sampling.

---

## 17. From DP to RLHF: A Grand Historical Arc

The entire evolution can be compressed into the following line:

### Phase 1: DP / V-function

Goal: stable evaluation and globally optimal solving under known models.

Key characteristics:

* Exact expectations;
* Stable V-function;
* Clean Bellman fixed point;
* But action optimization isn't direct.

### Phase 2: Q-learning / Qmax

Explicit action optimization begins.

Key characteristics:

* High data efficiency;
* Off-policy;
* Fast greedy propagation;
* But severe overestimation and instability.

### Phase 3: DQN Family

Extending Q-learning with deep networks, taming Qmax through replay / target networks / Double Q.

Key characteristics:

* Scalable;
* Still value-centric;
* Still centered on Bellman consistency;
* But heavy use of engineering stabilization tricks.

### Phase 4: Policy Gradient / Actor-Critic

Directly optimizing the policy distribution.

Key characteristics:

* Shift from value-centric to behavior-centric;
* No longer insisting on perfect Q;
* Adjusting behavior probabilities through advantage;
* More statistical, softer, more stable, but lower data efficiency.

### Phase 5: PPO / SAC / TD3

Fusing value and policy, balancing greediness and stability through constraints and regularization.

Key characteristics:

* PPO: stable policy updates;
* TD3: conservative Q critic;
* SAC: soft Q + stochastic policy + entropy;
* All addressing the tension between overly aggressive greediness and sampling noise.

### Phase 6: RLHF / GRPO

Reinforcement learning further becomes preference distribution shaping.

Key characteristics:

* Reward comes from human preferences or reward models;
* Preferences are not necessarily globally consistent;
* Algorithms resemble statistical preference learning;
* The goal is not absolute optimality, but placing the behavioral distribution within the zone of human preferences.

---

## 18. Final Summary: What Is the Essence of RL?

In one sentence:

**The essence of RL is not the Bellman equation itself, but how to stably amplify useful preferences under unknown environments, limited sampling, noisy bias, and distributional uncertainty -- while preventing the optimizer from exploiting its own estimation errors.**

More specifically:

1. DP is exact optimization under known models; RL is sampling-based optimization under unknown models.
2. Bellman optimality is an ideal fixed point; the training process is noisy nonlinear recursive iteration.
3. V is stable but conservative; Q enables explicit action optimization but more easily amplifies errors.
4. Qmax is data-efficient because it rapidly propagates high values; but it also preferentially selects positive noise.
5. Modern RL's core is not simply pursuing global optimality, but learning stable, useful, statistically superior policies under limited data.
6. Policy Gradient is a critical watershed: from finding the optimal value function to directly shaping behavior distributions.
7. Actor-Critic is the fusion of value estimation and policy optimization -- not simply "using Q" or "not using Q."
8. PPO / SAC / TD3 are essentially greedy optimization with brakes.
9. RLHF / GRPO further demonstrate: many real-world preferences are inherently ad-hoc, statistical, and context-dependent -- not suited for forcing into a globally Bellman-consistent true value function.
10. Therefore, modern RL more closely resembles statistical preference learning than classical analytical optimal control.

The entire thread can be compressed into one powerful statement:

**DP pursues the global optimal solution; RL pursues evolving good-enough behavioral preferences from limited, noisy, biased data.**

And this explains why RL's history has always oscillated:

* Too much averaging -- can't learn;
* Too much greediness -- collapses;
* The final answer is not choosing one side, but:

**Use greediness to amplify effective signals; use statistical constraints and regularization to prevent the system from being carried away by noise.**
