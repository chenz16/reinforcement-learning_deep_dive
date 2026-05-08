# Connections Between Reinforcement Learning and Other Models

> **Positioning**: This article discusses how RL does not exist in isolation -- it can be trained from scratch, fine-tuned on top of SL base models, combined with Diffusion/Flow Matching models, and even transferred between Sim and Real environments. From the perspective of model connections, this article outlines how RL interfaces with other paradigms across different scenarios.

---

## 1. Pure RL Training from Scratch

The most classic RL scenario: no pretrained base model, the model starts from random initialization and learns purely through interaction with the environment.

There are two approaches:

### 1.1 Learning Value Preferences (Value-based)

The model learns: what is the long-term value of each state-action pair.

```text
Q(s,a) → estimate of future returns
```

Representative algorithms: Q-learning, DQN, Double DQN, Dueling DQN

Decision method: argmax Q(s,a)

Essence: **Learn valuations first, then extract behavior from valuations**. The model compresses the value landscape.

### 1.2 Learning Action Preferences (Policy-based)

The model learns: at each state, what action should be taken with what probability.

```text
π(a|s) → behavior preference distribution
```

Representative algorithms: REINFORCE, PPO, SAC

Decision method: sample from π(a|s)

Essence: **Directly learn the preference distribution**. The model compresses the behavior preference.

### 1.3 Combining Both (Actor-Critic)

In practice, most modern RL algorithms are Actor-Critic:

```text
Critic: learns value (V or Q), provides preference signal
Actor: learns behavior distribution, receives guidance from Critic
```

Representative algorithms: A2C, DDPG, TD3, SAC, PPO (V-critic version)

Pure RL training from scratch is suitable for: game AI (Atari, Go), robot control, low-level control for autonomous driving.

### 1.4 Reward Sources: Must Be Prepared Before Training

The RL training loop requires reward signals, but **reward is not learned during RL training -- it must be prepared beforehand.** There are three sources:

```text
Source 1: Environment-provided reward (cleanest)
  Game scores, distance/collision/energy in physics simulation
  → Environment directly returns a scalar reward
  → No extra preparation needed

Source 2: Rule-based reward (engineered)
  Math answer correctness, code compilation pass rate, format compliance
  → Manually define a rule function R(output) → scalar
  → Commonly used with GRPO

Source 3: Reward Model learned from preference data (statistical approximation)
  Collect human preference pairs: (chosen, rejected)
  → Train a Reward Model to approximate human preferences
  → RM is essentially an SL model: input (prompt, response), output scalar score
  → Standard approach for PPO-RLHF
```

Key understanding:

```text
The Reward Model is not part of RL -- it is a precondition for RL training.
It is trained via SL on preference data, then frozen as a "scorer" during RL training.
```

Comparison of the three sources:

| Reward Source | Preparation | Pros | Cons | Typical Use |
|---|---|---|---|---|
| Environment-provided | None needed | Clean, unbiased | Only for environments with clear numerical feedback | Games, simulation |
| Rule-based | Manual rule writing | Precise, interpretable | Hard to cover complex preferences | Math/code verification, GRPO |
| Reward Model | Collect preferences + SL training | Captures fuzzy preferences | RM itself has bias and noise | RLHF, InstructGPT |

### 1.5 Architecture Diagram: Which Modules Are Used for Training vs. Inference

Pure RL architectures come in three typical patterns. The key distinction: **training requires more modules than inference -- the Critic exists only during training and is discarded at inference time.**

#### Pattern A: Value-based (DQN)

```text
┌─────────────────────────────────────────────────────────┐
│                Used for Training + Inference             │
│                                                         │
│   Observation ──→ ┌──────────────┐ ──→ ┌─────────────┐  │
│   (image/state)   │   Backbone   │     │  Q Head     │  │
│                    │  (CNN/MLP)   │     │ outputs Q   │  │
│                    │              │     │ for each    │  │
│                    └──────────────┘     │ action      │  │
│                                        └──────┬──────┘  │
│                                               │         │
│                                          argmax Q       │
│                                               │         │
│                                        select action    │
└─────────────────────────────────────────────────────────┘

Inference: entire network used (Backbone + Q Head + argmax)
Training:  same network + target network (frozen copy)
No separate policy network -- policy is implicit in argmax Q
```

#### Pattern B: Actor-Critic with Shared Backbone (common in PPO / A2C)

```text
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    ┌──────────────┐                          │
│   Observation ──→  │   Shared     │                          │
│   (image/state)    │   Backbone   │                          │
│                    │  (CNN/MLP)   │                          │
│                    └──────┬───────┘                          │
│                           │                                 │
│                     ┌─────┴──────┐                          │
│                     │            │                          │
│               ┌─────▼─────┐  ┌──▼──────────┐               │
│               │ Policy    │  │ Value       │               │
│               │ Head      │  │ Head        │               │
│               │           │  │             │               │
│               │ outputs   │  │ outputs V(s)│               │
│               │ π(a|s)    │  │ (scalar)    │               │
│               │ (softmax/ │  │             │               │
│               │  Gaussian)│  │             │               │
│               └─────┬─────┘  └──────┬──────┘               │
│                     │               │                      │
│                 ┌───┴───┐     ┌─────┴──────┐               │
│   Inference ✓   │sample │     │ compute    │  Training only ✗│
│                 │action │     │ advantage  │               │
│                 └────────┘     └────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Inference: only Backbone + Policy Head → output action
Training:  Backbone + Policy Head + Value Head → advantage guides policy update
Value Head is discarded at deployment
```

#### Pattern C: Actor-Critic with Separate Networks (common in DDPG / TD3 / SAC)

```text
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ┌─── Actor Network (used at inference ✓) ──────────────┐       │
│   │                                                      │       │
│   │  Observation ──→ ┌────────────┐ ──→ ┌─────────────┐  │       │
│   │                  │  Actor     │     │ Policy Head  │  │       │
│   │                  │  Backbone  │     │ outputs      │  │       │
│   │                  │  (MLP)     │     │ action       │  │       │
│   │                  │            │     │ (μ(s) or     │  │       │
│   │                  │            │     │  π(a|s))     │  │       │
│   │                  └────────────┘     └──────────────┘  │       │
│   └──────────────────────────────────────────────────────┘       │
│                                                                  │
│   ┌─── Critic Network (training only ✗) ─────────────────┐       │
│   │                                                      │       │
│   │  (s, a) ──→ ┌────────────┐ ──→ ┌────────────────┐   │       │
│   │              │  Critic    │     │ Q Head         │   │       │
│   │              │  Backbone  │     │ outputs Q(s,a) │   │       │
│   │              │  (MLP)     │     │ (scalar)       │   │       │
│   │              └────────────┘     └────────────────┘   │       │
│   │                                                      │       │
│   │  TD3/SAC typically have two Critics (double Q)       │       │
│   └──────────────────────────────────────────────────────┘       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

Inference: only Actor Network → output action
Training:  Actor + Critic → Critic's Q gradient guides Actor updates
Entire Critic Network is discarded at deployment
```

#### Summary: Training vs. Inference Module Comparison

| Architecture | Used During Training | Used During Inference | Discarded at Inference |
|---|---|---|---|
| **DQN** | Backbone + Q Head + Target Network | Backbone + Q Head | Target Network |
| **PPO (shared)** | Shared Backbone + Policy Head + Value Head | Shared Backbone + Policy Head | Value Head |
| **SAC/TD3 (separate)** | Actor Net + Critic Net(s) | Actor Net | Entire Critic Net |

Core principle:

```text
The Critic / Value Head is training "scaffolding" --
it provides gradient signals during training (advantage / Q gradient),
but is completely unnecessary at inference and is discarded.

The only thing deployed to production is the Policy / Actor network.
```

#### Value and Policy Are Two Independent Iteration Processes During Training

In Actor-Critic architectures, although the Critic and Actor are updated within the same training loop, they are **two independent optimization processes**, each with its own loss:

```text
Within the same training step:

  Step 1: Critic update (independent)
    loss_critic = (V(s) - V_target)^2     or  (Q(s,a) - Q_target)^2
    Only updates Critic parameters
    Goal: make value estimates more accurate

  Step 2: Actor update (independent)
    loss_actor = - log π(a|s) · A(s,a)    or  - Q(s, μ(s))
    Only updates Actor parameters
    Goal: make the policy better

  Two losses backpropagate separately, updating their own parameters.
  They share the same batch of data, but the optimization processes are isolated.
```

This isolation is important because:

```text
1. Critic's goal is accurate estimation (regression problem)
2. Actor's goal is better behavior (preference optimization problem)
3. The two losses are fundamentally different in nature -- mixing them would cause interference
4. TD3 deliberately makes Actor update less frequently than Critic (delayed policy update)
   → Let Critic stabilize first, then guide Actor
```

Even in shared-backbone architectures (PPO/A2C), the gradients for Policy Head and Value Head are computed and applied separately. The shared Backbone parameters receive gradients from both directions, but the two Heads' updates are still driven by independent losses.

#### Q/V and Policy: Coupled in Sampling, Decoupled in Gradients

This is an easily confused point. Q/V and Policy are both coupled and decoupled, but at different levels:

**At the sampling level: coupled**

Q/V's training data comes from policy rollouts. The policy determines what trajectories the agent takes, what states it sees, what actions it performs, what rewards it receives. So:

```text
policy changes → sampling distribution changes → Q/V's training data changes → Q/V changes too

Q/V and policy are indirectly coupled through the sampling process.
```

This is the essence of the RL closed-loop problem -- the "algorithm changes its own data distribution" discussed earlier.

**At the gradient level: decoupled**

But within any given training step's gradient computation, Q/V is a **constant** with respect to the policy gradient -- it does not participate in gradient computation:

```text
Policy gradient:
  ∇_θ J = E[ ∇_θ log π_θ(a|s) · A(s,a) ]
                  ↑                  ↑
          differentiate w.r.t. θ    A is a constant, not differentiated w.r.t. θ
                                    (A comes from Q/V, but Q/V's parameters don't
                                     participate in this gradient)

In other words:
  - log π_θ(a|s) is differentiated w.r.t. policy parameters θ  ✓
  - A(s,a) comes from Critic, but is treated as a fixed weight  ✗ (stop gradient)
```

Specifically for different algorithms:

```text
PPO:
  loss = - clip(ratio, 1-ε, 1+ε) · A(s,a)
  A(s,a) = r + γV(s') - V(s)
  → Backprop only differentiates w.r.t. π_θ; V's output is detached / stop_gradient

DDPG/TD3 (max Q loss — exception):
  loss_actor = - Q(s, μ_θ(s))
  → Here gradients DO flow through Q network to the action, then to the actor
  → But Q network's parameters are still NOT updated (frozen / detached)
  → Only actor parameters θ are updated

SAC:
  loss_actor = E[ α·log π(a|s) - Q(s,a) ]
  → Q's output participates in forward computation, but Q's parameters receive no gradients
```

So the complete understanding is:

```text
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Sampling phase:  Policy ←──coupled──→ Q/V                 │
│                   (policy determines data,                 │
│                    data affects Q/V learning)               │
│                                                            │
│  Gradient phase:  Policy ←──decoupled──→ Q/V               │
│                   (when updating policy, Q/V output         │
│                    is a constant)                           │
│                   (when updating Q/V, policy output         │
│                    is a constant)                           │
│                                                            │
│  They collaborate indirectly through alternating iteration: │
│    1. Sample data using current policy                      │
│    2. Update Q/V using the data (policy params frozen)      │
│    3. Update policy using updated Q/V (Q/V params frozen)   │
│    4. Go back to 1                                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

This also explains why Actor-Critic can be unstable -- two independent optimizers are chasing each other in alternating iteration: the Critic chases the policy's changing data distribution, while the Actor chases the Critic's changing value landscape. If one runs too fast, the other gets dragged off course.

---

## 2. RL Preference Fine-Tuning on SL Base Models

This is the core paradigm of LLM alignment: first train a powerful base model with supervised learning, then use RL methods for preference fine-tuning.

### 2.1 Overall Pipeline

```text
Phase 1: Pretraining (SL)
  Large-scale corpus → autoregressive language modeling → base model (pretrained LLM)

Phase 2: Supervised Fine-Tuning SFT (SL)
  Instruction-response pairs → supervised learning → SFT model

Phase 3: Preference Fine-Tuning (RL)
  Human preference data → RL methods → aligned model
```

Phase 3 RL methods have multiple implementations, with the core difference being which components are needed and how parameters are updated.

### 2.2 PPO Approach (InstructGPT Route)

PPO is the most classic LLM RL fine-tuning approach. It requires four models simultaneously in memory:

```text
┌─────────────────────────────────────────────────────────────┐
│  1. Policy Model (the LLM being trained)                    │
│  2. Reference Model (frozen copy of SFT model, for KL       │
│     constraint)                                             │
│  3. Reward Model (trained from preference data, frozen)     │
│  4. Value Head / Critic (estimates V(s), provides baseline) │
│     Usually initialized from Reward Model (not from scratch)│
└─────────────────────────────────────────────────────────────┘
```

Training loop:

```text
1. Policy generates responses
2. Reward Model scores them
3. Critic estimates baseline → compute advantage
4. PPO loss updates Policy (clipped ratio × advantage)
5. KL penalty prevents Policy from drifting too far from Reference
```

**Reward Model** is almost always a separate model. Typically initialized from an SFT checkpoint, replacing the language modeling head with a scalar reward head, trained on human preference data and then frozen.

**Critic (Value Function)** is usually initialized from the Reward Model -- because the Reward Model has already learned quality representations, and initializing from scratch would cause early training instability.

### 2.3 GRPO Approach (DeepSeek Route)

GRPO eliminates the Critic, greatly simplifying the architecture:

```text
┌──────────────────────────────────────────────────────────────┐
│  1. Policy Model (the LLM being trained)                     │
│  2. Reference Model (frozen copy of SFT model, for KL        │
│     constraint)                                              │
│  3. Reward source (Reward Model or rule-based verifier)      │
│     No Critic / Value Network needed                         │
└──────────────────────────────────────────────────────────────┘
```

Core mechanism:

```text
1. For each prompt, Policy generates a group of G responses
2. Reward scores each response
3. Within-group relative ranking serves as baseline (no Critic needed)
4. Policy gradient updates Policy
```

DeepSeek-R1 uses **full fine-tuning** (not LoRA) with GRPO, training all parameters.

GRPO's reward source can be:
- A trained Reward Model
- Rule-based verifiers (e.g., answer correctness checks for math problems, compilation/test pass rates for code)

### 2.4 DPO Approach (Stanford Route)

DPO goes even further, eliminating even the Reward Model:

```text
┌──────────────────────────────────────────────────────────────┐
│  1. Policy Model (the LLM being trained)                     │
│  2. Reference Model (frozen copy of SFT model)               │
│     No Reward Model needed, no Critic needed                 │
│     Trains directly on preference pairs                      │
└──────────────────────────────────────────────────────────────┘
```

DPO directly optimizes the policy on (chosen, rejected) preference pairs, using a classification-style loss:

```text
L_DPO = -log σ(β · (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
```

### 2.5 Parameter Update Methods: Full vs LoRA vs Adapter

Regardless of whether using PPO, GRPO, or DPO, there are choices for how to update the Policy Model's parameters:

| Method | Approach | Pros | Cons |
|---|---|---|---|
| **Full Fine-Tuning** | Update all parameters | Maximum expressiveness | Huge memory consumption, 4x model |
| **LoRA** | Freeze base, only train low-rank matrices A·B | Significantly reduced memory; Reference Model is free (i.e., the base itself) | Expressiveness limited by rank |
| **QLoRA** | Quantized base (4-bit) + LoRA | Can run RLHF on consumer GPUs | Quantization introduces precision loss |
| **Prefix Tuning** | Only train prefix token embeddings | Very few parameters | Performance typically inferior to LoRA |

**Key advantage of LoRA in RL fine-tuning**: The Reference Model does not need separate storage. Since the base parameters are frozen, a forward pass without the LoRA adapter produces the Reference Model's output. This eliminates the largest memory overhead among PPO's four models.

Typical choices in practice:

```text
PPO:   Full fine-tuning (InstructGPT) or LoRA (when resources are limited)
GRPO:  Full fine-tuning (DeepSeek-R1)
DPO:   LoRA is most common (DPO + LoRA is the most memory-efficient alignment approach)
```

### 2.6 Overall Comparison of Approaches

| Approach | Required Models | Bellman Equation | Preference Source | Complexity |
|---|---|---|---|---|
| PPO | Policy + Ref + RM + Critic (4) | Still used internally by Critic | Reward Model | High |
| GRPO | Policy + Ref + Reward source (2-3) | Not used at all | RM or rule-based verifier | Medium |
| DPO | Policy + Ref (2) | Not used at all | Preference pair data | Low |

---

## 3. RL with Diffusion / Flow Matching Models

Diffusion and Flow Matching models are fundamentally generative models -- they generate samples from noise through a multi-step denoising/flow process. When used as policies, RL can fine-tune them.

### 3.1 Diffusion Models as Policies

**Core idea**: Treat the multi-step denoising process of diffusion as a policy, where each denoising step is an action.

**Diffuser** (Janner et al., 2022): Pioneered using diffusion models for trajectory planning. The model generates complete state-action trajectories, using reward guidance at inference time to steer generation toward high-return trajectories.

**Diffusion Policy** (Chi et al., 2023): Uses DDPM to represent visuomotor policies. Outputs action chunks (sequences of continuous actions); diffusion's multimodal modeling capability allows it to handle diverse demonstration data better than Gaussian policies.

**Decision Diffuser** (Ajay et al., 2023): Uses classifier-free guidance for conditional generation, conditioning on return, constraints, and skills.

### 3.2 RL Fine-Tuning of Diffusion Models

Key challenge: reward is only available on the final generated output, but diffusion's "policy" spans T denoising steps.

**DDPO -- Denoising Diffusion Policy Optimization** (Black et al., 2023):

```text
Core idea: model the denoising process as an MDP
- State: current noisy image x_t
- Action: output of one denoising step
- Episode: one complete denoising chain x_T → x_0
- Reward: given only on the final x_0

Uses REINFORCE / PPO to estimate policy gradient for each step
No need to backpropagate through the entire denoising chain
```

DDPO has been successfully used for: fine-tuning text-to-image diffusion models with human preferences / aesthetic scores.

**DRaFT -- Differentiable Reward Fine-Tuning** (Clark et al., 2023): Directly backpropagates reward gradients through the (truncated) denoising chain. More sample-efficient than DDPO, but requires a differentiable reward model.

**DPPO -- Diffusion Policy Policy Optimization** (Ren et al., 2025): Specifically adapts PPO for diffusion policies in continuous control. Addresses cross-denoising-step clipping, high-dimensional action spaces, and other issues.

### 3.3 Flow Matching Models and RL

Flow Matching (Lipman et al., 2023) uses deterministic ODEs instead of stochastic SDEs, enabling faster inference (5-10 steps vs 50-100 steps), which is critical for real-time control.

Applications in RL:

```text
- Flow matching policy: generates actions from noise via ODE integration
- RL fine-tuning similar to DDPO: treat each discretized ODE step as an action in the MDP
- Action Flow Matching: uses conditional flow matching to learn action generation
```

### 3.4 Key Technical Challenges

| Challenge | Description | Common Solutions |
|---|---|---|
| Cross-step credit assignment | Reward only at final output, needs to be distributed across T steps | Model denoising chain as episodic MDP |
| Gradient variance | Long denoising chains cause high REINFORCE variance | Reduce steps / baseline / truncated backpropagation |
| Reward differentiability | Direct backpropagation requires differentiable reward | Policy gradient methods bypass this limitation |
| KL regularization | Prevent model collapse after fine-tuning | KL penalty similar to RLHF |
| Action consistency | Diffusion policy outputs action chunks, requiring temporal coherence | Overlapping window design |

### 3.5 Timeline

```text
Diffuser (2022) → Decision Diffuser, Diffusion Policy, DDPO, DRaFT (2023) → DPPO, Flow Matching Policy (2024-2025)
```

Trend: treating the denoising process as a multi-step MDP for policy gradient; flow matching is emerging as a faster inference alternative.

---

## 4. Agent RL vs One-Round RL

The interaction structure of RL can be divided into two modes:

### 4.1 Traditional RL: Single-Round Sequential Interaction

Traditional RL interaction is a continuous time-series rollout:

```text
s_0 → a_0 → r_0 → s_1 → a_1 → r_1 → ... → s_T
```

Each action directly acts on the environment, immediately producing the next state and reward. The entire rollout is a continuous, single-round episode.

Typical scenarios:
- **Autonomous driving**: perceive environment at each moment → output control commands (throttle, brake, steering) → vehicle moves → new environment state
- **Robot control**: each action chunk is a sequence of continuous actions, the agent continuously interacts with the physical environment
- **Atari games**: observe each frame → choose action → game state updates

### 4.2 Agentic RL: Multi-Round Conversational Interaction

Agentic RL interaction is multi-round and nested:

```text
Episode = multi-round dialogue/interaction

Round 1: agent observes → thinks → calls tools/outputs → environment feedback
Round 2: agent observes feedback → thinks → calls tools/outputs → environment feedback
...
Round N: agent observes feedback → final output → episode ends
```

Typical scenarios:
- **Agentic AI**: multi-round interaction with coding harness (read code → write code → run tests → see results → modify → rerun)
- **Dialogue systems**: adjust strategy based on user feedback across multiple turns
- **Tool-using agents**: search → read → summarize → search again → final answer

### 4.3 The Fundamental Connection

They seem different, but from RL's MDP framework, they are unified:

```text
Traditional RL:
  state = physical environment state
  action = control command
  reward = immediate/delayed reward
  episode = one continuous trajectory

Agentic RL:
  state = conversation history + environment state
  action = one complete agent output (may include tool calls)
  reward = per-round feedback / final task completion
  episode = one complete multi-round task
```

**Key insight: autonomous driving is essentially an agentic RL process.**

In autonomous driving:
- Each action chunk (e.g., control sequence for the next 2 seconds) is equivalent to a "conversation round" in agentic AI
- After each action chunk is executed, the agent observes the new environment state and makes a new decision
- The entire driving rollout is a multi-round interaction composed of chained action chunks

Comparison:

| Dimension | Autonomous Driving | Agentic AI |
|---|---|---|
| One "round" | One action chunk (~100ms-2s control sequence) | One agent output (tool call / text generation) |
| Environment feedback | New sensor observations | Tool execution results / user replies |
| Episode | One complete drive (minutes to hours) | One complete task (code modification / Q&A) |
| State space | Continuous, high-dimensional (images + point clouds + vehicle state) | Discrete, variable-length (text + tool state) |
| Action space | Continuous (throttle, brake, steering) | Discrete/mixed (token generation + tool selection) |

### 4.4 Differences in Training Methods

Although the MDP framework is unified, there are important differences in training methods:

**Traditional RL (autonomous driving/robotics)**:

```text
- Typically uses continuous action policy (Gaussian / Diffusion / Flow)
- Actor-Critic architecture is common (SAC / TD3 / PPO)
- Data comes from high-frequency simulator interactions
- Reward is usually engineered (distance, collision, comfort)
```

**Agentic RL (LLM Agent)**:

```text
- Typically uses autoregressive token policy (LLM)
- PPO / GRPO / DPO
- Data comes from multi-round agent-environment interaction
- Reward comes from task completion / human preference / verifiers
- Computational cost per "action" is far higher than traditional RL
```

### 4.5 Core Commonality

Whether autonomous driving or agentic AI, the core structure of RL is:

```text
agent interacts with environment → receives feedback → updates policy → changes future interaction behavior
```

The differences are only:
- Action granularity (millisecond-level control vs second/minute-level reasoning)
- Nature of the environment (physical world vs digital/text world)
- Interaction frequency (high-frequency continuous vs low-frequency discrete)

---

## 5. Sim & Real: RL Transfer Between Simulation and Reality

RL training typically requires extensive interaction, but interaction in real environments is expensive, dangerous, and slow. Therefore, Sim2Real and Real2Sim have become key technologies.

### 5.1 Sim2Real: From Simulation to Reality

The core challenge is the **reality gap** -- a policy that performs perfectly in simulation often fails when deployed to the real environment, because the physics engine, visual rendering, and dynamics parameters all differ from reality.

Main methods:

**Domain Randomization**

```text
Randomize simulator parameters during training:
- Physics parameters: friction, mass, latency, damping
- Visual parameters: lighting, textures, colors, camera positions
- Dynamics: transmission errors, sensor noise

→ The real world becomes "just another set of random parameters"
→ Policy learns to be robust to parameter variations
```

Milestone: OpenAI used Automatic Domain Randomization (ADR) to train a robotic hand to solve a Rubik's cube (2019), dynamically expanding the randomization range across thousands of parameters.

**System Identification**

```text
Measure real-world physics parameters → calibrate simulator
Can be combined with Domain Randomization: randomize around calibrated values rather than uniformly across the full range
```

**Domain Adaptation**

```text
Use CycleGAN / adversarial training to align visual distributions between simulation and reality
Can reduce the visual gap without paired data
```

**Progressive Transfer**

```text
Train in simulation first, then fine-tune in real environment with small amounts of data
Preserve features learned in simulation (Rusu et al., 2017 Progressive Nets)
```

### 5.2 Real2Sim: From Reality to Simulation

The reverse direction -- use real-world data to build or improve simulators, making simulation training more effective.

**Neural Scene Reconstruction**

```text
Use NeRF / 3D Gaussian Splatting to reconstruct scenes from real sensor data
→ Editable, photo-realistic simulation environments
Example: MARS (2023), UniSim (2023) reconstruct driving scenes for closed-loop simulation
```

**Learned World Models**

```text
Train generative models from real data to serve as implicit simulators
- GAIA-1 (Wayve, 2023): learns world model from driving videos, generates new scenes
- DayDreamer (Hafner et al., 2022): learns world model from real robot interactions, trains policy entirely within the model
```

**Digital Twins**

```text
High-fidelity digital replicas of real environments/equipment
Continuously updated with sensor data
Widely used in manufacturing and autonomous driving validation
```

**Sim Parameter Calibration**

```text
Optimize simulator parameters (friction, drag, latency) using real trajectory data
Methods: Bayesian optimization, differentiable simulation
Goal: minimize sim-real gap
```

### 5.3 Sim-Real Loop in Autonomous Driving

Autonomous driving is the most typical application of the Sim2Real / Real2Sim loop:

```text
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Real2Sim:                                              │
│   Real driving logs → reconstruct 3D scenes →            │
│   simulation environment                                 │
│   (Waymo SurfelGAN, NVIDIA DRIVE Sim,                    │
│    Tesla fleet data → simulation)                        │
│                                                          │
│          ↓                                               │
│                                                          │
│   Sim Training:                                          │
│   Train policy in reconstructed/synthetic environments   │
│   Domain randomize weather, traffic, sensor noise        │
│   Test dangerous/rare scenarios (edge cases)             │
│                                                          │
│          ↓                                               │
│                                                          │
│   Sim2Real:                                              │
│   Deploy to real vehicles                                │
│   Collect new real data                                  │
│                                                          │
│          ↓                                               │
│                                                          │
│   (Loop: new data improves sim → better training → ...)  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Trend: **continuous loop** -- real data improves simulation (Real2Sim), better simulation training improves real-world performance (Sim2Real), then collect new data, and repeat.

### 5.4 Sim-Real Method Comparison

| Direction | Method | Core Idea | Typical Application |
|---|---|---|---|
| Sim2Real | Domain Randomization | Randomize sim parameters to make policy robust to variations | Dexterous robot manipulation |
| Sim2Real | System Identification | Calibrate simulator parameters to match reality | Industrial control |
| Sim2Real | Domain Adaptation | Align sim/real visual distributions | Visual grasping |
| Sim2Real | Progressive Transfer | Train in sim first, then fine-tune in real | Mobile robots |
| Real2Sim | Neural Reconstruction | Reconstruct scenes from sensor data | Autonomous driving simulation |
| Real2Sim | Learned World Model | Learn implicit simulator from real data | Driving/robotics |
| Real2Sim | Digital Twin | High-fidelity digital replica | Manufacturing/driving validation |
| Real2Sim | Parameter Calibration | Calibrate sim parameters with real trajectories | Physics simulation |

---

## 6. Overview: RL's Role Across Different Scenarios

```text
┌──────────────────────────────────────────────────────────┐
│                      Role of RL                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  From Scratch       Fine-tune from SL    Combined with   │
│                     Base Model           Generative Models│
│  ┌─────────┐      ┌──────────┐        ┌──────────┐      │
│  │ Q-learn │      │ PPO+RM   │        │ DDPO     │      │
│  │ DQN     │      │ GRPO     │        │ DPPO     │      │
│  │ PPO     │      │ DPO      │        │ DRaFT    │      │
│  │ SAC     │      │ +LoRA    │        │ Flow RL  │      │
│  └─────────┘      └──────────┘        └──────────┘      │
│       ↕                 ↕                   ↕            │
│  Pure RL Env       LLM Alignment      Image/Control Gen  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Sim ←→ Real              Agent RL vs One-Round RL       │
│  ┌──────────┐             ┌──────────────────────┐      │
│  │ Domain   │             │ Driving ≈ Agentic    │      │
│  │ Random.  │             │ (action chunk = round)│      │
│  │ Real2Sim │             │ LLM Agent = multi-    │      │
│  │ World    │             │ round interaction     │      │
│  │ Models   │             │ All are closed-loop   │      │
│  │          │             │ MDPs at their core    │      │
│  └──────────┘             └──────────────────────┘      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```
