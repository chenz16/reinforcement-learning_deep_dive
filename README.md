# Truly Master Reinforcement Learning

**Stop memorizing algorithms. Start understanding why they exist.**

---

You've read Sutton & Barto. You know what Q-learning is. You can recite the Bellman equation. You've maybe even implemented PPO.

But something still feels off.

- Why did RL move from value functions to policy gradients?
- What exactly is the relationship between DQN, PPO, SAC, and RLHF?
- Why do modern methods look so *ad-hoc* compared to the elegant Bellman theory?
- Is there a deeper thread connecting all of this?

**Yes, there is.** And this repo lays it out.

---

## What This Repo Is

This is not another RL tutorial. This is a **conceptual deep dive** for people who already have the basics but can't yet see the full picture.

We approach RL from two complementary angles:

### The Macro View -- *Why did RL evolve this way?*

A single narrative tracing the entire arc from Bellman (1957) to GRPO (2025):

> The Bellman equation went from being *the entire optimization* to *an auxiliary tool* to *completely absent* in modern preference learning. This isn't regression -- it's the natural result of RL shifting from god's-eye analytical optimization to statistical preference shaping under limited sampling.

### The Micro View -- *What is each mechanism actually doing?*

A multi-dimensional dissection of RL internals:

- **Loss evolution**: MSE (TD) &rarr; Max Q (DDPG actor) &rarr; Entropy &rarr; Weighted log-likelihood
- **On/off-policy from first principles**: not by algorithm name, but by asking *"does the update formula require data from the current policy?"*
- **Bias management**: RL doesn't pursue unbiased optimization -- it manages which biases to accept
- **Q as value compression, Policy as preference compression**: why hard argmax is a fragile bridge between the two
- **RL vs SL**: same optimization framework, fundamentally different because *RL's algorithm changes its own data distribution*

### The Connections -- *How does RL relate to everything else?*

- RL from scratch vs. RL fine-tuning on SL base models (PPO / GRPO / DPO + LoRA)
- RL with Diffusion and Flow Matching models
- Agent RL vs. one-round RL (autonomous driving as agentic RL)
- Sim2Real and Real2Sim

---

## Who This Is For

| If you are... | You'll get... |
|---|---|
| A **student** who passed the RL course but can't connect the dots | The conceptual backbone that textbooks don't give you |
| An **engineer** building RL systems who wants deeper intuition | Understanding of *why* each technique exists, not just *how* |
| **Preparing for interviews** and tired of surface-level algorithm summaries | A unified framework that makes every algorithm a natural consequence |
| A **researcher** looking for a clean mental model | A thesis-level narrative from DP to preference learning |

---

## Document Map

```
docs/
├── zh/                                          # Chinese
│   ├── from-bellman-to-preference-learning.md   # Macro: the full evolution arc
│   ├── rl-evolution-diagram.md                  # Macro: visual evolution diagram (Mermaid)
│   ├── rl-understanding-framework.md            # Micro: multi-dimensional mechanism analysis
│   └── rl-connections-to-other-models.md        # Connections: RL meets other paradigms
│
└── en/                                          # English
    ├── from-bellman-to-preference-learning.md
    ├── rl-evolution-diagram.md
    ├── rl-understanding-framework.md
    └── rl-connections-to-other-models.md
```

### Reading Order

**If you want the big picture first:**

1. [Evolution Diagram](docs/en/rl-evolution-diagram.md) -- 5-minute visual overview of the 9-stage evolution
2. [From Bellman to Preference Learning](docs/en/from-bellman-to-preference-learning.md) -- the full macro narrative
3. [Understanding Framework](docs/en/rl-understanding-framework.md) -- dive into the micro mechanisms
4. [Connections to Other Models](docs/en/rl-connections-to-other-models.md) -- how RL connects to SL, diffusion, sim2real, and agents

**If you want to build intuition bottom-up:**

1. [Understanding Framework](docs/en/rl-understanding-framework.md) -- start with the mechanism-level analysis
2. [Evolution Diagram](docs/en/rl-evolution-diagram.md) -- see where each mechanism fits historically
3. [From Bellman to Preference Learning](docs/en/from-bellman-to-preference-learning.md) -- the macro synthesis
4. [Connections to Other Models](docs/en/rl-connections-to-other-models.md) -- extend to broader ML landscape

---

## The Core Thesis

If you take nothing else from this repo, take this:

> **DP solves for the global optimum under a known model.**
>
> **RL evolves good-enough behavioral preferences from limited, noisy, biased data.**
>
> **The entire history of RL is the tension between greedy amplification (needed for learning) and statistical constraints (needed for stability). Too much averaging -- can't learn. Too much greediness -- collapses. The answer is never one or the other, but: use greediness to amplify effective signals, use statistical constraints to prevent the system from being carried away by noise.**

---

## Contributing

This is a living document. If you find errors, have suggestions, or want to add a new analytical dimension, feel free to open an issue or PR.

---

*Built from deep thinking about why RL works the way it does, not just how.*
