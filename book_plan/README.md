# Book Plan: *Truly Master Reinforcement Learning*

## A ~100-page conceptual deep dive from Bellman to Preference Learning

---

## 1. Why This Book

### 1.1 Gap in Existing Literature

| Book | Pages | Approach | Gap |
|---|---|---|---|
| Sutton & Barto (2018) | ~550 | Algorithm-by-algorithm, textbook order | No unified narrative; slow to reach deep RL |
| Szepesvari (2010) | ~98 | Compact, mathematically rigorous | Pure theory, no intuition, no implementation |
| Bertsekas (2012) | ~1200 | Optimal control / DP foundations | Too formal, no connection to modern RL |
| Agarwal et al. (draft) | ~250 | Unified theoretical lens (PAC, regret) | Theory-heavy, no code, no RLHF |
| Lambert (2025) | TBD | RLHF / LLM alignment | Focused on LLM post-training, not RL fundamentals |
| Lapan "Hands-On" | ~600 | Practical / code-heavy | Long, no conceptual synthesis |

**No existing book offers**: a short (~100pp), conceptually unified, intuition-first treatment that traces WHY RL evolved the way it did, connects macro history to micro mechanisms, bridges to modern alignment methods, AND includes runnable code.

### 1.2 This Book's Unique Angle

```text
Not "here are 20 algorithms, learn them one by one"
But "here is ONE story: how RL went from Bellman to Preference Learning,
     and why every algorithm along the way is a natural consequence"
```

Three pillars:
1. **Macro narrative**: the evolution arc (Bellman → Q-learning → DQN → Policy Gradient → PPO/SAC → RLHF/GRPO)
2. **Micro mechanisms**: multi-dimensional analysis of WHY each technique exists
3. **Code as evidence**: 13 algorithms on a unified environment, read through the macro/micro lens

### 1.3 Target Audience

- Students who passed an RL course but can't connect the dots
- Engineers building RL systems who want deeper intuition
- Interview candidates tired of surface-level algorithm summaries
- Researchers seeking a clean mental model

---

## 2. Book Structure (~100 pages)

### Part I: The Big Picture (20 pages)

**Chapter 1: RL Is Not What You Think (5 pages)**
- RL vs SL: same optimization framework, fundamentally different sampling structure
- The closed loop: your algorithm changes its own data distribution
- RL is not "optimize a clean loss" — it is "biased preference optimization under shifting distributions"
- *Source: docs section 0 + section 13 (RL vs SL)*

**Chapter 2: From Bellman to Preference Learning — The Grand Arc (15 pages)**
- 9-stage evolution diagram with historical context
- Bellman equation's role: from "everything" to "auxiliary" to "gone"
- The core tension: averaging is too conservative, max is too aggressive
- Why Policy Gradient was the philosophical watershed
- *Source: from-bellman-to-preference-learning.md + rl-evolution-diagram.md*

### Part II: The Mechanisms — Why Each Technique Exists (45 pages)

**Chapter 3: The Loss Landscape of RL (10 pages)**
- MSE (TD) → Max Q (DDPG actor) → Entropy → Weighted log-likelihood
- Why value-based talks about "loss" but policy gradient talks about "gradient"
- The loss is constructed after the fact for policy gradient — gradient came first
- MLE → weighted MLE: the connection to statistical estimation
- *Source: docs section 1*
- *Code: DQN (01), DDPG (11), PPO (03), SAC (02)*

**Chapter 4: On-Policy, Off-Policy, and the Sampling Problem (8 pages)**
- First-principles criterion: does the update require data from the current policy?
- Why Q-learning is off-policy (Bellman target doesn't care who sampled)
- Why PPO is near on-policy (importance ratio + clipping)
- Data efficiency vs stability trade-off
- *Source: docs section 2 + section 4*
- *Code: DQN (01) vs PPO (03) vs SAC (02) comparison*

**Chapter 5: Exploration, Exploitation, and Why Argmax Needs Randomness (7 pages)**
- Greedy converts estimation noise into sampling bias
- ε-greedy → entropy → max entropy RL: from hard rules to soft objectives
- Entropy as sampling distribution regularization, not decoration
- *Source: docs section 3 + section 1.7*
- *Code: DQN ε-greedy (01) vs SAC entropy (02)*

**Chapter 6: Bias Management — RL's Real Game (8 pages)**
- RL doesn't pursue unbiased optimization — it manages which biases to accept
- Q-learning's bias (max overestimation), PG's bias (variance), PPO's bias (clipping), SAC's bias (entropy)
- "Trade bias for stability" as a design principle
- *Source: docs section 6*
- *Code: DQN (01) vs Double DQN (07) vs TD3 (12)*

**Chapter 7: Q as Value Compression, Policy as Preference Compression (7 pages)**
- Q compresses future value; Policy compresses behavioral preference
- Value-based: compress value first, then extract preference via argmax (fragile)
- Policy-based: compress preference directly (robust but data-hungry)
- Actor-Critic: value compressor guides preference compressor
- *Source: docs section 7*

**Chapter 8: Action Modeling — Discrete, Continuous, Deterministic, Stochastic (5 pages)**
- Discrete → argmax is trivial; continuous → need actor as differentiable argmax
- Deterministic policy = hard preference; stochastic policy = soft preference
- Why RL moved toward stochastic: policy is both decision-maker AND data sampler
- *Source: docs section 8*
- *Code: DQN discrete (01) vs DDPG continuous (11) vs SAC stochastic (02)*

### Part III: Beyond Classical RL (20 pages)

**Chapter 9: Optimal Control and RL — Same Problem, Different Trade-offs (8 pages)**
- Three dimensions: rollout (explicit vs implicit model), optimization (online vs offline), data (few parameters vs massive interactions)
- OC's hidden constraints: needs explicit reference (cascading dependency), DOF explosion, poor parallelizability
- RL model = compressed offline experience for fast online inference
- Where they fuse: model-based RL, hierarchical driving stacks, sim2real
- *Source: docs section 10*

**Chapter 10: RL Meets the Real World (12 pages)**
- RL fine-tuning on SL base models: PPO (4 models) → GRPO (no critic) → DPO (no RM)
- Parameter efficiency: full fine-tuning vs LoRA vs QLoRA
- Architecture diagram: what's for training vs inference (critic = scaffolding)
- Q/V and Policy: coupled in sampling, decoupled in gradients
- Reward sources: environment, rules, learned RM
- RL with Diffusion / Flow Matching: denoising as MDP
- Agent RL vs one-round RL: autonomous driving as agentic RL
- Sim2Real / Real2Sim loop
- *Source: rl-connections-to-other-models.md*

### Part IV: The Code — Reading Algorithms Through the Lens (15 pages)

**Chapter 11: 13 Algorithms, One Environment, One Story (15 pages)**

Each algorithm gets ~1 page: the macro context (where it sits in the evolution), the micro mechanism (which technique from Part II it embodies), and annotated code highlights.

| # | Algorithm | Macro Stage | Key Micro Mechanism | Code |
|---|---|---|---|---|
| 1 | DQN | Taming Qmax | MSE loss + replay + target net | 01_dqn |
| 2 | Double DQN | Stabilizing Qmax | Decouple selection & evaluation | 07_double_dqn |
| 3 | Dueling DQN | Structural Q decomposition | V + A separation | 08_dueling_dqn |
| 4 | REINFORCE | Policy gradient birth | Weighted log-likelihood | 09_reinforce |
| 5 | A2C | Actor-Critic fusion | Value baseline reduces variance | 10_a2c |
| 6 | DDPG | Continuous control | Max Q loss = differentiable argmax | 11_ddpg |
| 7 | TD3 | Conservative Q | Double critic + delayed update | 12_td3 |
| 8 | TRPO | Trust region | KL constraint on policy update | 13_trpo |
| 9 | PPO | Stable policy optimization | Clipped ratio + near on-policy | 03_ppo |
| 10 | SAC | Soft optimality | Entropy as objective, not trick | 02_sac |
| 11 | GRPO | No critic, group relative | Statistical preference, no Bellman | 04_grpo |
| 12 | DPO | Direct preference | No RM, no RL loop | 05_dpo |
| 13 | ProRL | Progressive RL | Curriculum-based training | 06_prorl |

**Not a code tutorial** — code is read through the conceptual framework established in Parts I-III.

---

## 3. Implementation Plan

### Phase 1: LaTeX Setup & Skeleton (Week 1)

- [ ] Set up LaTeX project structure (book class, chapters, bibliography)
- [ ] Create chapter files with section headings
- [ ] Set up code listing style (minted or listings package)
- [ ] Set up figure workflow (existing PNGs + new diagrams via TikZ or included Mermaid exports)
- [ ] Configure PDF build pipeline (latexmk or Makefile)

### Phase 2: Part I — The Big Picture (Week 2)

- [ ] Chapter 1: adapt from docs section 0 + section 13
- [ ] Chapter 2: adapt from macro essay + evolution diagram
- [ ] Convert Mermaid diagrams to TikZ or export as PDF figures
- [ ] First review pass

### Phase 3: Part II — The Mechanisms (Weeks 3-4)

- [ ] Chapters 3-8: adapt from docs sections 1-8
- [ ] Add LaTeX math formatting (replace code-block math with proper equations)
- [ ] Add cross-references between chapters
- [ ] Insert code snippets (key fragments, not full files)
- [ ] Second review pass

### Phase 4: Part III — Beyond Classical RL (Week 5)

- [ ] Chapter 9: adapt from docs section 10
- [ ] Chapter 10: adapt from connections doc
- [ ] Add architecture diagrams (TikZ or imported)
- [ ] Third review pass

### Phase 5: Part IV — Code Reading (Week 6)

- [ ] Chapter 11: write algorithm-by-algorithm annotations
- [ ] Select key code fragments (not full scripts — full code in appendix or GitHub link)
- [ ] Add output figures (reward curves, Q heatmaps)
- [ ] Fourth review pass

### Phase 6: Polish & Publish (Week 7)

- [ ] Table of contents, index, bibliography
- [ ] Consistent notation glossary
- [ ] Final formatting pass (margins, fonts, page count target ~100)
- [ ] Build final PDF
- [ ] Add to GitHub repo

---

## 4. Technical Setup

### LaTeX Structure

```text
book_plan/
├── README.md                    ← this file
├── main.tex                     ← master document
├── chapters/
│   ├── ch01_rl_is_not_what_you_think.tex
│   ├── ch02_grand_arc.tex
│   ├── ch03_loss_landscape.tex
│   ├── ch04_on_off_policy.tex
│   ├── ch05_exploration.tex
│   ├── ch06_bias_management.tex
│   ├── ch07_compression.tex
│   ├── ch08_action_modeling.tex
│   ├── ch09_optimal_control_vs_rl.tex
│   ├── ch10_rl_meets_real_world.tex
│   └── ch11_code_reading.tex
├── figures/
│   └── (diagrams, plots)
├── code_snippets/
│   └── (extracted key fragments)
├── bibliography.bib
└── Makefile
```

### Build Command

```bash
# Build PDF
cd book_plan && latexmk -pdf main.tex

# Or with Makefile
make pdf
```

### Page Budget

| Part | Chapters | Target Pages |
|---|---|---|
| Part I: Big Picture | Ch 1-2 | 20 |
| Part II: Mechanisms | Ch 3-8 | 45 |
| Part III: Beyond | Ch 9-10 | 20 |
| Part IV: Code | Ch 11 | 15 |
| Front/back matter | TOC, notation, bib | 5 |
| **Total** | | **~105** |

---

## 5. Differentiation from Existing Books

```text
Sutton & Barto:     "Here are the algorithms, in textbook order"
This book:          "Here is WHY each algorithm exists — one story, one arc"

Szepesvari:         "Here are the theorems and proofs"
This book:          "Here is the intuition and the engineering trade-offs"

Lambert (RLHF):     "Here is how to align LLMs"
This book:          "Here is how RL evolved FROM Bellman TO alignment — the full journey"

Lapan (Hands-On):   "Here is how to implement each algorithm"
This book:          "Here is how to READ each algorithm through a unified conceptual lens"
```

**One-sentence pitch:**

> *The shortest path from "I know what Q-learning is" to "I understand why the entire field evolved the way it did."*
