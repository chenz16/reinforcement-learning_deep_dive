# RL Learning Skills

Two interactive tools powered by Claude API, using the docs/ knowledge base.

## Setup

```bash
export ANTHROPIC_API_KEY="your-key-here"
cd rl_skills
```

## Skill 1: RL Q&A (`rl_qa.py`)

**You ask, it answers** — using the full docs as knowledge base.

```bash
python rl_qa.py
```

```
You: Why does PPO not need a Q network?
RL Tutor: [answers using macro/micro framework from docs]

You: How does SAC's entropy differ from epsilon-greedy?
RL Tutor: [connects exploration chapter + compression chapter + code]
```

Features:
- Multi-turn conversation with memory
- Streams responses in real-time
- Answers grounded in the docs knowledge base
- Connects macro evolution + micro mechanisms + code

## Skill 2: RL Coach (`rl_coach.py`)

**It asks YOU questions** — probes your understanding gaps.

```bash
python rl_coach.py
```

```
Generating your first set of questions...

[1] Why can PPO completely avoid maintaining a Q network,
    while SAC requires two Q networks?
    Dimension: connection
    Difficulty: intermediate

[2] In the "entangled iteration" of Actor-Critic, what happens
    if Critic updates much faster than Actor?
    Dimension: micro
    Difficulty: advanced
...

Your choice: 1
Exploring: Why can PPO completely avoid...

RL Coach: [deep answer with macro context + micro mechanism + code reference]
```

Commands:
- `[Enter]` — Generate new questions
- `[1-5]` — Select a question to explore
- `topic:PPO` — Focus questions on a specific topic
- `ask:your question` — Ask your own question
- `quit` — Exit

Features:
- Generates questions that probe CONNECTIONS between concepts
- Avoids repeating previously asked questions
- Ranges from foundational to advanced
- Answers include macro context + micro mechanism + code connection
