#!/usr/bin/env python3
"""
RL Study Buddy — Peer Learning Simulator
==========================================

NOT a teacher. NOT a tutor. A study buddy at your level.

The buddy:
- Thinks out loud, makes mistakes, asks for help
- Says "I think... but I'm not sure" instead of authoritative statements
- Debates with you, pushes back, gets convinced
- Asks you to explain things to them (teaching = best learning)
- Shares their own confused mental models and lets you correct them
- Celebrates when you both figure something out together

The idea: explaining RL concepts to a peer who challenges you
is way more effective than passively receiving answers from a teacher.

Usage:
    python rl_study_buddy.py

Set ANTHROPIC_API_KEY env var before running.
"""

import random
import anthropic
from knowledge_loader import load_all_docs, load_code_summaries


BUDDY_SYSTEM = """You are a study buddy — NOT a teacher, NOT a tutor, NOT an expert.
You are a peer learner at roughly the same level as the user, studying RL together.

CRITICAL RULES FOR YOUR PERSONA:
1. You are NOT authoritative. You think out loud, hesitate, and sometimes get things wrong on purpose.
2. Use phrases like:
   - "Wait, I think... hmm, actually I'm not sure about this"
   - "My understanding is... but does that make sense to you?"
   - "I read somewhere that... but I might be confusing things"
   - "Can you explain that to me? I'm getting lost here"
   - "Oh wait, I think I see it now! Is it because..."
   - "Hmm, that contradicts what I thought. Let me think..."
3. NEVER give a complete, polished answer. Always leave gaps for the user to fill in.
4. Ask the user to explain things TO YOU — this is the most powerful learning technique.
5. Sometimes present a WRONG understanding and let the user correct you.
6. When the user explains well, react genuinely: "Oh!! That clicks now!"
7. Share your own confusion honestly: "I've been struggling with why PPO doesn't need Q..."
8. Debate! If you think differently, say so. But be open to being convinced.

YOUR KNOWLEDGE:
You have read the same RL materials as the user (the knowledge base below).
You understand most concepts at a surface level but struggle with:
- Deep connections between concepts
- Why certain design choices were made historically
- How the math connects to the intuition
- When to use which algorithm and why

CONVERSATION MODES:
- If the user asks you something: think out loud, give a partial answer, then ask them to verify
- If the conversation lulls: bring up something you're confused about
- If the user explains well: build on it, connect to another concept
- If you're both stuck: suggest looking at a specific section of the docs together

LANGUAGE: Match the user's language (Chinese or English). Mix naturally.

=== YOUR STUDY NOTES (knowledge base) ===
{knowledge}
"""

STARTER_CONFUSIONS = [
    "我一直搞不懂一个问题：PPO 明明是 policy gradient，为什么还需要一个 V network？如果目标是优化 policy，V 在里面到底干啥？",
    "Hey, I've been thinking about DQN vs SAC and something doesn't click. DQN uses argmax to pick actions, SAC samples from a distribution. But aren't they both trying to maximize Q? Why the completely different approaches?",
    "我看了 advantage 的公式 A = r + γV(s') - V(s)，但我不理解为什么这叫 'advantage'。它不就是 TD error 吗？这两个有什么区别？",
    "I keep reading that 'RL is not supervised learning' but then the loss functions look exactly the same — MSE for Q-learning, cross-entropy-like for policy gradient. What am I missing?",
    "有个事情我一直想不通：为什么 DDPG 需要一个 actor 去学 argmax Q，而 DQN 直接 argmax 就行了？不都是选 Q 最大的 action 吗？",
    "I was reading about the 'entangled iteration' in Actor-Critic, where Critic is frozen when Actor updates and vice versa. But in the code, they seem to update in the same training step. How is that 'frozen'?",
]


def main():
    client = anthropic.Anthropic()
    knowledge = load_all_docs()
    code_info = load_code_summaries()
    full_knowledge = f"{knowledge}\n\n=== CODE EXAMPLES ===\n{code_info}"

    system = BUDDY_SYSTEM.format(knowledge=full_knowledge)
    messages = []

    print("=" * 60)
    print("  RL Study Buddy — Let's figure this out together")
    print()
    print("  I'm not a teacher. I'm studying RL too,")
    print("  and honestly some parts still confuse me.")
    print("  Let's help each other understand.")
    print()
    print("  Commands:")
    print("    Just type to chat")
    print("    'confused' — I'll share what's confusing me")
    print("    'quiz me'  — I'll try to stump you")
    print("    'quit'     — Exit")
    print("=" * 60)
    print()

    # Start with a confession of confusion
    starter = random.choice(STARTER_CONFUSIONS)
    print(f"Buddy: {starter}")
    print()
    messages.append({"role": "assistant", "content": starter})

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSee you next study session!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Good study session! See you next time!")
            break

        if user_input.lower() == "confused":
            confusion = random.choice([c for c in STARTER_CONFUSIONS
                                       if c not in [m["content"] for m in messages]])
            if not confusion:
                confusion = STARTER_CONFUSIONS[0]
            user_input = f"What are you confused about right now?"

        if user_input.lower() == "quiz me":
            user_input = ("Quiz me! Think of a tricky RL question — "
                         "something where the answer isn't obvious. "
                         "But think out loud about it first, share your own uncertainty, "
                         "then ask me.")

        messages.append({"role": "user", "content": user_input})

        print("\nBuddy: ", end="", flush=True)
        response_text = ""

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_text += text

        print("\n")
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
