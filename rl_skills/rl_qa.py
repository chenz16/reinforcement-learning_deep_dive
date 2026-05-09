#!/usr/bin/env python3
"""
RL Q&A Skill — Interactive RL Question Answering
=================================================

Uses the docs/ knowledge base to answer user's RL questions.
Knowledge comes from:
  - docs/en/01_rl_macro_evolution.md (macro arc)
  - docs/en/02_rl_evolution_diagram.md (evolution diagram)
  - docs/en/03_rl_micro_mechanism_framework.md (micro mechanisms)
  - docs/en/04_rl_training_with_foundation_model.md (foundation model connections)

Usage:
    python rl_qa.py

Set ANTHROPIC_API_KEY env var before running.
"""

import anthropic
from knowledge_loader import load_all_docs, load_code_summaries, SYSTEM_PROMPT_BASE


def build_system_prompt() -> list[dict]:
    """Build system prompt with knowledge base."""
    knowledge = load_all_docs()
    code_info = load_code_summaries()

    system_text = f"""{SYSTEM_PROMPT_BASE}

=== KNOWLEDGE BASE (docs) ===
{knowledge}

=== CODE EXAMPLES ===
{code_info}
"""
    return system_text


def main():
    client = anthropic.Anthropic()
    system = build_system_prompt()
    messages = []

    print("=" * 60)
    print("  RL Q&A — Ask anything about Reinforcement Learning")
    print("  Knowledge: Truly Master RL docs + 13 algorithm codes")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Stream the response
        print("\nRL Tutor: ", end="", flush=True)

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system,
            messages=messages,
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response_text += text

        print("\n")

        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
