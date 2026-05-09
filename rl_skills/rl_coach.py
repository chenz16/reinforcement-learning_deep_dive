#!/usr/bin/env python3
"""
RL Coach Skill — Proactive Question Generator + Answerer
=========================================================

Instead of waiting for the user to ask, this skill:
1. Analyzes the knowledge base and generates targeted questions
   that probe the user's understanding
2. Presents questions to the user
3. After user confirms/selects a question, provides a deep answer

This helps users discover gaps they didn't know they had.

Usage:
    python rl_coach.py

Set ANTHROPIC_API_KEY env var before running.
"""

import anthropic
from knowledge_loader import load_all_docs, load_code_summaries, SYSTEM_PROMPT_BASE


COACH_SYSTEM = """You are an RL learning coach. Your job is to help users deeply understand RL
by proactively generating insightful questions — the kind that reveal gaps in understanding.

You operate in two modes:

MODE 1: GENERATE QUESTIONS
When asked to generate questions, create 5 targeted questions that:
- Probe the connections BETWEEN concepts (not just definitions)
- Test whether the user understands WHY, not just WHAT
- Cover different dimensions: macro evolution, micro mechanisms, code implementation
- Range from foundational to advanced
- Are the kind of questions that make someone say "hmm, I never thought about that"

Format each question as:
[1] Question text
    Dimension: (macro/micro/code/connection)
    Difficulty: (foundational/intermediate/advanced)

MODE 2: ANSWER A QUESTION
When the user selects a question number or asks their own question,
provide a comprehensive answer using the knowledge base.
Structure your answer as:
- Direct answer (2-3 sentences)
- Macro context (where this fits in the evolution arc)
- Micro mechanism (which specific mechanism is involved)
- Code connection (which algorithm demonstrates this, if applicable)
- One insight the user probably hasn't considered
"""


def generate_questions(client, knowledge: str, topic: str = None, history: list = None) -> str:
    """Generate proactive questions for the user."""
    prompt = "Generate 5 insightful RL questions for me to think about."
    if topic:
        prompt += f" Focus on the topic: {topic}"
    if history:
        prompt += "\n\nAvoid repeating these previously asked questions:\n"
        prompt += "\n".join(f"- {q}" for q in history)

    system = f"""{COACH_SYSTEM}

=== KNOWLEDGE BASE ===
{knowledge}
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    return next(b.text for b in response.content if b.type == "text")


def answer_question(client, knowledge: str, question: str, messages: list) -> str:
    """Answer a selected question with deep analysis."""
    system = f"""{COACH_SYSTEM}

=== KNOWLEDGE BASE ===
{knowledge}
"""

    messages_copy = messages + [
        {"role": "user", "content": f"I want to explore this question:\n\n{question}\n\nGive me a deep answer."}
    ]

    result = ""
    print("\nRL Coach: ", end="", flush=True)
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=system,
        messages=messages_copy,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            result += text
    print("\n")
    return result


def main():
    client = anthropic.Anthropic()
    knowledge = load_all_docs()
    code_info = load_code_summaries()
    full_knowledge = f"{knowledge}\n\n=== CODE EXAMPLES ===\n{code_info}"

    asked_questions = []
    messages = []

    print("=" * 60)
    print("  RL Coach — I'll ask YOU the hard questions")
    print("  ")
    print("  Commands:")
    print("    [Enter]     - Generate new questions")
    print("    [1-5]       - Select a question to explore")
    print("    topic:XXX   - Generate questions on topic XXX")
    print("    ask:XXX     - Ask your own question")
    print("    quit        - Exit")
    print("=" * 60)
    print()

    current_questions = []

    # Generate initial questions
    print("Generating your first set of questions...\n")
    questions_text = generate_questions(client, full_knowledge, history=asked_questions)
    print(questions_text)
    print()

    # Parse question numbers from the generated text
    import re
    current_questions = re.findall(r'\[(\d)\]\s*(.+?)(?:\n|$)', questions_text)

    while True:
        try:
            user_input = input("Your choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            # Generate new questions
            print("\nGenerating new questions...\n")
            questions_text = generate_questions(client, full_knowledge, history=asked_questions)
            print(questions_text)
            print()
            current_questions = re.findall(r'\[(\d)\]\s*(.+?)(?:\n|$)', questions_text)
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if user_input.startswith("topic:"):
            topic = user_input[6:].strip()
            print(f"\nGenerating questions on: {topic}...\n")
            questions_text = generate_questions(client, full_knowledge, topic=topic, history=asked_questions)
            print(questions_text)
            print()
            current_questions = re.findall(r'\[(\d)\]\s*(.+?)(?:\n|$)', questions_text)
            continue

        if user_input.startswith("ask:"):
            question = user_input[4:].strip()
            asked_questions.append(question)
            answer = answer_question(client, full_knowledge, question, messages)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
            continue

        # Check if it's a number selection
        if user_input.isdigit():
            idx = int(user_input)
            matching = [q for num, q in current_questions if int(num) == idx]
            if matching:
                question = matching[0]
                print(f"\nExploring: {question}")
                asked_questions.append(question)
                answer = answer_question(client, full_knowledge, question, messages)
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
            else:
                print(f"No question #{idx} found. Try 1-5.")
            continue

        # Treat as a free-form question
        asked_questions.append(user_input)
        answer = answer_question(client, full_knowledge, user_input, messages)
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
