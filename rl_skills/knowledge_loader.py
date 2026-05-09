"""
Shared knowledge loader for RL skills.
Loads all docs from the docs/en/ folder as context for Claude API calls.
"""

import os
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs" / "en"


def load_all_docs() -> str:
    """Load all English docs as a single knowledge base string."""
    docs = []
    for f in sorted(DOCS_DIR.glob("*.md")):
        content = f.read_text(encoding="utf-8")
        docs.append(f"--- {f.name} ---\n{content}")
    return "\n\n".join(docs)


def load_code_summaries() -> str:
    """Load algorithm code filenames and their first docstring."""
    code_dir = Path(__file__).parent.parent / "rl_algorithm"
    summaries = []
    for f in sorted(code_dir.glob("*.py")):
        lines = f.read_text(encoding="utf-8").split("\n")
        # Extract the docstring (between first pair of triple quotes)
        in_doc = False
        doc_lines = []
        for line in lines:
            if '"""' in line and not in_doc:
                in_doc = True
                continue
            elif '"""' in line and in_doc:
                break
            elif in_doc:
                doc_lines.append(line)
        summary = "\n".join(doc_lines[:10]) if doc_lines else "(no docstring)"
        summaries.append(f"[{f.name}]\n{summary}")
    return "\n\n".join(summaries)


SYSTEM_PROMPT_BASE = """You are an expert RL tutor based on the "Truly Master Reinforcement Learning" knowledge base.

You have access to a comprehensive set of RL documents that cover:
- Macro evolution: from Bellman (1957) to Preference Learning (DPO/GRPO 2025)
- Micro mechanisms: loss design, on/off-policy, exploration, bias management, compression, action modeling
- Actor-Critic entangled iteration, advantage as causal inversion, PPO stability decomposition
- RL vs SL connections, RL vs Optimal Control unification
- RL with foundation models, diffusion/flow, sim2real, agentic RL

Your style:
- Use the macro/micro framework from the docs to answer
- Connect concepts across dimensions (e.g., "this relates to the compression view in section 7")
- Use concrete examples and code when helpful
- Be concise but thorough — compress, don't pad
- Use both English and Chinese naturally based on the user's language
"""
