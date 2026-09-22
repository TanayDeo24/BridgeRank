from __future__ import annotations

import json
import os
from pathlib import Path


SYSTEM_PROMPT = """You are reviewing a semantic-search evaluation. Identify systematic failure
patterns, distinguish lexical, semantic, corpus, and labeling problems, and propose prioritized,
testable experiments. Never invent evidence. Return concise Markdown with Observations, Hypotheses,
Experiments, and Risks sections."""


def build_prompt(evaluation: dict) -> str:
    return (
        "Analyze this retrieval evaluation. Compare variants and category slices, call out the "
        "quality/latency trade-offs, and recommend the next three ablations.\n\n"
        + json.dumps(evaluation, indent=2)
    )


def create_summary(input_path: Path, output_path: Path) -> None:
    evaluation = json.loads(input_path.read_text(encoding="utf-8"))
    prompt = build_prompt(evaluation)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        output = "# Claude evaluation prompt\n\n" + prompt
    else:
        from anthropic import Anthropic

        response = Anthropic(api_key=api_key).messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        output = "# Claude-assisted failure analysis\n\n" + response.content[0].text
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
