"""SKILL.md frontmatter -> OpenAI tool dicts, mirroring skills/_loader.py.

The pipeline builds pipecat FunctionSchemas from the same frontmatter; this
module reproduces the resulting OpenAI `tools` payload without importing
pipecat so the PoC venv stays tiny. Tools are returned sorted by name: the
tool list is rendered into the front of the prompt by the chat template, so
a stable order is a precondition for prefix-cache hits.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def parse_skill(path: Path) -> dict:
    m = _FRONTMATTER.match(path.read_text())
    if not m:
        raise ValueError(f"{path}: no YAML frontmatter")
    fm = yaml.safe_load(m.group(1))
    props, required = {}, []
    for pname, spec in (fm.get("parameters") or {}).items():
        prop = {"type": spec.get("type", "string")}
        if "description" in spec:
            prop["description"] = " ".join(str(spec["description"]).split())
        if "enum" in spec:
            prop["enum"] = spec["enum"]
        props[pname] = prop
        if spec.get("required"):
            required.append(pname)
    return {
        "name": fm["name"],
        "description": " ".join(str(fm.get("description", "")).split()),
        "enabled_when": fm.get("enabled_when"),
        "always_available": bool(fm.get("always_available", False)),
        "triggers": list(fm.get("triggers") or []),
        "parameters": {"type": "object", "properties": props, "required": required},
    }


def _enabled(skill: dict, enabled: dict[str, bool]) -> bool:
    key = skill.get("enabled_when")
    return True if not key else bool(enabled.get(key, False))


def load_skills(root: Path, enabled: dict[str, bool] | None = None) -> list[dict]:
    skills = [parse_skill(p) for p in sorted(root.glob("*/*/SKILL.md"))]
    if enabled is not None:
        skills = [s for s in skills if _enabled(s, enabled)]
    return sorted(skills, key=lambda s: s["name"])


def to_openai_tools(skills: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in sorted(skills, key=lambda s: s["name"])
    ]
