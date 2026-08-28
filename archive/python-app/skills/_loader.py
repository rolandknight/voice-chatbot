"""Discover, parse, and register file-based skills.

Each skill lives under `skills/<category>/<skill_name>/` and consists of:
- `SKILL.md`: YAML frontmatter (name, description, parameters, triggers, ...)
  followed by an optional markdown body of extended instructions
- `handler.py`: exports `async def handle(params, ctx)` that does the work

The loader walks the directory, parses frontmatter, builds FunctionSchemas,
imports each handler module, and exposes a SkillRegistry with a per-turn
filter that the SkillFilterProcessor uses to swap the LLM's tool set.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService

from config import Config
from skills._context import SkillContext

SKILLS_ROOT = Path(__file__).parent


_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class _Skill:
    name: str
    category: str
    description: str
    schema: FunctionSchema
    handle: Callable[[FunctionCallParams], Any]  # already bound to ctx
    triggers: list[str]  # raw trigger strings (for debug/logging)
    trigger_tokens: list[list[str]]  # pre-tokenized for fast matching
    always_available: bool
    body: str  # currently unused at runtime; reserved for future progressive disclosure


@dataclass
class SkillRegistry:
    skills_by_name: dict[str, _Skill] = field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.skills_by_name.keys())

    def by_category(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.skills_by_name.values():
            out.setdefault(s.category, []).append(s.name)
        return out

    def register(self, llm: LLMService, context: LLMContext) -> None:
        """Bind handlers on the LLM service and seed the context with the
        always-available tool set. The SkillFilterProcessor swaps tools per
        turn after this — registration here is for the cold start before the
        first user utterance.
        """
        for skill in self.skills_by_name.values():
            llm.register_function(skill.name, skill.handle)
        seeded = [s.schema for s in self.skills_by_name.values() if s.always_available]
        context.set_tools(ToolsSchema(standard_tools=seeded))

    def filter_for_turn(self, transcript: str, k: int = 15) -> list[FunctionSchema]:
        """Pick the tool subset visible to the LLM for this turn.

        Always-available skills are unconditional. The rest are scored by how
        many of their triggers appear as contiguous token subsequences of the
        transcript (word-boundary aware — so "use the" trigger doesn't match
        inside "pause the music"). Top-K by score, ties broken by name for
        determinism.
        """
        selected: set[str] = set()
        tokens = _tokenize(transcript or "")
        scored: list[tuple[int, str]] = []
        for skill in self.skills_by_name.values():
            if skill.always_available:
                selected.add(skill.name)
                continue
            if not tokens:
                continue
            score = sum(
                1 for tt in skill.trigger_tokens
                if tt and _contains_subseq(tokens, tt)
            )
            if score > 0:
                scored.append((-score, skill.name))  # negative for ascending sort
        scored.sort()
        for _, name in scored[:k]:
            selected.add(name)
        return [self.skills_by_name[n].schema for n in selected]


def _contains_subseq(tokens: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(tokens):
        return False
    if n == 1:
        return needle[0] in tokens
    last = len(tokens) - n + 1
    first = needle[0]
    for i in range(last):
        if tokens[i] == first and tokens[i:i + n] == needle:
            return True
    return False


def _parse_skill_md(path: Path) -> tuple[dict, str]:
    """Parse `--- yaml --- body` from a SKILL.md file."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: malformed YAML frontmatter")
    front = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")
    if not isinstance(front, dict):
        raise ValueError(f"{path}: frontmatter must be a YAML mapping")
    return front, body


def _build_schema(frontmatter: dict, source: Path) -> FunctionSchema:
    name = frontmatter.get("name")
    desc = frontmatter.get("description")
    if not name or not desc:
        raise ValueError(f"{source}: SKILL.md requires `name` and `description`")
    params_block = frontmatter.get("parameters") or {}
    if not isinstance(params_block, dict):
        raise ValueError(f"{source}: `parameters` must be a YAML mapping")

    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, pspec in params_block.items():
        if not isinstance(pspec, dict):
            raise ValueError(
                f"{source}: parameter {pname!r} must be a mapping"
            )
        spec = dict(pspec)
        if spec.pop("required", False):
            required.append(pname)
        properties[pname] = spec

    return FunctionSchema(
        name=name,
        description=" ".join(desc.split()) if isinstance(desc, str) else desc,
        properties=properties,
        required=required,
    )


def _import_handler(skill_dir: Path):
    handler_path = skill_dir / "handler.py"
    if not handler_path.is_file():
        raise FileNotFoundError(f"{skill_dir}: missing handler.py")
    mod_name = f"skills._loaded.{skill_dir.parent.name}_{skill_dir.name}"
    spec = importlib.util.spec_from_file_location(mod_name, handler_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load handler from {handler_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass and other introspection that walks
    # sys.modules[cls.__module__] resolves correctly during the handler's
    # module-level code execution.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    if not hasattr(module, "handle"):
        raise AttributeError(
            f"{handler_path}: must define `async def handle(params, ctx)`"
        )
    return module.handle


def _resolve_dotted(root: Any, dotted: str) -> Any:
    node = root
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def load_skills(ctx: SkillContext, cfg: Config, root: Path = SKILLS_ROOT) -> SkillRegistry:
    """Walk SKILLS_ROOT, parse each SKILL.md, return a populated registry.

    Skills are skipped (with a debug log) when:
    - their `enabled_when` dotted config path resolves to a falsy value, or
    - any name in their `requires` list is not satisfied by `ctx.has(name)`.

    `enabled_when` is a dotted attribute path into `cfg` (the Config tree),
    e.g. `skills.radio.enabled`. Skills with no `enabled_when` field load
    unconditionally.
    """
    registry = SkillRegistry()
    for skill_md in sorted(root.glob("*/*/SKILL.md")):
        skill_dir = skill_md.parent
        category = skill_dir.parent.name
        try:
            front, body = _parse_skill_md(skill_md)
        except Exception as e:
            logger.warning(f"Skipping {skill_md}: {e}")
            continue

        gate_path = front.get("enabled_when")
        if gate_path:
            try:
                gate_value = _resolve_dotted(cfg, gate_path)
            except AttributeError:
                logger.warning(
                    f"Skill {front.get('name')}: enabled_when={gate_path!r} "
                    f"does not resolve in config; skipping"
                )
                continue
            if not gate_value:
                logger.debug(
                    f"Skill {front.get('name')}: gated off by "
                    f"{gate_path}={gate_value!r}"
                )
                continue

        requires = front.get("requires") or []
        if isinstance(requires, str):
            requires = [requires]
        missing = [r for r in requires if not ctx.has(r)]
        if missing:
            logger.debug(
                f"Skill {front.get('name')}: missing context {missing}"
            )
            continue

        try:
            schema = _build_schema(front, skill_md)
            raw_handle = _import_handler(skill_dir)
        except Exception as e:
            logger.warning(f"Skipping {skill_md}: {e}")
            continue

        bound = _bind_ctx(raw_handle, ctx)
        triggers = [
            str(t).lower() for t in (front.get("triggers") or []) if t
        ]
        trigger_tokens = [_tokenize(t) for t in triggers]
        registry.skills_by_name[schema.name] = _Skill(
            name=schema.name,
            category=category,
            description=schema.description,
            schema=schema,
            handle=bound,
            triggers=triggers,
            trigger_tokens=trigger_tokens,
            always_available=bool(front.get("always_available", False)),
            body=body,
        )

    logger.info(
        f"Loaded {len(registry.skills_by_name)} skill(s): "
        f"{sorted(registry.skills_by_name.keys())}"
    )
    return registry


def _bind_ctx(handle, ctx: SkillContext):
    # The LLM service hands this wrapped callable to its function-call
    # dispatcher, so it's the single chokepoint every skill invocation
    # passes through. Logging here surfaces "did the LLM actually call a
    # tool, with what args, and did the handler return cleanly?" — useful
    # when diagnosing turns where Babel goes silent: a missing invoke log
    # means the LLM's tool-call JSON never reached dispatch (likely
    # truncated by max_tokens and dropped by json.loads in pipecat).
    async def _bound(params: FunctionCallParams):
        logger.info(
            f"Tool invoke -> {params.function_name}({dict(params.arguments)!r})"
        )
        try:
            result = await handle(params, ctx)
        except Exception:
            logger.exception(f"Tool error  <- {params.function_name} raised")
            raise
        logger.info(f"Tool return <- {params.function_name}")
        return result
    return _bound
