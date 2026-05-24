"""File-based skill registry for the babel voice assistant.

Each skill is one folder under `skills/<category>/<name>/` containing a
`SKILL.md` (Claude-style frontmatter + body) and a `handler.py`. The loader
discovers them at startup; the SkillFilterProcessor swaps the LLM's tool set
per turn so the model only sees ~15 relevant tools even when 200+ are loaded.
"""

from skills._context import SkillContext
from skills._filter import SkillFilterProcessor
from skills._loader import SkillRegistry, load_skills
from skills._tracker import BotSpeakingTracker

__all__ = [
    "BotSpeakingTracker",
    "SkillContext",
    "SkillFilterProcessor",
    "SkillRegistry",
    "load_skills",
]
