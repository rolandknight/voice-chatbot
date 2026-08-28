from __future__ import annotations

from pipecat.services.llm_service import FunctionCallParams

from persona_router import apply_skill_persona_switch
from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    requested = (params.arguments.get("persona") or "").strip().lower()
    if not requested:
        await params.result_callback("Tell me which persona to switch to.")
        return
    if not apply_skill_persona_switch(ctx.persona_config, ctx.persona_state, requested):
        options = ", ".join(sorted(ctx.persona_config.personas.keys()))
        await params.result_callback(
            f"I don't have a persona called {requested}. I can be {options}."
        )
        return
    await params.result_callback(f"Switched to {requested}.")
