from __future__ import annotations

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    # Loader-side `requires: [backend_state]` guarantees this is set, but
    # double-check for defensive safety since the skill mutates shared state.
    if ctx.backend_state is None:
        await params.result_callback(
            "Claude isn't available right now."
        )
        return
    ctx.backend_state["backend"] = "claude"
    await params.result_callback("Asking Claude. Go ahead.")
