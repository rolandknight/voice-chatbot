from __future__ import annotations

from datetime import datetime

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    spoken = datetime.now().strftime("%-I:%M %p")
    await params.result_callback(f"It's {spoken}.")
