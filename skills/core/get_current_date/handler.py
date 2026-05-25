from __future__ import annotations

from datetime import datetime

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    now = datetime.now()
    day_name = now.strftime("%A")
    month = now.strftime("%B")
    await params.result_callback(
        f"Today is {day_name}, {month} {_ordinal(now.day)}, {now.year}."
    )
