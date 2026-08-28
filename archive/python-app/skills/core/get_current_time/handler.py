from __future__ import annotations

from datetime import datetime

from pipecat.services.llm_service import FunctionCallParams

from skills._context import SkillContext

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty"]


def _two_digit_words(n: int) -> str:
    """Spell 0-59 in words (e.g. 5 -> 'five', 18 -> 'eighteen', 25 -> 'twenty
    five')."""
    if n < 20:
        return _ONES[n]
    if n % 10 == 0:
        return _TENS[n // 10]
    return f"{_TENS[n // 10]} {_ONES[n % 10]}"


def _spoken_time(now: datetime) -> str:
    """Render the clock time entirely in words so the TTS can't misread the
    digits/colon (e.g. '1:18' being voiced as 'one thousand eighteen')."""
    hour = now.hour % 12 or 12
    minute = now.minute
    meridiem = "in the morning" if now.hour < 12 else (
        "in the afternoon" if now.hour < 18 else "in the evening"
    )
    hour_words = _ONES[hour]
    if minute == 0:
        clock = f"{hour_words} o'clock"
    elif minute < 10:
        # Keep the leading "oh" so 1:05 is "one oh five", not "one five".
        clock = f"{hour_words} oh {_ONES[minute]}"
    else:
        clock = f"{hour_words} {_two_digit_words(minute)}"
    return f"{clock} {meridiem}"


async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    await params.result_callback(f"It's {_spoken_time(datetime.now())}.")
