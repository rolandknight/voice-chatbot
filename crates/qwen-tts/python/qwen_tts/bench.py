"""Bench sentences shared by every TTS PoC in the repo.

Same three sentences as poc-tts/poc_tts/bench.py so rows land next to the
Chatterbox Flash numbers. The Rust bench (qwen-tts-tester) reads SENTENCES.
"""

SENTENCES = [
    ("short", "Sure, the kitchen light is on."),
    ("medium", "I checked the calendar for tomorrow and you have three meetings, the first one starting at nine fifteen."),
    (
        "long",
        "Here is the summary you asked for. The build finished in about four minutes and all tests passed, "
        "except for one flaky integration test that succeeded on retry. I have also updated the dependency lock "
        "file, and the deployment to staging is scheduled for six o'clock this evening, so let me know if you "
        "want to hold it.",
    ),
]
