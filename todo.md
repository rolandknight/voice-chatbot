- do research into how we can add speaker recognition. first step is within a session. later, store voiceprints and identify people. this needs to be done with minimal impact on latency
- what needs to be done to add mcp client support? is gemma4 27b good at mcp? 

Because of the pre-roll, and it's working as designed — just with a side effect.

  Here's the mechanism (devices/rpi5/rpi_webrtc_voice.py:322): in --local-wake mode the Pi runs openWakeWord itself and stays disconnected until a wake word fires.
  openWakeWord fires at the end of the wake phrase ("hey one one"). To avoid clipping a command that you run straight into the wake word, on wake the client
  replays a ~500 ms buffer of audio captured before the wake instant into the stream (--preroll-ms, default 500; self._preroll replayed at line 601).

  The problem: 500 ms before the wake-fire instant is the wake phrase itself. So the stream the server receives is [hey one one][your command], and since the
  server sees "a normal push-mode client — no server-side wake" (line 326), Whisper just transcribes the whole thing → "Hey, one what?" as a user message. Nothing
  strips it, because the server never knew a wake word happened.

  So the wake phrase leaking in is the pre-roll being slightly longer than the gap between "wake fired" and "command starts."

  Why it matters here

  That "Hey, one what?" became its own user message in the LLM context — noise that almost certainly nudged the model toward web_search instead of play_spotify.
  Cleaning it up should improve tool selection.

  Fix — two knobs

  1. Lower the pre-roll so it stops dragging in the wake word:
  --preroll-ms 200
  1. 200–250 ms keeps the first phoneme of your command but usually excludes the wake tail — if you leave a beat between "hey one one" and the command. If you run
  them together ("hey-one-one-play-X" as one breath), any pre-roll that saves the command also catches the wake word; there's no clean split.
  2. Strip/ignore it server-side — either add a wake-prefix filter, or add a line to the system prompt telling the model to ignore a leading "hey one one / hey
  babel" in the user text. More robust for the run-together case.

  The --preroll-ms flag isn't currently plumbed through install_service.sh/the Makefile, so right now you'd add it to the unit's ExecStart by hand. Want me to (a)
  wire PREROLL_MS through install_service.sh + make install-service, and/or (b) add wake-prefix stripping on the server? I'd lean toward doing both — lower the
  pre-roll to ~200 ms and strip a leading wake phrase server-side, since together they cover both the spaced and run-together cases.
