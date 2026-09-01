# Autostarting the client on a Raspberry Pi

`make deploy-pi PI_HOST=pi@raspberrypi.local` from the repo root does the whole
thing: cross-build, ship, install, restart. It is the same command for the
first install and for every update.

```sh
# On the dev machine. The first run seeds /opt/voice-chatbot/.env.
make deploy-pi PI_HOST=pi@raspberrypi.local

# On the Pi, once, to point it at the server. The file belongs to the service
# account, so this needs no sudo when you are logged in as that user.
$EDITOR /opt/voice-chatbot/.env
sudo systemctl restart voice-chatbot-client
journalctl -u voice-chatbot-client -f
```

It rsyncs the cross-built binary, the `hey_*.onnx` wake heads and this
directory into `~/.cache/voice-chatbot-deploy` on the Pi, then runs
`install.sh` there under sudo. Overridable: `PI_DIR` (install location,
default `/opt/voice-chatbot`), `PI_SERVICE`, `PI_STAGE`. `install.sh` also runs
standalone on a Pi with the repo checked out and built natively.

The payload is only the binary plus the wake heads. The wake frontend
(melspectrogram + embedding) is compiled into the binary by `rust-embed`, and
inference is `tract`, not ONNX Runtime — so there is no shared library to
install and nothing to match versions with. `ffmpeg` is the one runtime
dependency (`apt install ffmpeg`); without it the client still talks, but
radio, shows and sound effects play nothing at all.

## Why a system unit and not `systemctl --user`

The usual reason for a user unit on a Pi is audio: PulseAudio and PipeWire live
in the user session, so a system service can't reach them. This client never
goes near either. `cpal` opens ALSA directly and deliberately ranks the raw
`plughw:` PCM above `default`/`sysdefault` (`alias_rank` in
`crates/client/src/audio.rs`) because `dmix`/`dsnoop` pin the period at 1024
frames and double-buffer into the latency budget. All it needs is `/dev/snd`,
which the unit grants with `SupplementaryGroups=audio`. No session, no linger,
no `pulse` PCM.

The corollary: on Raspberry Pi OS **Desktop**, PipeWire is running and may hold
the speakerphone, which blocks the exclusive `plughw:` open. Pi OS **Lite** has
no such contention and is the better target. Spotify is unaffected either way —
it plays on the client's own librespot endpoint, which is a separate service.

## The three unit settings that aren't boilerplate

- **`KillSignal=SIGINT`.** The client's only shutdown path is its Ctrl-C
  watcher (`run_call`). SIGTERM has no handler, so systemd's default would kill
  it mid-call instead of closing the peer and the ALSA streams.
- **`WorkingDirectory`.** Both runtime paths are relative to the CWD: the
  client loads `./.env` itself (`crates/env-file`, before clap reads its `env`
  fallbacks) and `WAKE_DIR` defaults to `./models/wakeword`.
- **`Restart=always`, 5 s.** Not for a late-booting server — the client already
  retries an unreachable one every 2 s, forever, on its own. It is for crashes
  and for the USB race: if the speakerphone hasn't enumerated when the unit
  starts, device selection fails and the restart picks it up.

## Config

`/opt/voice-chatbot/.env`, seeded from `env.example` on the first install and
never overwritten afterwards. It is *not* a copy of the repo-root `.env`: that
one holds the server's API keys, which have no business on a satellite. The
client reads `SERVER_URL`, `INPUT_DEVICE`, `OUTPUT_DEVICE`, `WAKE_DIR`,
`NO_WAKE`, `WAKE_THRESHOLD`, `WAKE_SESSION_SECS`, `LED` and `LOG_LEVEL`.

These names lost their `FLOWCAT_` prefix, and the client now *refuses to start*
with any `FLOWCAT_*` still set rather than silently running on the defaults.
`install.sh` checks the installed `.env` for them so the failure arrives with a
filename attached instead of in a 5-second restart loop.

`/etc/default/voice-chatbot-client` is read after `.env` as an optional ops
override; the unit tolerates it being absent.

## Speakerphone LEDs

The client shows chatbot activity on the Jabra's LED ring: dark when asleep,
solid green when listening or thinking, red while the mic
is gated. This rides the speakerphone's standard telephony HID interface
(docs/specs/jabra-led.md), not the audio path, and needs `/dev/hidraw*`
access: install.sh ships a udev rule opening Jabra hidraw nodes to the
`audio` group the service already runs in. `LED=off` in `.env` disables it;
`voice-chatbot-client led-test` (with the service stopped) cycles the states
for a look. Running the client by hand on a dev machine needs the same
access: run `sudo deploy/jabra-setup.sh` once (installs a desktop-friendly
`uaccess` + `audio`-group udev rule).

## Trimming boot time

A stock Pi OS Desktop image spends minutes of every boot on services a
satellite never uses — on one Pi, `docker.service` alone blamed 5min12s and
`rpi-eeprom-update.service` another 2min41s. `trim-boot.sh` turns them off
over ssh, from the dev machine:

```sh
deploy/rpi/trim-boot.sh pi@raspberrypi.local
```

It disables docker/containerd, the per-boot EEPROM update check, bluetooth
and cups, and sets the default boot target to the console (no lightdm). The
last one also helps audio on Desktop images: it is lightdm's autologin
session that starts the PipeWire which can hold the speakerphone (see above).
avahi stays — `deploy-pi` resolves the Pi's `*.local` name through it.

Everything is reversible (`sudo systemctl enable --now <unit>`,
`sudo systemctl set-default graphical.target`) and rerunning the script is a
no-op. With the boot check gone, run `sudo rpi-eeprom-update -a` by hand now
and then to keep the bootloader current.
