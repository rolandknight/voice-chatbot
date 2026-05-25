# ESP32-S3-BOX-3 sensor telemetry over MQTT

Reference design for streaming temperature, humidity, radar presence, and IR data between the Box-3 and the `voice-chatbot` backend. This sits alongside `docs/web-rtc.md` — that doc covers the voice path (WebRTC), this one covers the always-on telemetry path (MQTT).

The firmware side (`firmware/box3/main/sensors.*`, `mqtt_client.*`) is implemented today. The backend side is **deferred** and documented below so it can be picked up cleanly later.

---

## Why this design

The Box-3 voice client already does on-device wake-word and WebRTC streaming, but the WebRTC peer connection only exists during an active voice session (post-wake, pre-idle-timeout). Sensors need to be reachable **all the time** — the LLM should be able to answer "what's the temperature?" without first prompting the user to repeat the question, and radar presence events should fire whether or not anyone is talking to Babel.

The ESP32-S3-BOX-3-SENSOR add-on board (the official Espressif expansion) adds, in addition to the Box-3's own mics + speaker:

- **SHT40** temperature + humidity sensor (I2C, 0x44)
- **mmWave radar** for presence + range
- **IR LED + IR receiver** for emulating / capturing remote control codes (RMT peripheral)

The right transport for always-on, lightweight, pub/sub telemetry is **MQTT** to a broker on the same LAN as the voice-chatbot Mac. This decouples sensor data from voice entirely: the firmware can publish a radar event the instant presence changes, regardless of WebRTC state.

The intended outcome:

- A `query_sensors` skill lets the LLM answer "what's the temperature in here?" instantly by reading the cached MQTT value (no per-query round-trip to the device).
- Radar presence becomes a backend signal — e.g., the LLM volunteers "welcome home" on entry, or skips wake-word reminders when no one's around.
- IR raw codes can be captured + replayed via MQTT, opening the door to a "tell the AC to turn off" skill in the future.

---

## Topology

```
┌────────────────────── ESP32-S3-BOX-3 + SENSOR add-on ─────────────────────┐
│                                                                            │
│  ES7210 mics + ES8311 spkr ─── audio_pipeline ─── webrtc_client ───────────┼──── WebRTC (SRTP) ─────┐
│                                                                            │                        │
│  SHT40  (I2C)   ─┐                                                         │                        │
│  Radar  (UART)  ─┼── sensors.c ──── mqtt_client.c  ────────────────────────┼──── MQTT (TCP 1883) ─┐ │
│  IR     (RMT)   ─┘                                                         │                      │ │
│                                                                            │                      │ │
└────────────────────────────────────────────────────────────────────────────┘                      │ │
                                                                                                     │ │
                                                              ┌──── voice-chatbot host ──────────────┴─┴──┐
                                                              │                                            │
                                                              │  Mosquitto broker (docker)                 │
                                                              │     ↑                                      │
                                                              │     │ subscribe / publish                  │
                                                              │  mqtt_bridge.py  (deferred)                │
                                                              │     ↓                                      │
                                                              │  in-memory sensor cache                    │
                                                              │     ↓                                      │
                                                              │  skills/device/query_sensors/  (deferred)  │
                                                              │     ↓                                      │
                                                              │  Ollama / Claude (LLM)                     │
                                                              └────────────────────────────────────────────┘
```

Two **fully independent** channels:

- **WebRTC** carries voice (existing — see `docs/web-rtc.md`). Per-session, opened on wake.
- **MQTT** carries sensor data + device commands. Persistent — connects at boot, reconnects on WiFi flap, sets a last-will message so the broker marks the device offline cleanly when it dies.

---

## Topic schema

Device ID is the lower 48 bits of the WiFi MAC, formatted hex — e.g., `box3-a1b2c3d4e5f6`. Computed once at boot from `esp_efuse_mac_get_default()`.

| Topic | Direction | Payload | Retain | QoS |
|---|---|---|---|---|
| `babel/dev/{id}/status` | up | `"online"` / `"offline"` | yes (LWT) | 1 |
| `babel/dev/{id}/sensor/temp` | up | `{"c": 22.41, "ts": 1716528000}` | yes | 0 |
| `babel/dev/{id}/sensor/humidity` | up | `{"rh": 45.2, "ts": ...}` | yes | 0 |
| `babel/dev/{id}/sensor/radar` | up | `{"present": true, "distance_cm": 120, "energy": 87, "ts": ...}` | yes | 1 |
| `babel/dev/{id}/sensor/ir_rx` | up | `{"protocol":"nec", "code":"0x20DF10EF", "ts":...}` | no | 1 |
| `babel/dev/{id}/cmd/ir_tx` | down | `{"protocol":"nec", "code":"0x20DF10EF"}` | no | 1 |
| `babel/dev/{id}/cmd/query` | down | `{"sensor":"temp"}` (forces immediate publish) | no | 0 |
| `babel/dev/{id}/cmd/reboot` | down | `{}` | no | 1 |

### Publish cadence

- **Temperature + humidity** — every 30 s (configurable via `SENSOR_PUBLISH_INTERVAL_SEC`), and immediately on any change > 0.5 °C / 2% RH.
- **Radar** — on state change only (present↔absent, or distance bucket change > 20 cm). Debounced 500 ms to avoid chatter.
- **IR rx** — only when a valid frame decodes.

Retained messages mean any backend that subscribes mid-session gets the current state instantly without waiting for the next publish. The status topic uses MQTT's last-will so if the device drops off the network, the broker publishes `offline` automatically.

---

## Firmware (`firmware/box3/main/`)

### `sensors.{c,h}` — driver

Single module that owns the I2C bus, the UART for radar, and the RMT peripheral for IR. Exposes:

```c
typedef struct {
    float    temp_c;
    float    humidity_rh;
    bool     radar_present;
    int      radar_distance_cm;
    int      radar_energy;
    int64_t  ts_us;
} sensor_snapshot_t;

typedef void (*sensor_event_cb_t)(const sensor_snapshot_t *snap);
typedef void (*ir_rx_cb_t)(uint32_t code, const char *protocol);

esp_err_t sensors_init(sensor_event_cb_t on_change, ir_rx_cb_t on_ir);
esp_err_t sensors_read_now(sensor_snapshot_t *out);
esp_err_t sensors_ir_tx(uint32_t code, const char *protocol);
```

Internals:

- **SHT40.** ESP-IDF I2C master, address 0x44. Driver from `espressif/sht4x` managed component — no need to write the driver from scratch. Polled at the publish cadence.
- **Radar.** UART at 256 kbps for HLK-LD2410-style framing, or I2C if the sensor add-on uses Espressif's onboard radar IC. The exact driver lives in esp-bsp's `box_3_sensor` package — pull that in via `idf_component.yml` and call the BSP init.
- **IR.** RMT peripheral. Outbound drives the LED at 38 kHz carrier with NEC / RC5 / Sony framing. Inbound uses RMT capture mode + the `espressif/esp_ir_protocols` decoder.

A single `sensors_task` polls SHT40 every publish period, watches radar continuously, and dispatches changes through the `on_change` callback. Wake-word + audio paths are unaffected — sensors live on their own task.

### `mqtt_client.{c,h}` — broker glue

Thin wrapper over `esp_mqtt_client_*` (esp-mqtt is part of ESP-IDF core).

```c
typedef void (*mqtt_cmd_cb_t)(const char *subtopic, const cJSON *payload);

esp_err_t mqtt_client_start(const char *device_id, mqtt_cmd_cb_t on_cmd);
esp_err_t mqtt_publish_sensor(const sensor_snapshot_t *snap, const char *which);
esp_err_t mqtt_publish_ir_rx(uint32_t code, const char *protocol);
```

Init sequence:

1. `esp_mqtt_client_init()` with broker URI from `config.h`, LWT topic = `babel/dev/{id}/status`, LWT payload = `"offline"`, retain = true.
2. On `MQTT_EVENT_CONNECTED`: publish `"online"` retained; subscribe to `babel/dev/{id}/cmd/+`.
3. On `MQTT_EVENT_DATA`: parse JSON with cJSON, dispatch to `on_cmd` with the last topic segment (e.g., `ir_tx`, `query`, `reboot`).
4. Reconnect is handled inside esp-mqtt; just log it.

### `main.c` — wiring

After `webrtc_client_init()`:

```c
static void on_sensor_change(const sensor_snapshot_t *snap) {
    mqtt_publish_sensor(snap, "temp");
    mqtt_publish_sensor(snap, "humidity");
    mqtt_publish_sensor(snap, "radar");
}

static void on_ir_rx(uint32_t code, const char *proto) {
    mqtt_publish_ir_rx(code, proto);
}

static void on_mqtt_cmd(const char *sub, const cJSON *payload) {
    if (strcmp(sub, "ir_tx") == 0) {
        cJSON *c = cJSON_GetObjectItem(payload, "code");
        cJSON *p = cJSON_GetObjectItem(payload, "protocol");
        if (cJSON_IsString(c) && cJSON_IsString(p)) {
            sensors_ir_tx(strtoul(c->valuestring, NULL, 0), p->valuestring);
        }
    } else if (strcmp(sub, "query") == 0) {
        sensor_snapshot_t snap;
        if (sensors_read_now(&snap) == ESP_OK) on_sensor_change(&snap);
    } else if (strcmp(sub, "reboot") == 0) {
        esp_restart();
    }
}

char device_id[24];
make_device_id(device_id, sizeof(device_id));   // hex MAC
sensors_init(on_sensor_change, on_ir_rx);
mqtt_client_start(device_id, on_mqtt_cmd);
```

### `config.h` — new vars

```c
#define MQTT_BROKER_URI              "mqtt://192.168.1.10:1883"
#define MQTT_USERNAME                ""
#define MQTT_PASSWORD                ""
#define SENSOR_PUBLISH_INTERVAL_SEC  30
#define SENSOR_TEMP_DELTA_C          0.5f
#define SENSOR_HUMIDITY_DELTA_RH     2.0f
#define RADAR_DEBOUNCE_MS            500
```

### `idf_component.yml` — new dependencies

```yaml
  espressif/sht4x:
    version: "^0.4.0"
  espressif/esp_box_sensor:
    version: "^0.2.0"         # or the BSP variant if the registry name differs
  espressif/esp_ir_protocols:
    version: "^0.1.0"
```

esp-mqtt is part of ESP-IDF core — no managed component entry needed.

### `CMakeLists.txt` — sources + REQUIRES

```cmake
SRCS
    ...
    "sensors.c"
    "mqtt_client.c"
REQUIRES
    ...
    mqtt
    driver
```

---

## Backend changes (deferred — to implement later)

### Broker

Add `infra/mosquitto/mosquitto.conf` and a compose entry:

```yaml
services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports: ["1883:1883"]
    volumes:
      - ./infra/mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf
      - mosquitto_data:/mosquitto/data
```

LAN-only deployment, no auth in v1. Add username/password to `mosquitto.conf` once the protocol works.

### `mqtt_bridge.py` — sensor cache (new)

Small asyncio module using `aiomqtt`. On start, connects, subscribes to `babel/dev/+/sensor/+`, and maintains an in-memory dict per device:

```python
class SensorCache:
    def __init__(self):
        self._state: dict[str, dict] = {}

    async def run(self):
        async with aiomqtt.Client(MQTT_BROKER_URL) as c:
            await c.subscribe("babel/dev/+/sensor/+")
            async for msg in c.messages:
                ...

    def latest(self, device_id: str | None = None) -> dict:
        ...

    async def send_ir(self, device_id: str, protocol: str, code: int):
        ...
```

Lifecycle managed by the Pipecat runner — start at boot, stop on shutdown. Single-device deployments default `device_id` to "the only one we've heard from."

### `skills/device/query_sensors/` (new)

Following the pattern in `skills/spotify/whats_playing/` and `skills/shows/play_bbc_show/`:

**`SKILL.md`:**

```yaml
name: query_sensors
description: Read the current temperature, humidity, or presence from the room sensor.
category: device
enabled_when: BABEL_SENSORS_ENABLED
requires: [sensor_cache]
parameters:
  metric:
    type: string
    description: "temp | humidity | radar | all"
    required: false
```

**`handler.py`:**

```python
async def handle(params: FunctionCallParams, ctx: SkillContext) -> None:
    metric = params.arguments.get("metric", "all")
    state = ctx.sensor_cache.latest()
    if metric == "temp":
        await params.result_callback(f"It's {state['temp']['c']:.1f} degrees Celsius.")
    elif metric == "humidity":
        await params.result_callback(f"Humidity is {state['humidity']['rh']:.0f} percent.")
    elif metric == "radar":
        presence = "occupied" if state["radar"]["present"] else "empty"
        await params.result_callback(f"The room is {presence}.")
    else:
        await params.result_callback(
            f"It's {state['temp']['c']:.1f}°C, {state['humidity']['rh']:.0f}% humidity, "
            f"and the room is {'occupied' if state['radar']['present'] else 'empty'}."
        )
```

Register through `skills/_loader.py` like the existing skills.

### `.env.example` additions

```
BABEL_SENSORS_ENABLED=true
MQTT_BROKER_URL=mqtt://localhost:1883
MQTT_DEVICE_FILTER=                # blank = accept all device IDs
```

### Optional follow-ups (beyond v1)

- **Presence-triggered behaviors.** `mqtt_bridge.py` fires Pipecat frames on radar transitions: "person entered" → unsolicited greeting via TTS.
- **IR control skill.** `skills/device/send_ir/` with a YAML codebook (`tv_off: {protocol: nec, code: 0x20DF10EF}`) and an `mqtt_publish` to the device's `cmd/ir_tx`.

---

## Build order

1. **Firmware sensor module standalone.** Build `sensors.c`, flash, log raw values to UART. No MQTT yet. Confirm SHT40 reads sane temps, radar fires on hand-wave.
2. **MQTT client + topic plumbing.** Add `mqtt_client.c`, point at a local Mosquitto, watch with `mosquitto_sub -t 'babel/#' -v`.
3. **Command path.** `mosquitto_pub` to `cmd/query` and `cmd/ir_tx`, confirm device responds.
4. **Backend deferred work.** Mosquitto in compose, `mqtt_bridge.py`, `query_sensors` skill, wire `ctx.sensor_cache` into the `SkillContext`.
5. **End-to-end voice query.** "Hey babel, what's the temperature?" → skill reads cache → LLM responds.

---

## Verification

### Firmware alone, no backend

```sh
# Run a local broker on the dev box
docker run --rm -it -p 1883:1883 eclipse-mosquitto:2

# Subscribe to everything from any Box-3
mosquitto_sub -h localhost -t 'babel/dev/+/#' -v

# Flash + watch
idf.py -p /dev/ttyACM0 flash monitor
```

Expected, within a few seconds of boot:

```
babel/dev/box3-a1b2c3d4e5f6/status online
babel/dev/box3-a1b2c3d4e5f6/sensor/temp     {"c":22.4,"ts":...}
babel/dev/box3-a1b2c3d4e5f6/sensor/humidity {"rh":45.2,"ts":...}
babel/dev/box3-a1b2c3d4e5f6/sensor/radar    {"present":false,"distance_cm":0,"energy":0,"ts":...}
```

Wave a hand in front of the radar:

```
babel/dev/.../sensor/radar {"present":true,"distance_cm":80,"energy":92,"ts":...}
```

Force a publish:

```sh
mosquitto_pub -h localhost -t 'babel/dev/<id>/cmd/query' -m '{"sensor":"all"}'
# device republishes all three sensor topics
```

Test IR loopback (point the device at itself or a known remote):

```sh
mosquitto_sub -h localhost -t 'babel/dev/+/sensor/ir_rx' -v
# press buttons on a NEC-protocol remote, expect:
#   babel/dev/.../sensor/ir_rx {"protocol":"nec","code":"0x20DF10EF",...}
```

Yank power from the device, watch the LWT fire:

```
babel/dev/box3-a1b2c3d4e5f6/status offline
```

### End-to-end (after deferred backend work lands)

- Start broker + voice-chatbot with `BABEL_SENSORS_ENABLED=true`.
- Say "hey babel, what's the temperature?" → expect TTS response within ~1 s reading from the cache (no per-query round-trip to the device).
- Walk away from the sensor → radar `present:false` after debounce → cache reflects it.
- "Hey babel, is anyone home?" → "The room is empty." (or "occupied" if standing in front).
