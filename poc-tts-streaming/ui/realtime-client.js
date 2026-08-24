// Minimal OpenAI-Realtime-over-WebRTC client for text-to-speech.
// The only thing that should differ between this server and api.openai.com
// is `baseUrl` (and, there, a real API key): the flow below is the one
// OpenAI documents -- client secret, SDP offer to /v1/realtime/calls,
// "oai-events" data channel, audio on the media track.
class RealtimeTtsClient {
  constructor({ baseUrl = "", apiKey = null, session = {}, model = "chatterbox-flash", iceServers = [] } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.sessionPatch = session;
    this.model = model;
    this.iceServers = iceServers;
    this.pc = null; this.dc = null; this.callLocation = null;
    this.remoteStream = null;
    this.state = { pc: "new", ice: "new", dc: "closed" };
    this._handlers = new Map();
    this._pending = new Map();   // client event_id -> {resolve, reject, okType}
    this._seq = 0;
  }

  on(type, fn) { (this._handlers.get(type) || this._handlers.set(type, []).get(type)).push(fn); return this; }
  off(type, fn) { const h = this._handlers.get(type) || []; this._handlers.set(type, h.filter(x => x !== fn)); }
  _emit(type, ev) { for (const fn of [...(this._handlers.get(type) || []), ...(this._handlers.get("*") || [])]) fn(ev); }

  async _clientSecret() {
    if (this.apiKey) return this.apiKey;   // against OpenAI, mint the ephemeral key server-side; here a raw key is fine for a manual check
    const r = await fetch(`${this.baseUrl}/v1/realtime/client_secrets`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ session: { type: "realtime", model: this.model, ...this.sessionPatch } }),
    });
    if (!r.ok) throw new Error(`client_secrets ${r.status}: ${await r.text()}`);
    return (await r.json()).value;
  }

  async connect() {
    const key = await this._clientSecret();
    const pc = new RTCPeerConnection({ iceServers: this.iceServers });
    this.pc = pc;
    pc.addEventListener("connectionstatechange", () => { this.state.pc = pc.connectionState; this._emit("state", this.state); });
    pc.addEventListener("iceconnectionstatechange", () => { this.state.ice = pc.iceConnectionState; this._emit("state", this.state); });
    pc.addEventListener("track", (e) => { this.remoteStream = e.streams[0]; this._emit("track", e.streams[0]); });
    pc.addTransceiver("audio", { direction: "recvonly" });

    const dc = pc.createDataChannel("oai-events");
    this.dc = dc;
    const ready = new Promise((resolve, reject) => {
      this._onceType("conversation.created", resolve);
      dc.addEventListener("close", () => reject(new Error("data channel closed")), { once: true });
    });
    dc.addEventListener("open", () => { this.state.dc = "open"; this._emit("state", this.state); });
    dc.addEventListener("close", () => { this.state.dc = "closed"; this._emit("state", this.state); });
    dc.addEventListener("message", (e) => this._onServerEvent(JSON.parse(e.data)));

    await pc.setLocalDescription(await pc.createOffer());
    await this._waitIce(pc);
    const r = await fetch(`${this.baseUrl}/v1/realtime/calls?model=${encodeURIComponent(this.model)}`, {
      method: "POST", headers: { "content-type": "application/sdp", authorization: `Bearer ${key}` },
      body: pc.localDescription.sdp,
    });
    if (!r.ok) throw new Error(`calls ${r.status}: ${await r.text()}`);
    this.callLocation = r.headers.get("location");
    await pc.setRemoteDescription({ type: "answer", sdp: await r.text() });
    await ready;
  }

  _waitIce(pc) {
    if (pc.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve) => {
      const check = () => { if (pc.iceGatheringState === "complete") { pc.removeEventListener("icegatheringstatechange", check); resolve(); } };
      pc.addEventListener("icegatheringstatechange", check);
      setTimeout(resolve, 1500);
    });
  }

  _onceType(type, fn) { const h = (ev) => { this.off(type, h); fn(ev); }; this.on(type, h); }

  _onServerEvent(ev) {
    this._emit(ev.type, ev);
    if (ev.type === "error" && ev.error?.event_id && this._pending.has(ev.error.event_id)) {
      this._pending.get(ev.error.event_id).reject(new Error(`${ev.error.code}: ${ev.error.message}`));
      this._pending.delete(ev.error.event_id);
    }
    for (const [id, p] of this._pending) {
      if (ev.type === p.okType) { this._pending.delete(id); p.resolve(ev); break; }  // Map iterates in insertion order: oldest first
    }
  }

  send(event, okType = null) {
    if (!this.dc || this.dc.readyState !== "open") return Promise.reject(new Error("not connected"));
    const event_id = `evt_${++this._seq}`;
    const full = { event_id, ...event };
    this.dc.send(JSON.stringify(full));
    this._emit("client-event", full);
    if (!okType) return Promise.resolve(null);
    return new Promise((resolve, reject) => this._pending.set(event_id, { resolve, reject, okType }));
  }

  updateSession(patch) { return this.send({ type: "session.update", session: patch }, "session.updated"); }

  async speak(text, responsePatch = {}) {
    await this.send({ type: "conversation.item.create", item: {
      type: "message", role: "user", content: [{ type: "input_text", text }] } }, "conversation.item.done");
    const created = await this.send({ type: "response.create", response: responsePatch }, "response.created");
    return created.response.id;
  }

  cancel() { return this.send({ type: "response.cancel" }); }
  clear() { return this.send({ type: "output_audio_buffer.clear" }); }

  async disconnect() {
    try { if (this.callLocation) await fetch(`${this.baseUrl}${this.callLocation}`, { method: "DELETE" }); } catch {}
    try { this.dc?.close(); } catch {}
    try { this.pc?.close(); } catch {}
    this.pc = null; this.dc = null; this.remoteStream = null;
    this.state = { pc: "new", ice: "new", dc: "closed" };
    this._emit("state", this.state);
  }
}
window.RealtimeTtsClient = RealtimeTtsClient;
