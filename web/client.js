// voice-chatbot browser client.
//
// WebRTC peer connection + JSON-over-DataChannel control protocol that
// matches docs/web-rtc.md. Sends a `hello` advertising kind=simple, then
// renders `state` / `transcript` / `ready` messages from the server.

const els = {
  connect: document.getElementById("connect"),
  listen: document.getElementById("listen"),
  disconnect: document.getElementById("disconnect"),
  send: document.getElementById("send"),
  msg: document.getElementById("msg"),
  remote: document.getElementById("remote"),
  log: document.getElementById("log"),
  stateBig: document.getElementById("state-big"),
  transcript: document.getElementById("transcript"),
  pillPc: document.getElementById("pill-pc"),
  pillIce: document.getElementById("pill-ice"),
  pillDc: document.getElementById("pill-dc"),
  pillMode: document.getElementById("pill-mode"),
  pillWake: document.getElementById("pill-wake"),
  backend: document.getElementById("backend"),
  persona: document.getElementById("persona"),
};

let serverModes = ["push"]; // populated from /api/options

let pc = null;
let dc = null;
let localStream = null;

function log(kind, msg) {
  const ts = new Date().toISOString().split("T")[1].replace("Z", "");
  const text = typeof msg === "string" ? msg : JSON.stringify(msg);
  const line = document.createElement("div");
  line.innerHTML = `<span class="ts">${ts}</span> <span class="${kind}">${escape(text)}</span>`;
  els.log.appendChild(line);
  els.log.scrollTop = els.log.scrollHeight;
}

function escape(s) {
  return s.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function setPill(el, label, cls) {
  el.textContent = label;
  el.className = "pill" + (cls ? " " + cls : "");
}

function setState(state) {
  els.stateBig.textContent = state;
  els.stateBig.className = ["listening", "thinking", "speaking"].includes(state) ? state : "";
}

async function connect(mode = "push") {
  els.connect.disabled = true;
  els.listen.disabled = true;
  setPill(els.pillMode, `mode: ${mode}`, mode === "wake" ? "warm" : "ok");
  if (mode === "wake") {
    els.pillWake.style.display = "";
    setPill(els.pillWake, "wake: asleep");
  } else {
    els.pillWake.style.display = "none";
  }

  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    log("err",
      `mic access blocked: this page is not a secure context (origin = ${location.origin}). ` +
      `Browsers only expose getUserMedia on https:// or localhost. ` +
      `Restart the server with \`make run-server-lan\` and open the https:// URL.`);
    teardown();
    return;
  }

  try {
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (e) {
    log("err", `getUserMedia failed: ${e.message}`);
    teardown();
    return;
  }

  pc = new RTCPeerConnection({
    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
  });

  pc.addEventListener("connectionstatechange", () => {
    setPill(els.pillPc, `pc: ${pc.connectionState}`,
      pc.connectionState === "connected" ? "ok" :
      pc.connectionState === "failed" ? "err" : "");
    log("warn", `pc state: ${pc.connectionState}`);
    if (pc.connectionState === "failed" || pc.connectionState === "closed") {
      teardown();
    }
  });

  pc.addEventListener("iceconnectionstatechange", () => {
    setPill(els.pillIce, `ice: ${pc.iceConnectionState}`,
      pc.iceConnectionState === "connected" || pc.iceConnectionState === "completed" ? "ok" :
      pc.iceConnectionState === "failed" ? "err" : "");
  });

  pc.addEventListener("track", (e) => {
    log("recv", `inbound track kind=${e.track.kind}`);
    els.remote.srcObject = e.streams[0];
  });

  dc = pc.createDataChannel("control");
  dc.addEventListener("open", () => {
    setPill(els.pillDc, "dc: open", "ok");
    els.send.disabled = false;
    els.backend.disabled = false;
    els.persona.disabled = false;
    log("warn", "datachannel open");
    sendControl({
      type: "hello",
      kind: "simple",
      capabilities: ["audio"],
    });
  });
  dc.addEventListener("close", () => {
    setPill(els.pillDc, "dc: closed");
    els.send.disabled = true;
    els.backend.disabled = true;
    els.persona.disabled = true;
    log("warn", "datachannel closed");
  });
  dc.addEventListener("message", (e) => {
    log("recv", e.data);
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type === "state" && typeof msg.state === "string") {
      setState(msg.state);
    } else if (msg.type === "transcript" && typeof msg.text === "string") {
      els.transcript.textContent = msg.text;
    } else if (msg.type === "ready") {
      // Server is source of truth for current backend/persona/mode — sync UI.
      if (Array.isArray(msg.available_backends)) {
        fillSelect(els.backend, msg.available_backends, msg.backend);
      } else if (typeof msg.backend === "string") {
        els.backend.value = msg.backend;
      }
      if (Array.isArray(msg.available_personas)) {
        fillSelect(els.persona, msg.available_personas, msg.persona);
      } else if (typeof msg.persona === "string") {
        els.persona.value = msg.persona;
      }
      if (typeof msg.mode === "string") {
        setPill(els.pillMode, `mode: ${msg.mode}`, msg.mode === "wake" ? "warm" : "ok");
        els.pillWake.style.display = msg.mode === "wake" ? "" : "none";
      }
    } else if (msg.type === "backend" && typeof msg.name === "string") {
      els.backend.value = msg.name;
    } else if (msg.type === "persona" && typeof msg.name === "string") {
      els.persona.value = msg.name;
    } else if (msg.type === "wake" && typeof msg.state === "string") {
      const tag = msg.state === "awake"
        ? `wake: awake${msg.model ? " ("+msg.model+")" : ""}`
        : "wake: asleep";
      setPill(els.pillWake, tag, msg.state === "awake" ? "ok" : "");
    }
  });

  for (const track of localStream.getTracks()) {
    pc.addTrack(track, localStream);
  }

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitIceGatheringComplete(pc);

  const res = await fetch("/api/offer", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      mode,
    }),
  });
  if (!res.ok) {
    log("err", `offer failed: ${res.status} ${await res.text()}`);
    teardown();
    return;
  }
  const answer = await res.json();
  await pc.setRemoteDescription(answer);
  log("warn", "remote answer applied");

  els.disconnect.disabled = false;
}

function waitIceGatheringComplete(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    pc.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 1500);
  });
}

function sendControl(obj) {
  if (!dc || dc.readyState !== "open") return;
  const text = JSON.stringify(obj);
  dc.send(text);
  log("send", text);
}

function teardown() {
  if (dc) { try { dc.close(); } catch {} dc = null; }
  if (pc) { try { pc.close(); } catch {} pc = null; }
  if (localStream) {
    for (const t of localStream.getTracks()) t.stop();
    localStream = null;
  }
  els.remote.srcObject = null;
  els.connect.disabled = false;
  els.listen.disabled = !serverModes.includes("wake");
  els.disconnect.disabled = true;
  els.send.disabled = true;
  setPill(els.pillPc, "pc: new");
  setPill(els.pillIce, "ice: new");
  setPill(els.pillDc, "dc: closed");
  setPill(els.pillMode, "mode: —");
  els.pillWake.style.display = "none";
  setState("idle");
}

async function populateOptions() {
  try {
    const r = await fetch("/api/options");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const opts = await r.json();
    fillSelect(els.backend, opts.backends, opts.default_backend);
    fillSelect(els.persona, opts.personas, opts.default_persona);
    serverModes = Array.isArray(opts.modes) ? opts.modes : ["push"];
    els.listen.disabled = !serverModes.includes("wake");
    if (!serverModes.includes("wake")) {
      els.listen.title = "wake mode unavailable — no usable wake models loaded on the server";
    } else {
      els.listen.title = `wake words: ${(opts.wake_models || []).join(", ") || "configured"}`;
    }
    log("warn",
      `options: backends=${opts.backends.join(",")} personas=${opts.personas.join(",")} ` +
      `modes=${serverModes.join(",")}` +
      (opts.wake_models?.length ? ` wake=${opts.wake_models.join(",")}` : "")
    );
  } catch (e) {
    log("err", `failed to load /api/options: ${e.message}`);
  }
}

function fillSelect(select, values, defaultValue) {
  select.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === defaultValue) opt.selected = true;
    select.appendChild(opt);
  }
}

els.backend.addEventListener("change", () => {
  sendControl({ type: "backend", name: els.backend.value });
});
els.persona.addEventListener("change", () => {
  sendControl({ type: "persona", name: els.persona.value });
});

els.connect.addEventListener("click", () => { connect("push").catch((e) => log("err", e.message)); });
els.listen.addEventListener("click", () => { connect("wake").catch((e) => log("err", e.message)); });
els.disconnect.addEventListener("click", () => { log("warn", "disconnect"); teardown(); });
els.send.addEventListener("click", () => {
  const raw = els.msg.value.trim();
  if (!raw) return;
  let obj;
  try { obj = JSON.parse(raw); }
  catch (e) { log("err", `not JSON: ${e.message}`); return; }
  sendControl(obj);
});
els.msg.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !els.send.disabled) els.send.click();
});

populateOptions();
