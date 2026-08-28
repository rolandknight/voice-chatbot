// WebRTC smoke-test browser client.
//
// Captures the mic, opens a peer connection to the local aiortc server,
// adds a "control" DataChannel that speaks the same JSON protocol the real
// backend will speak, and plays the looped-back audio.

const els = {
  connect: document.getElementById("connect"),
  disconnect: document.getElementById("disconnect"),
  send: document.getElementById("send"),
  msg: document.getElementById("msg"),
  remote: document.getElementById("remote"),
  log: document.getElementById("log"),
  pillPc: document.getElementById("pill-pc"),
  pillIce: document.getElementById("pill-ice"),
  pillDc: document.getElementById("pill-dc"),
};

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

async function connect() {
  els.connect.disabled = true;

  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    log("err",
      `mic access blocked: this page is not a secure context (origin = ${location.origin}). ` +
      `Browsers only expose getUserMedia on https:// or localhost. ` +
      `Restart the server with \`make run-webrtc-smoke-lan\` and open the https:// URL.`);
    els.connect.disabled = false;
    return;
  }

  try {
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (e) {
    log("err", `getUserMedia failed: ${e.message}`);
    els.connect.disabled = false;
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
    log("warn", "datachannel closed");
  });
  dc.addEventListener("message", (e) => {
    log("recv", e.data);
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
  els.disconnect.disabled = true;
  els.send.disabled = true;
  setPill(els.pillPc, "pc: new");
  setPill(els.pillIce, "ice: new");
  setPill(els.pillDc, "dc: closed");
}

els.connect.addEventListener("click", () => { connect().catch((e) => log("err", e.message)); });
els.disconnect.addEventListener("click", () => { log("warn", "disconnect"); teardown(); });
els.send.addEventListener("click", () => {
  const raw = els.msg.value.trim() || '{"type":"hello","kind":"simple","capabilities":["audio"]}';
  let obj;
  try { obj = JSON.parse(raw); }
  catch (e) { log("err", `not JSON: ${e.message}`); return; }
  sendControl(obj);
});
els.msg.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !els.send.disabled) els.send.click();
});
