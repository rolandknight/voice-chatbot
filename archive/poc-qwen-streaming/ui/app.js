// poc-qwen-streaming browser client.
//
// One WebSocket per page; each Generate sends a JSON request and then plays
// int16 PCM frames the moment they arrive through Web Audio, scheduling each
// buffer to start where the previous one ends. Client-side TTFA is measured
// from the send of the request to the first binary frame. When the server
// says done, the frames are also assembled into a WAV for the <audio> replay.

const $ = (id) => document.getElementById(id);
const state = { ws: null, ctx: null, nextT: 0, frames: [], sr: 24000, t0: 0, ttfa: null, busy: false, ref: null, rec: null };

// ---- catalogue -------------------------------------------------------------
async function loadCatalog() {
  const c = await (await fetch('/api/catalog')).json();
  for (const sel of document.querySelectorAll('select.lang')) {
    sel.innerHTML = c.languages.map((l) => `<option>${l}</option>`).join('');
  }
  $('t_speaker').innerHTML = c.speakers.map((s) => `<option>${s}</option>`).join('');
  $('t_size').innerHTML = c.sizes.map((s) => `<option ${s === '1.7B' ? 'selected' : ''}>${s}</option>`).join('');
  $('c_size').innerHTML = c.sizes.map((s) => `<label><input type="radio" name="c_size" value="${s}" ${s === '1.7B' ? 'checked' : ''}>${s}</label>`).join('');
  $('c_preset').innerHTML = '<option value="">—</option>' + c.voices.map((v) => `<option value="${v.name}" data-transcript="${encodeURIComponent(v.transcript)}">${v.name}</option>`).join('');
}

async function refreshInfo() {
  try {
    const i = await (await fetch('/api/info')).json();
    const resident = (i.resident || []).map((m) => m.split('/').pop()).join(', ') || 'none';
    const p = i.preload || {};
    const pre = p.state === 'running' ? ` · ⏳ preloading ${p.done.length}/${p.done.length + p.pending.length} (${p.pending[0] || ''})` : p.state === 'done' ? ` · preloaded in ${p.s} s${p.errors?.length ? ` (${p.errors.length} errors)` : ''}` : '';
    $('info').textContent = `${i.chip || '?'} · mlx ${i.mlx || '?'} · mlx-audio ${i.mlx_audio || '?'} · resident: ${resident} · active ${i.active_gb ?? '?'} GiB · peak ${i.peak_gb ?? '?'} GiB${pre}`;
    if (p.state === 'running') setTimeout(refreshInfo, 1000);
  } catch (e) {
    $('info').textContent = `info failed: ${e}`;
  }
}

// ---- tabs ------------------------------------------------------------------
for (const b of document.querySelectorAll('.tab')) {
  b.onclick = () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === b));
    document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${b.dataset.tab}`));
  };
}

// ---- reference audio (clone tab) ------------------------------------------
function setRef(ref, hint) {
  state.ref = ref; // {preset} | {path}
  $('c_ref_hint').textContent = hint;
}

$('c_preset').onchange = (e) => {
  const opt = e.target.selectedOptions[0];
  if (!opt.value) return setRef(null, 'No reference selected.');
  setRef({ preset: opt.value }, `Preset: ${opt.value}`);
  $('c_ref_player').src = `/voice/${encodeURIComponent(opt.value)}`;
  $('c_ref_text').value = decodeURIComponent(opt.dataset.transcript || '');
  if (!$('c_ref_text').value) transcribeRef();
};

async function uploadBlob(blob, name) {
  const fd = new FormData();
  fd.append('file', blob, name);
  const r = await (await fetch('/api/upload', { method: 'POST', body: fd })).json();
  if (r.error) throw new Error(r.error);
  return r.path;
}

$('c_file').onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $('c_preset').value = '';
  $('c_ref_player').src = URL.createObjectURL(f);
  const path = await uploadBlob(f, f.name);
  setRef({ path }, `Uploaded: ${f.name}`);
  $('c_ref_text').value = '';
};

// Mic: capture raw PCM with a ScriptProcessor and encode WAV ourselves, so
// the server gets a container mlx-audio can load without ffmpeg guesswork.
$('c_rec').onclick = async () => {
  if (state.rec) return stopRec();
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  proc.onaudioprocess = (ev) => chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
  src.connect(proc);
  proc.connect(ctx.destination);
  state.rec = { stream, ctx, proc, chunks };
  $('c_rec').textContent = '■ Stop';
  $('c_rec').classList.add('rec');
};

async function stopRec() {
  const { stream, ctx, proc, chunks } = state.rec;
  state.rec = null;
  proc.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const sr = ctx.sampleRate;
  await ctx.close();
  $('c_rec').textContent = '● Record';
  $('c_rec').classList.remove('rec');
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const pcm = new Int16Array(n);
  let o = 0;
  for (const c of chunks) for (let i = 0; i < c.length; i++) pcm[o++] = Math.max(-1, Math.min(1, c[i])) * 32767;
  const blob = wavBlob(pcm, sr);
  $('c_preset').value = '';
  $('c_ref_player').src = URL.createObjectURL(blob);
  const path = await uploadBlob(blob, 'mic.wav');
  setRef({ path }, `Recorded ${(n / sr).toFixed(1)} s`);
  $('c_ref_text').value = '';
}

async function transcribeRef() {
  if (!state.ref) return;
  let path = state.ref.path;
  status('Transcribing…');
  const body = state.ref.preset ? { preset: state.ref.preset } : { path };
  // Presets are resolved server-side via /voice; for transcribe we need the path: ask the catalog.
  if (state.ref.preset) {
    const c = await (await fetch('/api/catalog')).json();
    path = (c.voices.find((v) => v.name === state.ref.preset) || {}).path;
  }
  const r = await (await fetch('/api/transcribe', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ path }) })).json();
  if (r.error) return status(r.error, true);
  $('c_ref_text').value = r.text;
  status('Transcribed.');
}
$('c_transcribe').onclick = transcribeRef;

// ---- WAV encode --------------------------------------------------------------
function wavBlob(pcm, sr) {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: 'audio/wav' });
}

// ---- generation over WebSocket ----------------------------------------------
function status(msg, err = false) {
  $('status').innerHTML = msg;
  $('status').classList.toggle('err', err);
}

function ws() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) return Promise.resolve(state.ws);
  return new Promise((res, rej) => {
    const s = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
    s.binaryType = 'arraybuffer';
    s.onopen = () => res(s);
    s.onerror = (e) => rej(e);
    s.onmessage = onMessage;
    s.onclose = () => { state.ws = null; };
    state.ws = s;
  });
}

function playFrame(i16) {
  const ctx = state.ctx;
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  const buf = ctx.createBuffer(1, f32.length, state.sr);
  buf.getChannelData(0).set(f32);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const now = ctx.currentTime;
  if (state.nextT < now) state.nextT = now + 0.02; // (re)start with a 20 ms cushion
  src.start(state.nextT);
  state.nextT += buf.duration;
}

function onMessage(ev) {
  if (ev.data instanceof ArrayBuffer) {
    const i16 = new Int16Array(ev.data);
    if (state.ttfa === null) {
      state.ttfa = (performance.now() - state.t0) / 1000;
      status(`First audio in <b>${state.ttfa.toFixed(3)} s</b> — streaming…`);
    }
    state.frames.push(i16);
    playFrame(i16);
    return;
  }
  const m = JSON.parse(ev.data);
  if (m.type === 'start') {
    state.sr = m.sample_rate;
    if (!state.ctx || state.ctx.sampleRate !== m.sample_rate) {
      if (state.ctx) state.ctx.close();
      state.ctx = new AudioContext({ sampleRate: m.sample_rate });
    }
    state.ctx.resume();
    state.nextT = 0;
  } else if (m.type === 'done') {
    const t = m.timings;
    const n = state.frames.reduce((a, f) => a + f.length, 0);
    const all = new Int16Array(n);
    let o = 0;
    for (const f of state.frames) { all.set(f, o); o += f.length; }
    $('out').src = URL.createObjectURL(wavBlob(all, state.sr));
    status(`<b>${(t.model || '').split('/').pop()}</b> · ${t.chars} chars → ${t.audio_s.toFixed(2)} s audio in ${t.gen_s.toFixed(2)} s (RTF ${t.rtf?.toFixed(2)}, ${t.chunks} chunks) · TTFA browser <b>${state.ttfa?.toFixed(3)} s</b> / server ${t.ttfa_s?.toFixed(3)} s`);
    finish();
  } else if (m.type === 'error') {
    status(`❌ ${m.message}`, true);
    finish();
  }
}

function finish() {
  state.busy = false;
  $('cancel').disabled = true;
  document.querySelectorAll('.generate').forEach((b) => (b.disabled = false));
  refreshInfo();
}

function params(tab) {
  if (tab === 'design') return { text: $('d_text').value, language: $('d_lang').value, instruct: $('d_instruct').value };
  if (tab === 'custom') return { text: $('t_text').value, language: $('t_lang').value, speaker: $('t_speaker').value, instruct: $('t_instruct').value, size: $('t_size').value };
  const size = document.querySelector('input[name=c_size]:checked')?.value || '1.7B';
  const p = { text: $('c_text').value, language: $('c_lang').value, size, ref_text: $('c_ref_text').value, xvector_only: $('c_xvec').checked };
  if (!state.ref) throw new Error('Upload or record a reference clip, or pick a preset voice.');
  if (state.ref.preset) p.preset = state.ref.preset; else p.ref_audio = state.ref.path;
  return p;
}

for (const b of document.querySelectorAll('.generate')) {
  b.onclick = async () => {
    if (state.busy) return;
    let p;
    try { p = params(b.dataset.tab); } catch (e) { return status(`❌ ${e.message}`, true); }
    state.busy = true;
    state.frames = [];
    state.ttfa = null;
    document.querySelectorAll('.generate').forEach((x) => (x.disabled = true));
    $('cancel').disabled = false;
    status('Generating…');
    // Create the AudioContext on the click so autoplay policy lets us play immediately.
    if (!state.ctx) state.ctx = new AudioContext({ sampleRate: 24000 });
    state.ctx.resume();
    const s = await ws();
    state.t0 = performance.now();
    s.send(JSON.stringify({ type: 'generate', tab: b.dataset.tab, ...p }));
  };
}

$('cancel').onclick = () => {
  if (state.ws) state.ws.send(JSON.stringify({ type: 'cancel' }));
  status('Stopped.');
  finish();
};

$('unload').onclick = async () => {
  await fetch('/api/unload', { method: 'POST' });
  refreshInfo();
};

loadCatalog().then(refreshInfo);
