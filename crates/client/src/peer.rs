//! A single-owner, sans-I/O driver for the native WebRTC audio peer.
//!
//! `str0m` deliberately does not own sockets or clocks.  The important rule in
//! this module is that every mutation of `Rtc` is immediately followed by a
//! complete `poll_output` drain.  Only after that drain do we await socket I/O
//! or another application event.

use std::collections::VecDeque;
use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use audiopus::coder::{Decoder, Encoder};
use audiopus::{Application, Channels, SampleRate};
use str0m::change::{SdpAnswer, SdpPendingOffer};
use str0m::format::Codec;
use str0m::media::{Direction, Frequency, MediaData, MediaKind, MediaTime, Mid, Pt};
use str0m::net::{Protocol, Receive, Transmit};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use crate::resampler::StreamingResampler;

const RTP_RATE: u32 = 48_000;
const OPUS_FRAME_SAMPLES: usize = 960;
const OPUS_MAX_PACKET_BYTES: usize = 1_275;
const OPUS_MAX_DECODE_SAMPLES: usize = 5_760;
const UDP_RECEIVE_BYTES: usize = 2_000;
const MEDIA_FRAME_DURATION: Duration = Duration::from_millis(20);
const CAPTURE_BUFFER_FRAMES: usize = 5;
const MAX_PLAYBACK_SECONDS: usize = 10;

/// An offer-side peer whose SDP answer has not yet been applied.
///
/// The retained `Mid` and `SdpPendingOffer` belong to the exact `Rtc` which
/// generated `offer_sdp`; keeping them together prevents an answer from being
/// accidentally applied to a different negotiation.
pub struct PendingPeer {
    rtc: Rtc,
    socket: UdpSocket,
    local_addr: SocketAddr,
    mid: Mid,
    pending: SdpPendingOffer,
    offer_sdp: String,
    actions: VecDeque<DrainedAction>,
}

/// The local interface address that routes to `server` — a connected UDP
/// socket's local address; no packet is sent. Loopback servers yield
/// 127.0.0.1, LAN servers the LAN interface, so the ICE host candidate we
/// advertise is one the server can actually reach.
pub fn local_ip_toward(server: SocketAddr) -> Result<IpAddr> {
    let probe = std::net::UdpSocket::bind(match server {
        SocketAddr::V4(_) => SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
        SocketAddr::V6(_) => SocketAddr::from((std::net::Ipv6Addr::UNSPECIFIED, 0)),
    })
    .context("bind route-probe UDP socket")?;
    probe
        .connect(server)
        .with_context(|| format!("no route to WebRTC server {server}"))?;
    Ok(probe.local_addr().context("read route-probe address")?.ip())
}

impl PendingPeer {
    /// Bind a loopback UDP socket and build an Opus-only, send/receive offer
    /// (same-machine server; kept for tests and the default URL).
    pub async fn create() -> Result<Self> {
        Self::create_on(IpAddr::V4(Ipv4Addr::LOCALHOST)).await
    }

    /// Bind on the interface that routes to `server` and advertise it as the
    /// host ICE candidate, so a server on another machine can pair with us.
    pub async fn create_toward(server: SocketAddr) -> Result<Self> {
        Self::create_on(local_ip_toward(server)?).await
    }

    async fn create_on(ip: IpAddr) -> Result<Self> {
        let socket = UdpSocket::bind((ip, 0))
            .await
            .with_context(|| format!("bind WebRTC UDP socket on {ip}"))?;
        let local_addr = socket
            .local_addr()
            .context("read WebRTC UDP socket address")?;

        let mut rtc = Rtc::builder()
            .clear_codecs()
            .enable_opus(true)
            .build(Instant::now());
        let mut actions = VecDeque::new();

        let candidate =
            Candidate::host(local_addr, "udp").context("construct WebRTC host ICE candidate")?;
        let candidate_added = rtc.add_local_candidate(candidate).is_some();
        // add_local_candidate is a mutation, even when it rejects a candidate.
        let candidate_drain = drain_rtc(&mut rtc, &mut actions);
        if !candidate_added {
            bail!("str0m rejected the WebRTC host ICE candidate {local_addr}");
        }
        let _ = candidate_drain?;

        let mut change = rtc.sdp_api();
        let mid = change.add_media(MediaKind::Audio, Direction::SendRecv, None, None, None);
        let offer_result = change
            .apply()
            .ok_or_else(|| anyhow!("str0m did not generate an SDP offer"));
        // Applying the SDP change is a mutation.
        let offer_drain = drain_rtc(&mut rtc, &mut actions);
        let (offer, pending) = offer_result?;
        let _ = offer_drain?;

        Ok(Self {
            rtc,
            socket,
            local_addr,
            mid,
            pending,
            offer_sdp: offer.to_sdp_string(),
            actions,
        })
    }

    /// The raw SDP body to send to the server's `/webrtc/offer` endpoint.
    pub fn offer_sdp(&self) -> &str {
        &self.offer_sdp
    }

    /// Parse and apply the server's raw SDP answer.
    pub fn accept_answer(self, answer_sdp: &str) -> Result<Peer> {
        let answer = SdpAnswer::from_sdp_string(answer_sdp)
            .context("parse the server's WebRTC SDP answer")?;
        let Self {
            mut rtc,
            socket,
            local_addr,
            mid,
            pending,
            offer_sdp: _,
            mut actions,
        } = self;

        let accept_result = rtc.sdp_api().accept_answer(pending, answer);
        // accept_answer may partially mutate before reporting an error, so the
        // drain is unconditional and happens before propagating either result.
        let deadline_result = drain_rtc(&mut rtc, &mut actions);
        accept_result.context("apply the server's WebRTC SDP answer")?;
        let _ = deadline_result?;

        // Resolve and retain the negotiated payload type once.  `writer` takes
        // `&mut Rtc`, so conservatively drain again before handing out Peer.
        let opus_pt_result = negotiated_opus_pt(&mut rtc, mid);
        let deadline_result = drain_rtc(&mut rtc, &mut actions);
        let opus_pt = opus_pt_result?;
        let deadline = deadline_result?;

        Ok(Peer {
            rtc,
            socket,
            local_addr,
            mid,
            opus_pt,
            deadline,
            actions,
        })
    }
}

/// A negotiated WebRTC audio peer.
pub struct Peer {
    rtc: Rtc,
    socket: UdpSocket,
    local_addr: SocketAddr,
    mid: Mid,
    opus_pt: Pt,
    deadline: Instant,
    actions: VecDeque<DrainedAction>,
}

impl Peer {
    /// Drive WebRTC and bridge mono device PCM until `cancel` completes.
    ///
    /// `input` is expected to be a bounded Tokio channel fed by the capture
    /// callback. `output` is a bounded standard-library synchronous channel,
    /// and `voice_recycle` carries its spent buffers back to be refilled here:
    /// the callback must never free one, since `free` can take the allocator's
    /// lock.
    /// Incoming RTP bursts are paced into that channel at their decoded sample
    /// rate, and neither direction blocks the RTC loop behind an audio device.
    pub async fn run<F>(
        mut self,
        input_rate: u32,
        output_rate: u32,
        mut input: mpsc::Receiver<Vec<i16>>,
        output: SyncSender<Vec<i16>>,
        voice_recycle: Receiver<Vec<i16>>,
        cancel: F,
    ) -> Result<()>
    where
        F: Future<Output = ()>,
    {
        let runtime = AudioRuntime::new(input_rate, output_rate, voice_recycle);
        let mut runtime = match runtime {
            Ok(runtime) => runtime,
            Err(error) => {
                if let Err(close_error) = self.close_and_drain().await {
                    tracing::warn!(%close_error, "failed to close peer after audio setup error");
                }
                return Err(error);
            }
        };
        let mut udp_buffer = vec![0_u8; UDP_RECEIVE_BYTES];
        let cancel = cancel;
        tokio::pin!(cancel);

        let session_result = 'session: loop {
            if let Err(error) = self.process_actions(&mut runtime).await {
                break 'session Err(error);
            }
            if !self.rtc.is_alive() {
                break 'session Err(anyhow!("WebRTC peer is no longer alive"));
            }

            if runtime.connected && runtime.clock.next_wallclock() <= Instant::now() {
                // Give cancellation priority even while capture has queued many
                // frames, otherwise a permanently-ready audio source could delay
                // Ctrl-C indefinitely.
                let cancelled = tokio::select! {
                    biased;
                    _ = &mut cancel => true,
                    _ = std::future::ready(()) => false,
                };
                if cancelled {
                    break 'session Ok(());
                }

                let now = Instant::now();
                runtime.clock.resync_if_late(now);
                let frame = runtime.frames.pop_frame_or_silence();
                let write_result =
                    runtime.encode_and_write(&mut self.rtc, self.mid, self.opus_pt, &frame);
                // Writer::write is a mutation. Drain even if encoding/writing
                // reports an error, then preserve the primary error.
                let drain_result = drain_rtc(&mut self.rtc, &mut self.actions);
                if let Err(error) = write_result {
                    let _ = drain_result;
                    break 'session Err(error);
                }
                match drain_result {
                    Ok(deadline) => self.deadline = deadline,
                    Err(error) => break 'session Err(error),
                }
                continue;
            }

            if runtime.playback.is_due(Instant::now()) {
                let cancelled = tokio::select! {
                    biased;
                    _ = &mut cancel => true,
                    _ = std::future::ready(()) => false,
                };
                if cancelled {
                    break 'session Ok(());
                }
                runtime.playback.deliver_due(&output, Instant::now())?;
                continue;
            }

            let timer = tokio::time::sleep_until(self.deadline.into());
            tokio::pin!(timer);
            let audio_timer = tokio::time::sleep_until(runtime.clock.next_wallclock().into());
            tokio::pin!(audio_timer);
            let playback_timer = tokio::time::sleep_until(runtime.playback.deadline().into());
            tokio::pin!(playback_timer);

            enum Wake {
                Cancel,
                Timeout,
                AudioTick,
                PlaybackTick,
                Datagram(std::io::Result<(usize, SocketAddr)>),
                Audio(Option<Vec<i16>>),
            }

            let wake = tokio::select! {
                biased;
                _ = &mut cancel => Wake::Cancel,
                _ = &mut timer => Wake::Timeout,
                _ = &mut audio_timer, if runtime.connected => Wake::AudioTick,
                _ = &mut playback_timer, if runtime.playback.has_pending() => Wake::PlaybackTick,
                datagram = self.socket.recv_from(&mut udp_buffer) => Wake::Datagram(datagram),
                audio = input.recv() => Wake::Audio(audio),
            };

            match wake {
                Wake::Cancel => break 'session Ok(()),
                Wake::AudioTick => continue,
                Wake::PlaybackTick => continue,
                Wake::Audio(Some(pcm)) => {
                    // str0m drops writes before ICE/DTLS is connected. Discard
                    // pre-connect capture instead of presenting it later as
                    // stale speech after the call becomes live.
                    if !runtime.connected {
                        continue;
                    }
                    let converted = match runtime
                        .capture_resampler
                        .process(&pcm)
                        .context("resample captured audio to 48 kHz")
                    {
                        Ok(converted) => converted,
                        Err(error) => break 'session Err(error),
                    };
                    runtime.frames.push(&converted);
                    // No Rtc mutation occurred. Looping lets a complete frame
                    // take the normal write -> drain path above.
                    continue;
                }
                Wake::Audio(None) => {
                    break 'session Err(anyhow!("audio input channel disconnected"));
                }
                Wake::Timeout => {
                    let result = self.rtc.handle_input(Input::Timeout(Instant::now()));
                    let drain_result = drain_rtc(&mut self.rtc, &mut self.actions);
                    if let Err(error) = result {
                        let _ = drain_result;
                        break 'session Err(error).context("advance WebRTC timeout");
                    }
                    match drain_result {
                        Ok(deadline) => self.deadline = deadline,
                        Err(error) => break 'session Err(error),
                    }
                }
                Wake::Datagram(Err(error)) => {
                    break 'session Err(error).context("receive WebRTC UDP datagram");
                }
                Wake::Datagram(Ok((length, source))) => {
                    let receive = match Receive::new(
                        Protocol::Udp,
                        source,
                        self.local_addr,
                        &udp_buffer[..length],
                    ) {
                        Ok(receive) => receive,
                        Err(error) => {
                            tracing::debug!(%error, %source, "ignoring invalid WebRTC UDP datagram");
                            continue;
                        }
                    };
                    let result = self
                        .rtc
                        .handle_input(Input::Receive(Instant::now(), receive));
                    let drain_result = drain_rtc(&mut self.rtc, &mut self.actions);
                    if let Err(error) = result {
                        let _ = drain_result;
                        break 'session Err(error).context("handle WebRTC UDP datagram");
                    }
                    match drain_result {
                        Ok(deadline) => self.deadline = deadline,
                        Err(error) => break 'session Err(error),
                    }
                }
            }
        };

        // Cancellation is intentionally successful even if the best-effort
        // close_notify cannot be sent (for example, Ctrl-C before DTLS starts).
        if let Err(error) = self.close_and_drain().await {
            tracing::warn!(%error, "WebRTC close did not complete cleanly");
        }
        session_result
    }

    async fn process_actions(&mut self, runtime: &mut AudioRuntime) -> Result<()> {
        while let Some(action) = self.actions.pop_front() {
            match action {
                DrainedAction::Transmit(transmit) => self.send_transmit(transmit).await?,
                DrainedAction::Event(event) => {
                    handle_event(*event, self.local_addr, runtime)?;
                }
            }
        }
        Ok(())
    }

    async fn send_transmit(&self, transmit: Transmit) -> Result<()> {
        if transmit.proto != Protocol::Udp {
            bail!("str0m requested unsupported {} transport", transmit.proto);
        }
        if transmit.source != self.local_addr {
            bail!(
                "str0m requested unbound source address {} (bound {})",
                transmit.source,
                self.local_addr
            );
        }
        self.socket
            .send_to(&transmit.contents, transmit.destination)
            .await
            .with_context(|| format!("send WebRTC UDP datagram to {}", transmit.destination))?;
        Ok(())
    }

    async fn close_and_drain(&mut self) -> Result<()> {
        let close_result = self.rtc.close();
        let drain_result = drain_rtc(&mut self.rtc, &mut self.actions);

        // The RTC is fully drained before any send is awaited. Continue past an
        // individual send error so a later DTLS close_notify still gets a chance.
        let mut send_error = None;
        while let Some(action) = self.actions.pop_front() {
            if let DrainedAction::Transmit(transmit) = action {
                if let Err(error) = self.send_transmit(transmit).await {
                    send_error.get_or_insert(error);
                }
            }
        }

        close_result.context("close WebRTC peer")?;
        let _ = drain_result?;
        if let Some(error) = send_error {
            return Err(error);
        }
        Ok(())
    }
}

enum DrainedAction {
    Transmit(Transmit),
    Event(Box<Event>),
}

/// Exhaust all currently available output. This function never awaits.
fn drain_rtc(rtc: &mut Rtc, actions: &mut VecDeque<DrainedAction>) -> Result<Instant> {
    loop {
        match rtc.poll_output().context("poll WebRTC output")? {
            Output::Timeout(deadline) => return Ok(deadline),
            Output::Transmit(transmit) => {
                actions.push_back(DrainedAction::Transmit(transmit));
            }
            Output::Event(event) => actions.push_back(DrainedAction::Event(Box::new(event))),
        }
    }
}

fn negotiated_opus_pt(rtc: &mut Rtc, mid: Mid) -> Result<Pt> {
    let writer = rtc
        .writer(mid)
        .ok_or_else(|| anyhow!("negotiated audio media is not send-capable"))?;
    let pt = writer
        .payload_params()
        .find(|params| params.spec().codec == Codec::Opus)
        .map(|params| params.pt())
        .ok_or_else(|| anyhow!("the server's SDP answer did not negotiate Opus audio"));
    pt
}

fn handle_event(event: Event, local_addr: SocketAddr, runtime: &mut AudioRuntime) -> Result<()> {
    match event {
        Event::Connected => {
            if !runtime.connected {
                runtime.mark_connected(Instant::now());
                tracing::info!(%local_addr, "WebRTC peer connected; audio is live");
            }
        }
        Event::IceConnectionStateChange(IceConnectionState::Disconnected) => {
            bail!("WebRTC ICE connection disconnected");
        }
        Event::IceConnectionStateChange(state) => {
            tracing::debug!(?state, "WebRTC ICE state changed");
        }
        Event::Closed => bail!("remote WebRTC peer closed the connection"),
        Event::MediaData(data) => runtime.decode_and_buffer(data, Instant::now())?,
        other => tracing::trace!(?other, "WebRTC event"),
    }
    Ok(())
}

struct AudioRuntime {
    encoder: Encoder,
    decoder: Decoder,
    capture_resampler: StreamingResampler,
    playback_resampler: StreamingResampler,
    frames: PcmFrames,
    clock: RtpClock,
    playback: PlayoutQueue,
    opus_packet: Vec<u8>,
    decoded: Vec<i16>,
    connected: bool,
    received_audio: bool,
}

impl AudioRuntime {
    fn new(input_rate: u32, output_rate: u32, voice_recycle: Receiver<Vec<i16>>) -> Result<Self> {
        let encoder = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)
            .context("create mono Opus VoIP encoder")?;
        let decoder = Decoder::new(SampleRate::Hz48000, Channels::Mono)
            .context("create mono Opus decoder")?;
        Ok(Self {
            encoder,
            decoder,
            capture_resampler: StreamingResampler::new(input_rate, RTP_RATE)
                .context("create capture resampler")?,
            playback_resampler: StreamingResampler::new(RTP_RATE, output_rate)
                .context("create playback resampler")?,
            frames: PcmFrames::default(),
            clock: RtpClock::new(Instant::now()),
            playback: PlayoutQueue::new(output_rate, voice_recycle),
            opus_packet: vec![0; OPUS_MAX_PACKET_BYTES],
            decoded: vec![0; OPUS_MAX_DECODE_SAMPLES],
            connected: false,
            received_audio: false,
        })
    }

    fn encode_and_write(&mut self, rtc: &mut Rtc, mid: Mid, pt: Pt, frame: &[i16]) -> Result<()> {
        if frame.len() != OPUS_FRAME_SAMPLES {
            bail!(
                "internal Opus frame has {} samples, expected {}",
                frame.len(),
                OPUS_FRAME_SAMPLES
            );
        }
        let encoded = self
            .encoder
            .encode(frame, &mut self.opus_packet)
            .context("encode captured PCM as Opus")?;
        let (wallclock, media_time) = self.clock.take_frame();
        rtc.writer(mid)
            .ok_or_else(|| anyhow!("negotiated audio media is no longer send-capable"))?
            .write(
                pt,
                wallclock,
                media_time,
                self.opus_packet[..encoded].to_vec(),
            )
            .context("write Opus frame to WebRTC")?;
        Ok(())
    }

    fn mark_connected(&mut self, now: Instant) {
        self.connected = true;
        self.frames.clear();
        self.clock.reset(now);
    }

    fn decode_and_buffer(&mut self, data: MediaData, now: Instant) -> Result<()> {
        if data.params.spec().codec != Codec::Opus {
            tracing::warn!(
                ?data.pt,
                codec = ?data.params.spec().codec,
                "ignoring non-Opus media on Opus-only peer"
            );
            return Ok(());
        }

        let samples =
            match self
                .decoder
                .decode(Some(data.data.as_ref()), &mut self.decoded[..], false)
            {
                Ok(samples) => samples,
                Err(error) => {
                    tracing::warn!(%error, "dropping invalid incoming Opus frame");
                    return Ok(());
                }
            };
        if !self.received_audio {
            self.received_audio = true;
            tracing::info!("receiving remote Opus audio");
        }
        let converted = self
            .playback_resampler
            .process(&self.decoded[..samples])
            .context("resample decoded audio to output device rate")?;
        if converted.is_empty() {
            return Ok(());
        }

        match self.playback.push(&converted, now) {
            None => tracing::warn!(
                max_seconds = MAX_PLAYBACK_SECONDS,
                "decoded audio frame exceeds the playback buffer; dropping it"
            ),
            Some(0) => {}
            Some(dropped_chunks) => tracing::warn!(
                dropped_chunks,
                max_seconds = MAX_PLAYBACK_SECONDS,
                "playback fell behind; dropping oldest decoded audio"
            ),
        }
        Ok(())
    }
}

#[derive(Default)]
struct PcmFrames {
    samples: VecDeque<i16>,
}

impl PcmFrames {
    fn push(&mut self, samples: &[i16]) {
        self.samples.extend(samples.iter().copied());
        let max_samples = OPUS_FRAME_SAMPLES * CAPTURE_BUFFER_FRAMES;
        if self.samples.len() > max_samples {
            self.samples.drain(..self.samples.len() - max_samples);
        }
    }

    fn has_frame(&self) -> bool {
        self.samples.len() >= OPUS_FRAME_SAMPLES
    }

    fn pop_frame(&mut self) -> Option<Vec<i16>> {
        if !self.has_frame() {
            return None;
        }
        Some(self.samples.drain(..OPUS_FRAME_SAMPLES).collect::<Vec<_>>())
    }

    fn pop_frame_or_silence(&mut self) -> Vec<i16> {
        self.pop_frame()
            .unwrap_or_else(|| vec![0; OPUS_FRAME_SAMPLES])
    }

    fn clear(&mut self) {
        self.samples.clear();
    }
}

struct PlayoutQueue {
    chunks: VecDeque<Vec<i16>>,
    queued_samples: usize,
    sample_rate: u32,
    max_samples: usize,
    next_wallclock: Option<Instant>,
    /// Buffers the audio callback has finished with. Refilled here rather than
    /// freed there: `free` can take the allocator's lock, and that callback is
    /// the one thread that must never block.
    recycle: Receiver<Vec<i16>>,
}

impl PlayoutQueue {
    fn new(sample_rate: u32, recycle: Receiver<Vec<i16>>) -> Self {
        Self::with_max_samples_and_recycling(
            sample_rate,
            sample_rate as usize * MAX_PLAYBACK_SECONDS,
            recycle,
        )
    }

    /// A queue with nothing being handed back: every chunk allocates, as it
    /// does before the callback has returned its first buffer.
    #[cfg(test)]
    fn with_max_samples(sample_rate: u32, max_samples: usize) -> Self {
        let (_tx, rx) = std::sync::mpsc::sync_channel(1);
        Self::with_max_samples_and_recycling(sample_rate, max_samples, rx)
    }

    fn with_max_samples_and_recycling(
        sample_rate: u32,
        max_samples: usize,
        recycle: Receiver<Vec<i16>>,
    ) -> Self {
        Self {
            chunks: VecDeque::new(),
            queued_samples: 0,
            sample_rate,
            max_samples,
            next_wallclock: None,
            recycle,
        }
    }

    /// Queue a decoded chunk and return how many older chunks were evicted.
    /// `None` means the new chunk itself is larger than the entire bound.
    ///
    /// Takes a slice rather than a `Vec` so the samples can be copied into a
    /// buffer the audio callback has handed back, instead of a fresh
    /// allocation. Copying costs a memcpy on this thread and saves a `free` on
    /// the callback's.
    fn push(&mut self, samples: &[i16], now: Instant) -> Option<usize> {
        if samples.is_empty() {
            return Some(0);
        }
        if samples.len() > self.max_samples {
            return None;
        }

        let mut dropped_stale_audio = 0;
        while self.queued_samples.saturating_add(samples.len()) > self.max_samples {
            let Some(stale) = self.chunks.pop_front() else {
                break;
            };
            self.queued_samples -= stale.len();
            dropped_stale_audio += 1;
        }
        if dropped_stale_audio > 0 {
            self.next_wallclock = Some(now);
        } else {
            self.next_wallclock = Some(
                self.next_wallclock
                    .filter(|deadline| *deadline > now)
                    .unwrap_or(now),
            );
        }
        let mut chunk = self.recycle.try_recv().unwrap_or_default();
        chunk.clear();
        chunk.extend_from_slice(samples);
        self.queued_samples += chunk.len();
        self.chunks.push_back(chunk);
        Some(dropped_stale_audio)
    }

    fn has_pending(&self) -> bool {
        !self.chunks.is_empty()
    }

    fn deadline(&self) -> Instant {
        self.next_wallclock
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(24 * 60 * 60))
    }

    fn is_due(&self, now: Instant) -> bool {
        self.has_pending() && self.next_wallclock.is_some_and(|deadline| deadline <= now)
    }

    fn deliver_due(&mut self, output: &SyncSender<Vec<i16>>, now: Instant) -> Result<()> {
        if !self.is_due(now) {
            return Ok(());
        }
        let chunk = self
            .chunks
            .pop_front()
            .expect("pending playout always has a front chunk");
        let samples = chunk.len();
        match output.try_send(chunk) {
            Ok(()) => {
                self.queued_samples -= samples;
                let duration = Duration::from_secs_f64(samples as f64 / self.sample_rate as f64);
                let scheduled = self.next_wallclock.expect("queued chunk has a deadline");
                self.next_wallclock =
                    Some(if now.saturating_duration_since(scheduled) >= duration {
                        now + duration
                    } else {
                        scheduled + duration
                    });
                Ok(())
            }
            Err(TrySendError::Full(chunk)) => {
                self.chunks.push_front(chunk);
                self.next_wallclock = Some(now + Duration::from_millis(1));
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => bail!("audio output channel disconnected"),
        }
    }
}

struct RtpClock {
    epoch: Instant,
    next_timestamp: u64,
}

impl RtpClock {
    fn new(epoch: Instant) -> Self {
        Self {
            epoch,
            next_timestamp: 0,
        }
    }

    fn take_frame(&mut self) -> (Instant, MediaTime) {
        let timestamp = self.next_timestamp;
        self.next_timestamp = self.next_timestamp.wrapping_add(OPUS_FRAME_SAMPLES as u64);
        let wallclock =
            self.epoch + std::time::Duration::from_secs_f64(timestamp as f64 / RTP_RATE as f64);
        (
            wallclock,
            MediaTime::new(timestamp, Frequency::FORTY_EIGHT_KHZ),
        )
    }

    fn resync_if_late(&mut self, now: Instant) {
        let next = self.next_wallclock();
        if now.saturating_duration_since(next) >= MEDIA_FRAME_DURATION {
            let media_offset =
                Duration::from_secs_f64(self.next_timestamp as f64 / RTP_RATE as f64);
            self.epoch = now.checked_sub(media_offset).unwrap_or(now);
        }
    }

    fn next_wallclock(&self) -> Instant {
        self.epoch
            + std::time::Duration::from_secs_f64(self.next_timestamp as f64 / RTP_RATE as f64)
    }

    fn reset(&mut self, epoch: Instant) {
        self.epoch = epoch;
        self.next_timestamp = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn offer_is_loopback_sendrecv_and_opus_only() {
        let peer = PendingPeer::create().await.unwrap();
        let sdp = peer.offer_sdp();

        assert_eq!(
            sdp.lines()
                .filter(|line| line.starts_with("m=audio "))
                .count(),
            1
        );
        assert!(!sdp.lines().any(|line| line.starts_with("m=video ")));
        assert!(sdp.contains("a=sendrecv"));
        assert!(sdp.to_ascii_lowercase().contains("opus/48000/2"));
        let candidate = sdp
            .lines()
            .find(|line| line.starts_with("a=candidate:"))
            .expect("offer contains a host candidate");
        assert!(candidate.contains("127.0.0.1"), "{candidate}");
    }

    #[tokio::test]
    async fn invalid_answer_is_rejected() {
        let peer = PendingPeer::create().await.unwrap();
        let error = peer
            .accept_answer("this is not SDP")
            .err()
            .expect("invalid SDP must fail");
        assert!(error.to_string().contains("parse the server's"));
    }

    #[tokio::test]
    async fn negotiated_peer_cancels_cleanly_before_connecting() {
        use str0m::change::SdpOffer;

        let pending = PendingPeer::create().await.unwrap();
        let offer = SdpOffer::from_sdp_string(pending.offer_sdp()).unwrap();
        let mut remote = Rtc::builder()
            .clear_codecs()
            .enable_opus(true)
            .build(Instant::now());
        remote.add_local_candidate(Candidate::host("127.0.0.1:9".parse().unwrap(), "udp").unwrap());
        let answer = remote.sdp_api().accept_offer(offer).unwrap();
        let peer = pending.accept_answer(&answer.to_sdp_string()).unwrap();

        let (_input_tx, input_rx) = mpsc::channel(1);
        let (output_tx, _output_rx) = std::sync::mpsc::sync_channel(1);
        let (_recycle_tx, recycle_rx) = std::sync::mpsc::sync_channel(1);
        peer.run(
            RTP_RATE,
            RTP_RATE,
            input_rx,
            output_tx,
            recycle_rx,
            std::future::ready(()),
        )
        .await
        .unwrap();
    }

    #[test]
    fn pcm_framer_preserves_order_and_remainder() {
        let mut frames = PcmFrames::default();
        frames.push(&(0..500).map(|value| value as i16).collect::<Vec<_>>());
        assert!(!frames.has_frame());
        frames.push(
            &(500..(OPUS_FRAME_SAMPLES + 11))
                .map(|value| value as i16)
                .collect::<Vec<_>>(),
        );

        let frame = frames.pop_frame().unwrap();
        assert_eq!(frame.len(), OPUS_FRAME_SAMPLES);
        assert_eq!(frame[0], 0);
        assert_eq!(frame[OPUS_FRAME_SAMPLES - 1], 959);
        assert!(!frames.has_frame());
        assert_eq!(
            frames.samples.iter().copied().collect::<Vec<_>>(),
            (960..971).map(|value| value as i16).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pcm_framer_sends_silence_on_underflow_and_drops_stale_capture() {
        let mut frames = PcmFrames::default();
        assert_eq!(frames.pop_frame_or_silence(), vec![0; OPUS_FRAME_SAMPLES]);

        let received_frames = CAPTURE_BUFFER_FRAMES + 2;
        let samples = (0..OPUS_FRAME_SAMPLES * received_frames)
            .map(|sample| sample as i16)
            .collect::<Vec<_>>();
        frames.push(&samples);

        assert_eq!(
            frames.samples.len(),
            OPUS_FRAME_SAMPLES * CAPTURE_BUFFER_FRAMES
        );
        let newest_window_start = OPUS_FRAME_SAMPLES * 2;
        let frame = frames.pop_frame().unwrap();
        assert_eq!(frame[0], newest_window_start as i16);
        assert_eq!(
            frame[OPUS_FRAME_SAMPLES - 1],
            (newest_window_start + OPUS_FRAME_SAMPLES - 1) as i16
        );
    }

    /// The audio callback returns spent buffers rather than freeing them on
    /// its own thread. That only helps if this queue refills them, so assert
    /// the very allocation handed back is the one that goes out again.
    #[test]
    fn playout_refills_a_returned_buffer_instead_of_allocating() {
        let epoch = Instant::now();
        let (recycle_tx, recycle_rx) = std::sync::mpsc::sync_channel(4);
        let mut queue = PlayoutQueue::with_max_samples_and_recycling(
            RTP_RATE,
            OPUS_FRAME_SAMPLES * 4,
            recycle_rx,
        );
        let spare = vec![0i16; OPUS_FRAME_SAMPLES];
        let spare_alloc = spare.as_ptr();
        recycle_tx.try_send(spare).expect("hand a buffer back");

        assert_eq!(queue.push(&[7; OPUS_FRAME_SAMPLES], epoch), Some(0));
        let (output, received) = std::sync::mpsc::sync_channel(4);
        queue.deliver_due(&output, epoch).unwrap();
        let delivered = received.try_recv().expect("a delivered chunk");
        assert_eq!(
            delivered.as_ptr(),
            spare_alloc,
            "the returned allocation must be refilled, not replaced"
        );
        assert_eq!(delivered, vec![7; OPUS_FRAME_SAMPLES]);
    }

    #[test]
    fn playout_is_paced_and_evicts_oldest_audio_when_bounded() {
        let epoch = Instant::now();
        let mut queue = PlayoutQueue::with_max_samples(RTP_RATE, OPUS_FRAME_SAMPLES * 2);
        let first = vec![1; OPUS_FRAME_SAMPLES];
        let second = vec![2; OPUS_FRAME_SAMPLES];
        let third = vec![3; OPUS_FRAME_SAMPLES];

        assert_eq!(queue.push(&first, epoch), Some(0));
        assert_eq!(queue.push(&second, epoch), Some(0));
        assert_eq!(queue.push(&third, epoch), Some(1));
        assert_eq!(queue.queued_samples, OPUS_FRAME_SAMPLES * 2);
        assert_eq!(queue.push(&vec![4; OPUS_FRAME_SAMPLES * 3], epoch), None);

        let (output, received) = std::sync::mpsc::sync_channel(4);
        queue.deliver_due(&output, epoch).unwrap();
        assert_eq!(received.try_recv().unwrap(), second);
        assert!(!queue.is_due(epoch + Duration::from_millis(19)));

        queue
            .deliver_due(&output, epoch + MEDIA_FRAME_DURATION)
            .unwrap();
        assert_eq!(received.try_recv().unwrap(), third);
        assert!(!queue.has_pending());

        // Even though the queue was momentarily empty, a packet arriving early
        // retains the existing sample clock instead of playing immediately.
        assert_eq!(
            queue.push(&[4; OPUS_FRAME_SAMPLES], epoch + Duration::from_millis(25)),
            Some(0)
        );
        assert!(!queue.is_due(epoch + Duration::from_millis(39)));
        assert!(queue.is_due(epoch + Duration::from_millis(40)));
    }

    #[test]
    fn playout_retries_without_losing_audio_when_device_channel_is_full() {
        let epoch = Instant::now();
        let mut queue = PlayoutQueue::with_max_samples(RTP_RATE, OPUS_FRAME_SAMPLES);
        let audio = vec![7; OPUS_FRAME_SAMPLES];
        assert_eq!(queue.push(&audio, epoch), Some(0));

        let (output, received) = std::sync::mpsc::sync_channel(1);
        output.try_send(vec![9]).unwrap();
        queue.deliver_due(&output, epoch).unwrap();
        assert!(queue.has_pending());
        assert_eq!(queue.queued_samples, OPUS_FRAME_SAMPLES);
        assert_eq!(received.try_recv().unwrap(), vec![9]);

        queue
            .deliver_due(&output, epoch + Duration::from_millis(1))
            .unwrap();
        assert_eq!(received.try_recv().unwrap(), audio);
        assert!(!queue.has_pending());
    }

    #[test]
    fn rtp_clock_advances_by_exact_twenty_ms_frames() {
        let epoch = Instant::now();
        let mut clock = RtpClock::new(epoch);
        assert_eq!(clock.next_wallclock(), epoch);
        let (wallclock0, time0) = clock.take_frame();
        assert_eq!(clock.next_wallclock().duration_since(epoch).as_millis(), 20);
        let (wallclock1, time1) = clock.take_frame();
        let (_, time2) = clock.take_frame();

        assert_eq!(wallclock0, epoch);
        assert_eq!(wallclock1.duration_since(epoch).as_millis(), 20);
        assert_eq!(time0.numer(), 0);
        assert_eq!(time1.numer(), 960);
        assert_eq!(time2.numer(), 1_920);
        assert_eq!(time2.denom(), RTP_RATE);
    }

    #[test]
    fn rtp_clock_skips_wallclock_catch_up_without_skipping_media_time() {
        let epoch = Instant::now();
        let mut clock = RtpClock::new(epoch);
        let _ = clock.take_frame();
        let late = epoch + Duration::from_millis(100);

        clock.resync_if_late(late);
        assert_eq!(clock.next_wallclock(), late);
        let (wallclock, media_time) = clock.take_frame();
        assert_eq!(wallclock, late);
        assert_eq!(media_time.numer(), OPUS_FRAME_SAMPLES as u64);
    }
}
