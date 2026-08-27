//! Realtime pacing for outbound bot audio.
//!
//! The WebRTC transport writes every frame the moment the pipeline hands it
//! over, and the TTS backends produce audio faster than realtime (Qwen at
//! ~3x, Kokoro/Chatterbox as one burst). A long reply therefore reaches the
//! client seconds ahead of playback, and the native client's playout queue
//! evicts the oldest *unplayed* audio once more than its bound (10 s) is
//! queued — the start and the end of the reply play, the middle vanishes.
//! It also means a barge-in cannot stop what has already been bursted.
//!
//! [`PacedTransport`] wraps any [`MediaTransport`] and releases outbound
//! audio on a playout clock, at most [`LEAD`] ahead of realtime. A single
//! task owns the inner transport and interleaves inbound `recv` with the
//! timed sends, so pacing never starves caller audio (the pipeline's shared
//! transport driver prioritises outbound commands, which is why the delay
//! cannot live inside `send_audio` itself). `send_audio` only enqueues and
//! returns immediately, so the sink's own playout estimate is unaffected.

use std::collections::VecDeque;
use std::time::Duration;

use async_trait::async_trait;
use flowcat_core::transport::{MediaIn, MediaTransport};
use flowcat_core::types::AudioChunk;
use flowcat_core::{FlowcatError, Result};
use tokio::sync::mpsc;
use tokio::time::Instant;

/// How far ahead of realtime audio may be sent: jitter margin for the client,
/// and the most stale audio a barge-in can leave queued there.
pub const LEAD: Duration = Duration::from_millis(500);

enum Command {
    Audio(AudioChunk),
    Clear,
}

pub struct PacedTransport {
    carrier_rate: u32,
    cmd_tx: mpsc::UnboundedSender<Command>,
    inbound_rx: mpsc::UnboundedReceiver<MediaIn>,
}

impl PacedTransport {
    /// Wrap `inner`, spawning the task that owns it.
    pub fn new<T: MediaTransport + 'static>(inner: T) -> Self {
        let carrier_rate = inner.carrier_rate();
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        tokio::spawn(run(inner, cmd_rx, inbound_tx));
        Self {
            carrier_rate,
            cmd_tx,
            inbound_rx,
        }
    }

    fn command(&self, cmd: Command) -> Result<()> {
        self.cmd_tx
            .send(cmd)
            .map_err(|_| FlowcatError::Transport("paced transport stopped".into()))
    }
}

/// Moment the audio sent so far finishes playing, or `None` when idle.
fn advance(play_clock: Option<Instant>, now: Instant, chunk: &AudioChunk) -> Instant {
    let start = play_clock.filter(|t| *t > now).unwrap_or(now);
    let dur = Duration::from_secs_f64(chunk.pcm.len() as f64 / chunk.sample_rate.max(1) as f64);
    start + dur
}

async fn run<T: MediaTransport>(
    mut inner: T,
    mut cmd_rx: mpsc::UnboundedReceiver<Command>,
    inbound_tx: mpsc::UnboundedSender<MediaIn>,
) {
    let mut pending: VecDeque<AudioChunk> = VecDeque::new();
    let mut play_clock: Option<Instant> = None;
    loop {
        let now = Instant::now();
        // Next chunk is due once the playout clock is within LEAD of now.
        let due = play_clock
            .and_then(|t| t.checked_sub(LEAD))
            .unwrap_or(now)
            .max(now);
        tokio::select! {
            biased;
            cmd = cmd_rx.recv() => match cmd {
                None => break, // the pipeline dropped the transport
                Some(Command::Audio(chunk)) => pending.push_back(chunk),
                Some(Command::Clear) => {
                    pending.clear();
                    play_clock = None;
                    if let Err(e) = inner.send_clear().await {
                        tracing::warn!(error = %e, "paced transport: clear failed");
                        break;
                    }
                }
            },
            _ = tokio::time::sleep_until(due), if !pending.is_empty() => {
                let chunk = pending.pop_front().expect("guarded by is_empty");
                play_clock = Some(advance(play_clock, Instant::now(), &chunk));
                if let Err(e) = inner.send_audio(chunk).await {
                    tracing::warn!(error = %e, "paced transport: send failed");
                    break;
                }
            },
            incoming = inner.recv() => match incoming {
                Some(event) => {
                    let stopped = matches!(event, MediaIn::Stop);
                    if inbound_tx.send(event).is_err() || stopped {
                        break;
                    }
                }
                None => break,
            },
            _ = inbound_tx.closed() => break,
        }
    }
}

#[async_trait]
impl MediaTransport for PacedTransport {
    async fn recv(&mut self) -> Option<MediaIn> {
        self.inbound_rx.recv().await
    }

    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<()> {
        self.command(Command::Audio(chunk))
    }

    async fn send_clear(&mut self) -> Result<()> {
        self.command(Command::Clear)
    }

    fn carrier_rate(&self) -> u32 {
        self.carrier_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    const RATE: u32 = 16_000;
    const FRAME: Duration = Duration::from_millis(20);

    /// Records when each chunk was sent; inbound comes from a channel.
    struct Fake {
        sent: Sent,
        cleared: Arc<Mutex<usize>>,
        inbound: mpsc::UnboundedReceiver<MediaIn>,
    }

    #[async_trait]
    impl MediaTransport for Fake {
        async fn recv(&mut self) -> Option<MediaIn> {
            self.inbound.recv().await
        }
        async fn send_audio(&mut self, chunk: AudioChunk) -> Result<()> {
            self.sent
                .lock()
                .unwrap()
                .push((Instant::now(), chunk.pcm.len()));
            Ok(())
        }
        async fn send_clear(&mut self) -> Result<()> {
            *self.cleared.lock().unwrap() += 1;
            Ok(())
        }
        fn carrier_rate(&self) -> u32 {
            RATE
        }
    }

    type Sent = Arc<Mutex<Vec<(Instant, usize)>>>;

    fn setup() -> (
        PacedTransport,
        Sent,
        Arc<Mutex<usize>>,
        mpsc::UnboundedSender<MediaIn>,
    ) {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let cleared = Arc::new(Mutex::new(0));
        let (in_tx, in_rx) = mpsc::unbounded_channel();
        let fake = Fake {
            sent: sent.clone(),
            cleared: cleared.clone(),
            inbound: in_rx,
        };
        (PacedTransport::new(fake), sent, cleared, in_tx)
    }

    fn frame() -> AudioChunk {
        AudioChunk::new(vec![0; RATE as usize / 50], RATE)
    }

    #[tokio::test(start_paused = true)]
    async fn a_burst_is_released_at_realtime_with_lead() {
        let (mut t, sent, _, _in_tx) = setup();
        let t0 = Instant::now();
        // 5 s of audio handed over instantly.
        for _ in 0..250 {
            t.send_audio(frame()).await.unwrap();
        }
        tokio::time::sleep(Duration::from_secs(6)).await;
        let sent = sent.lock().unwrap();
        assert_eq!(sent.len(), 250);
        // Everything within LEAD of the playout clock goes out at once...
        let immediate = sent.iter().filter(|(at, _)| *at == t0).count();
        assert_eq!(
            immediate,
            (LEAD.as_millis() / FRAME.as_millis()) as usize + 1
        );
        // ...and the last frame leaves LEAD before it is due to play.
        let last = sent.last().unwrap().0;
        let expected = t0 + FRAME * 250 - LEAD - FRAME;
        assert!(
            last.saturating_duration_since(expected)
                .max(expected.saturating_duration_since(last))
                <= Duration::from_millis(2),
            "last frame sent at +{:?}, expected +{:?}",
            last - t0,
            expected - t0
        );
    }

    #[tokio::test(start_paused = true)]
    async fn clear_drops_queued_audio_and_reaches_the_inner_transport() {
        let (mut t, sent, cleared, _in_tx) = setup();
        for _ in 0..250 {
            t.send_audio(frame()).await.unwrap();
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
        t.send_clear().await.unwrap();
        tokio::time::sleep(Duration::from_secs(6)).await;
        assert_eq!(*cleared.lock().unwrap(), 1);
        let n = sent.lock().unwrap().len();
        assert!(n < 250, "queued audio must be dropped on clear, sent {n}");
        // A fresh utterance after the clear starts immediately.
        let before = Instant::now();
        t.send_audio(frame()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!(sent.lock().unwrap().last().unwrap().0, before);
    }

    #[tokio::test(start_paused = true)]
    async fn inbound_is_forwarded_while_outbound_audio_waits() {
        let (mut t, _, _, in_tx) = setup();
        for _ in 0..250 {
            t.send_audio(frame()).await.unwrap();
        }
        in_tx.send(MediaIn::Audio(frame())).unwrap();
        let got = tokio::time::timeout(Duration::from_millis(50), t.recv())
            .await
            .expect("inbound must not wait for the outbound queue");
        assert!(matches!(got, Some(MediaIn::Audio(_))));
        in_tx.send(MediaIn::Stop).unwrap();
        assert_eq!(t.recv().await, Some(MediaIn::Stop));
        assert_eq!(t.recv().await, None);
        // The task stopped with the call; later sends fail instead of hanging.
        assert!(t.send_audio(frame()).await.is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn advance_restarts_the_clock_after_a_gap() {
        let now = Instant::now();
        let c = frame();
        let end = advance(None, now, &c);
        assert_eq!(end, now + FRAME);
        let end2 = advance(Some(end), now, &c);
        assert_eq!(end2, now + FRAME * 2);
        let stale = advance(Some(now - Duration::from_secs(5)), now, &c);
        assert_eq!(stale, now + FRAME);
    }
}
