//! The WS2812 strip (ADR-0008, "Vocabulary"): what the bot is doing, as
//! light. Asleep is one dim green pixel sweeping slowly end to end;
//! listening is every LED dim green; thinking is the Larson scanner in the
//! palette's colours; speaking is a soft warm glow; a gated mic paints both
//! ends red over any of those; no server is one amber pixel blinking. Two
//! brightness levels: `brightness` for the states that last seconds,
//! `idle_brightness` for the ones that last hours.
//!
//! [`Strip`] owns the device and a thread that renders the [`Animator`]
//! between the indications it is handed. It lives for the whole call, so the
//! retry loop can show Offline between sessions, while each session drives
//! it through a [`StripHandle`], its [`LedSink`]. Dropping the `Strip` clears
//! it and ends the thread.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use voice_chatbot_ws2812::color::{paint, Rgb, PALETTE};
use voice_chatbot_ws2812::larson::Larson;
use voice_chatbot_ws2812::strip::{Order, Ws2812Spi};

use crate::led::{Indication, LedSink, Phase};

/// One end-to-end pass of the thinking scanner, the demo's pace.
const SWEEP: Duration = Duration::from_millis(700);
/// One end-to-end pass of the idle pixel: slow enough to ignore.
const IDLE_SWEEP: Duration = Duration::from_secs(5);
/// Half a period of the offline blink.
const BLINK: Duration = Duration::from_secs(1);
/// The strip's SPI clock (ADR-0008, "The signal").
const SPI_HZ: u32 = 3_200_000;

const GREEN: Rgb = Rgb::new(0, 255, 0);
const RED: Rgb = Rgb::new(255, 0, 0);
/// Soft warm white for the bot's voice.
const WARM: Rgb = Rgb::new(255, 120, 40);
/// The fault colour. A WS2812's green is its strongest emitter, so amber
/// leans well into red.
const AMBER: Rgb = Rgb::new(255, 150, 0);
/// The kit's eye, 1:4:2:1, for thinking.
const EYE: [u32; 4] = [1, 4, 2, 1];
/// A lone pixel for the idle sweep.
const LONE_PIXEL: [u32; 4] = [0, 4, 0, 0];

/// Where the strip is and how to drive it.
#[derive(Clone, Debug, PartialEq)]
pub struct StripConfig {
    /// The spidev node; SPI0 MOSI is GPIO 10 on a Pi.
    pub device: PathBuf,
    pub count: usize,
    /// 0..=1, for thinking, speaking and the mute overlay.
    pub brightness: f32,
    /// 0..=1, for asleep, listening and offline: the states that last hours.
    pub idle_brightness: f32,
}

/// A frame writer: the spidev strip, or a fake in tests.
pub trait Pixels: Send {
    fn write(&mut self, pixels: &[Rgb]) -> Result<()>;
}

impl Pixels for Ws2812Spi {
    fn write(&mut self, pixels: &[Rgb]) -> Result<()> {
        Ws2812Spi::write(self, pixels)
    }
}

/// The frames for each phase (the module doc has the vocabulary). Static
/// phases render the same frame every call; animated ones advance.
pub struct Animator {
    count: usize,
    brightness: f32,
    idle_brightness: f32,
    shown: Option<Indication>,
    scanner: Larson,
    pass: usize,
    blink_on: bool,
    frame: Vec<Rgb>,
}

impl Animator {
    /// `count` LEDs (at least two), brightnesses 0..=1.
    pub fn new(count: usize, brightness: f32, idle_brightness: f32) -> Self {
        Self {
            count,
            brightness,
            idle_brightness,
            shown: None,
            scanner: Larson::new(count),
            pass: 0,
            blink_on: true,
            frame: vec![Rgb::BLACK; count],
        }
    }

    /// Take in a settled indication. Entering a phase restarts its motion;
    /// repeating one does not.
    pub fn show(&mut self, indication: Indication) {
        if self.shown.map(|shown| shown.phase) != Some(indication.phase) {
            match indication.phase {
                Phase::Thinking => {
                    self.scanner = Larson::with_weights(self.count, EYE);
                    self.pass = 0;
                }
                Phase::Asleep => self.scanner = Larson::with_weights(self.count, LONE_PIXEL),
                Phase::Offline => self.blink_on = true,
                Phase::Listening | Phase::Speaking => {}
            }
        }
        self.shown = Some(indication);
    }

    /// Whether the current phase wants a frame every [`Self::step`].
    pub fn animated(&self) -> bool {
        matches!(
            self.shown.map(|shown| shown.phase),
            Some(Phase::Thinking | Phase::Asleep | Phase::Offline)
        )
    }

    pub fn steps_per_pass(&self) -> u32 {
        self.scanner.steps_per_pass()
    }

    /// Time between frames of an animated phase.
    pub fn step(&self) -> Duration {
        match self.shown.map(|shown| shown.phase) {
            Some(Phase::Thinking) => SWEEP / self.scanner.steps_per_pass(),
            Some(Phase::Asleep) => IDLE_SWEEP / self.scanner.steps_per_pass(),
            Some(Phase::Offline) => BLINK,
            _ => Duration::ZERO,
        }
    }

    /// The frame to show now, advancing an animated phase to its next one.
    pub fn frame(&mut self) -> &[Rgb] {
        let Some(shown) = self.shown else {
            self.frame.fill(Rgb::BLACK);
            return &self.frame;
        };
        match shown.phase {
            Phase::Asleep => {
                let levels = self.scanner.levels();
                paint(&levels, GREEN, self.idle_brightness, 1.0, &mut self.frame);
                self.scanner.step();
            }
            Phase::Listening => self.frame.fill(GREEN.scaled(self.idle_brightness)),
            Phase::Thinking => {
                let color = PALETTE[self.pass % PALETTE.len()];
                let levels = self.scanner.levels();
                paint(&levels, color, self.brightness, 1.0, &mut self.frame);
                if self.scanner.step() {
                    self.pass += 1;
                }
            }
            Phase::Speaking => self.frame.fill(WARM.scaled(self.brightness)),
            Phase::Offline => {
                self.frame.fill(Rgb::BLACK);
                if self.blink_on {
                    self.frame[0] = AMBER.scaled(self.idle_brightness);
                }
                self.blink_on = !self.blink_on;
            }
        }
        if shown.muted {
            let red = RED.scaled(self.brightness);
            self.frame[0] = red;
            self.frame[self.count - 1] = red;
        }
        &self.frame
    }

    /// Every LED off.
    pub fn dark(&self) -> Vec<Rgb> {
        vec![Rgb::BLACK; self.count]
    }
}

/// The strip: a thread owns the device and the [`Animator`].
pub struct Strip {
    updates: Option<mpsc::Sender<Indication>>,
    thread: Option<JoinHandle<()>>,
    description: String,
}

/// A session's way to drive the strip: an [`LedSink`] that does not own it,
/// so dropping it (at session end) neither clears nor stops the strip.
#[derive(Clone)]
pub struct StripHandle(mpsc::Sender<Indication>);

impl Strip {
    /// Open the strip; fails when SPI is off, the node is not writable, or
    /// this is not Linux — the caller runs without a strip then.
    pub fn open(config: &StripConfig) -> Result<Self> {
        let pixels = Ws2812Spi::open(&config.device, SPI_HZ, Order::Grb, config.count)
            .with_context(|| format!("open the led strip on {}", config.device.display()))?;
        let mut strip = Self::from_pixels(
            Box::new(pixels),
            config.count,
            config.brightness,
            config.idle_brightness,
        );
        strip.description = format!(
            "ws2812 strip, {} LEDs on {}",
            config.count,
            config.device.display()
        );
        Ok(strip)
    }

    /// Drive any frame writer; what the tests use.
    pub fn from_pixels(
        pixels: Box<dyn Pixels>,
        count: usize,
        brightness: f32,
        idle_brightness: f32,
    ) -> Self {
        let (updates, changes) = mpsc::channel();
        let animator = Animator::new(count, brightness, idle_brightness);
        let thread = thread::Builder::new()
            .name("led-strip".into())
            .spawn(move || run(pixels, changes, animator))
            .expect("spawn the led-strip thread");
        Self {
            updates: Some(updates),
            thread: Some(thread),
            description: format!("ws2812 strip, {count} LEDs"),
        }
    }

    /// One console-worthy line: what was opened.
    pub fn describe(&self) -> &str {
        &self.description
    }

    /// A sink for a session.
    pub fn handle(&self) -> StripHandle {
        StripHandle(self.updates.clone().expect("a live strip has its channel"))
    }

    /// Show an indication from the owner (the call loop, or led-test).
    pub fn show(&self, indication: Indication) -> Result<()> {
        self.updates
            .as_ref()
            .context("led strip already stopped")?
            .send(indication)
            .context("led strip thread stopped (a write failed)")
    }
}

impl LedSink for Strip {
    fn set(&mut self, indication: Indication) -> Result<()> {
        self.show(indication)
    }
}

impl LedSink for StripHandle {
    fn set(&mut self, indication: Indication) -> Result<()> {
        self.0
            .send(indication)
            .context("led strip thread stopped (a write failed)")
    }
}

impl Drop for Strip {
    fn drop(&mut self) {
        // Closing the channel is the stop signal; the thread clears the
        // strip on its way out, and its writes are sub-millisecond.
        drop(self.updates.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The strip thread: a frame every step while the phase is animated, one
/// write when a static phase arrives, nothing in between. A failed write
/// ends the thread; the next `set` then fails and the driver drops its
/// handle.
fn run(mut pixels: Box<dyn Pixels>, changes: mpsc::Receiver<Indication>, mut animator: Animator) {
    let mut due = Instant::now();
    loop {
        let next = if animator.animated() {
            if !write(pixels.as_mut(), animator.frame()) {
                return;
            }
            // Measured from the previous deadline, not from now, so the
            // write time does not stretch every step.
            due += animator.step();
            changes.recv_timeout(due.saturating_duration_since(Instant::now()))
        } else {
            changes
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
        };
        match next {
            Ok(indication) => {
                animator.show(indication);
                if animator.animated() {
                    due = Instant::now();
                } else if !write(pixels.as_mut(), animator.frame()) {
                    return;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let _ = pixels.write(&animator.dark());
}

fn write(pixels: &mut dyn Pixels, frame: &[Rgb]) -> bool {
    match pixels.write(frame) {
        Ok(()) => true,
        Err(error) => {
            tracing::debug!(%error, "led strip write failed; no more frames this session");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::led::{Indication, LedSink, Phase};
    use std::collections::BTreeSet;
    use std::sync::mpsc;
    use std::time::Duration;
    use voice_chatbot_ws2812::color::Rgb;

    const COUNT: usize = 8;
    const BRIGHT: f32 = 0.2;
    const IDLE: f32 = 0.05;
    /// Green at the idle brightness: 255 x 0.05 rounds to 13.
    const IDLE_GREEN: Rgb = Rgb::new(0, 13, 0);

    const fn at(phase: Phase) -> Indication {
        Indication {
            phase,
            muted: false,
        }
    }

    fn lit(frame: &[Rgb]) -> bool {
        frame.iter().any(|px| *px != Rgb::BLACK)
    }

    fn lit_indices(frame: &[Rgb]) -> Vec<usize> {
        frame
            .iter()
            .enumerate()
            .filter(|(_, px)| **px != Rgb::BLACK)
            .map(|(i, _)| i)
            .collect()
    }

    fn animator() -> Animator {
        Animator::new(COUNT, BRIGHT, IDLE)
    }

    #[test]
    fn starts_dark_until_told_otherwise() {
        let mut anim = animator();
        assert!(!anim.animated());
        assert!(!lit(anim.frame()));
        assert_eq!(anim.dark(), vec![Rgb::BLACK; COUNT]);
    }

    #[test]
    fn asleep_sweeps_one_dim_green_pixel_end_to_end() {
        let mut anim = animator();
        anim.show(at(Phase::Asleep));
        assert!(anim.animated());
        assert_eq!(anim.step(), IDLE_SWEEP / anim.steps_per_pass());
        let mut visited = BTreeSet::new();
        for _ in 0..(2 * anim.steps_per_pass()) {
            let frame = anim.frame().to_vec();
            let on = lit_indices(&frame);
            assert!(
                !on.is_empty() && on.len() <= 2,
                "one pixel, or two while it crosses between LEDs: {on:?}"
            );
            if on.len() == 2 {
                assert_eq!(on[1], on[0] + 1, "the two are neighbours");
            }
            for i in &on {
                assert_eq!((frame[*i].r, frame[*i].b), (0, 0), "green only");
                assert!(
                    frame[*i].g <= IDLE_GREEN.g,
                    "never above the idle brightness"
                );
            }
            if on.len() == 1 && frame[on[0]] == IDLE_GREEN {
                visited.insert(on[0]);
            }
        }
        assert_eq!(
            visited.len(),
            COUNT,
            "over an up and a down pass the pixel sits fully on every LED: {visited:?}"
        );
    }

    #[test]
    fn listening_is_every_led_dim_green_and_static() {
        let mut anim = animator();
        anim.show(at(Phase::Listening));
        assert!(!anim.animated());
        assert_eq!(anim.frame(), &[IDLE_GREEN; COUNT][..]);
    }

    #[test]
    fn thinking_runs_the_scanner() {
        let mut anim = animator();
        anim.show(at(Phase::Thinking));
        assert!(anim.animated());
        assert_eq!(anim.step(), SWEEP / anim.steps_per_pass());
        assert!(lit(anim.frame()));
        anim.show(Indication {
            phase: Phase::Thinking,
            muted: true,
        });
        assert!(anim.animated(), "a mute overlay does not stop the scanner");
        anim.show(at(Phase::Speaking));
        assert!(!anim.animated());
    }

    #[test]
    fn thinking_restarts_the_scanner_each_time_it_begins() {
        let mut anim = animator();
        anim.show(at(Phase::Thinking));
        let first = anim.frame().to_vec();
        for _ in 0..10 {
            anim.frame();
        }
        assert_ne!(anim.frame().to_vec(), first, "the eye moves");
        anim.show(at(Phase::Listening));
        anim.show(at(Phase::Thinking));
        assert_eq!(
            anim.frame().to_vec(),
            first,
            "a new thinking spell starts from the low end again"
        );
    }

    #[test]
    fn thinking_changes_colour_every_pass() {
        let mut anim = Animator::new(COUNT, 1.0, IDLE);
        anim.show(at(Phase::Thinking));
        let brightest = |frame: &[Rgb]| {
            frame
                .iter()
                .copied()
                .max_by_key(|px| u32::from(px.r) + u32::from(px.g) + u32::from(px.b))
                .unwrap()
        };
        assert_eq!(
            brightest(anim.frame()),
            Rgb::new(255, 0, 0),
            "first pass is red"
        );
        for _ in 0..anim.steps_per_pass() {
            anim.frame();
        }
        assert_eq!(
            brightest(anim.frame()),
            Rgb::new(255, 80, 0),
            "the second pass takes the palette's next colour"
        );
    }

    #[test]
    fn speaking_is_a_soft_even_warm_glow() {
        let mut anim = animator();
        anim.show(at(Phase::Speaking));
        assert!(!anim.animated());
        let frame = anim.frame().to_vec();
        assert!(frame.iter().all(|px| *px == frame[0]), "steady and even");
        assert_eq!(frame[0], WARM.scaled(BRIGHT));
        assert!(
            frame[0].r > frame[0].g && frame[0].g > frame[0].b,
            "warm: red over green over blue"
        );
    }

    #[test]
    fn mute_paints_both_ends_red_over_the_phase() {
        let mut anim = animator();
        anim.show(Indication {
            phase: Phase::Listening,
            muted: true,
        });
        let red = Rgb::new(255, 0, 0).scaled(BRIGHT);
        let frame = anim.frame().to_vec();
        assert_eq!((frame[0], frame[COUNT - 1]), (red, red));
        assert_eq!(
            frame[1], IDLE_GREEN,
            "the phase still shows between the ends"
        );
        anim.show(Indication {
            phase: Phase::Thinking,
            muted: true,
        });
        let frame = anim.frame().to_vec();
        assert_eq!(
            (frame[0], frame[COUNT - 1]),
            (red, red),
            "even over the scanner"
        );
    }

    #[test]
    fn offline_blinks_one_amber_pixel_slowly() {
        let mut anim = animator();
        anim.show(at(Phase::Offline));
        assert!(anim.animated());
        assert_eq!(anim.step(), BLINK);
        let on = anim.frame().to_vec();
        let off = anim.frame().to_vec();
        assert_eq!(lit_indices(&on), [0]);
        assert_eq!(on[0], AMBER.scaled(IDLE));
        assert!(!lit(&off));
        assert_eq!(anim.frame().to_vec(), on, "and on again");
        anim.show(at(Phase::Offline));
        assert_eq!(
            anim.frame().to_vec(),
            off,
            "repeating the state does not restart the blink"
        );
    }

    struct FakePixels(mpsc::Sender<Vec<Rgb>>);

    impl Pixels for FakePixels {
        fn write(&mut self, pixels: &[Rgb]) -> anyhow::Result<()> {
            self.0.send(pixels.to_vec()).unwrap();
            Ok(())
        }
    }

    #[test]
    fn strip_animates_thinking_and_writes_static_phases_once() {
        let timeout = Duration::from_secs(5);
        let (frames_tx, frames) = mpsc::channel();
        let mut strip = Strip::from_pixels(Box::new(FakePixels(frames_tx)), COUNT, BRIGHT, IDLE);
        strip.set(at(Phase::Thinking)).unwrap();
        // The port's first two frames are the same (the kit's start-of-pass
        // dwell), so look a little further for movement.
        let first: Vec<Vec<Rgb>> = (0..6)
            .map(|_| frames.recv_timeout(timeout).unwrap())
            .collect();
        assert!(
            first.iter().all(|frame| lit(frame)),
            "thinking lights the strip"
        );
        assert!(
            first.iter().any(|frame| *frame != first[0]),
            "the scanner moves between frames"
        );

        strip.set(at(Phase::Listening)).unwrap();
        let green = vec![IDLE_GREEN; COUNT];
        let settled =
            std::iter::from_fn(|| frames.recv_timeout(timeout).ok()).find(|f| *f == green);
        assert!(settled.is_some(), "listening lands as its static frame");
        assert!(
            frames.recv_timeout(Duration::from_millis(150)).is_err(),
            "and nothing more is written for a static phase"
        );

        strip.set(at(Phase::Speaking)).unwrap();
        assert_eq!(
            frames.recv_timeout(timeout).unwrap(),
            vec![WARM.scaled(BRIGHT); COUNT]
        );

        drop(strip);
        assert_eq!(
            frames.recv_timeout(timeout).unwrap(),
            vec![Rgb::BLACK; COUNT],
            "dropping the strip clears it"
        );
        assert!(
            matches!(
                frames.recv_timeout(Duration::from_millis(500)),
                Err(mpsc::RecvTimeoutError::Disconnected)
            ),
            "and ends its thread"
        );
    }

    #[test]
    fn a_handle_drives_the_strip_but_does_not_own_it() {
        let timeout = Duration::from_secs(5);
        let (frames_tx, frames) = mpsc::channel();
        let strip = Strip::from_pixels(Box::new(FakePixels(frames_tx)), COUNT, BRIGHT, IDLE);
        let mut handle = strip.handle();
        handle.set(at(Phase::Speaking)).unwrap();
        assert_eq!(
            frames.recv_timeout(timeout).unwrap(),
            vec![WARM.scaled(BRIGHT); COUNT]
        );
        drop(handle);
        assert!(
            frames.recv_timeout(Duration::from_millis(150)).is_err(),
            "dropping a handle neither clears nor stops the strip"
        );
        strip.show(at(Phase::Listening)).unwrap();
        assert_eq!(
            frames.recv_timeout(timeout).unwrap(),
            vec![IDLE_GREEN; COUNT]
        );
        drop(strip);
        assert_eq!(
            frames.recv_timeout(timeout).unwrap(),
            vec![Rgb::BLACK; COUNT]
        );
    }
}
