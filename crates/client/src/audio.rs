//! Native audio device discovery, selection, capture, and playback.
//!
//! Audio crossing this module's channel boundary is always mono `i16`.  The
//! callbacks only perform sample-format conversion, channel mixing, and
//! non-blocking channel operations.  Resampling and codec work belong on the
//! consumer side of [`AudioIo`].

use std::cmp::Ordering;
use std::fmt;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, SupportedBufferSize};
use tokio::sync::mpsc as tokio_mpsc;

/// Number of native capture buffers that may wait for the async consumer.
///
/// This is intentionally shallow: retaining old microphone audio is worse than
/// dropping a buffer if the real-time consumer falls substantially behind.
pub const INPUT_CHANNEL_CAPACITY: usize = 8;

/// Number of mono playback chunks that may wait for the native output callback.
pub const OUTPUT_CHANNEL_CAPACITY: usize = 8;

/// Name fragment of the speakerphone to open when no device is asked for.
///
/// The Jabra USB speakerphone is this project's reference hardware: its
/// hardware echo cancellation is what lets the wake word survive the
/// assistant's own voice coming out of the same box. Starting a call on
/// whatever the desktop happens to have set as its default -- laptop mic into
/// laptop speakers -- gives a much worse call, so an unspecified device
/// resolves to the speakerphone whenever the host has one plugged in.
/// `--input-device default` / `--output-device default` asks for the system
/// default explicitly.
const PREFERRED_DEVICE_NAME: &str = "jabra";

/// Period this client asks each device for, in both directions.
///
/// [`cpal::BufferSize::Default`] lets ALSA choose, and on a USB `plughw:` PCM
/// it chooses about 241 frames: a 5 ms period, which CPAL double-buffers into
/// a **10 ms** ring. The worker that has to refill that ring is an ordinary
/// thread -- CPAL's `realtime` feature is off, and promoting it needs an
/// `rtprio` allowance this project cannot assume (`ulimit -r` is 0 on a stock
/// desktop, with rtkit inactive) -- so any scheduling hiccup longer than 10 ms
/// overruns capture or underruns playback.
///
/// 20 ms is the granularity the call already runs on: it is one Opus frame, so
/// it costs a frame of latency and buys a 40 ms ring, four times the slack.
const TARGET_PERIOD: Duration = Duration::from_millis(20);

/// The relevant part of a device's default CPAL stream configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioStreamConfig {
    pub channels: u16,
    pub sample_rate: u32,
    pub sample_format: SampleFormat,
    pub buffer_size: SupportedBufferSize,
}

impl From<cpal::SupportedStreamConfig> for AudioStreamConfig {
    fn from(config: cpal::SupportedStreamConfig) -> Self {
        Self {
            channels: config.channels(),
            sample_rate: config.sample_rate(),
            sample_format: config.sample_format(),
            buffer_size: *config.buffer_size(),
        }
    }
}

impl AudioStreamConfig {
    fn stream_config(self, buffer_size: cpal::BufferSize) -> cpal::StreamConfig {
        cpal::StreamConfig {
            channels: self.channels,
            sample_rate: self.sample_rate,
            buffer_size,
        }
    }

    /// [`TARGET_PERIOD`] in this device's frames, then CPAL's default.
    ///
    /// Both are returned because only building the stream validates a period
    /// against the device. The `SupportedBufferSize` in a default config is
    /// the *buffer* range, not the period range -- a `plughw:` PCM advertises
    /// `4..=268435455` while accepting only `48..=48000` -- so the request
    /// cannot be clamped up front. A `dmix:` PCM pins its period outright
    /// (`1024..=1024`) and falls back to the default.
    fn buffer_sizes(self) -> [cpal::BufferSize; 2] {
        let frames = (u64::from(self.sample_rate) * TARGET_PERIOD.as_millis() as u64) / 1_000;
        [
            cpal::BufferSize::Fixed(frames.clamp(1, u64::from(u32::MAX)) as u32),
            cpal::BufferSize::Default,
        ]
    }
}

impl fmt::Display for AudioStreamConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} Hz, {} channel{}, {}",
            self.sample_rate,
            self.channels,
            if self.channels == 1 { "" } else { "s" },
            self.sample_format
        )
    }
}

/// Stable, displayable information about one input or output device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioDeviceInfo {
    /// Deterministic, one-based index shown to the user.
    pub index: usize,
    /// CPAL's serialised stable [`cpal::DeviceId`].
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub config: AudioStreamConfig,
}

#[derive(Clone)]
struct AudioDevice {
    info: AudioDeviceInfo,
    device: cpal::Device,
}

#[derive(Clone, Copy)]
enum Direction {
    Input,
    Output,
}

impl fmt::Display for Direction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => formatter.write_str("input"),
            Self::Output => formatter.write_str("output"),
        }
    }
}

/// A deterministic snapshot of the default host's input and output devices.
pub struct AudioDevices {
    inputs: Vec<AudioDevice>,
    outputs: Vec<AudioDevice>,
}

impl AudioDevices {
    /// Enumerate input and output devices separately using CPAL's default host.
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let default_input_id = default_device_id(host.default_input_device());
        let default_output_id = default_device_id(host.default_output_device());

        let inputs = enumerate_devices(&host, Direction::Input, default_input_id.as_deref())?;
        let outputs = enumerate_devices(&host, Direction::Output, default_output_id.as_deref())?;

        Ok(Self { inputs, outputs })
    }

    /// Input devices in deterministic display order.
    pub fn input_devices(&self) -> impl ExactSizeIterator<Item = &AudioDeviceInfo> {
        self.inputs.iter().map(|device| &device.info)
    }

    /// Output devices in deterministic display order.
    pub fn output_devices(&self) -> impl ExactSizeIterator<Item = &AudioDeviceInfo> {
        self.outputs.iter().map(|device| &device.info)
    }

    /// Print both device lists, including stable IDs and default configurations.
    pub fn print(&self) {
        print_devices("Input devices", &self.inputs);
        print_devices("Output devices", &self.outputs);
    }

    /// Open the selected devices using each device's default stream config.
    ///
    /// `None` or an empty selector asks for no device in particular: that
    /// picks the [`PREFERRED_DEVICE_NAME`] speakerphone when one is plugged
    /// in, and the system default otherwise. `"default"` always selects the
    /// system default. Other selectors are resolved in this order: exact
    /// stable ID, one-based display index, exact name, then a case-insensitive
    /// name substring. A selector matching several ALSA aliases of one device
    /// resolves by [`alias_rank`] rather than being reported as ambiguous.
    pub fn open(
        &self,
        input_selector: Option<&str>,
        output_selector: Option<&str>,
    ) -> Result<AudioIo> {
        self.open_with_capacities(
            input_selector,
            output_selector,
            INPUT_CHANNEL_CAPACITY,
            OUTPUT_CHANNEL_CAPACITY,
        )
    }

    /// Like [`Self::open`], with explicit bounded-channel capacities.
    pub fn open_with_capacities(
        &self,
        input_selector: Option<&str>,
        output_selector: Option<&str>,
        input_capacity: usize,
        output_capacity: usize,
    ) -> Result<AudioIo> {
        if input_capacity == 0 {
            bail!("input audio channel capacity must be greater than zero");
        }
        if output_capacity == 0 {
            bail!("output audio channel capacity must be greater than zero");
        }

        let input = select_entry(&self.inputs, input_selector, Direction::Input)?;
        let output = select_entry(&self.outputs, output_selector, Direction::Output)?;

        ensure_pcm_format(input.info.config.sample_format, Direction::Input)
            .with_context(|| format!("cannot open input device {:?}", input.info.name))?;
        ensure_pcm_format(output.info.config.sample_format, Direction::Output)
            .with_context(|| format!("cannot open output device {:?}", output.info.name))?;

        let (input_tx, input_rx) = tokio_mpsc::channel(input_capacity);

        let input_stream = build_input_stream(input, input_tx)?;
        let (output_stream, output_tx, media_tx, media_gain) =
            build_output_stream(output, output_capacity)?;

        Ok(AudioIo {
            input_rate: input.info.config.sample_rate,
            output_rate: output.info.config.sample_rate,
            input_device: input.info.clone(),
            output_device: output.info.clone(),
            input_rx,
            output_tx,
            media_tx,
            media_gain,
            streams: AudioStreams {
                input_stream,
                output_stream,
            },
        })
    }
}

/// Open native streams and the mono `i16` queues connected to them.
pub struct AudioIo {
    /// Native capture rate. Consumers resample from this rate as needed.
    pub input_rate: u32,
    /// Native playback rate. Producers resample to this rate as needed.
    pub output_rate: u32,
    /// The selected capture device and its opened default configuration.
    pub input_device: AudioDeviceInfo,
    /// The selected playback device and its opened default configuration.
    pub output_device: AudioDeviceInfo,
    /// Native capture chunks, downmixed to mono `i16`.
    pub input_rx: tokio_mpsc::Receiver<Vec<i16>>,
    /// Mono `i16` playback chunks, duplicated to every hardware channel.
    pub output_tx: SyncSender<Vec<i16>>,
    /// Mono `i16` media chunks, summed with `output_tx` under a ramped gain.
    pub media_tx: SyncSender<Vec<i16>>,
    /// Ramped gain applied to `media_tx` only.
    pub media_gain: crate::media::gain::Gain,
    streams: AudioStreams,
}

/// CPAL stream ownership guard. Keep this alive for the duration of a call.
pub struct AudioStreams {
    input_stream: cpal::Stream,
    output_stream: cpal::Stream,
}

impl AudioStreams {
    /// Start playback and capture. Dropping this guard stops both streams.
    pub fn start(&self) -> Result<()> {
        self.output_stream
            .play()
            .context("failed to start output audio stream")?;

        if let Err(error) = self.input_stream.play() {
            let _ = self.output_stream.pause();
            return Err(error).context("failed to start input audio stream");
        }

        Ok(())
    }
}

/// Ownership-safe pieces returned by [`AudioIo::into_parts`].
pub struct AudioIoParts {
    pub streams: AudioStreams,
    pub input_rate: u32,
    pub output_rate: u32,
    pub input_device: AudioDeviceInfo,
    pub output_device: AudioDeviceInfo,
    pub input_rx: tokio_mpsc::Receiver<Vec<i16>>,
    pub output_tx: SyncSender<Vec<i16>>,
    pub media_tx: SyncSender<Vec<i16>>,
    pub media_gain: crate::media::gain::Gain,
}

impl AudioIo {
    /// Start playback and capture while retaining the bundled channels.
    pub fn start(&self) -> Result<()> {
        self.streams.start()
    }

    /// Split the channels from the stream guard without shortening stream lifetime.
    ///
    /// Destructure the returned value, keep `streams` in scope, and move
    /// `input_rx`/`output_tx` into the async peer task.
    pub fn into_parts(self) -> AudioIoParts {
        AudioIoParts {
            streams: self.streams,
            input_rate: self.input_rate,
            output_rate: self.output_rate,
            input_device: self.input_device,
            output_device: self.output_device,
            input_rx: self.input_rx,
            output_tx: self.output_tx,
            media_tx: self.media_tx,
            media_gain: self.media_gain,
        }
    }
}

/// Resolve a selector against device metadata without touching audio hardware.
///
/// See [`AudioDevices::open`] for what each selector means; `None` auto-selects
/// the preferred speakerphone.
pub fn select_device<'a>(
    devices: &'a [AudioDeviceInfo],
    selector: Option<&str>,
) -> Result<&'a AudioDeviceInfo> {
    select_device_with_label(devices, selector, "audio")
}

fn default_device_id(device: Option<cpal::Device>) -> Option<String> {
    device.and_then(|device| device.id().ok().map(|id| id.to_string()))
}

fn enumerate_devices(
    host: &cpal::Host,
    direction: Direction,
    default_id: Option<&str>,
) -> Result<Vec<AudioDevice>> {
    let devices: Vec<_> = match direction {
        Direction::Input => host
            .input_devices()
            .with_context(|| format!("failed to enumerate {direction} audio devices"))?
            .collect(),
        Direction::Output => host
            .output_devices()
            .with_context(|| format!("failed to enumerate {direction} audio devices"))?
            .collect(),
    };

    let mut available = Vec::new();
    for device in devices {
        let inspected = (|| -> Result<AudioDevice> {
            let id = device
                .id()
                .with_context(|| format!("failed to read {direction} audio device ID"))?
                .to_string();
            if is_alsa_software_mixer(&id) {
                bail!("ALSA software mixing plugin PCMs are never probed");
            }
            let name = device
                .description()
                .with_context(|| format!("failed to describe {direction} audio device {id:?}"))?
                .name()
                .to_owned();
            let config = match direction {
                Direction::Input => device.default_input_config(),
                Direction::Output => device.default_output_config(),
            }
            .with_context(|| {
                format!("failed to read default {direction} config for {name:?} ({id})")
            })?;

            Ok(AudioDevice {
                info: AudioDeviceInfo {
                    index: 0,
                    is_default: default_id == Some(id.as_str()),
                    id,
                    name,
                    config: config.into(),
                },
                device,
            })
        })();

        match inspected {
            Ok(device) => available.push(device),
            Err(error) => {
                tracing::debug!(%error, %direction, "skipping unavailable audio device")
            }
        }
    }

    available.sort_by(|left, right| compare_device_info(&left.info, &right.info));
    for (zero_based_index, device) in available.iter_mut().enumerate() {
        device.info.index = zero_based_index + 1;
    }

    Ok(available)
}

/// ALSA `dsnoop`/`dmix` plugin PCMs.
///
/// These are userspace mixers layered directly on a hardware PCM. Probing
/// their configuration opens and prepares that hardware behind the sound
/// server's back; on Realtek HDA codecs the Alt Analog `dsnoop` probe leaves
/// the main ADC producing no frames for the next capture stream. They are
/// never a useful selection for this client, so they are excluded before any
/// probe happens.
fn is_alsa_software_mixer(id: &str) -> bool {
    id.strip_prefix("alsa:")
        .is_some_and(|pcm| pcm.starts_with("dsnoop:") || pcm.starts_with("dmix:"))
}

fn compare_device_info(left: &AudioDeviceInfo, right: &AudioDeviceInfo) -> Ordering {
    left.id
        .cmp(&right.id)
        .then_with(|| left.name.cmp(&right.name))
        .then_with(|| left.config.sample_rate.cmp(&right.config.sample_rate))
        .then_with(|| left.config.channels.cmp(&right.config.channels))
        .then_with(|| left.config.sample_format.cmp(&right.config.sample_format))
}

fn print_devices(heading: &str, devices: &[AudioDevice]) {
    println!("{heading}:");
    if devices.is_empty() {
        println!("  (none)");
        return;
    }

    for device in devices {
        let default_marker = if device.info.is_default {
            " [default]"
        } else {
            ""
        };
        println!(
            "  {}. {}{}\n     ID: {}\n     Config: {}",
            device.info.index, device.info.name, default_marker, device.info.id, device.info.config
        );
    }
}

fn select_entry<'a>(
    devices: &'a [AudioDevice],
    selector: Option<&str>,
    direction: Direction,
) -> Result<&'a AudioDevice> {
    let infos = devices
        .iter()
        .map(|device| device.info.clone())
        .collect::<Vec<_>>();
    let selected = select_device_with_label(&infos, selector, &direction.to_string())?;
    let selected_index = selected.index;

    // Display indexes are assigned after sorting and are therefore contiguous.
    Ok(&devices[selected_index - 1])
}

fn select_device_with_label<'a>(
    devices: &'a [AudioDeviceInfo],
    selector: Option<&str>,
    direction: &str,
) -> Result<&'a AudioDeviceInfo> {
    if devices.is_empty() {
        bail!("no {direction} devices are available");
    }

    let selector = selector.unwrap_or_default().trim();
    // Nothing asked for: take the speakerphone if this host has one.
    if selector.is_empty() {
        if let Some(preferred) = preferred_device(devices) {
            tracing::debug!(
                %direction,
                device = %preferred.name,
                id = %preferred.id,
                "no selector given; auto-selected the preferred speakerphone"
            );
            return Ok(preferred);
        }
    }
    if selector.is_empty() || selector.eq_ignore_ascii_case("default") {
        return unique_match(
            devices.iter().filter(|device| device.is_default).collect(),
            "default",
            direction,
            "system default",
        );
    }

    let exact_ids = devices
        .iter()
        .filter(|device| device.id == selector)
        .collect::<Vec<_>>();
    if !exact_ids.is_empty() {
        return unique_match(exact_ids, selector, direction, "stable ID");
    }

    if let Ok(index) = selector.parse::<usize>() {
        return devices
            .iter()
            .find(|device| device.index == index)
            .with_context(|| {
                format!(
                    "no {direction} device has display index {index}; valid indexes are 1..={}",
                    devices.len()
                )
            });
    }

    let exact_names = devices
        .iter()
        .filter(|device| device.name == selector)
        .collect::<Vec<_>>();
    if !exact_names.is_empty() {
        return unique_match(exact_names, selector, direction, "exact name");
    }

    let folded_selector = selector.to_lowercase();
    let substring_matches = devices
        .iter()
        .filter(|device| device.name.to_lowercase().contains(&folded_selector))
        .collect::<Vec<_>>();
    if substring_matches.is_empty() {
        bail!(
            "no {direction} device matches selector {selector:?}; use a displayed index, stable ID, or device name"
        );
    }

    unique_match(
        substring_matches,
        selector,
        direction,
        "case-insensitive name substring",
    )
}

/// The best [`PREFERRED_DEVICE_NAME`] device, if this host has one.
///
/// One speakerphone appears once on CoreAudio but many times on ALSA -- once
/// per PCM alias of its card, all carrying the same name -- so matching the
/// name alone is ambiguous. Rank the aliases and take the best; equal ranks
/// keep the deterministic display order.
fn preferred_device(devices: &[AudioDeviceInfo]) -> Option<&AudioDeviceInfo> {
    devices
        .iter()
        .filter(|device| device.name.to_lowercase().contains(PREFERRED_DEVICE_NAME))
        .min_by_key(|device| (alias_rank(&device.id), device.index))
}

/// Preference among the ALSA PCM aliases of one card; lower is better.
///
/// `plughw` is the plug layer over the card's hardware PCM: it converts rate,
/// format and channel count, and it lets ALSA size the period from the
/// hardware, which is what makes full duplex work. `front` and `hw` reach the
/// same hardware without conversion, so they only open when the device's own
/// default format is usable.
///
/// `default` and `sysdefault` come last of the usable paths, even though they
/// are the ones that mix with other raw-ALSA clients. They resolve to
/// `dmix`/`dsnoop`, which pin the playback period at 1024 frames; CPAL's
/// `BufferSize::Default` then double-buffers that into a 42.7 ms ring, while
/// capture on the same card runs 64 ms periods. The playback ring drains
/// before the worker is next serviced and the stream underruns once per
/// capture period -- measured on a Jabra Speak2 40 as 93 XRUNs in 6 s of
/// duplex, against none on `plughw`. The period is not negotiable: CPAL
/// reports the supported range as exactly `1024..=1024`, so the ring cannot be
/// widened to cover it.
///
/// Aliases that are not a plain analog path at all (`iec958`, `surround40`,
/// ...) rank last. Non-ALSA IDs all rank first and equal: on those hosts the
/// device is listed exactly once.
fn alias_rank(id: &str) -> u8 {
    let Some(pcm) = id.strip_prefix("alsa:") else {
        return 0;
    };
    match pcm.split(':').next().unwrap_or_default() {
        "plughw" => 0,
        "front" => 1,
        "hw" => 2,
        "default" | "sysdefault" => 3,
        _ => 4,
    }
}

/// The best-ranked match when every candidate is an ALSA alias of one device.
///
/// ALSA lists a single card once per PCM alias, all under the same name, so a
/// selector that hits several of them is not really ambiguous: it is one piece
/// of hardware reachable by several paths, and [`alias_rank`] knows which path
/// to take. Candidates that differ by name, or that come from a host which
/// lists each device once, stay ambiguous for the caller to report.
fn best_alias<'a>(matches: &[&'a AudioDeviceInfo]) -> Option<&'a AudioDeviceInfo> {
    let first = matches.first()?;
    if !matches
        .iter()
        .all(|device| device.name == first.name && device.id.starts_with("alsa:"))
    {
        return None;
    }
    matches
        .iter()
        .copied()
        .min_by_key(|device| (alias_rank(&device.id), device.index))
}

fn unique_match<'a>(
    matches: Vec<&'a AudioDeviceInfo>,
    selector: &str,
    direction: &str,
    match_kind: &str,
) -> Result<&'a AudioDeviceInfo> {
    match matches.as_slice() {
        [device] => return Ok(device),
        [] if match_kind == "system default" => {
            bail!("no system-default {direction} device is available")
        }
        [] => bail!("no {direction} device matches selector {selector:?}"),
        _ => {}
    }

    // Several ALSA aliases of one device: rank them rather than refusing.
    if let Some(device) = best_alias(&matches) {
        return Ok(device);
    }

    let candidates = matches
        .iter()
        .map(|device| format!("{}: {} ({})", device.index, device.name, device.id))
        .collect::<Vec<_>>()
        .join(", ");
    bail!("{direction} device selector {selector:?} is ambiguous ({match_kind}); matches: {candidates}")
}

fn ensure_pcm_format(format: SampleFormat, direction: Direction) -> Result<()> {
    match format {
        SampleFormat::I8
        | SampleFormat::I16
        | SampleFormat::I24
        | SampleFormat::I32
        | SampleFormat::I64
        | SampleFormat::U8
        | SampleFormat::U16
        | SampleFormat::U24
        | SampleFormat::U32
        | SampleFormat::U64
        | SampleFormat::F32
        | SampleFormat::F64 => Ok(()),
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => {
            bail!("{direction} format {format} is DSD; this client accepts PCM device formats only")
        }
        _ => bail!("unsupported {direction} sample format {format}"),
    }
}

fn build_input_stream(
    selected: &AudioDevice,
    sender: tokio_mpsc::Sender<Vec<i16>>,
) -> Result<cpal::Stream> {
    let format = selected.info.config.sample_format;
    match format {
        SampleFormat::I8 => build_input_stream_typed::<i8>(selected, sender),
        SampleFormat::I16 => build_input_stream_typed::<i16>(selected, sender),
        SampleFormat::I24 => build_input_stream_typed::<cpal::I24>(selected, sender),
        SampleFormat::I32 => build_input_stream_typed::<i32>(selected, sender),
        SampleFormat::I64 => build_input_stream_typed::<i64>(selected, sender),
        SampleFormat::U8 => build_input_stream_typed::<u8>(selected, sender),
        SampleFormat::U16 => build_input_stream_typed::<u16>(selected, sender),
        SampleFormat::U24 => build_input_stream_typed::<cpal::U24>(selected, sender),
        SampleFormat::U32 => build_input_stream_typed::<u32>(selected, sender),
        SampleFormat::U64 => build_input_stream_typed::<u64>(selected, sender),
        SampleFormat::F32 => build_input_stream_typed::<f32>(selected, sender),
        SampleFormat::F64 => build_input_stream_typed::<f64>(selected, sender),
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => {
            bail!("input format {format} is DSD; this client accepts PCM device formats only")
        }
        _ => bail!("unsupported input sample format {format}"),
    }
}

fn build_input_stream_typed<T>(
    selected: &AudioDevice,
    sender: tokio_mpsc::Sender<Vec<i16>>,
) -> Result<cpal::Stream>
where
    T: SizedSample + Send + 'static,
    i16: FromSample<T>,
{
    let config = selected.info.config;
    let channels = usize::from(config.channels);

    let mut refused = None;
    for buffer_size in config.buffer_sizes() {
        let sender = sender.clone();
        let device_id = selected.info.id.clone();
        match selected.device.build_input_stream::<T, _, _>(
            config.stream_config(buffer_size),
            move |samples, _| {
                let mono = downmix_to_mono_i16(samples, channels);
                if !mono.is_empty() {
                    // A full or closed queue must never stall the real-time thread.
                    let _ = sender.try_send(mono);
                }
            },
            move |error| {
                tracing::error!(%error, %device_id, "input audio stream error");
            },
            None,
        ) {
            Ok(stream) => return Ok(stream),
            // Only a refused period is worth retrying at the device's own size.
            Err(error) if matches!(error.kind(), cpal::ErrorKind::UnsupportedConfig) => {
                tracing::debug!(%error, id = %selected.info.id, ?buffer_size, "input period refused");
                refused = Some(error);
            }
            Err(error) => return Err(error).with_context(|| open_failure("input", selected)),
        }
    }
    Err(refused.expect("a refusal for every candidate period"))
        .with_context(|| open_failure("input", selected))
}

fn open_failure(direction: &str, selected: &AudioDevice) -> String {
    format!(
        "failed to open {direction} device {:?} ({}) with default config {}",
        selected.info.name, selected.info.id, selected.info.config
    )
}

fn downmix_to_mono_i16<T>(samples: &[T], channels: usize) -> Vec<i16>
where
    T: Sample,
    i16: FromSample<T>,
{
    if channels == 0 {
        return Vec::new();
    }

    samples
        .chunks_exact(channels)
        .map(|frame| {
            let sum = frame
                .iter()
                .map(|&sample| i64::from(i16::from_sample(sample)))
                .sum::<i64>();
            (sum / channels as i64) as i16
        })
        .collect()
}

/// The stream plus its two feed channels and the media gain: `(stream,
/// voice_tx, media_tx, gain)`.
type OutputStreamParts = (
    cpal::Stream,
    SyncSender<Vec<i16>>,
    SyncSender<Vec<i16>>,
    crate::media::gain::Gain,
);

/// Build the playback stream together with the queue that feeds it.
///
/// The queue is created here, not by the caller, because a refused period
/// leaves its [`Receiver`] inside the dropped callback: each attempt needs a
/// fresh channel, and only the successful one's sender may escape.
fn build_output_stream(selected: &AudioDevice, capacity: usize) -> Result<OutputStreamParts> {
    let format = selected.info.config.sample_format;
    match format {
        SampleFormat::I8 => build_output_stream_typed::<i8>(selected, capacity),
        SampleFormat::I16 => build_output_stream_typed::<i16>(selected, capacity),
        SampleFormat::I24 => build_output_stream_typed::<cpal::I24>(selected, capacity),
        SampleFormat::I32 => build_output_stream_typed::<i32>(selected, capacity),
        SampleFormat::I64 => build_output_stream_typed::<i64>(selected, capacity),
        SampleFormat::U8 => build_output_stream_typed::<u8>(selected, capacity),
        SampleFormat::U16 => build_output_stream_typed::<u16>(selected, capacity),
        SampleFormat::U24 => build_output_stream_typed::<cpal::U24>(selected, capacity),
        SampleFormat::U32 => build_output_stream_typed::<u32>(selected, capacity),
        SampleFormat::U64 => build_output_stream_typed::<u64>(selected, capacity),
        SampleFormat::F32 => build_output_stream_typed::<f32>(selected, capacity),
        SampleFormat::F64 => build_output_stream_typed::<f64>(selected, capacity),
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => {
            bail!("output format {format} is DSD; this client accepts PCM device formats only")
        }
        _ => bail!("unsupported output sample format {format}"),
    }
}

fn build_output_stream_typed<T>(
    selected: &AudioDevice,
    capacity: usize,
) -> Result<OutputStreamParts>
where
    T: SizedSample + FromSample<i16> + Send + 'static,
{
    let config = selected.info.config;
    let channels = usize::from(config.channels);

    let mut refused = None;
    for buffer_size in config.buffer_sizes() {
        let device_id = selected.info.id.clone();
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let (media_sender, media_receiver) = mpsc::sync_channel(capacity);
        let gain = crate::media::gain::Gain::new(crate::media::gain::FULL);
        let mut queued_audio = OutputMixer::new(
            receiver,
            media_receiver,
            gain.clone(),
            crate::media::gain::step_for(config.sample_rate),
        );
        match selected.device.build_output_stream::<T, _, _>(
            config.stream_config(buffer_size),
            move |output, _| {
                fill_output_buffer(output, channels, || queued_audio.next_sample());
            },
            move |error| {
                tracing::error!(%error, %device_id, "output audio stream error");
            },
            None,
        ) {
            Ok(stream) => return Ok((stream, sender, media_sender, gain)),
            // Only a refused period is worth retrying at the device's own size.
            Err(error) if matches!(error.kind(), cpal::ErrorKind::UnsupportedConfig) => {
                tracing::debug!(%error, id = %selected.info.id, ?buffer_size, "output period refused");
                refused = Some(error);
            }
            Err(error) => return Err(error).with_context(|| open_failure("output", selected)),
        }
    }
    Err(refused.expect("a refusal for every candidate period"))
        .with_context(|| open_failure("output", selected))
}

/// One producer feeding the output callback: a queue plus a read cursor.
struct Source {
    receiver: Receiver<Vec<i16>>,
    current: Vec<i16>,
    offset: usize,
}

impl Source {
    fn new(receiver: Receiver<Vec<i16>>) -> Self {
        Self {
            receiver,
            current: Vec::new(),
            offset: 0,
        }
    }

    fn next_sample(&mut self) -> Option<i16> {
        loop {
            if let Some(&sample) = self.current.get(self.offset) {
                self.offset += 1;
                return Some(sample);
            }
            match self.receiver.try_recv() {
                Ok(chunk) => {
                    self.current = chunk;
                    self.offset = 0;
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return None,
            }
        }
    }
}

/// Sums the call's voice with the media player, scaling media by a ramped
/// gain. The hardware callback pulls this, so the ramp advances on the audio
/// clock and needs no timer.
struct OutputMixer {
    voice: Source,
    media: Source,
    gain: crate::media::gain::Gain,
    current_gain: f32,
    step: f32,
}

impl OutputMixer {
    fn new(
        voice: Receiver<Vec<i16>>,
        media: Receiver<Vec<i16>>,
        gain: crate::media::gain::Gain,
        step: f32,
    ) -> Self {
        let current_gain = gain.target();
        Self {
            voice: Source::new(voice),
            media: Source::new(media),
            gain,
            current_gain,
            step,
        }
    }

    fn next_sample(&mut self) -> Option<i16> {
        if self.gain.take_flush() {
            // Drop the previous source's audio rather than playing it under
            // the new one. Bounded by the channel capacity; only ever runs on
            // an explicit stop or station change, never per callback.
            self.media.current.clear();
            self.media.offset = 0;
            while self.media.receiver.try_recv().is_ok() {}
        }

        let voice = self.voice.next_sample();
        let media = self.media.next_sample();

        // Advance the ramp every sample, so a gap in the media queue cannot
        // strand the gain mid-fade.
        let target = self.gain.target();
        self.current_gain = if self.gain.take_jump() {
            target
        } else {
            crate::media::gain::advance(self.current_gain, target, self.step)
        };

        if voice.is_none() && media.is_none() {
            return None;
        }
        let media = f32::from(media.unwrap_or(0)) * self.current_gain;
        let mixed = i32::from(voice.unwrap_or(0)) + media as i32;
        Some(mixed.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16)
    }
}

fn fill_output_buffer<T>(
    output: &mut [T],
    channels: usize,
    mut next_sample: impl FnMut() -> Option<i16>,
) where
    T: Sample + FromSample<i16>,
{
    if channels == 0 {
        output.fill(T::EQUILIBRIUM);
        return;
    }

    let mut frames = output.chunks_mut(channels);
    while let Some(frame) = frames.next() {
        let Some(sample) = next_sample() else {
            frame.fill(T::EQUILIBRIUM);
            for remaining in frames {
                remaining.fill(T::EQUILIBRIUM);
            }
            return;
        };
        frame.fill(T::from_sample(sample));
    }
}

#[cfg(test)]
mod live_tests {
    //! Opens the real auto-selected speakerphone:
    //! `cargo test -p voice-chatbot-client -- --ignored live`.
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

    /// Counts the ERROR events the stream error callbacks emit on CPAL's
    /// worker threads, which a thread-local subscriber would never see.
    struct CountStreamErrors(Arc<AtomicUsize>);

    impl<S: tracing::Subscriber> Layer<S> for CountStreamErrors {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            if *event.metadata().level() == tracing::Level::ERROR {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Full duplex on the auto-selected device, with no stream errors.
    ///
    /// A smoke test, not a reliable regression guard: on `sysdefault:` (dmix)
    /// the underrun this checks for is *intermittent* -- measured at 93 XRUNs
    /// in 6 s on two runs out of three, and none on the third -- so a pass
    /// here does not prove the alias ranking is right. A failure does prove it
    /// is wrong. `plughw:` was clean on every run. See [`alias_rank`].
    #[test]
    #[ignore]
    fn live_duplex_on_the_auto_selected_device_does_not_underrun() {
        let xruns = Arc::new(AtomicUsize::new(0));
        // Global, because the callbacks fire on CPAL's threads. If another
        // test in this binary already installed one we can only check capture.
        let counting = tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(CountStreamErrors(xruns.clone())),
        )
        .is_ok();

        let devices = AudioDevices::new().unwrap();
        let audio = devices
            .open(None, None)
            .expect("open auto-selected devices");
        eprintln!(
            "input:  {} ({}, {})\noutput: {} ({}, {})",
            audio.input_device.name,
            audio.input_device.id,
            audio.input_device.config,
            audio.output_device.name,
            audio.output_device.id,
            audio.output_device.config,
        );

        let parts = audio.into_parts();
        parts.streams.start().expect("start streams");

        let mut input_rx = parts.input_rx;
        let captured = Arc::new(AtomicUsize::new(0));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(6);
        while std::time::Instant::now() < deadline {
            let _ = parts.output_tx.try_send(vec![0i16; 480]);
            if let Ok(chunk) = input_rx.try_recv() {
                captured.fetch_add(chunk.len(), Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        drop(parts.streams);

        let frames = captured.load(Ordering::Relaxed);
        assert!(
            frames > parts.input_rate as usize,
            "expected over a second of capture, got {frames} samples at {} Hz",
            parts.input_rate
        );

        if counting {
            let xruns = xruns.load(Ordering::Relaxed);
            assert_eq!(
                xruns, 0,
                "{xruns} stream errors in 6 s of duplex on {}; see alias_rank",
                parts.output_device.id
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn mixer_of(voice: Vec<i16>, media: Vec<i16>, gain_value: f32) -> OutputMixer {
        let (voice_tx, voice_rx) = mpsc::sync_channel(4);
        let (media_tx, media_rx) = mpsc::sync_channel(4);
        if !voice.is_empty() {
            voice_tx.try_send(voice).expect("queue voice");
        }
        if !media.is_empty() {
            media_tx.try_send(media).expect("queue media");
        }
        drop(voice_tx);
        drop(media_tx);
        let gain = crate::media::gain::Gain::new(gain_value);
        // A step of 1.0 settles the ramp on the first sample, so these tests
        // assert mixing rather than ramp timing.
        OutputMixer::new(voice_rx, media_rx, gain, 1.0)
    }

    #[test]
    fn mixer_is_silent_only_when_both_sources_are_dry() {
        let mut empty = mixer_of(vec![], vec![], 1.0);
        assert_eq!(empty.next_sample(), None);

        let mut voice_only = mixer_of(vec![100], vec![], 1.0);
        assert_eq!(voice_only.next_sample(), Some(100));
        assert_eq!(voice_only.next_sample(), None);

        let mut media_only = mixer_of(vec![], vec![100], 1.0);
        assert_eq!(media_only.next_sample(), Some(100));
        assert_eq!(media_only.next_sample(), None);
    }

    #[test]
    fn mixer_sums_both_sources_and_scales_only_the_media_one() {
        let mut full = mixer_of(vec![1000], vec![1000], 1.0);
        assert_eq!(full.next_sample(), Some(2000));

        // The voice is untouched by the media gain.
        let mut ducked = mixer_of(vec![1000], vec![1000], 0.5);
        assert_eq!(ducked.next_sample(), Some(1500));

        // Fully ducked media still keeps the stream alive.
        let mut silent = mixer_of(vec![], vec![1000], 0.0);
        assert_eq!(silent.next_sample(), Some(0));
    }

    #[test]
    fn mixer_saturates_instead_of_wrapping() {
        let mut hot = mixer_of(vec![30000], vec![30000], 1.0);
        assert_eq!(hot.next_sample(), Some(i16::MAX));

        let mut cold = mixer_of(vec![-30000], vec![-30000], 1.0);
        assert_eq!(cold.next_sample(), Some(i16::MIN));
    }

    #[test]
    fn mixer_ramps_the_media_gain_and_a_jump_skips_the_ramp() {
        let (voice_tx, voice_rx) = mpsc::sync_channel(4);
        let (media_tx, media_rx) = mpsc::sync_channel(4);
        media_tx.try_send(vec![1000; 4]).expect("queue media");
        drop(voice_tx);
        drop(media_tx);
        let gain = crate::media::gain::Gain::new(0.0);
        let mut mixer = OutputMixer::new(voice_rx, media_rx, gain.clone(), 0.25);

        // Ramping up from 0: the first sample is one step in, not the target.
        gain.ramp_to(1.0);
        assert_eq!(mixer.next_sample(), Some(250));
        assert_eq!(mixer.next_sample(), Some(500));

        // A jump lands on the target immediately.
        gain.jump_to(0.0);
        assert_eq!(mixer.next_sample(), Some(0));
    }

    #[test]
    fn a_flush_discards_media_queued_by_the_previous_source() {
        let (voice_tx, voice_rx) = mpsc::sync_channel(4);
        let (media_tx, media_rx) = mpsc::sync_channel(4);
        media_tx.try_send(vec![1000; 2]).expect("queue media");
        media_tx.try_send(vec![2000; 2]).expect("queue more media");
        drop(voice_tx);
        drop(media_tx);
        let gain = crate::media::gain::Gain::new(1.0);
        let mut mixer = OutputMixer::new(voice_rx, media_rx, gain.clone(), 1.0);

        // One sample of the old source is consumed before the switch.
        assert_eq!(mixer.next_sample(), Some(1000));

        // Everything still queued belongs to the previous stream.
        gain.flush();
        assert_eq!(
            mixer.next_sample(),
            None,
            "queued audio from the previous source must not be heard"
        );
    }

    fn config() -> AudioStreamConfig {
        AudioStreamConfig {
            channels: 2,
            sample_rate: 48_000,
            sample_format: SampleFormat::I16,
            buffer_size: SupportedBufferSize::Unknown,
        }
    }

    fn device(index: usize, id: &str, name: &str, is_default: bool) -> AudioDeviceInfo {
        AudioDeviceInfo {
            index,
            id: id.to_owned(),
            name: name.to_owned(),
            is_default,
            config: config(),
        }
    }

    #[test]
    fn requested_period_is_twenty_milliseconds_of_the_device_rate() {
        let sizes = |sample_rate| {
            let config = AudioStreamConfig {
                sample_rate,
                ..config()
            };
            config.buffer_sizes()
        };

        for (rate, frames) in [(48_000, 960), (44_100, 882), (16_000, 320), (8_000, 160)] {
            assert!(
                matches!(sizes(rate)[0], cpal::BufferSize::Fixed(n) if n == frames),
                "{rate} Hz should ask for {frames} frames"
            );
        }
        // A device that refuses the period falls back to its own choice.
        assert!(matches!(sizes(48_000)[1], cpal::BufferSize::Default));
    }

    #[test]
    fn selector_obeys_precedence_and_one_based_indexes() {
        let devices = vec![
            device(1, "host:alpha", "2", true),
            device(2, "1", "Studio Mic", false),
            device(3, "host:studio", "Studio Mic Pro", false),
        ];

        assert_eq!(select_device(&devices, None).unwrap().index, 1);
        assert_eq!(select_device(&devices, Some("default")).unwrap().index, 1);
        // Exact IDs take precedence over numeric indexes.
        assert_eq!(select_device(&devices, Some("1")).unwrap().index, 2);
        assert_eq!(select_device(&devices, Some("2")).unwrap().index, 2);
        // Exact names take precedence over substring matching.
        assert_eq!(
            select_device(&devices, Some("Studio Mic")).unwrap().index,
            2
        );
        assert_eq!(select_device(&devices, Some("mic pro")).unwrap().index, 3);
    }

    /// The ALSA aliases one Jabra card produces, plus an unrelated default.
    fn jabra_host() -> Vec<AudioDeviceInfo> {
        vec![
            device(1, "alsa:default", "Default ALSA Output", true),
            device(
                2,
                "alsa:front:CARD=UC,DEV=0",
                "Jabra Speak2 40 UC, USB Audio",
                false,
            ),
            device(
                3,
                "alsa:hw:CARD=2,DEV=0",
                "Jabra Speak2 40 UC, USB Audio",
                false,
            ),
            device(
                4,
                "alsa:iec958:CARD=UC,DEV=0",
                "Jabra Speak2 40 UC, USB Audio",
                false,
            ),
            device(
                5,
                "alsa:plughw:CARD=UC,DEV=0",
                "Jabra Speak2 40 UC, USB Audio",
                false,
            ),
            device(
                6,
                "alsa:sysdefault:CARD=UC",
                "Jabra Speak2 40 UC, USB Audio",
                false,
            ),
        ]
    }

    #[test]
    fn unspecified_selector_prefers_the_speakerphone_over_the_system_default() {
        let devices = jabra_host();

        for selector in [None, Some(""), Some("   ")] {
            let selected = select_device(&devices, selector).unwrap();
            assert_eq!(selected.id, "alsa:plughw:CARD=UC,DEV=0");
        }

        // "default" is an explicit request, and still means the system default.
        assert_eq!(
            select_device(&devices, Some("default")).unwrap().id,
            "alsa:default"
        );
        assert_eq!(
            select_device(&devices, Some("DEFAULT")).unwrap().id,
            "alsa:default"
        );
    }

    #[test]
    fn unspecified_selector_falls_back_to_the_system_default_without_a_speakerphone() {
        let devices = vec![
            device(1, "alsa:default", "Default ALSA Output", true),
            device(
                2,
                "alsa:hw:CARD=PCH,DEV=0",
                "HDA Intel PCH, ALC1220 Analog",
                false,
            ),
        ];

        assert_eq!(select_device(&devices, None).unwrap().id, "alsa:default");
        assert!(select_device(
            &[device(1, "alsa:hw:CARD=PCH,DEV=0", "ALC1220", false)],
            None
        )
        .unwrap_err()
        .to_string()
        .contains("no system-default"));
    }

    #[test]
    fn speakerphone_alias_ranking_prefers_the_pcm_that_survives_duplex() {
        // `plughw` lets ALSA size the period from the hardware. `sysdefault`
        // resolves to dmix/dsnoop, whose 1024-frame period double-buffers into
        // a 42.7 ms ring that underruns once per 64 ms capture period.
        assert!(alias_rank("alsa:plughw:CARD=UC,DEV=0") < alias_rank("alsa:front:CARD=UC,DEV=0"));
        assert!(alias_rank("alsa:front:CARD=UC,DEV=0") < alias_rank("alsa:hw:CARD=UC,DEV=0"));
        assert!(alias_rank("alsa:hw:CARD=UC,DEV=0") < alias_rank("alsa:sysdefault:CARD=UC"));
        assert!(alias_rank("alsa:sysdefault:CARD=UC") < alias_rank("alsa:iec958:CARD=UC,DEV=0"));
        // A host that lists the device once needs no ranking at all.
        assert_eq!(alias_rank("coreaudio:Jabra Speak2 40 UC"), 0);
        // `default` resolves through the same mixing chain as `sysdefault`.
        assert_eq!(
            alias_rank("alsa:default"),
            alias_rank("alsa:sysdefault:CARD=UC")
        );

        // Each tier wins once the better ones are gone, and equal ranks keep
        // the deterministic display order.
        let mut devices = jabra_host();
        for expected in [
            "alsa:plughw:CARD=UC,DEV=0",
            "alsa:front:CARD=UC,DEV=0",
            "alsa:hw:CARD=2,DEV=0",
            "alsa:sysdefault:CARD=UC",
            "alsa:iec958:CARD=UC,DEV=0",
        ] {
            assert_eq!(preferred_device(&devices).unwrap().id, expected);
            devices.retain(|device| device.id != expected);
        }
        assert_eq!(preferred_device(&devices), None, "only the default is left");
    }

    #[test]
    fn speakerphone_is_matched_case_insensitively_anywhere_in_the_name() {
        let devices = vec![
            device(1, "alsa:default", "Default ALSA Output", true),
            device(2, "coreaudio:x", "JABRA Evolve2 65", false),
        ];
        assert_eq!(select_device(&devices, None).unwrap().index, 2);
    }

    #[test]
    fn explicit_selectors_still_win_over_the_speakerphone() {
        let devices = jabra_host();

        assert_eq!(select_device(&devices, Some("3")).unwrap().index, 3);
        assert_eq!(
            select_device(&devices, Some("alsa:plughw:CARD=UC,DEV=0"))
                .unwrap()
                .index,
            5
        );
    }

    #[test]
    fn naming_one_device_ranks_its_alsa_aliases_instead_of_refusing() {
        let devices = jabra_host();

        // A substring, and an exact name, that hit every alias of one card.
        for selector in ["jabra", "JABRA", "Jabra Speak2 40 UC, USB Audio"] {
            assert_eq!(
                select_device(&devices, Some(selector)).unwrap().id,
                "alsa:plughw:CARD=UC,DEV=0",
                "{selector:?} names one device reachable by several PCM paths"
            );
        }
    }

    #[test]
    fn ranking_never_papers_over_a_real_ambiguity() {
        // Same name, but not ALSA aliases: these are two separate devices and
        // the caller has to say which one.
        let two_devices = vec![
            device(1, "coreaudio:a", "USB Mic", false),
            device(2, "coreaudio:b", "USB Mic", false),
        ];
        assert!(select_device(&two_devices, Some("USB Mic"))
            .unwrap_err()
            .to_string()
            .contains("ambiguous"));

        // ALSA aliases, but of devices that differ by name.
        let two_cards = vec![
            device(1, "alsa:plughw:CARD=UC,DEV=0", "Jabra Speak2 40 UC", false),
            device(2, "alsa:plughw:CARD=PCH,DEV=0", "Jabra Evolve2 65", false),
        ];
        assert!(select_device(&two_cards, Some("jabra"))
            .unwrap_err()
            .to_string()
            .contains("ambiguous"));
    }

    #[test]
    fn selector_reports_ambiguous_substrings_and_exact_names() {
        let devices = vec![
            device(1, "host:a", "USB Mic", false),
            device(2, "host:b", "USB Mic", false),
            device(3, "host:c", "USB Headset", true),
        ];

        let exact_error = select_device(&devices, Some("USB Mic"))
            .unwrap_err()
            .to_string();
        assert!(exact_error.contains("ambiguous (exact name)"));

        let substring_error = select_device(&devices, Some("usb"))
            .unwrap_err()
            .to_string();
        assert!(substring_error.contains("ambiguous (case-insensitive name substring)"));
        assert!(substring_error.contains("1: USB Mic (host:a)"));
        assert!(substring_error.contains("3: USB Headset (host:c)"));
    }

    #[test]
    fn selector_reports_missing_default_index_and_name() {
        let devices = vec![device(1, "host:a", "Built-in Mic", false)];

        assert!(select_device(&devices, None)
            .unwrap_err()
            .to_string()
            .contains("no system-default"));
        assert!(select_device(&devices, Some("0"))
            .unwrap_err()
            .to_string()
            .contains("display index 0"));
        assert!(select_device(&devices, Some("missing"))
            .unwrap_err()
            .to_string()
            .contains("no audio device matches"));
    }

    #[test]
    fn enumeration_never_probes_alsa_software_mixing_plugins() {
        // Probing dsnoop/dmix opens the underlying hardware PCM directly behind
        // PipeWire's back; on an ALC1220 the Alt Analog dsnoop probe leaves the
        // codec's main ADC silent for the next capture stream.
        for id in [
            "alsa:dsnoop:CARD=PCH,DEV=2",
            "alsa:dsnoop:CARD=PCH,DEV=0",
            "alsa:dmix:CARD=PCH,DEV=0",
            "alsa:dmix:CARD=NVidia,DEV=3",
        ] {
            assert!(is_alsa_software_mixer(id), "{id} must be skipped");
        }
        for id in [
            "alsa:default",
            "alsa:pipewire",
            "alsa:hw:CARD=PCH,DEV=2",
            "alsa:plughw:CARD=PCH,DEV=0",
            "alsa:front:CARD=PCH,DEV=0",
            "coreaudio:dsnoop",
        ] {
            assert!(!is_alsa_software_mixer(id), "{id} must be probed");
        }
    }

    #[test]
    fn device_order_uses_stable_id() {
        let mut devices = [
            device(0, "host:z", "beta", false),
            device(0, "host:b", "Alpha", false),
            device(0, "host:a", "Alpha", false),
            device(0, "host:c", "alpha", false),
        ];
        devices.sort_by(compare_device_info);

        let ids = devices
            .iter()
            .map(|device| device.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["host:a", "host:b", "host:c", "host:z"]);
    }

    #[test]
    fn input_downmixes_interleaved_pcm_without_overflow() {
        assert_eq!(
            downmix_to_mono_i16(&[3_000_i16, 1_000, -1_000, -3_000], 2),
            vec![2_000, -2_000]
        );
        assert_eq!(
            downmix_to_mono_i16(&[i16::MAX, i16::MAX, i16::MIN, i16::MIN], 2),
            vec![i16::MAX, i16::MIN]
        );
        assert!(downmix_to_mono_i16(&[1_i16, 2], 0).is_empty());
    }

    #[test]
    fn input_conversion_accepts_every_pcm_sample_type() {
        fn assert_silence<T>(sample: T)
        where
            T: Sample,
            i16: FromSample<T>,
        {
            assert_eq!(downmix_to_mono_i16(&[sample], 1), vec![0]);
        }

        assert_silence(0_i8);
        assert_silence(0_i16);
        assert_silence(cpal::I24::new(0).unwrap());
        assert_silence(0_i32);
        assert_silence(0_i64);
        assert_silence(128_u8);
        assert_silence(32_768_u16);
        assert_silence(cpal::U24::new(1 << 23).unwrap());
        assert_silence(1_u32 << 31);
        assert_silence(1_u64 << 63);
        assert_silence(0.0_f32);
        assert_silence(0.0_f64);
    }

    #[test]
    fn output_duplicates_mono_samples_and_silences_underflow() {
        let mut source = [Some(1_000_i16), Some(-2_000)].into_iter();
        let mut stereo = [123_i16; 6];
        fill_output_buffer(&mut stereo, 2, || source.next().flatten());
        assert_eq!(stereo, [1_000, 1_000, -2_000, -2_000, 0, 0]);

        let mut unsigned = [0_u16; 2];
        fill_output_buffer(&mut unsigned, 2, || None);
        assert_eq!(unsigned, [u16::EQUILIBRIUM; 2]);
    }

    #[test]
    fn output_checks_an_empty_queue_only_once_per_callback() {
        let mut polls = 0;
        let mut output = [123_i16; 256];
        fill_output_buffer(&mut output, 2, || {
            polls += 1;
            None
        });

        assert_eq!(polls, 1);
        assert_eq!(output, [0; 256]);
    }

    #[test]
    fn dsd_formats_have_clear_errors() {
        for format in [
            SampleFormat::DsdU8,
            SampleFormat::DsdU16,
            SampleFormat::DsdU32,
        ] {
            let error = ensure_pcm_format(format, Direction::Input)
                .unwrap_err()
                .to_string();
            assert!(error.contains("DSD"));
            assert!(error.contains("PCM"));
        }
    }
}
