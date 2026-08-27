//! Native audio device discovery, selection, capture, and playback.
//!
//! Audio crossing this module's channel boundary is always mono `i16`.  The
//! callbacks only perform sample-format conversion, channel mixing, and
//! non-blocking channel operations.  Resampling and codec work belong on the
//! consumer side of [`AudioIo`].

use std::cmp::Ordering;
use std::fmt;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};

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
    fn stream_config(self) -> cpal::StreamConfig {
        cpal::StreamConfig {
            channels: self.channels,
            sample_rate: self.sample_rate,
            buffer_size: cpal::BufferSize::Default,
        }
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
    /// `None`, an empty selector, or `"default"` selects the system default.
    /// Other selectors are resolved in this order: exact stable ID, one-based
    /// display index, exact name, then a unique case-insensitive name substring.
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
        let (output_tx, output_rx) = mpsc::sync_channel(output_capacity);

        let input_stream = build_input_stream(input, input_tx)?;
        let output_stream = build_output_stream(output, output_rx)?;

        Ok(AudioIo {
            input_rate: input.info.config.sample_rate,
            output_rate: output.info.config.sample_rate,
            input_device: input.info.clone(),
            output_device: output.info.clone(),
            input_rx,
            output_tx,
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
        }
    }
}

/// Resolve a selector against device metadata without touching audio hardware.
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

    let selector = selector.unwrap_or("default").trim();
    if selector.is_empty() || selector.eq_ignore_ascii_case("default") {
        return unique_match(
            devices.iter().filter(|device| device.is_default).collect(),
            selector,
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

fn unique_match<'a>(
    matches: Vec<&'a AudioDeviceInfo>,
    selector: &str,
    direction: &str,
    match_kind: &str,
) -> Result<&'a AudioDeviceInfo> {
    match matches.as_slice() {
        [device] => Ok(*device),
        [] if match_kind == "system default" => {
            bail!("no system-default {direction} device is available")
        }
        [] => bail!("no {direction} device matches selector {selector:?}"),
        _ => {
            let candidates = matches
                .iter()
                .map(|device| format!("{}: {} ({})", device.index, device.name, device.id))
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "{direction} device selector {selector:?} is ambiguous ({match_kind}); matches: {candidates}"
            )
        }
    }
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
    let channels = usize::from(selected.info.config.channels);
    let device_id = selected.info.id.clone();
    selected
        .device
        .build_input_stream::<T, _, _>(
            selected.info.config.stream_config(),
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
        )
        .with_context(|| {
            format!(
                "failed to open input device {:?} ({}) with default config {}",
                selected.info.name, selected.info.id, selected.info.config
            )
        })
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

fn build_output_stream(
    selected: &AudioDevice,
    receiver: Receiver<Vec<i16>>,
) -> Result<cpal::Stream> {
    let format = selected.info.config.sample_format;
    match format {
        SampleFormat::I8 => build_output_stream_typed::<i8>(selected, receiver),
        SampleFormat::I16 => build_output_stream_typed::<i16>(selected, receiver),
        SampleFormat::I24 => build_output_stream_typed::<cpal::I24>(selected, receiver),
        SampleFormat::I32 => build_output_stream_typed::<i32>(selected, receiver),
        SampleFormat::I64 => build_output_stream_typed::<i64>(selected, receiver),
        SampleFormat::U8 => build_output_stream_typed::<u8>(selected, receiver),
        SampleFormat::U16 => build_output_stream_typed::<u16>(selected, receiver),
        SampleFormat::U24 => build_output_stream_typed::<cpal::U24>(selected, receiver),
        SampleFormat::U32 => build_output_stream_typed::<u32>(selected, receiver),
        SampleFormat::U64 => build_output_stream_typed::<u64>(selected, receiver),
        SampleFormat::F32 => build_output_stream_typed::<f32>(selected, receiver),
        SampleFormat::F64 => build_output_stream_typed::<f64>(selected, receiver),
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => {
            bail!("output format {format} is DSD; this client accepts PCM device formats only")
        }
        _ => bail!("unsupported output sample format {format}"),
    }
}

fn build_output_stream_typed<T>(
    selected: &AudioDevice,
    receiver: Receiver<Vec<i16>>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<i16> + Send + 'static,
{
    let channels = usize::from(selected.info.config.channels);
    let device_id = selected.info.id.clone();
    let mut queued_audio = OutputQueue::new(receiver);

    selected
        .device
        .build_output_stream::<T, _, _>(
            selected.info.config.stream_config(),
            move |output, _| {
                fill_output_buffer(output, channels, || queued_audio.next_sample());
            },
            move |error| {
                tracing::error!(%error, %device_id, "output audio stream error");
            },
            None,
        )
        .with_context(|| {
            format!(
                "failed to open output device {:?} ({}) with default config {}",
                selected.info.name, selected.info.id, selected.info.config
            )
        })
}

struct OutputQueue {
    receiver: Receiver<Vec<i16>>,
    current: Vec<i16>,
    offset: usize,
}

impl OutputQueue {
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
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

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
