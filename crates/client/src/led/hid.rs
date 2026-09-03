//! Finding a Jabra telephony HID interface and the output-report bits that
//! drive its LEDs. Layouts differ across Jabra models, so report IDs and
//! bit positions come from the interface's own report descriptor — never
//! from hardcoded bytes (docs/specs/jabra-led.md, R3).

use std::ffi::CStr;

use anyhow::{bail, Context, Result};
use hidapi::{HidApi, HidDevice, MAX_REPORT_DESCRIPTOR_SIZE};
use hidreport::{Field, Report, ReportDescriptor};

use crate::led::{Indication, LedSink};

/// GN Audio (Jabra) USB vendor ID.
pub(crate) const JABRA_VENDOR_ID: u16 = 0x0b0e;
/// HID LED usage page, and the telephony LEDs on it (HID Usage Tables §11).
const LED_PAGE: u16 = 0x08;
pub(crate) const USAGE_MUTE: u16 = 0x09;
pub(crate) const USAGE_OFF_HOOK: u16 = 0x17;
pub(crate) const USAGE_RING: u16 = 0x18;

/// One LED bit inside an output report: which report, how many bytes one
/// write of it is (report ID included), and the bit's index counted from
/// the start of that buffer (hidreport's convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedBit {
    pub report_id: u8,
    pub report_len: usize,
    pub bit: usize,
}

/// Where the three telephony LEDs live on one interface, if they exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LedMap {
    pub off_hook: Option<LedBit>,
    pub ring: Option<LedBit>,
    pub mute: Option<LedBit>,
}

/// Pull the three LED bits out of a descriptor's numbered output reports.
pub fn map_leds(descriptor: &[u8]) -> Result<LedMap> {
    let parsed = ReportDescriptor::try_from(descriptor)
        .map_err(|error| anyhow::anyhow!("parse report descriptor: {error:?}"))?;
    let mut map = LedMap::default();
    for report in parsed.output_reports() {
        // Jabra telephony collections number their reports; hidapi needs the
        // ID as the write's first byte, so unnumbered ones are skipped.
        let Some(report_id) = report.report_id() else {
            continue;
        };
        let report_id = u8::from(report_id);
        let report_len = report.size_in_bytes();
        for field in report.fields() {
            let Field::Variable(var) = field else {
                continue;
            };
            if u16::from(var.usage.usage_page) != LED_PAGE {
                continue;
            }
            let led = LedBit {
                report_id,
                report_len,
                bit: var.bits.start,
            };
            match u16::from(var.usage.usage_id) {
                USAGE_OFF_HOOK => map.off_hook.get_or_insert(led),
                USAGE_RING => map.ring.get_or_insert(led),
                USAGE_MUTE => map.mute.get_or_insert(led),
                _ => continue,
            };
        }
    }
    Ok(map)
}

/// Build the output-report buffers (report ID first, as hidapi wants) that
/// show exactly the given bits. Every report carrying a mapped LED is
/// emitted, so clearing a bit rewrites its report with that bit zero.
pub fn compose(map: &LedMap, off_hook: bool, ring: bool, mute: bool) -> Vec<Vec<u8>> {
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    for (led, on) in [(map.off_hook, off_hook), (map.ring, ring), (map.mute, mute)] {
        let Some(led) = led else { continue };
        let buffer = match buffers.iter_mut().find(|buffer| buffer[0] == led.report_id) {
            Some(buffer) => buffer,
            None => {
                buffers.push(vec![0u8; led.report_len]);
                let buffer = buffers.last_mut().unwrap();
                buffer[0] = led.report_id;
                buffer
            }
        };
        if on {
            buffer[led.bit / 8] |= 1 << (led.bit % 8);
        }
    }
    buffers
}

/// A Jabra telephony HID interface with its LED layout.
pub struct TelephonyLeds {
    device: HidDevice,
    map: LedMap,
    product: String,
}

impl TelephonyLeds {
    /// One console-worthy line: what was opened and where its LEDs sit.
    pub fn describe(&self) -> String {
        format!("{} ({:?})", self.product, self.map)
    }
}

impl LedSink for TelephonyLeds {
    fn set(&mut self, indication: Indication) -> Result<()> {
        let (off_hook, ring, mute) = indication.bits();
        for buffer in compose(&self.map, off_hook, ring, mute) {
            self.device
                .write(&buffer)
                .with_context(|| format!("write led report {:#04x}", buffer[0]))?;
        }
        Ok(())
    }
}

/// Belt-and-braces fallback for when `get_report_descriptor` fails on the
/// linux-native backend: the descriptor is also a plain sysfs file, keyed
/// by the hidraw node name.
#[cfg(target_os = "linux")]
fn descriptor_from_sysfs(path: &CStr) -> Option<Vec<u8>> {
    let node = std::path::Path::new(path.to_str().ok()?)
        .file_name()?
        .to_str()?;
    std::fs::read(format!("/sys/class/hidraw/{node}/device/report_descriptor")).ok()
}

#[cfg(not(target_os = "linux"))]
fn descriptor_from_sysfs(_path: &CStr) -> Option<Vec<u8>> {
    None
}

/// Find the first Jabra HID interface whose descriptor offers an Off-Hook
/// LED output — that is the telephony collection. A Jabra presents several
/// HID interfaces (consumer volume, vendor pages); content, not interface
/// number, is what identifies the right one.
pub fn open() -> Result<TelephonyLeds> {
    let api = HidApi::new().context("initialize hidapi")?;
    let mut tried = Vec::new();
    for info in api
        .device_list()
        .filter(|d| d.vendor_id() == JABRA_VENDOR_ID)
    {
        let path = info.path();
        let shown = path.to_string_lossy().into_owned();
        let device = match api.open_path(path) {
            Ok(device) => device,
            Err(error) => {
                tried.push(format!("{shown}: open: {error}"));
                continue;
            }
        };
        let mut buffer = [0u8; MAX_REPORT_DESCRIPTOR_SIZE];
        let descriptor = match device.get_report_descriptor(&mut buffer) {
            Ok(length) => buffer[..length].to_vec(),
            Err(_) => match descriptor_from_sysfs(path) {
                Some(descriptor) => descriptor,
                None => {
                    tried.push(format!("{shown}: no report descriptor"));
                    continue;
                }
            },
        };
        let map = match map_leds(&descriptor) {
            Ok(map) => map,
            Err(error) => {
                tried.push(format!("{shown}: {error}"));
                continue;
            }
        };
        if map.off_hook.is_none() {
            tried.push(format!("{shown}: no off-hook led"));
            continue;
        }
        let product = info.product_string().unwrap_or("Jabra").trim().to_string();
        return Ok(TelephonyLeds {
            device,
            map,
            product,
        });
    }
    if tried.is_empty() {
        bail!("no Jabra usb hid device present (vendor {JABRA_VENDOR_ID:#06x})");
    }
    bail!("no Jabra telephony led interface: {}", tried.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal telephony collection shaped like a Jabra's: one input
    /// report (hook switch + mute button, which the mapper must ignore),
    /// one output report with the three LEDs, and a third output report on
    /// the Telephony page (a ringer control) the mapper must also ignore —
    /// it proves the LED-usage-page filter actually excludes something,
    /// not just report 1's input fields. Synthetic — the real Speak2 40
    /// descriptor is archived by the hardware-validation task.
    pub(super) const SYNTHETIC_TELEPHONY_DESCRIPTOR: &[u8] = &[
        0x05, 0x0B, // Usage Page (Telephony)
        0x09, 0x05, // Usage (Headset)
        0xA1, 0x01, // Collection (Application)
        0x85, 0x01, //   Report ID (1)
        0x0B, 0x20, 0x00, 0x0B, 0x00, //   Usage (Telephony: Hook Switch)
        0x0B, 0x2F, 0x00, 0x0B, 0x00, //   Usage (Telephony: Phone Mute)
        0x75, 0x01, //   Report Size (1)
        0x95, 0x02, //   Report Count (2)
        0x81, 0x02, //   Input (Data,Var,Abs)
        0x75, 0x06, //   Report Size (6)
        0x95, 0x01, //   Report Count (1)
        0x81, 0x01, //   Input (Const) — padding
        0x85, 0x02, //   Report ID (2)
        0x05, 0x08, //   Usage Page (LED)
        0x09, 0x17, //   Usage (Off-Hook)
        0x09, 0x09, //   Usage (Mute)
        0x09, 0x18, //   Usage (Ring)
        0x75, 0x01, //   Report Size (1)
        0x95, 0x03, //   Report Count (3)
        0x91, 0x02, //   Output (Data,Var,Abs)
        0x75, 0x05, //   Report Size (5)
        0x95, 0x01, //   Report Count (1)
        0x91, 0x01, //   Output (Const) — padding
        0x85, 0x03, //   Report ID (3): non-LED output the mapper must skip
        0x05, 0x0B, //   Usage Page (Telephony)
        0x09, 0x9E, //   Usage (Ringer)
        0x75, 0x01, //   Report Size (1)
        0x95, 0x01, //   Report Count (1)
        0x91, 0x02, //   Output (Data,Var,Abs)
        0x75, 0x07, //   Report Size (7)
        0x95, 0x01, //   Report Count (1)
        0x91, 0x01, //   Output (Const) — padding
        0xC0, // End Collection
    ];

    #[test]
    fn maps_the_three_leds_from_a_telephony_descriptor() {
        let map = map_leds(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let off_hook = map.off_hook.expect("off-hook led");
        let mute = map.mute.expect("mute led");
        let ring = map.ring.expect("ring led");
        assert_eq!(off_hook.report_id, 2);
        assert_eq!(off_hook.report_len, 2, "report id byte + one data byte");
        // Descriptor order is off-hook, mute, ring: consecutive bits. Absolute
        // offsets follow hidreport's convention; relative order is what matters.
        assert_eq!(mute.bit, off_hook.bit + 1);
        assert_eq!(ring.bit, off_hook.bit + 2);
        // All three come from report 2's LED-page fields; report 3's
        // Telephony-page ringer field must not leak into the map.
        assert_eq!(mute.report_id, 2);
        assert_eq!(ring.report_id, 2);
    }

    #[test]
    fn a_descriptor_without_led_outputs_maps_to_nothing() {
        // Consumer-control page, input only — like a Jabra's volume interface.
        const NO_LEDS: &[u8] = &[
            0x05, 0x0C, // Usage Page (Consumer)
            0x09, 0x01, // Usage (Consumer Control)
            0xA1, 0x01, // Collection (Application)
            0x85, 0x03, //   Report ID (3)
            0x09, 0xE9, //   Usage (Volume Up)
            0x09, 0xEA, //   Usage (Volume Down)
            0x75, 0x01, 0x95, 0x02, //   1 bit x 2
            0x81, 0x02, //   Input (Data,Var,Abs)
            0x75, 0x06, 0x95, 0x01, 0x81, 0x01, // padding
            0xC0, // End Collection
        ];
        assert_eq!(map_leds(NO_LEDS).unwrap(), LedMap::default());
    }

    #[test]
    fn a_descriptor_without_report_ids_maps_to_nothing() {
        // LED-page output fields, but no Report ID item anywhere: an
        // unnumbered report. hidapi needs a report ID as the write's first
        // byte, so the mapper must skip it rather than guess one.
        const UNNUMBERED_LED_OUTPUT: &[u8] = &[
            0x05, 0x08, // Usage Page (LED)
            0x09, 0x17, // Usage (Off-Hook)
            0xA1, 0x01, // Collection (Application)
            0x09, 0x17, //   Usage (Off-Hook)
            0x09, 0x09, //   Usage (Mute)
            0x09, 0x18, //   Usage (Ring)
            0x75, 0x01, //   Report Size (1)
            0x95, 0x03, //   Report Count (3)
            0x91, 0x02, //   Output (Data,Var,Abs)
            0x75, 0x05, //   Report Size (5)
            0x95, 0x01, //   Report Count (1)
            0x91, 0x01, //   Output (Const) — padding
            0xC0, // End Collection
        ];
        assert_eq!(map_leds(UNNUMBERED_LED_OUTPUT).unwrap(), LedMap::default());
    }

    #[test]
    fn a_numeric_usage_id_match_on_the_wrong_page_is_not_mapped() {
        // Same numeric usage ID as USAGE_OFF_HOOK, but on the Telephony
        // page rather than LED. Confirms the mapper filters by usage page,
        // not just by numeric usage ID — a fixture with the LED page
        // present elsewhere can't tell a real filter from a no-op one
        // (get_or_insert only fills an empty slot, so a matching ID that
        // shares a report with real LED fields is masked either way).
        const WRONG_PAGE_SAME_ID: &[u8] = &[
            0x05, 0x0B, // Usage Page (Telephony)
            0x09, 0x05, // Usage (Headset)
            0xA1, 0x01, // Collection (Application)
            0x85, 0x01, //   Report ID (1)
            0x09, 0x17, //   Usage (numerically == USAGE_OFF_HOOK, wrong page)
            0x75, 0x01, //   Report Size (1)
            0x95, 0x01, //   Report Count (1)
            0x91, 0x02, //   Output (Data,Var,Abs)
            0x75, 0x07, //   Report Size (7)
            0x95, 0x01, //   Report Count (1)
            0x91, 0x01, //   Output (Const) — padding
            0xC0, // End Collection
        ];
        assert_eq!(map_leds(WRONG_PAGE_SAME_ID).unwrap(), LedMap::default());
    }

    #[test]
    fn composed_reports_round_trip_through_the_descriptor() {
        let parsed = ReportDescriptor::try_from(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let map = map_leds(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let buffers = compose(&map, true, false, true);
        assert_eq!(buffers.len(), 1, "all three leds live in one report");
        let buffer = &buffers[0];
        assert_eq!(buffer[0], 2, "hidapi wants the report id first");
        assert_eq!(buffer.len(), 2);
        // Read the buffer back through hidreport itself: pins the bit-index
        // convention without hardcoding it.
        let report = parsed.find_output_report(buffer).expect("report 2");
        let mut lit = std::collections::HashMap::new();
        for field in report.fields() {
            if let Field::Variable(var) = field {
                let value: u32 = var.extract(buffer).unwrap().into();
                lit.insert(u16::from(var.usage.usage_id), value);
            }
        }
        assert_eq!(lit[&USAGE_OFF_HOOK], 1);
        assert_eq!(lit[&USAGE_RING], 0);
        assert_eq!(lit[&USAGE_MUTE], 1);
    }

    #[test]
    fn clearing_writes_the_report_as_zeros() {
        let map = map_leds(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let buffers = compose(&map, false, false, false);
        assert_eq!(
            buffers.len(),
            1,
            "a mapped report is written even all-clear"
        );
        assert_eq!(buffers[0][0], 2);
        assert!(buffers[0][1..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn an_empty_map_composes_no_writes() {
        assert!(compose(&LedMap::default(), true, true, true).is_empty());
    }
}
