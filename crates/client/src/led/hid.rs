//! Finding a Jabra telephony HID interface and the output-report bits that
//! drive its LEDs. Layouts differ across Jabra models, so report IDs and
//! bit positions come from the interface's own report descriptor — never
//! from hardcoded bytes (docs/specs/jabra-led.md, R3).

use anyhow::Result;
use hidreport::{Field, Report, ReportDescriptor};

/// GN Audio (Jabra) USB vendor ID. Not consumed by this module — reserved
/// for the device-discovery task that matches USB devices before parsing
/// their descriptors.
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal telephony collection shaped like a Jabra's: one input
    /// report (hook switch + mute button, which the mapper must ignore) and
    /// one output report with the three LEDs. Synthetic — the real Speak2 40
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
}
