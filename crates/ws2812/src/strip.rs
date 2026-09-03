//! WS2812B over SPI.
//!
//! Each LED bit becomes four SPI bits, `1000` for a 0 and `1110` for a 1, so
//! at 3.2 MHz a bit is 1.25 µs with a 0.31 µs or 0.94 µs high pulse -- inside
//! the WS2812B's windows. A run of zero bytes after the frame holds the line
//! low for the >280 µs reset that latches it. The whole frame goes out as one
//! spidev transfer: the LED measures the high time of each pulse, so a gap
//! between bytes only matters once it is long enough to count as a reset --
//! which a write per byte, an ioctl each, easily is.

use std::fmt;
use std::str::FromStr;

use crate::color::Rgb;

/// Bytes on the wire per LED: 24 colour bits x 4 SPI bits.
pub const BYTES_PER_LED: usize = 12;
/// Idle-low time after a frame that latches it (WS2812B-V5 needs >= 280 µs).
pub const RESET_US: u64 = 400;
/// spidev's default ceiling for one transfer (`spidev.bufsiz`).
pub const SPIDEV_BUFSIZ: usize = 4096;

const PATTERNS: [u8; 4] = [0b1000_1000, 0b1000_1110, 0b1110_1000, 0b1110_1110];

/// The byte order a strip expects. WS2812B is GRB; some clones are RGB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Order {
    #[default]
    Grb,
    Rgb,
}

impl Order {
    fn bytes(self, px: Rgb) -> [u8; 3] {
        match self {
            Order::Grb => [px.g, px.r, px.b],
            Order::Rgb => [px.r, px.g, px.b],
        }
    }
}

impl FromStr for Order {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "grb" => Ok(Order::Grb),
            "rgb" => Ok(Order::Rgb),
            other => Err(format!("{other:?} is not a colour order: use grb or rgb")),
        }
    }
}

impl fmt::Display for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Order::Grb => "grb",
            Order::Rgb => "rgb",
        })
    }
}

/// Zero bytes that hold the line low for [`RESET_US`] at `hz`.
pub fn reset_bytes(hz: u32) -> usize {
    (u64::from(hz) * RESET_US / 8 / 1_000_000) as usize + 1
}

/// Bytes one frame of `count` LEDs puts on the wire at `hz`.
pub fn frame_bytes(count: usize, hz: u32) -> usize {
    count * BYTES_PER_LED + reset_bytes(hz)
}

/// Encode `pixels` for the wire into `out` (cleared first), followed by
/// `reset` zero bytes.
pub fn encode(pixels: &[Rgb], order: Order, reset: usize, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(pixels.len() * BYTES_PER_LED + reset);
    for px in pixels {
        for byte in order.bytes(*px) {
            for shift in [6, 4, 2, 0] {
                out.push(PATTERNS[usize::from((byte >> shift) & 0b11)]);
            }
        }
    }
    out.resize(out.len() + reset, 0);
}

pub use platform::Ws2812Spi;

#[cfg(target_os = "linux")]
mod platform {
    use std::path::Path;

    use anyhow::{bail, Context, Result};
    use spidev::{SpiModeFlags, Spidev, SpidevOptions, SpidevTransfer};

    use super::{encode, frame_bytes, reset_bytes, Order, SPIDEV_BUFSIZ};
    use crate::color::Rgb;

    /// A strip of `count` WS2812s on a spidev node.
    pub struct Ws2812Spi {
        dev: Spidev,
        order: Order,
        reset: usize,
        count: usize,
        frame: Vec<u8>,
    }

    impl Ws2812Spi {
        /// Open `path` (on a Pi, `/dev/spidev0.0` is SPI0 with MOSI on
        /// GPIO 10) at `hz`, which should sit in 2.0-3.8 MHz.
        pub fn open(path: &Path, hz: u32, order: Order, count: usize) -> Result<Self> {
            let bytes = frame_bytes(count, hz);
            if bytes > SPIDEV_BUFSIZ {
                bail!(
                    "{count} LEDs need {bytes} bytes per frame, over spidev's {SPIDEV_BUFSIZ}-byte \
                     transfer limit (raise it with the spidev.bufsiz kernel parameter)"
                );
            }
            let mut dev = Spidev::open(path).with_context(|| match path.exists() {
                false => format!(
                    "{} does not exist: SPI is off. Put `dtparam=spi=on` in \
                     /boot/firmware/config.txt and reboot (run-on-pi.sh does this); do not \
                     apply it live with dtparam or raspi-config, which hung a Pi 5 run",
                    path.display()
                ),
                true => format!(
                    "cannot open {}: add yourself to the spi group (`sudo usermod -aG spi $USER`, \
                     then log in again)",
                    path.display()
                ),
            })?;
            dev.configure(
                &SpidevOptions::new()
                    .bits_per_word(8)
                    .max_speed_hz(hz)
                    .mode(SpiModeFlags::SPI_MODE_0)
                    .build(),
            )
            .with_context(|| format!("configuring {} for {hz} Hz", path.display()))?;
            Ok(Ws2812Spi {
                dev,
                order,
                reset: reset_bytes(hz),
                count,
                frame: Vec::with_capacity(bytes),
            })
        }

        /// Show `pixels` (the first `count` of them; missing ones are dark).
        pub fn write(&mut self, pixels: &[Rgb]) -> Result<()> {
            let mut padded;
            let pixels = if pixels.len() == self.count {
                pixels
            } else {
                padded = vec![Rgb::BLACK; self.count];
                let n = pixels.len().min(self.count);
                padded[..n].copy_from_slice(&pixels[..n]);
                &padded
            };
            encode(pixels, self.order, self.reset, &mut self.frame);
            let mut transfer = SpidevTransfer::write(&self.frame);
            self.dev
                .transfer(&mut transfer)
                .context("writing a frame to the strip")
        }

        /// Turn every LED off.
        pub fn clear(&mut self) -> Result<()> {
            self.write(&[])
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use std::path::Path;

    use anyhow::{bail, Result};

    use super::Order;
    use crate::color::Rgb;

    /// Placeholder off Linux so the workspace builds everywhere; spidev is a
    /// Linux interface.
    pub struct Ws2812Spi {
        _private: (),
    }

    impl Ws2812Spi {
        pub fn open(_path: &Path, _hz: u32, _order: Order, _count: usize) -> Result<Self> {
            bail!("WS2812 over spidev is only available on Linux")
        }

        pub fn write(&mut self, _pixels: &[Rgb]) -> Result<()> {
            bail!("WS2812 over spidev is only available on Linux")
        }

        pub fn clear(&mut self) -> Result<()> {
            self.write(&[])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_two_bits_per_byte_in_grb_order() {
        let mut out = Vec::new();
        encode(&[Rgb::new(0xff, 0x00, 0x81)], Order::Grb, 2, &mut out);
        assert_eq!(
            out,
            [
                0x88, 0x88, 0x88, 0x88, // g = 0x00
                0xee, 0xee, 0xee, 0xee, // r = 0xff
                0xe8, 0x88, 0x88, 0x8e, // b = 0x81 = 10 00 00 01
                0x00, 0x00, // reset
            ]
        );
    }

    #[test]
    fn rgb_order_swaps_the_first_two_bytes() {
        let mut grb = Vec::new();
        let mut rgb = Vec::new();
        let px = [Rgb::new(0x12, 0x34, 0x56)];
        encode(&px, Order::Grb, 0, &mut grb);
        encode(&px, Order::Rgb, 0, &mut rgb);
        assert_eq!(&grb[..4], &rgb[4..8]);
        assert_eq!(&grb[4..8], &rgb[..4]);
        assert_eq!(&grb[8..], &rgb[8..]);
    }

    #[test]
    fn frame_size_is_twelve_bytes_per_led_plus_reset() {
        let mut out = vec![0xaa; 3];
        encode(&[Rgb::BLACK; 8], Order::Grb, 10, &mut out);
        assert_eq!(out.len(), 8 * BYTES_PER_LED + 10);
        assert!(out[8 * BYTES_PER_LED..].iter().all(|&b| b == 0));
        assert_eq!(frame_bytes(8, 3_200_000), 96 + 161);
    }

    #[test]
    fn reset_holds_the_line_low_long_enough() {
        // 400 µs at 3.2 MHz is 1280 bits = 160 bytes, plus one for rounding.
        assert_eq!(reset_bytes(3_200_000), 161);
        assert_eq!(reset_bytes(2_000_000), 101);
        let low_us = reset_bytes(3_800_000) as u64 * 8 * 1_000_000 / 3_800_000;
        assert!(low_us >= RESET_US);
    }

    #[test]
    fn parses_orders() {
        assert_eq!("GRB".parse::<Order>().unwrap(), Order::Grb);
        assert_eq!("rgb".parse::<Order>().unwrap(), Order::Rgb);
        assert!("bgr".parse::<Order>().is_err());
        assert_eq!(Order::default().to_string(), "grb");
    }
}
