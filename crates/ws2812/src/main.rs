//! `ws2812-poc`: light a WS2812B strip on a Raspberry Pi's SPI0 MOSI (GPIO 10)
//! with a Larson scanner, or run a wiring check. Ctrl-C clears the strip.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use voice_chatbot_ws2812::color::{paint, Rgb, PALETTE};
use voice_chatbot_ws2812::larson::Larson;
use voice_chatbot_ws2812::strip::{Order, Ws2812Spi};

#[derive(Debug, Parser)]
#[command(
    name = "ws2812-poc",
    version,
    about = "Larson scanner on a WS2812B strip wired to a Raspberry Pi's SPI0 MOSI (GPIO 10, pin 19)"
)]
struct Cli {
    /// LEDs on the strip.
    #[arg(long, default_value_t = 8)]
    count: usize,

    /// spidev node the strip's data line is on (SPI0 MOSI is GPIO 10, header pin 19).
    #[arg(long, default_value = "/dev/spidev0.0")]
    spi: PathBuf,

    /// SPI clock in Hz. 2.0-3.8 MHz keeps the WS2812 pulse widths in spec.
    #[arg(long, default_value_t = 3_200_000)]
    spi_hz: u32,

    /// Colour byte order the strip expects: grb (WS2812B) or rgb.
    #[arg(long, default_value_t = Order::Grb)]
    order: Order,

    /// Overall brightness, 0-1. Full white on 8 LEDs is 480 mA and painful to look at.
    #[arg(long, default_value_t = 0.2)]
    brightness: f32,

    /// Exponent shaping the eye's brightness curve; 1 is the kit's linear PWM.
    #[arg(long, default_value_t = 1.0)]
    gamma: f32,

    /// Milliseconds for one end-to-end pass of the eye.
    #[arg(long, default_value_t = 700)]
    sweep_ms: u64,

    /// Colours the eye cycles through, one per pass: names or #rrggbb, comma-separated.
    #[arg(long, value_delimiter = ',', default_values_t = PALETTE)]
    colors: Vec<Rgb>,

    /// Stop after this many seconds (0 = run until Ctrl-C).
    #[arg(long, default_value_t = 0)]
    seconds: u64,

    /// What to show.
    #[arg(long, value_enum, default_value_t = Pattern::Larson)]
    pattern: Pattern,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Pattern {
    /// The scanner, cycling through --colors.
    Larson,
    /// Wiring check: every LED red, green, blue, white for a second each,
    /// then one LED at a time from index 0. Wrong colours mean --order.
    Wiring,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.count < 2 {
        bail!("--count must be at least 2");
    }
    if !(0.0..=1.0).contains(&cli.brightness) {
        bail!("--brightness must be within 0 and 1");
    }
    if cli.gamma <= 0.0 {
        bail!("--gamma must be positive");
    }
    if cli.sweep_ms == 0 {
        bail!("--sweep-ms must be positive");
    }
    if cli.colors.is_empty() {
        bail!("--colors needs at least one colour");
    }
    if !(1_000_000..=10_000_000).contains(&cli.spi_hz) {
        bail!(
            "--spi-hz {} is outside anything a WS2812 could decode",
            cli.spi_hz
        );
    }
    if !(2_000_000..=3_800_000).contains(&cli.spi_hz) {
        eprintln!(
            "warning: --spi-hz {} is outside 2.0-3.8 MHz; the pulse widths may not decode",
            cli.spi_hz
        );
    }

    let stop = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let stop = Arc::clone(&stop);
        move || stop.store(true, Ordering::SeqCst)
    })
    .context("installing the Ctrl-C handler")?;

    let mut strip = Ws2812Spi::open(&cli.spi, cli.spi_hz, cli.order, cli.count)?;
    strip.clear()?;
    let result = match cli.pattern {
        Pattern::Larson => run_larson(&cli, &mut strip, &stop),
        Pattern::Wiring => run_wiring(&cli, &mut strip, &stop),
    };
    // Whatever happened, leave the strip dark rather than frozen mid-frame.
    let cleared = strip.clear();
    result?;
    cleared?;
    if stop.load(Ordering::SeqCst) {
        println!("stopped; strip cleared");
    }
    Ok(())
}

fn run_larson(cli: &Cli, strip: &mut Ws2812Spi, stop: &AtomicBool) -> Result<()> {
    let mut scanner = Larson::new(cli.count);
    let step = Duration::from_millis(cli.sweep_ms) / scanner.steps_per_pass();
    let colors: Vec<String> = cli.colors.iter().map(ToString::to_string).collect();
    println!(
        "larson scanner on {} LEDs via {} at {} Hz: {} ms per pass ({} steps of {:?}), \
         brightness {}, gamma {}, colours {}{}",
        cli.count,
        cli.spi.display(),
        cli.spi_hz,
        cli.sweep_ms,
        scanner.steps_per_pass(),
        step,
        cli.brightness,
        cli.gamma,
        colors.join(", "),
        if cli.seconds > 0 {
            format!("; stopping after {} s", cli.seconds)
        } else {
            "; Ctrl-C to stop".to_owned()
        }
    );

    let deadline = (cli.seconds > 0).then(|| Instant::now() + Duration::from_secs(cli.seconds));
    let mut pixels = vec![Rgb::BLACK; cli.count];
    let mut pass = 0usize;
    let mut next = Instant::now();
    while !stop.load(Ordering::Relaxed) && deadline.is_none_or(|d| Instant::now() < d) {
        let color = cli.colors[pass % cli.colors.len()];
        paint(
            &scanner.levels(),
            color,
            cli.brightness,
            cli.gamma,
            &mut pixels,
        );
        strip.write(&pixels)?;
        if scanner.step() {
            pass += 1;
        }
        // A fixed cadence from the previous deadline, not from "now", so the
        // write time does not stretch every step.
        next += step;
        thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    Ok(())
}

fn run_wiring(cli: &Cli, strip: &mut Ws2812Spi, stop: &AtomicBool) -> Result<()> {
    println!(
        "wiring check on {} LEDs via {} (order {}): each colour below should match what you see",
        cli.count,
        cli.spi.display(),
        cli.order
    );
    let mut pixels = vec![Rgb::BLACK; cli.count];
    for (name, color) in [
        ("red", Rgb::new(255, 0, 0)),
        ("green", Rgb::new(0, 255, 0)),
        ("blue", Rgb::new(0, 0, 255)),
        ("white", Rgb::WHITE),
    ] {
        println!("  all LEDs {name}");
        pixels.fill(color.scaled(cli.brightness));
        strip.write(&pixels)?;
        if !pause(Duration::from_secs(1), stop) {
            return Ok(());
        }
    }
    println!("  one LED at a time, index 0 first (the end the data wire enters)");
    for i in 0..cli.count {
        pixels.fill(Rgb::BLACK);
        pixels[i] = Rgb::WHITE.scaled(cli.brightness);
        strip.write(&pixels)?;
        if !pause(Duration::from_millis(250), stop) {
            return Ok(());
        }
    }
    println!("done");
    Ok(())
}

/// Sleep for `total`, waking early if `stop` is raised; false when it was.
fn pause(total: Duration, stop: &AtomicBool) -> bool {
    let until = Instant::now() + total;
    while Instant::now() < until {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        thread::sleep(Duration::from_millis(20));
    }
    !stop.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_parse() {
        let cli = Cli::try_parse_from(["ws2812-poc"]).unwrap();
        assert_eq!(cli.count, 8);
        assert_eq!(cli.spi, PathBuf::from("/dev/spidev0.0"));
        assert_eq!(cli.order, Order::Grb);
        assert_eq!(cli.colors, PALETTE, "the demo shows the shared palette");
        assert_eq!(cli.pattern, Pattern::Larson);
    }

    #[test]
    fn colours_and_order_come_from_the_command_line() {
        let cli = Cli::try_parse_from([
            "ws2812-poc",
            "--colors",
            "blue,#102030",
            "--order",
            "rgb",
            "--pattern",
            "wiring",
        ])
        .unwrap();
        assert_eq!(
            cli.colors,
            [Rgb::new(0, 0, 255), Rgb::new(0x10, 0x20, 0x30)]
        );
        assert_eq!(cli.order, Order::Rgb);
        assert_eq!(cli.pattern, Pattern::Wiring);
        assert!(Cli::try_parse_from(["ws2812-poc", "--colors", "mauve"]).is_err());
    }
}
