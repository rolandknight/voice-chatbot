//! Colours for the strip: an 8-bit RGB triple, the `red` / `#ff8800` spelling
//! the command line accepts, and the brightness-curve painting that turns a
//! scanner frame into pixels.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const BLACK: Rgb = Rgb::new(0, 0, 0);
    pub const WHITE: Rgb = Rgb::new(255, 255, 255);

    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Rgb { r, g, b }
    }

    /// Every channel scaled by `level`, clamped to 0..=1.
    pub fn scaled(self, level: f32) -> Rgb {
        let k = level.clamp(0.0, 1.0);
        let scale = |c: u8| (f32::from(c) * k).round() as u8;
        Rgb::new(scale(self.r), scale(self.g), scale(self.b))
    }
}

impl fmt::Display for Rgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match NAMED.iter().find(|(_, c)| c == self) {
            Some((name, _)) => f.write_str(name),
            None => write!(f, "#{:02x}{:02x}{:02x}", self.r, self.g, self.b),
        }
    }
}

/// Names accepted on the command line. Orange and yellow lean red because a
/// WS2812's green is the strongest of its three emitters.
pub const NAMED: &[(&str, Rgb)] = &[
    ("red", Rgb::new(255, 0, 0)),
    ("orange", Rgb::new(255, 80, 0)),
    ("yellow", Rgb::new(255, 200, 0)),
    ("green", Rgb::new(0, 255, 0)),
    ("cyan", Rgb::new(0, 255, 255)),
    ("blue", Rgb::new(0, 0, 255)),
    ("magenta", Rgb::new(255, 0, 255)),
    ("purple", Rgb::new(128, 0, 255)),
    ("white", Rgb::WHITE),
];

/// The scanner's colours, one per pass: what the demo shows by default and
/// what the client shows while the bot is thinking.
pub const PALETTE: [Rgb; 8] = [
    Rgb::new(255, 0, 0),
    Rgb::new(255, 80, 0),
    Rgb::new(255, 200, 0),
    Rgb::new(0, 255, 0),
    Rgb::new(0, 255, 255),
    Rgb::new(0, 0, 255),
    Rgb::new(255, 0, 255),
    Rgb::WHITE,
];

impl FromStr for Rgb {
    type Err = String;

    /// A name from [`NAMED`] or `rrggbb` / `#rrggbb` hex.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if let Some((_, c)) = NAMED.iter().find(|(n, _)| n.eq_ignore_ascii_case(s)) {
            return Ok(*c);
        }
        let hex = s.strip_prefix('#').unwrap_or(s);
        let bad = || {
            let names: Vec<&str> = NAMED.iter().map(|(n, _)| *n).collect();
            format!(
                "{s:?} is not a colour: use #rrggbb or one of {}",
                names.join(", ")
            )
        };
        if hex.len() != 6 {
            return Err(bad());
        }
        let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| bad());
        Ok(Rgb::new(channel(0)?, channel(2)?, channel(4)?))
    }
}

/// Turn per-LED brightness levels (0..=1) into pixels of one colour.
/// `gamma` shapes the curve (1 = linear, the kit's PWM), `brightness` caps it.
pub fn paint(levels: &[f32], color: Rgb, brightness: f32, gamma: f32, out: &mut [Rgb]) {
    for (px, level) in out.iter_mut().zip(levels) {
        *px = color.scaled(brightness * level.clamp(0.0, 1.0).powf(gamma));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_and_hex() {
        assert_eq!("red".parse::<Rgb>().unwrap(), Rgb::new(255, 0, 0));
        assert_eq!(" Cyan ".parse::<Rgb>().unwrap(), Rgb::new(0, 255, 255));
        assert_eq!("#ff8000".parse::<Rgb>().unwrap(), Rgb::new(255, 128, 0));
        assert_eq!("0A0b0C".parse::<Rgb>().unwrap(), Rgb::new(10, 11, 12));
    }

    #[test]
    fn rejects_garbage() {
        for bad in ["", "#ff", "reddish", "#gg0000", "#1234567"] {
            let err = bad.parse::<Rgb>().unwrap_err();
            assert!(err.contains("not a colour"), "{bad:?}: {err}");
        }
    }

    #[test]
    fn displays_names_or_hex() {
        assert_eq!(Rgb::new(0, 0, 255).to_string(), "blue");
        assert_eq!(Rgb::new(1, 2, 3).to_string(), "#010203");
    }

    #[test]
    fn scales_and_clamps() {
        assert_eq!(Rgb::new(200, 100, 0).scaled(0.5), Rgb::new(100, 50, 0));
        assert_eq!(Rgb::WHITE.scaled(2.0), Rgb::WHITE);
        assert_eq!(Rgb::WHITE.scaled(-1.0), Rgb::BLACK);
    }

    #[test]
    fn paints_linear_and_gamma_curves() {
        let levels = [1.0, 0.5, 0.0];
        let mut out = [Rgb::BLACK; 3];
        paint(&levels, Rgb::new(255, 0, 0), 1.0, 1.0, &mut out);
        assert_eq!(out, [Rgb::new(255, 0, 0), Rgb::new(128, 0, 0), Rgb::BLACK]);
        paint(&levels, Rgb::new(255, 0, 0), 0.2, 2.0, &mut out);
        assert_eq!(out, [Rgb::new(51, 0, 0), Rgb::new(13, 0, 0), Rgb::BLACK]);
    }
}
