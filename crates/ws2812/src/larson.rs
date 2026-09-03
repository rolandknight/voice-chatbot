//! The Larson scanner: a port of Evil Mad Scientist's `larson.c` (v1.4,
//! <https://github.com/evil-mad/larsonscanner>), generalised from the kit's
//! nine LEDs to any strip length.
//!
//! The "eye" is four parts with relative brightness 1:4:2:1 -- leading edge,
//! peak, then two steps of tail -- moving 1/16 of an LED per step. Each part
//! is split between the two LEDs it sits across in proportion to the sub-step,
//! which is what makes the motion smooth, and parts that run off the end are
//! reflected back onto the strip, which folds the eye up at the ends before it
//! turns around. Everything here is integer arithmetic on the kit's own scale
//! (a part's full brightness is 15, the sum is clipped at 60) so the frames
//! can be checked against the firmware by hand.

/// Sub-steps per LED: the kit's 4-bit fractional position.
const SUBSTEPS: u32 = 16;
/// Relative brightness of the kit's eye parts, leading edge first (`LEDBright`).
const KIT_EYE: [u32; 4] = [1, 4, 2, 1];
/// Where the kit clips the summed brightness (`while (j < 60)`).
const MAX_LEVEL: u32 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Towards higher LED indices.
    Up,
    /// Towards LED 0.
    Down,
}

#[derive(Clone, Debug)]
pub struct Larson {
    count: usize,
    position: u32,
    direction: Direction,
    weights: [u32; 4],
}

impl Larson {
    /// The kit's scanner over `count` LEDs (at least two), starting at the
    /// low end and moving up.
    pub fn new(count: usize) -> Self {
        Self::with_weights(count, KIT_EYE)
    }

    /// A scanner with its own eye: the relative brightness of the four
    /// parts, leading edge first, each at most 4 so the sum stays on the
    /// kit's scale. The kit's "skinny" mode is `[0, 4, 1, 0]`; a lone
    /// pixel is `[0, 4, 0, 0]`.
    pub fn with_weights(count: usize, weights: [u32; 4]) -> Self {
        assert!(count >= 2, "a Larson scanner needs at least two LEDs");
        assert!(
            weights.iter().all(|w| *w <= 4),
            "eye weights are on the kit's scale, at most 4"
        );
        Larson {
            count,
            position: 0,
            direction: Direction::Up,
            weights,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Sub-steps in one end-to-end pass (128 on the nine-LED kit).
    pub fn steps_per_pass(&self) -> u32 {
        SUBSTEPS * (self.count as u32 - 1)
    }

    /// Advance one sub-step. Returns true when the eye has reached the end
    /// of the strip and turned around.
    pub fn step(&mut self) -> bool {
        self.position += 1;
        if self.position < self.steps_per_pass() {
            return false;
        }
        self.position = 0;
        self.direction = match self.direction {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
        };
        true
    }

    /// Brightness of every LED at the current position, 0..=1.
    pub fn levels(&self) -> Vec<f32> {
        self.raw_levels()
            .into_iter()
            .map(|v| v.min(MAX_LEVEL) as f32 / MAX_LEVEL as f32)
            .collect()
    }

    /// The kit's summed brightness per LED, before clipping.
    fn raw_levels(&self) -> Vec<u32> {
        let last = self.count as i32 - 1;
        let sub_max = SUBSTEPS - 1;
        // The LED the eye is anchored on, and how each part's brightness is
        // split between its LED (`this`) and the next one along (`next`) --
        // the firmware's ILED, RLED and MLED.
        let (anchor, this, next) = match self.direction {
            Direction::Up => {
                let p = sub_max + self.position;
                (p / SUBSTEPS, p % SUBSTEPS, sub_max - p % SUBSTEPS)
            }
            Direction::Down => {
                let p = self.steps_per_pass() - 1 - self.position;
                (p / SUBSTEPS, sub_max - p % SUBSTEPS, p % SUBSTEPS)
            }
        };
        let anchor = anchor as i32;
        // Where each part sits, leading edge first: two ahead of the anchor
        // down to two behind, reflected back inside the strip at both ends.
        let mut at = [0usize; 5];
        for (j, slot) in at.iter_mut().enumerate() {
            let offset = 2 - j as i32;
            let loc = match self.direction {
                Direction::Up => anchor + offset,
                Direction::Down => anchor - offset,
            };
            *slot = reflect(loc, last) as usize;
        }
        let mut leds = vec![0u32; self.count];
        for (j, weight) in self.weights.iter().enumerate() {
            leds[at[j]] += weight * this;
            leds[at[j + 1]] += weight * next;
        }
        leds
    }
}

/// Fold an LED index back into `0..=last`, mirroring at each end.
fn reflect(mut loc: i32, last: i32) -> i32 {
    while loc < 0 || loc > last {
        if loc < 0 {
            loc = -loc;
        }
        if loc > last {
            loc = 2 * last - loc;
        }
    }
    loc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_after(count: usize, steps: u32) -> Vec<u32> {
        let mut s = Larson::new(count);
        for _ in 0..steps {
            s.step();
        }
        s.raw_levels()
    }

    #[test]
    fn matches_the_kit_at_the_start_of_a_pass() {
        // larson.c at position 0, direction 0: the peak (weight 4) on LED 1,
        // the tail (weight 2) on LED 0, the leading edge (1) on LED 2, and
        // the far tail (1) reflected from -1 back onto LED 1.
        assert_eq!(raw_after(9, 0), [30, 75, 15, 0, 0, 0, 0, 0, 0]);
        let levels = Larson::new(9).levels();
        assert_eq!(&levels[..3], [0.5, 1.0, 0.25]);
    }

    #[test]
    fn peak_reaches_the_far_end_before_folding_back() {
        // Position 112 on the kit: ILED = 7, RLED = 15, so the peak (4 x 15)
        // lands on LED 8; LED 7 gets the leading edge reflected from 9 plus
        // the first tail step (15 + 30).
        assert_eq!(raw_after(9, 112), [0, 0, 0, 0, 0, 0, 15, 45, 60]);
        // Fifteen steps later the eye has folded up against the end.
        assert_eq!(raw_after(9, 127), [0, 0, 0, 0, 0, 0, 15, 73, 32]);
    }

    #[test]
    fn turns_around_after_one_pass() {
        let mut s = Larson::new(8);
        assert_eq!(s.steps_per_pass(), 112);
        for _ in 0..111 {
            assert!(!s.step());
            assert_eq!(s.direction(), Direction::Up);
        }
        assert!(s.step());
        assert_eq!(s.direction(), Direction::Down);
        assert_eq!(s.count(), 8);
    }

    #[test]
    fn the_down_pass_mirrors_the_up_pass_one_step_later() {
        // The firmware anchors the up pass at 15 + position and the down pass
        // at 127 - position, so down frame N is the mirror of up frame N + 1,
        // and the last down frame repeats the first up frame: the eye dwells
        // one step at the low end before setting off again.
        let count = 8;
        let mut up = Larson::new(count);
        up.step();
        let mut down = Larson::new(count);
        for _ in 0..down.steps_per_pass() {
            down.step();
        }
        assert_eq!(down.direction(), Direction::Down);
        for pos in 0..down.steps_per_pass() - 1 {
            let mut mirrored = up.levels();
            mirrored.reverse();
            assert_eq!(down.levels(), mirrored, "down position {pos}");
            assert!(!up.step() || pos == down.steps_per_pass() - 2);
            assert!(!down.step());
        }
        assert_eq!(down.levels(), Larson::new(count).levels());
        assert!(down.step());
        assert_eq!(down.direction(), Direction::Up);
    }

    #[test]
    fn levels_are_within_range_and_never_dark() {
        let mut s = Larson::new(5);
        for _ in 0..(2 * s.steps_per_pass()) {
            let levels = s.levels();
            assert!(levels.iter().all(|l| (0.0..=1.0).contains(l)));
            assert!(levels.iter().any(|&l| l > 0.0));
            s.step();
        }
    }

    #[test]
    fn short_strips_stay_in_bounds() {
        for count in 2..=4 {
            let mut s = Larson::new(count);
            for _ in 0..(3 * s.steps_per_pass()) {
                assert_eq!(s.levels().len(), count);
                s.step();
            }
        }
    }

    #[test]
    fn a_single_pixel_eye_splits_across_two_leds_as_it_moves() {
        let mut s = Larson::with_weights(8, [0, 4, 0, 0]);
        assert_eq!(
            s.levels(),
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "one pixel, on LED 1 at the start"
        );
        for _ in 0..8 {
            s.step();
        }
        let lit: Vec<(usize, f32)> = s
            .levels()
            .into_iter()
            .enumerate()
            .filter(|(_, level)| *level > 0.0)
            .collect();
        assert_eq!(
            lit.len(),
            2,
            "half way between LEDs it lights both: {lit:?}"
        );
        assert_eq!((lit[0].0, lit[1].0), (1, 2));
        assert!(
            (lit[0].1 + lit[1].1 - 1.0).abs() < 1e-6,
            "the brightness is shared, not doubled"
        );
    }

    #[test]
    fn reflect_folds_into_range() {
        assert_eq!(reflect(9, 8), 7);
        assert_eq!(reflect(-1, 8), 1);
        assert_eq!(reflect(3, 1), 1);
        assert_eq!(reflect(-2, 1), 0);
        assert_eq!(reflect(4, 8), 4);
    }
}
