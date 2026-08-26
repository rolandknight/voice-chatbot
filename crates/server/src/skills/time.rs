//! `get_current_time` / `get_current_date` — local clock, rendered in words so
//! the TTS can't misread digits (port of skills/core/get_current_{time,date}).

use async_trait::async_trait;
use chrono::{Datelike, Local, NaiveTime, Timelike};
use serde_json::Value;

use super::{CallCtx, Skill};

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 6] = ["", "", "twenty", "thirty", "forty", "fifty"];

/// Spell 0-59 in words (5 -> "five", 25 -> "twenty five").
fn two_digit_words(n: u32) -> String {
    let n = n as usize;
    if n < 20 {
        ONES[n].to_string()
    } else if n.is_multiple_of(10) {
        TENS[n / 10].to_string()
    } else {
        format!("{} {}", TENS[n / 10], ONES[n % 10])
    }
}

/// "one oh five in the afternoon" — entirely in words.
pub fn spoken_time(t: NaiveTime) -> String {
    let hour24 = t.hour();
    let hour = match hour24 % 12 {
        0 => 12,
        h => h,
    } as usize;
    let minute = t.minute();
    let meridiem = if hour24 < 12 {
        "in the morning"
    } else if hour24 < 18 {
        "in the afternoon"
    } else {
        "in the evening"
    };
    let clock = if minute == 0 {
        format!("{} o'clock", ONES[hour])
    } else if minute < 10 {
        // Keep the leading "oh" so 1:05 is "one oh five", not "one five".
        format!("{} oh {}", ONES[hour], ONES[minute as usize])
    } else {
        format!("{} {}", ONES[hour], two_digit_words(minute))
    };
    format!("{clock} {meridiem}")
}

fn ordinal(n: u32) -> String {
    let suffix = if (10..=20).contains(&(n % 100)) {
        "th"
    } else {
        match n % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    };
    format!("{n}{suffix}")
}

pub fn spoken_date(d: chrono::NaiveDate) -> String {
    format!(
        "Today is {}, {} {}, {}.",
        d.format("%A"),
        d.format("%B"),
        ordinal(d.day()),
        d.year()
    )
}

pub struct GetCurrentTime;

#[async_trait]
impl Skill for GetCurrentTime {
    fn name(&self) -> &str {
        "get_current_time"
    }
    async fn call(&self, _args: &Value, _ctx: &CallCtx) -> String {
        format!("It's {}.", spoken_time(Local::now().time()))
    }
}

pub struct GetCurrentDate;

#[async_trait]
impl Skill for GetCurrentDate {
    fn name(&self) -> &str {
        "get_current_date"
    }
    async fn call(&self, _args: &Value, _ctx: &CallCtx) -> String {
        spoken_date(Local::now().date_naive())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn t(h: u32, m: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, 0).unwrap()
    }

    #[test]
    fn times_are_spoken_in_words() {
        assert_eq!(spoken_time(t(1, 5)), "one oh five in the morning");
        assert_eq!(spoken_time(t(13, 18)), "one eighteen in the afternoon");
        assert_eq!(spoken_time(t(0, 0)), "twelve o'clock in the morning");
        assert_eq!(spoken_time(t(12, 30)), "twelve thirty in the afternoon");
        assert_eq!(spoken_time(t(20, 45)), "eight forty five in the evening");
        assert_eq!(spoken_time(t(18, 0)), "six o'clock in the evening");
    }

    #[test]
    fn dates_use_ordinals() {
        assert_eq!(
            spoken_date(NaiveDate::from_ymd_opt(2026, 8, 26).unwrap()),
            "Today is Wednesday, August 26th, 2026."
        );
        assert_eq!(ordinal(1), "1st");
        assert_eq!(ordinal(2), "2nd");
        assert_eq!(ordinal(3), "3rd");
        assert_eq!(ordinal(11), "11th");
        assert_eq!(ordinal(12), "12th");
        assert_eq!(ordinal(13), "13th");
        assert_eq!(ordinal(21), "21st");
        assert_eq!(ordinal(22), "22nd");
        assert_eq!(ordinal(23), "23rd");
        assert_eq!(ordinal(31), "31st");
    }
}
