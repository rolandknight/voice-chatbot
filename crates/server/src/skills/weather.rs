//! `get_weather` — Open-Meteo, no API key (port of skills/core/get_weather).
//!
//! Location precedence: the tool argument → configured default → current
//! location (macOS `CoreLocationCLI`, then IP geolocation) → ask for a city.
//! The resolved current location is cached for the process.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use super::{arg_str, CallCtx, Skill};

const HTTP_TIMEOUT: Duration = Duration::from_secs(6);
const CORELOCATION_TIMEOUT: Duration = Duration::from_secs(5);
const IPGEO_TIMEOUT: Duration = Duration::from_secs(3);

/// WMO weather interpretation codes -> short spoken phrases.
fn describe_wmo(code: i64) -> &'static str {
    match code {
        0 => "clear",
        1 => "mostly clear",
        2 => "partly cloudy",
        3 => "overcast",
        45 => "foggy",
        48 => "foggy with rime",
        51 => "drizzling lightly",
        53 => "drizzling",
        55 => "drizzling heavily",
        61 => "raining lightly",
        63 => "raining",
        65 => "raining heavily",
        66 | 67 => "freezing rain",
        71 => "snowing lightly",
        73 => "snowing",
        75 => "snowing heavily",
        77 => "snow grains",
        80 => "light rain showers",
        81 => "rain showers",
        82 => "heavy rain showers",
        85 => "light snow showers",
        86 => "snow showers",
        95 => "thunderstorms",
        96 | 99 => "thunderstorms with hail",
        _ => "",
    }
}

#[derive(Clone, Debug)]
struct Resolved {
    lat: f64,
    lon: f64,
    place_name: String,
    country_code: Option<String>,
}

fn country_code(raw: Option<&str>) -> Option<String> {
    let cc = raw.unwrap_or("").trim().to_ascii_uppercase();
    (cc.len() == 2 && cc.chars().all(|c| c.is_ascii_alphabetic())).then_some(cc)
}

fn use_imperial(cc: Option<&str>) -> bool {
    matches!(cc, None | Some("US"))
}

/// "Partly cloudy, 18 degrees, wind around 20 miles per hour in Paris right now."
fn format_reply(current: &Value, place_name: &str, cc: Option<&str>) -> String {
    let imperial = use_imperial(cc);
    let mut parts: Vec<String> = Vec::new();
    if let Some(code) = current.get("weather_code").and_then(Value::as_i64) {
        let d = describe_wmo(code);
        if !d.is_empty() {
            let mut c = d.chars();
            let cap = c
                .next()
                .map(|f| f.to_uppercase().collect::<String>() + c.as_str());
            parts.push(cap.unwrap_or_default());
        }
    }
    if let Some(t) = current.get("temperature_2m").and_then(Value::as_f64) {
        parts.push(format!("{} degrees", t.round() as i64));
    }
    if let Some(w) = current.get("wind_speed_10m").and_then(Value::as_f64) {
        let (unit, threshold) = if imperial {
            ("miles per hour", 15.0)
        } else {
            ("kilometers per hour", 24.0)
        };
        if w >= threshold {
            parts.push(format!("wind around {} {unit}", w.round() as i64));
        }
    }
    let body = if parts.is_empty() {
        "conditions unavailable".to_string()
    } else {
        parts.join(", ")
    };
    format!("{body} in {place_name} right now.")
}

pub struct GetWeather {
    http: reqwest::Client,
    default_location: String,
    cache: Mutex<Option<Resolved>>,
}

impl GetWeather {
    pub fn new(default_location: String) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(HTTP_TIMEOUT)
                .build()
                .expect("reqwest client"),
            default_location,
            cache: Mutex::new(None),
        }
    }

    async fn resolve_corelocation() -> Option<Resolved> {
        if !cfg!(target_os = "macos") {
            return None;
        }
        let child = tokio::process::Command::new("CoreLocationCLI")
            .args(["-once", "-format", "%latitude\t%longitude\t%ISOcountryCode"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .ok()?;
        let out = match tokio::time::timeout(CORELOCATION_TIMEOUT, child.wait_with_output()).await {
            Ok(Ok(o)) if o.status.success() => o,
            Ok(_) => return None,
            Err(_) => {
                tracing::debug!("CoreLocationCLI timed out");
                return None;
            }
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let mut parts = text.trim().split('\t');
        let lat: f64 = parts.next()?.parse().ok()?;
        let lon: f64 = parts.next()?.parse().ok()?;
        Some(Resolved {
            lat,
            lon,
            place_name: "your location".to_string(),
            country_code: country_code(parts.next()),
        })
    }

    async fn resolve_ip_geolocation(&self) -> Option<Resolved> {
        let data: Value = self
            .http
            .get("https://ipwho.is/")
            .timeout(IPGEO_TIMEOUT)
            .send()
            .await
            .ok()?
            .error_for_status()
            .ok()?
            .json()
            .await
            .ok()?;
        if !data
            .get("success")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            tracing::debug!(message = ?data.get("message"), "IP geolocation returned error");
            return None;
        }
        let place_name = ["city", "region"]
            .iter()
            .find_map(|k| {
                data.get(*k)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or("your location")
            .to_string();
        Some(Resolved {
            lat: data.get("latitude")?.as_f64()?,
            lon: data.get("longitude")?.as_f64()?,
            place_name,
            country_code: country_code(data.get("country_code").and_then(Value::as_str)),
        })
    }

    async fn resolve_current(&self) -> Option<Resolved> {
        let mut cache = self.cache.lock().await;
        if let Some(r) = cache.as_ref() {
            return Some(r.clone());
        }
        let resolved = match Self::resolve_corelocation().await {
            Some(r) => Some(r),
            None => self.resolve_ip_geolocation().await,
        };
        *cache = resolved.clone();
        resolved
    }

    async fn geocode(&self, location: &str) -> Result<Option<Resolved>, reqwest::Error> {
        let data: Value = self
            .http
            .get("https://geocoding-api.open-meteo.com/v1/search")
            .query(&[
                ("name", location),
                ("count", "1"),
                ("language", "en"),
                ("format", "json"),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let Some(place) = data
            .get("results")
            .and_then(Value::as_array)
            .and_then(|r| r.first())
        else {
            return Ok(None);
        };
        Ok(Some(Resolved {
            lat: place.get("latitude").and_then(Value::as_f64).unwrap_or(0.0),
            lon: place
                .get("longitude")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            place_name: place
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(location)
                .to_string(),
            country_code: country_code(place.get("country_code").and_then(Value::as_str)),
        }))
    }

    async fn current_weather(&self, r: &Resolved) -> Result<Value, reqwest::Error> {
        let imperial = use_imperial(r.country_code.as_deref());
        let data: Value = self
            .http
            .get("https://api.open-meteo.com/v1/forecast")
            .query(&[
                ("latitude", r.lat.to_string()),
                ("longitude", r.lon.to_string()),
                (
                    "current",
                    "temperature_2m,weather_code,wind_speed_10m".to_string(),
                ),
                (
                    "temperature_unit",
                    if imperial { "fahrenheit" } else { "celsius" }.to_string(),
                ),
                (
                    "wind_speed_unit",
                    if imperial { "mph" } else { "kmh" }.to_string(),
                ),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        Ok(data.get("current").cloned().unwrap_or(Value::Null))
    }
}

#[async_trait]
impl Skill for GetWeather {
    fn name(&self) -> &str {
        "get_weather"
    }

    async fn call(&self, args: &Value, _ctx: &CallCtx) -> String {
        let mut location = arg_str(args, "location").to_string();
        if location.is_empty() {
            location = self.default_location.trim().to_string();
        }
        let resolved = if location.is_empty() {
            match self.resolve_current().await {
                Some(r) => r,
                None => {
                    return "I couldn't figure out where you are. Try asking again with a city name."
                        .to_string()
                }
            }
        } else {
            match self.geocode(&location).await {
                Ok(Some(r)) => r,
                Ok(None) => return format!("I couldn't find a place called {location}."),
                Err(e) => {
                    tracing::warn!(error = %e, "weather geocode failed");
                    return "I couldn't reach the weather service right now.".to_string();
                }
            }
        };
        match self.current_weather(&resolved).await {
            Ok(current) => format_reply(
                &current,
                &resolved.place_name,
                resolved.country_code.as_deref(),
            ),
            Err(e) => {
                tracing::warn!(error = %e, "weather lookup failed");
                "I couldn't reach the weather service right now.".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reply_phrasing_matches_python() {
        let cur = json!({"temperature_2m": 17.6, "weather_code": 2, "wind_speed_10m": 30.2});
        assert_eq!(
            format_reply(&cur, "Paris", Some("FR")),
            "Partly cloudy, 18 degrees, wind around 30 kilometers per hour in Paris right now."
        );
        // Below the wind threshold the wind clause is dropped; US is imperial.
        let cur = json!({"temperature_2m": 64.4, "weather_code": 0, "wind_speed_10m": 5.0});
        assert_eq!(
            format_reply(&cur, "Brooklyn", Some("US")),
            "Clear, 64 degrees in Brooklyn right now."
        );
        assert_eq!(
            format_reply(&json!({}), "here", None),
            "conditions unavailable in here right now."
        );
    }

    #[test]
    fn country_codes_are_validated() {
        assert_eq!(country_code(Some(" gb ")).as_deref(), Some("GB"));
        assert_eq!(country_code(Some("USA")), None);
        assert_eq!(country_code(Some("")), None);
        assert_eq!(country_code(None), None);
        assert!(use_imperial(None));
        assert!(use_imperial(Some("US")));
        assert!(!use_imperial(Some("GB")));
    }
}

#[cfg(test)]
mod network_tests {
    //! Real Open-Meteo calls: `cargo test -p voice-chatbot-server -- --ignored network`.
    use super::*;
    use serde_json::json;

    #[tokio::test]
    #[ignore]
    async fn network_weather_for_a_named_city() {
        let w = GetWeather::new(String::new());
        let out = w
            .call(
                &json!({"location": "Paris, France"}),
                &CallCtx { run_id: 0 },
            )
            .await;
        assert!(out.ends_with("in Paris right now."), "{out}");
        assert!(out.contains("degrees"), "{out}");
        let out = w
            .call(&json!({"location": "Xyzzyqwv"}), &CallCtx { run_id: 0 })
            .await;
        assert_eq!(out, "I couldn't find a place called Xyzzyqwv.");
    }
}
