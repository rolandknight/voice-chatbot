//! `play_bbc_radio` / `stop_bbc_radio` — live BBC streams on the client
//! (port of scripts/radio.py + skills/radio/*).

use async_trait::async_trait;
use serde_json::Value;

use super::alias::AliasTable;
use super::{arg_str, CallCtx, Skill};

// BBC HLS endpoints. Each station has a distinct Akamai pool, so we list URLs
// explicitly rather than templating off the slug. If a station starts 404'ing,
// BBC has reshuffled pools — refresh from https://gist.github.com/bpsib/67089b959e4fa898af69fea59ad74bc3
// or https://github.com/groupmsl/BBCRadioStreams (community-maintained).
// Sports Extra is uk-live and will 403 outside the UK without a VPN.
const BASE_WW: &str = "http://as-hls-ww-live.akamaized.net";
const BASE_UK: &str = "http://as-hls-uk-live.akamaized.net";

fn ww(pool: &str, slug: &str) -> String {
    format!("{BASE_WW}/{pool}/live/ww/{slug}/{slug}.isml/{slug}-audio%3d96000.norewind.m3u8")
}

fn uk(pool: &str, slug: &str) -> String {
    format!("{BASE_UK}/{pool}/live/uk/{slug}/{slug}.isml/{slug}-audio%3d96000.norewind.m3u8")
}

pub struct Station {
    pub key: &'static str,
    pub display: &'static str,
    pub url: String,
    pub aliases: &'static [&'static str],
}

pub fn stations() -> Vec<Station> {
    let s = |key, display, url, aliases| Station {
        key,
        display,
        url,
        aliases,
    };
    vec![
        s(
            "radio_5_sports_extra",
            "BBC Radio 5 Sports Extra",
            uk("pool_47700285", "bbc_radio_five_live_sports_extra"),
            &[
                "5 sports extra",
                "five sports extra",
                "radio 5 sports extra",
            ],
        ),
        s(
            "radio_4_extra",
            "BBC Radio 4 Extra",
            ww("pool_26173715", "bbc_radio_four_extra"),
            &["radio 4 extra", "radio four extra", "4 extra", "four extra"],
        ),
        s(
            "radio_1xtra",
            "BBC Radio 1Xtra",
            ww("pool_92079267", "bbc_1xtra"),
            &[
                "1xtra",
                "one xtra",
                "1 xtra",
                "radio 1 xtra",
                "radio one xtra",
            ],
        ),
        s(
            "radio_5_live",
            "BBC Radio 5 Live",
            ww("pool_89021708", "bbc_radio_five_live"),
            &["5 live", "five live", "radio 5 live", "radio five live"],
        ),
        s(
            "radio_6_music",
            "BBC Radio 6 Music",
            ww("pool_81827798", "bbc_6music"),
            &[
                "6 music",
                "six music",
                "radio 6 music",
                "radio six music",
                "radio 6",
                "radio six",
            ],
        ),
        s(
            "radio_1",
            "BBC Radio 1",
            ww("pool_01505109", "bbc_radio_one"),
            &["radio 1", "radio one", "radio won", "r1"],
        ),
        s(
            "radio_2",
            "BBC Radio 2",
            ww("pool_74208725", "bbc_radio_two"),
            &["radio 2", "radio two", "radio too", "radio to"],
        ),
        s(
            "radio_3",
            "BBC Radio 3",
            ww("pool_23461179", "bbc_radio_three"),
            &["radio 3", "radio three", "radio free"],
        ),
        s(
            "radio_4",
            "BBC Radio 4",
            ww("pool_55057080", "bbc_radio_fourfm"),
            &["radio 4", "radio four", "radio for"],
        ),
        s(
            "asian_network",
            "BBC Asian Network",
            ww("pool_22108647", "bbc_asian_network"),
            &["asian network"],
        ),
        s(
            "world_service",
            "BBC World Service",
            ww("pool_87948813", "bbc_world_service"),
            &["world service"],
        ),
    ]
}

pub struct StationTable {
    stations: Vec<Station>,
    aliases: AliasTable,
}

impl StationTable {
    pub fn new() -> Self {
        let stations = stations();
        let aliases = AliasTable::new(stations.iter().enumerate().map(|(i, s)| (i, s.aliases)));
        Self { stations, aliases }
    }

    /// Exact key (`radio_4`) first, then the longest spoken alias in `text`.
    pub fn resolve(&self, text: &str) -> Option<&Station> {
        self.stations
            .iter()
            .find(|s| s.key == text)
            .or_else(|| self.aliases.find(text).map(|i| &self.stations[i]))
    }
}

impl Default for StationTable {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PlayBbcRadio {
    table: StationTable,
}

impl PlayBbcRadio {
    pub fn new() -> Self {
        Self {
            table: StationTable::new(),
        }
    }
}

impl Default for PlayBbcRadio {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Skill for PlayBbcRadio {
    fn name(&self) -> &str {
        "play_bbc_radio"
    }

    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        let raw = arg_str(args, "station");
        if raw.is_empty() {
            return "Tell me which BBC station to play.".to_string();
        }
        let Some(station) = self.table.resolve(raw) else {
            return format!("I don't have a BBC station called {raw}.");
        };
        let Some(media) = ctx.media.as_ref() else {
            return format!("I couldn't start {}.", station.display);
        };
        ctx.stop_other_audio().await;
        media.play_stream(&station.url, station.display);
        format!("Playing {}.", station.display)
    }
}

/// Stops whatever BBC audio is playing (live or on-demand); also stops
/// Spotify, mirroring the Python cross-stop.
pub struct StopBbcRadio;

#[async_trait]
impl Skill for StopBbcRadio {
    fn name(&self) -> &str {
        "stop_bbc_radio"
    }

    async fn call(&self, _args: &Value, ctx: &CallCtx) -> String {
        let stopped_media = ctx.media.as_ref().and_then(|m| m.stop()).is_some();
        let stopped_spotify = ctx.stop_spotify().await;
        if stopped_media || stopped_spotify {
            "Stopped.".to_string()
        } else {
            "Nothing's playing.".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn station_urls_follow_the_python_builders() {
        let t = StationTable::new();
        assert_eq!(
            t.resolve("radio_4").unwrap().url,
            "http://as-hls-ww-live.akamaized.net/pool_55057080/live/ww/bbc_radio_fourfm/bbc_radio_fourfm.isml/bbc_radio_fourfm-audio%3d96000.norewind.m3u8"
        );
        assert_eq!(
            t.resolve("radio_5_sports_extra").unwrap().url,
            "http://as-hls-uk-live.akamaized.net/pool_47700285/live/uk/bbc_radio_five_live_sports_extra/bbc_radio_five_live_sports_extra.isml/bbc_radio_five_live_sports_extra-audio%3d96000.norewind.m3u8"
        );
    }

    #[test]
    fn alias_cases_from_the_python_module() {
        let t = StationTable::new();
        let d = |s: &str| t.resolve(s).map(|s| s.display);
        assert_eq!(d("BBC Radio 4"), Some("BBC Radio 4"));
        assert_eq!(d("Radio 4 Extra"), Some("BBC Radio 4 Extra"));
        assert_eq!(d("put on 4 extra"), Some("BBC Radio 4 Extra"));
        assert_eq!(d("five live"), Some("BBC Radio 5 Live"));
        assert_eq!(d("five sports extra"), Some("BBC Radio 5 Sports Extra"));
        assert_eq!(d("radio for"), Some("BBC Radio 4"));
        assert_eq!(d("6 Music"), Some("BBC Radio 6 Music"));
        assert_eq!(d("radio six"), Some("BBC Radio 6 Music"));
        assert_eq!(d("1Xtra"), Some("BBC Radio 1Xtra"));
        assert_eq!(d("the World Service, please"), Some("BBC World Service"));
        assert_eq!(d("radio 1"), Some("BBC Radio 1"));
        assert_eq!(d("radio 10"), None);
        assert_eq!(d("Classic FM"), None);
    }
}
