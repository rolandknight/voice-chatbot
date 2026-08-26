//! `play_bbc_show` — on-demand BBC Sounds programmes and podcast episodes
//! (port of scripts/bbc_shows.py + skills/shows/play_bbc_show).
//!
//! Two-layer strategy:
//!   1. Curated RSS list — fast lookup of favourites; most BBC talk/drama
//!      shows publish a feed at https://podcasts.files.bbci.co.uk/<pid>.rss.
//!   2. Fallback — BBC Sounds search (public-but-undocumented endpoint) for
//!      an episode pid, then `yt-dlp` (a system binary, run as a subprocess)
//!      turns the play page into a stream URL. Without `yt-dlp` installed the
//!      fallback is simply unavailable.

use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, FixedOffset, NaiveDate};
use serde_json::Value;

use super::alias::AliasTable;
use super::{arg_str, CallCtx, Skill};

const HTTP_TIMEOUT: Duration = Duration::from_secs(8);
const YTDLP_TIMEOUT: Duration = Duration::from_secs(15);

pub struct BbcShow {
    pub key: &'static str,
    pub display: &'static str,
    /// Preferred when present: one GET, with episode metadata for date/query
    /// filtering. If 404'ing, refresh the PID from https://www.bbc.co.uk/programmes/<pid>.
    pub rss_url: &'static str,
    pub aliases: &'static [&'static str],
}

// Curated favourites. Each PID was verified against the feed returning 200
// with the matching channel title. PIDs occasionally rotate at the BBC's end;
// if a feed starts 404'ing, search BBC Sounds for the show and pull the brand
// PID from the URL. Shows without a public RSS feed (Today programme, More or
// Less, …) are not listed — they fall through to the search + yt-dlp path.
const CURATED: &[BbcShow] = &[
    BbcShow {
        key: "archers_omnibus",
        display: "The Archers Omnibus",
        rss_url: "https://podcasts.files.bbci.co.uk/b006qnkc.rss",
        aliases: &["the archers omnibus", "archers omnibus"],
    },
    BbcShow {
        key: "archers",
        display: "The Archers",
        rss_url: "https://podcasts.files.bbci.co.uk/b006qpgr.rss",
        aliases: &["the archers", "archers"],
    },
    BbcShow {
        key: "in_our_time",
        display: "In Our Time",
        rss_url: "https://podcasts.files.bbci.co.uk/b006qykl.rss",
        aliases: &["in our time"],
    },
    BbcShow {
        key: "desert_island_discs",
        display: "Desert Island Discs",
        rss_url: "https://podcasts.files.bbci.co.uk/b006qnmr.rss",
        aliases: &["desert island discs"],
    },
    BbcShow {
        key: "front_row",
        display: "Front Row",
        rss_url: "https://podcasts.files.bbci.co.uk/b006qsq5.rss",
        aliases: &["front row"],
    },
    BbcShow {
        key: "thinking_allowed",
        display: "Thinking Allowed",
        rss_url: "https://podcasts.files.bbci.co.uk/b006qy05.rss",
        aliases: &["thinking allowed"],
    },
    BbcShow {
        key: "just_a_minute",
        display: "Just A Minute",
        rss_url: "https://podcasts.files.bbci.co.uk/b006s5dp.rss",
        aliases: &["just a minute"],
    },
    BbcShow {
        key: "friday_night_comedy",
        display: "Friday Night Comedy",
        rss_url: "https://podcasts.files.bbci.co.uk/p02pc9pj.rss",
        aliases: &["friday night comedy"],
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedEpisode {
    pub url: String,
    pub display: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RssItem {
    pub title: String,
    pub description: String,
    pub enclosure_url: Option<String>,
    pub pub_date: Option<DateTime<FixedOffset>>,
}

/// `<item>`s of a podcast feed, in document order (newest first at the BBC).
pub fn parse_rss(xml: &str) -> Vec<RssItem> {
    let doc = match roxmltree::Document::parse(xml) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(error = %e, "RSS parse failed");
            return Vec::new();
        }
    };
    let child_text = |item: roxmltree::Node, name: &str| -> String {
        item.children()
            .find(|c| c.is_element() && c.tag_name().name() == name)
            .and_then(|c| c.text())
            .unwrap_or("")
            .trim()
            .to_string()
    };
    doc.descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "item")
        .map(|item| RssItem {
            title: child_text(item, "title"),
            description: child_text(item, "description"),
            enclosure_url: item
                .children()
                .find(|c| c.is_element() && c.tag_name().name() == "enclosure")
                .and_then(|e| e.attribute("url"))
                .map(str::to_string),
            pub_date: DateTime::parse_from_rfc2822(child_text(item, "pubDate").as_str()).ok(),
        })
        .collect()
}

/// "Sunday 24 August".
pub fn pretty_pub_date(dt: Option<&DateTime<FixedOffset>>) -> String {
    dt.map(|d| d.format("%A %-d %B").to_string())
        .unwrap_or_default()
}

/// The episode for an ISO date, else the first whose title/description
/// contains `query`, else the newest. A date or query with no match is `None`.
pub fn pick_item<'a>(items: &'a [RssItem], date: &str, query: &str) -> Option<&'a RssItem> {
    if items.is_empty() {
        return None;
    }
    if !date.is_empty() {
        let target = NaiveDate::parse_from_str(date.trim(), "%Y-%m-%d").ok()?;
        return items
            .iter()
            .find(|i| i.pub_date.map(|d| d.date_naive()) == Some(target));
    }
    if !query.is_empty() {
        let needle = query.to_lowercase();
        return items.iter().find(|i| {
            format!("{} {}", i.title, i.description)
                .to_lowercase()
                .contains(&needle)
        });
    }
    items.first()
}

/// First `urn:bbc:radio:episode:<pid>` anywhere in the search response.
pub fn first_episode_pid(node: &Value) -> Option<String> {
    match node {
        Value::Object(map) => {
            if let Some(pid) = map
                .get("urn")
                .and_then(Value::as_str)
                .and_then(|urn| urn.strip_prefix("urn:bbc:radio:episode:"))
                .filter(|pid| {
                    pid.len() == 8
                        && pid.starts_with(|c: char| c.is_ascii_lowercase())
                        && pid
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                })
            {
                return Some(pid.to_string());
            }
            map.values().find_map(first_episode_pid)
        }
        Value::Array(items) => items.iter().find_map(first_episode_pid),
        _ => None,
    }
}

pub struct PlayBbcShow {
    http: reqwest::Client,
    aliases: AliasTable,
}

impl PlayBbcShow {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(HTTP_TIMEOUT)
                .user_agent("babel-voice-bot/1.0")
                .build()
                .expect("reqwest client"),
            aliases: AliasTable::new(CURATED.iter().enumerate().map(|(i, s)| (i, s.aliases))),
        }
    }

    fn curated(&self, show: &str) -> Option<&'static BbcShow> {
        CURATED
            .iter()
            .find(|s| s.key == show)
            .or_else(|| self.aliases.find(show).map(|i| &CURATED[i]))
    }

    async fn fetch_rss(&self, url: &str) -> Result<Vec<RssItem>, reqwest::Error> {
        let text = self
            .http
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        Ok(parse_rss(&text))
    }

    /// Play URL of the top episode hit on BBC Sounds search.
    async fn sounds_search(&self, query: &str) -> Option<String> {
        let data: Value = match self
            .http
            .get("https://rms.api.bbc.co.uk/v2/experience/inline/search")
            .query(&[("q", query), ("stations", "all")])
            .send()
            .await
            .and_then(|r| r.error_for_status())
        {
            Ok(r) => r.json().await.ok()?,
            Err(e) => {
                tracing::warn!(error = %e, "BBC Sounds search failed");
                return None;
            }
        };
        first_episode_pid(&data).map(|pid| format!("https://www.bbc.co.uk/sounds/play/{pid}"))
    }

    /// `yt-dlp -j` on a BBC Sounds page → (stream url, title). Brand/series
    /// pages yield a playlist stub; re-run on its first entry's page.
    async fn ytdlp_resolve(url: &str) -> Result<ResolvedEpisode, String> {
        async fn dump(url: &str) -> Result<Value, String> {
            let out = tokio::time::timeout(
                YTDLP_TIMEOUT,
                tokio::process::Command::new("yt-dlp")
                    .args([
                        "-j",
                        "--no-warnings",
                        "--no-playlist",
                        "--playlist-items",
                        "1",
                        "-f",
                        "bestaudio/best",
                        url,
                    ])
                    .stdin(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .map_err(|_| "yt-dlp timed out".to_string())?
            .map_err(|e| format!("yt-dlp not runnable: {e}"))?;
            if !out.status.success() {
                return Err(format!("yt-dlp exited with {}", out.status));
            }
            let first_line = String::from_utf8_lossy(&out.stdout)
                .lines()
                .find(|l| l.trim_start().starts_with('{'))
                .map(str::to_string)
                .ok_or("yt-dlp returned no entries")?;
            serde_json::from_str(&first_line).map_err(|e| format!("yt-dlp json: {e}"))
        }
        let mut info = dump(url).await?;
        if info.get("url").and_then(Value::as_str).is_none() {
            let leaf = info
                .get("webpage_url")
                .or_else(|| info.get("url"))
                .and_then(Value::as_str)
                .filter(|u| *u != url)
                .ok_or("yt-dlp returned no playable URL")?
                .to_string();
            info = dump(&leaf).await?;
        }
        let stream_url = info
            .get("url")
            .and_then(Value::as_str)
            .ok_or("yt-dlp returned no playable URL")?
            .to_string();
        let display = info
            .get("title")
            .or_else(|| info.get("webpage_url"))
            .and_then(Value::as_str)
            .unwrap_or(url)
            .to_string();
        Ok(ResolvedEpisode {
            url: stream_url,
            display,
        })
    }

    pub async fn resolve(&self, show: &str, date: &str, query: &str) -> Option<ResolvedEpisode> {
        let curated = self.curated(show);
        if let Some(c) = curated {
            let items = match self.fetch_rss(c.rss_url).await {
                Ok(items) => items,
                Err(e) => {
                    tracing::warn!(show = c.display, error = %e, "RSS fetch failed");
                    Vec::new()
                }
            };
            if let Some(url) = pick_item(&items, date, query).and_then(|i| {
                i.enclosure_url
                    .as_ref()
                    .map(|u| (u.clone(), pretty_pub_date(i.pub_date.as_ref())))
            }) {
                let (url, suffix) = url;
                let display = if suffix.is_empty() {
                    c.display.to_string()
                } else {
                    format!("{}, {suffix}", c.display)
                };
                return Some(ResolvedEpisode { url, display });
            }
            tracing::info!(
                show = c.display,
                "RSS empty/no-match; falling back to search"
            );
        }
        // Fallback: BBC Sounds search + yt-dlp. The curated display name is a
        // better search term than the raw user text when we matched one.
        let mut terms = vec![curated.map(|c| c.display).unwrap_or(show).to_string()];
        if !query.is_empty() {
            terms.push(query.to_string());
        }
        let play_url = self.sounds_search(&terms.join(" ")).await?;
        match Self::ytdlp_resolve(&play_url).await {
            Ok(ep) => Some(ep),
            Err(e) => {
                tracing::warn!(%play_url, error = %e, "yt-dlp resolve failed");
                None
            }
        }
    }
}

impl Default for PlayBbcShow {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Skill for PlayBbcShow {
    fn name(&self) -> &str {
        "play_bbc_show"
    }

    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        let show = arg_str(args, "show");
        let date = arg_str(args, "date");
        let query = arg_str(args, "query");
        if show.is_empty() {
            return "Tell me which BBC show to play.".to_string();
        }
        let Some(episode) = self.resolve(show, date, query).await else {
            return if !date.is_empty() {
                format!("I couldn't find an episode of {show} for that date.")
            } else if !query.is_empty() {
                format!("I couldn't find a {show} episode about {query}.")
            } else {
                format!("I couldn't find {show} on BBC Sounds.")
            };
        };
        let Some(media) = ctx.media.as_ref() else {
            return format!("I couldn't play {}.", episode.display);
        };
        ctx.stop_other_audio().await;
        media.play_stream(&episode.url, &episode.display);
        format!("Playing {}.", episode.display)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FEED: &str = r#"<?xml version="1.0"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" version="2.0"><channel>
<title>In Our Time</title>
<item><title>Spinoza</title><description>The philosopher.</description>
  <enclosure url="http://x/spinoza.mp3" type="audio/mpeg"/><pubDate>Thu, 21 Aug 2025 09:00:00 +0000</pubDate></item>
<item><title>Climate</title><description>Weather over time.</description>
  <enclosure url="http://x/climate.mp3" type="audio/mpeg"/><pubDate>Thu, 14 Aug 2025 09:00:00 +0000</pubDate></item>
</channel></rss>"#;

    #[test]
    fn parses_items_and_picks_by_date_query_or_latest() {
        let items = parse_rss(FEED);
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].enclosure_url.as_deref(),
            Some("http://x/spinoza.mp3")
        );
        assert_eq!(
            pretty_pub_date(items[0].pub_date.as_ref()),
            "Thursday 21 August"
        );
        assert_eq!(pick_item(&items, "", "").unwrap().title, "Spinoza");
        assert_eq!(
            pick_item(&items, "2025-08-14", "").unwrap().title,
            "Climate"
        );
        assert!(pick_item(&items, "2025-08-15", "").is_none());
        assert!(pick_item(&items, "not-a-date", "").is_none());
        assert_eq!(pick_item(&items, "", "weather").unwrap().title, "Climate");
        assert!(pick_item(&items, "", "spaceships").is_none());
        assert!(parse_rss("<not xml").is_empty());
    }

    #[test]
    fn finds_first_episode_pid_in_search_tree() {
        let data = json!({"results": [
            {"urn": "urn:bbc:radio:brand:b006qykl"},
            {"data": [{"urn": "urn:bbc:radio:episode:m002abcd", "title": "x"}]}
        ]});
        assert_eq!(first_episode_pid(&data).as_deref(), Some("m002abcd"));
        assert_eq!(
            first_episode_pid(&json!({"urn": "urn:bbc:radio:episode:TOOLONG12"})),
            None
        );
    }

    #[test]
    fn curated_lookup_by_key_or_alias() {
        let p = PlayBbcShow::new();
        assert_eq!(p.curated("in_our_time").unwrap().display, "In Our Time");
        assert_eq!(
            p.curated("play the Archers omnibus").unwrap().display,
            "The Archers Omnibus"
        );
        assert_eq!(p.curated("the archers").unwrap().display, "The Archers");
        assert!(p.curated("Today programme").is_none());
    }
}

#[cfg(test)]
mod network_tests {
    //! Real BBC calls: `cargo test -p voice-chatbot-server -- --ignored network`.
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn network_curated_rss_resolves_latest_episode() {
        let p = PlayBbcShow::new();
        let ep = p.resolve("in our time", "", "").await.expect("episode");
        assert!(ep.url.starts_with("http"), "{ep:?}");
        assert!(ep.display.starts_with("In Our Time, "), "{ep:?}");
    }

    #[tokio::test]
    #[ignore]
    async fn network_search_and_ytdlp_fallback() {
        // Not curated → BBC Sounds search → yt-dlp. Needs yt-dlp on PATH.
        let p = PlayBbcShow::new();
        let ep = p
            .resolve("More or Less", "", "")
            .await
            .expect("episode via yt-dlp");
        assert!(ep.url.starts_with("http"), "{ep:?}");
        assert!(!ep.display.is_empty(), "{ep:?}");
    }
}
