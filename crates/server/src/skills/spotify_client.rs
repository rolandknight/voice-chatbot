//! Spotify Web API control for the librespot "Babel" Connect endpoint (port
//! of scripts/spotify.py). Control-only: audio plays on the client's
//! librespot; we only issue commands targeting its device id.
//!
//! Auth is the PKCE token spotipy cached at `~/.config/babel/spotify_token.json`
//! (same file, same fields, written back the same way), refreshed here with
//! the client id alone — no secret. First-time auth: `voice-chatbot-server
//! spotify-login` (Rust) or the legacy `python scripts/spotify.py --bootstrap`.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const SCOPE: &str = "user-modify-playback-state user-read-playback-state \
user-read-currently-playing playlist-read-private playlist-read-collaborative";
pub const DEVICE_NAME: &str = "Babel";
const API: &str = "https://api.spotify.com/v1";
const TOKEN_URL: &str = "https://accounts.spotify.com/api/token";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(4);
const DEVICE_POLL: Duration = Duration::from_secs(5);

pub fn config_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".config").join("babel")
}

/// spotipy's `CacheFileHandler` layout.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Token {
    pub access_token: String,
    #[serde(default = "bearer")]
    pub token_type: String,
    #[serde(default)]
    pub expires_in: i64,
    pub refresh_token: String,
    #[serde(default)]
    pub scope: String,
    /// Unix seconds.
    pub expires_at: i64,
}

fn bearer() -> String {
    "Bearer".to_string()
}

fn unix_now() -> i64 {
    chrono::Utc::now().timestamp()
}

impl Token {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// spotipy refreshes when fewer than 60 s remain.
    pub fn is_expiring(&self, now: i64) -> bool {
        self.expires_at - now < 60
    }

    /// Fold a token endpoint response (`access_token`, `expires_in`, optional
    /// rotated `refresh_token`) into this token.
    pub fn apply_response(&mut self, resp: &Value, now: i64) -> Result<(), String> {
        self.access_token = resp
            .get("access_token")
            .and_then(Value::as_str)
            .ok_or("token response without access_token")?
            .to_string();
        self.expires_in = resp
            .get("expires_in")
            .and_then(Value::as_i64)
            .unwrap_or(3600);
        self.expires_at = now + self.expires_in;
        if let Some(t) = resp.get("token_type").and_then(Value::as_str) {
            self.token_type = t.to_string();
        }
        if let Some(s) = resp.get("scope").and_then(Value::as_str) {
            self.scope = s.to_string();
        }
        if let Some(r) = resp.get("refresh_token").and_then(Value::as_str) {
            self.refresh_token = r.to_string();
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ApiError {
    /// Spotify's signal that the device id we sent no longer exists.
    DeviceNotFound,
    Other(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::DeviceNotFound => write!(f, "device not found (404)"),
            ApiError::Other(s) => write!(f, "{s}"),
        }
    }
}

#[derive(Default)]
struct PlayState {
    /// Playback started this session (no local process to probe: librespot
    /// lives on the client), so cross-stops know whether to pause.
    playing: bool,
    user_paused: bool,
}

pub struct SpotifyClient {
    http: reqwest::Client,
    client_id: String,
    token_path: PathBuf,
    device_path: PathBuf,
    device_name: String,
    token: Mutex<Option<Token>>,
    device_id: Mutex<Option<String>>,
    state: Mutex<PlayState>,
    now_playing_cache: Mutex<Option<(Instant, Option<Value>)>>,
}

/// Where the last `Babel` device id was cached.
fn read_device_cache(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

impl SpotifyClient {
    /// Fails when there is no cached token (nothing to refresh from): the
    /// skills are then gated off with a startup log line.
    pub fn new(client_id: String) -> Result<Self, String> {
        let dir = config_dir();
        Self::with_paths(
            client_id,
            dir.join("spotify_token.json"),
            dir.join("spotify_device.txt"),
        )
    }

    pub fn with_paths(
        client_id: String,
        token_path: PathBuf,
        device_path: PathBuf,
    ) -> Result<Self, String> {
        if client_id.trim().is_empty() {
            return Err("SPOTIPY_CLIENT_ID is not set".to_string());
        }
        let token = Token::load(&token_path).map_err(|e| {
            format!("Spotify isn't authorised yet ({e}); run `voice-chatbot-server spotify-login`")
        })?;
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .expect("reqwest client"),
            client_id,
            token_path,
            device_id: Mutex::new(read_device_cache(&device_path)),
            device_path,
            device_name: DEVICE_NAME.to_string(),
            token: Mutex::new(Some(token)),
            state: Mutex::new(PlayState::default()),
            now_playing_cache: Mutex::new(None),
        })
    }

    // ---------- auth ----------

    async fn access_token(&self) -> Result<String, ApiError> {
        let (needs_refresh, refresh_token) = {
            let t = self.token.lock().unwrap();
            let t = t
                .as_ref()
                .ok_or_else(|| ApiError::Other("no token".into()))?;
            (t.is_expiring(unix_now()), t.refresh_token.clone())
        };
        if !needs_refresh {
            return Ok(self
                .token
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .access_token
                .clone());
        }
        let resp: Value = self
            .http
            .post(TOKEN_URL)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.as_str()),
                ("client_id", self.client_id.as_str()),
            ])
            .send()
            .await
            .map_err(|e| ApiError::Other(format!("token refresh: {e}")))?
            .error_for_status()
            .map_err(|e| ApiError::Other(format!("token refresh: {e}")))?
            .json()
            .await
            .map_err(|e| ApiError::Other(format!("token refresh body: {e}")))?;
        let mut guard = self.token.lock().unwrap();
        let token = guard.as_mut().unwrap();
        token
            .apply_response(&resp, unix_now())
            .map_err(ApiError::Other)?;
        if let Err(e) = token.save(&self.token_path) {
            tracing::warn!(error = %e, "spotify: could not persist refreshed token");
        }
        tracing::info!("spotify: access token refreshed");
        Ok(token.access_token.clone())
    }

    /// One Web API request. `Ok(Value::Null)` for empty (204) responses.
    async fn api(
        &self,
        method: reqwest::Method,
        path: &str,
        query: &[(&str, &str)],
        body: Option<Value>,
    ) -> Result<Value, ApiError> {
        let token = self.access_token().await?;
        let mut req = self
            .http
            .request(method, format!("{API}{path}"))
            .bearer_auth(token)
            .query(query);
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ApiError::Other(e.to_string()))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(ApiError::DeviceNotFound);
        }
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ApiError::Other(format!("{path}: HTTP {status} {text}")));
        }
        let text = resp.text().await.unwrap_or_default();
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| ApiError::Other(format!("{path}: bad json: {e}")))
    }

    // ---------- device ----------

    fn save_device_id(&self, id: &str) {
        *self.device_id.lock().unwrap() = Some(id.to_string());
        if let Some(dir) = self.device_path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = std::fs::write(&self.device_path, id) {
            tracing::debug!(error = %e, "spotify: could not write device cache");
        }
    }

    /// The "Babel" device's Connect id, polling for up to ~5 s (librespot can
    /// take a couple of seconds to register). `None` when it never shows up.
    pub async fn resolve_device(&self, force_refresh: bool) -> Option<String> {
        if !force_refresh {
            if let Some(id) = self.device_id.lock().unwrap().clone() {
                return Some(id);
            }
        }
        let deadline = Instant::now() + DEVICE_POLL;
        loop {
            let devices = match self
                .api(reqwest::Method::GET, "/me/player/devices", &[], None)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(error = %e, "spotify: devices() failed");
                    return None;
                }
            };
            if let Some(id) = devices
                .get("devices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .find(|d| d.get("name").and_then(Value::as_str) == Some(self.device_name.as_str()))
                .and_then(|d| d.get("id").and_then(Value::as_str))
            {
                self.save_device_id(id);
                return Some(id.to_string());
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        }
    }

    fn cached_device_id(&self) -> Option<String> {
        self.device_id.lock().unwrap().clone()
    }

    // ---------- playback state ----------

    pub fn is_playing(&self) -> bool {
        let s = self.state.lock().unwrap();
        s.playing && !s.user_paused
    }

    fn mark_started(&self) {
        let mut s = self.state.lock().unwrap();
        s.playing = true;
        s.user_paused = false;
        *self.now_playing_cache.lock().unwrap() = None;
    }

    async fn current_playback(&self) -> Option<Value> {
        if let Some((at, cached)) = self.now_playing_cache.lock().unwrap().as_ref() {
            if at.elapsed() < Duration::from_secs(2) {
                return cached.clone();
            }
        }
        let info = match self
            .api(reqwest::Method::GET, "/me/player", &[], None)
            .await
        {
            Ok(Value::Null) => None,
            Ok(v) => Some(v),
            Err(e) => {
                tracing::debug!(error = %e, "spotify: current_playback failed");
                return None;
            }
        };
        *self.now_playing_cache.lock().unwrap() = Some((Instant::now(), info.clone()));
        info
    }

    /// "Track by Artist", or `None` when nothing is playing.
    pub async fn now_playing(&self) -> Option<String> {
        let info = self.current_playback().await?;
        let item = info.get("item")?;
        if item.is_null() {
            return None;
        }
        let track = item.get("name").and_then(Value::as_str).unwrap_or("");
        let artists = artists_str(item);
        if !track.is_empty() && artists != "unknown" {
            Some(format!("{track} by {artists}"))
        } else if !track.is_empty() {
            Some(track.to_string())
        } else {
            None
        }
    }

    async fn transfer_if_needed(&self, device_id: &str) {
        let Some(current) = self.current_playback().await else {
            return;
        };
        if current.pointer("/device/id").and_then(Value::as_str) != Some(device_id) {
            if let Err(e) = self
                .api(
                    reqwest::Method::PUT,
                    "/me/player",
                    &[],
                    Some(json!({"device_ids": [device_id], "play": false})),
                )
                .await
            {
                tracing::debug!(error = %e, "spotify: transfer_playback failed");
            }
        }
    }

    async fn start_playback(&self, device_id: &str, body: Option<Value>) -> Result<(), ApiError> {
        self.api(
            reqwest::Method::PUT,
            "/me/player/play",
            &[("device_id", device_id)],
            body,
        )
        .await
        .map(|_| ())
    }

    const NO_DEVICE: &'static str = "Spotify can't see the Babel device yet. Make sure librespot is running on the speaker, or pick Babel from the Connect menu.";
    const LOST_DEVICE: &'static str =
        "Spotify lost the Babel device. Open Spotify on your phone and pick Babel from Connect.";

    /// Search, pick, play. Returns (success, spoken reply).
    pub async fn search_and_play(&self, query: &str, kind: &str) -> (bool, String) {
        let query = query.trim();
        if query.is_empty() {
            return (false, "Tell me what to play.".to_string());
        }
        let Some(mut device_id) = self.resolve_device(false).await else {
            return (false, Self::NO_DEVICE.to_string());
        };
        let kind = match kind.trim().to_lowercase().as_str() {
            k @ ("track" | "album" | "artist") => k.to_string(),
            _ => "any".to_string(),
        };
        let search_type = if kind == "any" {
            "track,album,artist"
        } else {
            kind.as_str()
        };
        // No `market=from_token`: it needs the user-read-private scope. Global
        // results are always playable on a Premium account.
        let results = match self
            .api(
                reqwest::Method::GET,
                "/search",
                &[("q", query), ("type", search_type), ("limit", "5")],
                None,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "spotify: search failed");
                return (false, "I couldn't reach Spotify search.".to_string());
            }
        };
        let Some((pick_kind, item)) = pick_result(&results, &kind) else {
            return (false, format!("I couldn't find {query} on Spotify."));
        };
        self.transfer_if_needed(&device_id).await;

        // For artists, fetch the top tracks once; the play body is then fixed.
        let (body, spoken) = match pick_kind {
            "track" => (
                json!({"uris": [item["uri"]]}),
                format!(
                    "Playing {} by {}.",
                    item["name"].as_str().unwrap_or(""),
                    artists_str(&item)
                ),
            ),
            "album" => (
                json!({"context_uri": item["uri"]}),
                format!(
                    "Playing {} by {}.",
                    item["name"].as_str().unwrap_or(""),
                    artists_str(&item)
                ),
            ),
            _ => {
                let name = item["name"].as_str().unwrap_or("").to_string();
                let id = item["id"].as_str().unwrap_or("");
                let top = self
                    .api(
                        reqwest::Method::GET,
                        &format!("/artists/{id}/top-tracks"),
                        &[],
                        None,
                    )
                    .await
                    .ok()
                    .and_then(|v| v.get("tracks").and_then(Value::as_array).cloned())
                    .unwrap_or_default();
                if top.is_empty() {
                    return (false, format!("I couldn't find any tracks for {name}."));
                }
                let uris: Vec<Value> = top
                    .iter()
                    .take(10)
                    .filter_map(|t| t.get("uri").cloned())
                    .collect();
                (
                    json!({"uris": uris}),
                    format!("Playing top tracks from {name}."),
                )
            }
        };

        match self.start_playback(&device_id, Some(body.clone())).await {
            Ok(()) => {}
            Err(ApiError::DeviceNotFound) => {
                // Cached device id is stale. Re-resolve and retry once.
                tracing::info!("spotify: start_playback got 404; refreshing device id");
                match self.resolve_device(true).await {
                    Some(id) => device_id = id,
                    None => return (false, Self::LOST_DEVICE.to_string()),
                }
                if let Err(e) = self.start_playback(&device_id, Some(body)).await {
                    tracing::warn!(error = %e, "spotify: start_playback retry failed");
                    return (false, "Spotify wouldn't start playback.".to_string());
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "spotify: start_playback failed");
                return (false, "Spotify wouldn't start playback.".to_string());
            }
        }
        self.mark_started();
        (true, spoken)
    }

    pub async fn play_playlist(&self, name: &str) -> (bool, String) {
        let name = name.trim();
        if name.is_empty() {
            return (false, "Tell me which playlist to play.".to_string());
        }
        let Some(mut device_id) = self.resolve_device(false).await else {
            return (false, Self::NO_DEVICE.to_string());
        };
        let Some(target) = self.match_playlist(name).await else {
            return (false, format!("I couldn't find a playlist called {name}."));
        };
        let target_name = target
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(name)
            .to_string();
        let body = json!({"context_uri": target["uri"]});
        self.transfer_if_needed(&device_id).await;
        match self.start_playback(&device_id, Some(body.clone())).await {
            Ok(()) => {}
            Err(ApiError::DeviceNotFound) => {
                tracing::info!("spotify: start_playback (playlist) got 404; refreshing device id");
                match self.resolve_device(true).await {
                    Some(id) => device_id = id,
                    None => return (false, Self::LOST_DEVICE.to_string()),
                }
                if let Err(e) = self.start_playback(&device_id, Some(body)).await {
                    tracing::warn!(error = %e, "spotify: playlist retry failed");
                    return (false, format!("Spotify wouldn't start {target_name}."));
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "spotify: playlist start failed");
                return (false, format!("Spotify wouldn't start {target_name}."));
            }
        }
        self.mark_started();
        (true, format!("Playing {target_name}."))
    }

    /// The user's own playlists (all pages), then a public search as fallback.
    async fn match_playlist(&self, name: &str) -> Option<Value> {
        let mut owned: Vec<Value> = Vec::new();
        let mut offset = 0;
        loop {
            let page = match self
                .api(
                    reqwest::Method::GET,
                    "/me/playlists",
                    &[("limit", "50"), ("offset", &offset.to_string())],
                    None,
                )
                .await
            {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!(error = %e, "spotify: current_user_playlists failed");
                    break;
                }
            };
            let items = page
                .get("items")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let n = items.len();
            owned.extend(items);
            if n == 0 || page.get("next").map(Value::is_null).unwrap_or(true) {
                break;
            }
            offset += 50;
        }
        if let Some(pl) = match_owned_playlist(&owned, name) {
            return Some(pl);
        }
        match self
            .api(
                reqwest::Method::GET,
                "/search",
                &[("q", name), ("type", "playlist"), ("limit", "5")],
                None,
            )
            .await
        {
            Ok(r) => r
                .pointer("/playlists/items")
                .and_then(Value::as_array)
                .and_then(|items| items.iter().find(|i| !i.is_null()).cloned()),
            Err(e) => {
                tracing::debug!(error = %e, "spotify: playlist search fallback failed");
                None
            }
        }
    }

    async fn device_command(&self, method: reqwest::Method, path: &str) -> bool {
        let device = self.cached_device_id();
        let query: Vec<(&str, &str)> = device
            .as_deref()
            .map(|d| ("device_id", d))
            .into_iter()
            .collect();
        match self.api(method, path, &query, None).await {
            Ok(_) => {
                *self.now_playing_cache.lock().unwrap() = None;
                true
            }
            Err(e) => {
                tracing::debug!(error = %e, path, "spotify: command failed");
                false
            }
        }
    }

    pub async fn pause(&self) -> bool {
        let ok = self
            .device_command(reqwest::Method::PUT, "/me/player/pause")
            .await;
        if ok {
            self.state.lock().unwrap().user_paused = true;
        }
        ok
    }

    pub async fn resume(&self) -> bool {
        let ok = self
            .device_command(reqwest::Method::PUT, "/me/player/play")
            .await;
        if ok {
            self.mark_started();
        }
        ok
    }

    pub async fn skip_next(&self) -> bool {
        self.device_command(reqwest::Method::POST, "/me/player/next")
            .await
    }

    pub async fn skip_previous(&self) -> bool {
        self.device_command(reqwest::Method::POST, "/me/player/previous")
            .await
    }

    /// Stop: best-effort API pause so playback halts on the client, then
    /// forget the session state.
    pub async fn stop(&self) {
        if self.is_playing() && self.cached_device_id().is_some() {
            self.device_command(reqwest::Method::PUT, "/me/player/pause")
                .await;
        }
        let mut s = self.state.lock().unwrap();
        s.playing = false;
        s.user_paused = false;
    }
}

/// "A, B" from an item's `artists`, or "unknown".
pub fn artists_str(item: &Value) -> String {
    let names: Vec<&str> = item
        .get("artists")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| a.get("name").and_then(Value::as_str))
        .filter(|n| !n.is_empty())
        .collect();
    if names.is_empty() {
        "unknown".to_string()
    } else {
        names.join(", ")
    }
}

/// For kind=any pick the bucket with the highest popularity, tie-break
/// track > album > artist (users usually want one song).
pub fn pick_result(results: &Value, kind: &str) -> Option<(&'static str, Value)> {
    let first = |bucket: &str| -> Option<Value> {
        results
            .get(bucket)
            .and_then(|b| b.get("items"))
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .cloned()
    };
    let buckets: [(&'static str, &str, i32); 3] = [
        ("track", "tracks", 3),
        ("album", "albums", 2),
        ("artist", "artists", 1),
    ];
    if let Some((k, b, _)) = buckets.iter().find(|(k, _, _)| *k == kind) {
        return first(b).map(|item| (*k, item));
    }
    let mut best: Option<(&'static str, Value, i64, i32)> = None;
    for (k, b, prio) in buckets {
        let Some(item) = first(b) else { continue };
        let score = item.get("popularity").and_then(Value::as_i64).unwrap_or(0);
        let better = match &best {
            None => true,
            Some((_, _, s, p)) => score > *s || (score == *s && prio > *p),
        };
        if better {
            best = Some((k, item, score, prio));
        }
    }
    best.map(|(k, item, _, _)| (k, item))
}

/// Exact normalised name, then substring, then Jaccard ≥ 0.4 on tokens.
pub fn match_owned_playlist(owned: &[Value], name: &str) -> Option<Value> {
    let target = super::alias::normalise(name);
    let target_tokens: std::collections::HashSet<&str> = target.split_whitespace().collect();
    let pl_name =
        |pl: &Value| super::alias::normalise(pl.get("name").and_then(Value::as_str).unwrap_or(""));
    if let Some(pl) = owned.iter().find(|pl| pl_name(pl) == target) {
        return Some(pl.clone());
    }
    if !target.is_empty() {
        if let Some(pl) = owned.iter().find(|pl| pl_name(pl).contains(&target)) {
            return Some(pl.clone());
        }
    }
    let mut best: Option<(&Value, f64)> = None;
    for pl in owned {
        let n = pl_name(pl);
        let tokens: std::collections::HashSet<&str> = n.split_whitespace().collect();
        if tokens.is_empty() || target_tokens.is_empty() {
            continue;
        }
        let score = tokens.intersection(&target_tokens).count() as f64
            / tokens.union(&target_tokens).count() as f64;
        if best.map(|(_, s)| score > s).unwrap_or(true) {
            best = Some((pl, score));
        }
    }
    best.filter(|(_, s)| *s >= 0.4).map(|(pl, _)| pl.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_roundtrips_spotipy_format_and_refresh_rules() {
        let dir = std::env::temp_dir().join(format!("vc-spotify-test-{}", std::process::id()));
        let path = dir.join("spotify_token.json");
        let raw = r#"{"access_token":"a","token_type":"Bearer","expires_in":3600,"refresh_token":"r","scope":"s","expires_at":1000}"#;
        let mut t: Token = serde_json::from_str(raw).unwrap();
        assert!(t.is_expiring(950));
        assert!(!t.is_expiring(900));
        t.apply_response(&json!({"access_token": "b", "expires_in": 100}), 2000)
            .unwrap();
        assert_eq!(t.access_token, "b");
        assert_eq!(t.expires_at, 2100);
        assert_eq!(t.refresh_token, "r", "refresh token kept when not rotated");
        t.apply_response(&json!({"access_token": "c", "refresh_token": "r2"}), 3000)
            .unwrap();
        assert_eq!(t.refresh_token, "r2");
        t.save(&path).unwrap();
        assert_eq!(Token::load(&path).unwrap(), t);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn picks_by_popularity_with_track_tiebreak() {
        let results = json!({
            "tracks": {"items": [{"name": "T", "popularity": 50, "uri": "t"}]},
            "albums": {"items": [{"name": "A", "popularity": 50, "uri": "a"}]},
            "artists": {"items": [{"name": "R", "popularity": 80, "uri": "r"}]}
        });
        assert_eq!(pick_result(&results, "any").unwrap().0, "artist");
        let results = json!({
            "tracks": {"items": [{"name": "T", "popularity": 50}]},
            "albums": {"items": [{"name": "A", "popularity": 50}]}
        });
        assert_eq!(pick_result(&results, "any").unwrap().0, "track");
        assert_eq!(pick_result(&results, "album").unwrap().1["name"], "A");
        assert!(pick_result(&results, "artist").is_none());
        assert!(pick_result(&json!({}), "any").is_none());
    }

    #[test]
    fn artists_and_playlist_matching() {
        assert_eq!(
            artists_str(&json!({"artists": [{"name": "A"}, {"name": "B"}]})),
            "A, B"
        );
        assert_eq!(artists_str(&json!({})), "unknown");
        let owned = vec![
            json!({"name": "Jazz Classics", "uri": "1"}),
            json!({"name": "Morning Workout Mix", "uri": "2"}),
            json!({"name": "Discover Weekly", "uri": "3"}),
        ];
        assert_eq!(
            match_owned_playlist(&owned, "discover weekly").unwrap()["uri"],
            "3"
        );
        assert_eq!(match_owned_playlist(&owned, "workout").unwrap()["uri"], "2");
        assert_eq!(
            match_owned_playlist(&owned, "classics jazz").unwrap()["uri"],
            "1"
        );
        assert!(match_owned_playlist(&owned, "chill").is_none());
    }
}

#[cfg(test)]
mod network_tests {
    //! Real Spotify calls with the cached token (read-only: refresh, devices,
    //! now-playing). `cargo test -p voice-chatbot-server -- --ignored network`.
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn network_spotify_token_refresh_and_devices() {
        voice_chatbot_env_file::load_if_unset(std::path::Path::new("../../.env"));
        let client_id = std::env::var("SPOTIPY_CLIENT_ID").expect("SPOTIPY_CLIENT_ID in .env");
        let c = SpotifyClient::new(client_id).expect("cached token");
        // Force a refresh regardless of expiry so the PKCE refresh path runs.
        c.token.lock().unwrap().as_mut().unwrap().expires_at = 0;
        let token = c.access_token().await.expect("refresh");
        assert!(!token.is_empty());
        let devices = c
            .api(reqwest::Method::GET, "/me/player/devices", &[], None)
            .await
            .expect("devices");
        let names: Vec<String> = devices["devices"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|d| d["name"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        eprintln!("visible devices: {names:?}");
        eprintln!("babel device: {:?}", c.resolve_device(true).await);
        eprintln!("now playing: {:?}", c.now_playing().await);
    }
}
