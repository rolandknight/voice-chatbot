//! The seven Spotify tools (port of skills/spotify/*), all thin wrappers over
//! [`SpotifyClient`]. Playback happens on the client's librespot "Babel"
//! endpoint; these only issue Web API commands.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::spotify_client::SpotifyClient;
use super::{arg_str, CallCtx, Skill};

pub fn all(client: Arc<SpotifyClient>) -> Vec<Arc<dyn Skill>> {
    vec![
        Arc::new(PlaySpotify(client.clone())),
        Arc::new(PlaySpotifyPlaylist(client.clone())),
        Arc::new(PauseSpotify(client.clone())),
        Arc::new(ResumeSpotify(client.clone())),
        Arc::new(SkipSpotify(client.clone())),
        Arc::new(StopSpotify(client.clone())),
        Arc::new(WhatsPlaying(client)),
    ]
}

pub struct PlaySpotify(Arc<SpotifyClient>);

#[async_trait]
impl Skill for PlaySpotify {
    fn name(&self) -> &str {
        "play_spotify"
    }
    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        let query = arg_str(args, "query");
        let kind = arg_str(args, "kind");
        if let Some(m) = &ctx.media {
            m.stop();
        }
        self.0
            .search_and_play(query, if kind.is_empty() { "any" } else { kind })
            .await
            .1
    }
}

pub struct PlaySpotifyPlaylist(Arc<SpotifyClient>);

#[async_trait]
impl Skill for PlaySpotifyPlaylist {
    fn name(&self) -> &str {
        "play_spotify_playlist"
    }
    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        if let Some(m) = &ctx.media {
            m.stop();
        }
        self.0.play_playlist(arg_str(args, "name")).await.1
    }
}

pub struct PauseSpotify(Arc<SpotifyClient>);

#[async_trait]
impl Skill for PauseSpotify {
    fn name(&self) -> &str {
        "pause_spotify"
    }
    async fn call(&self, _args: &Value, _ctx: &CallCtx) -> String {
        if self.0.pause().await {
            "Paused.".to_string()
        } else {
            "I couldn't pause Spotify.".to_string()
        }
    }
}

pub struct ResumeSpotify(Arc<SpotifyClient>);

#[async_trait]
impl Skill for ResumeSpotify {
    fn name(&self) -> &str {
        "resume_spotify"
    }
    async fn call(&self, _args: &Value, _ctx: &CallCtx) -> String {
        if self.0.resume().await {
            "Resumed.".to_string()
        } else {
            "I couldn't resume Spotify.".to_string()
        }
    }
}

pub struct SkipSpotify(Arc<SpotifyClient>);

#[async_trait]
impl Skill for SkipSpotify {
    fn name(&self) -> &str {
        "skip_spotify"
    }
    async fn call(&self, args: &Value, _ctx: &CallCtx) -> String {
        let (ok, done) = if arg_str(args, "direction").eq_ignore_ascii_case("previous") {
            (self.0.skip_previous().await, "Went back.")
        } else {
            (self.0.skip_next().await, "Skipped.")
        };
        if ok {
            done.to_string()
        } else {
            "I couldn't skip the track.".to_string()
        }
    }
}

/// Stops Spotify and, like the Python handler, any BBC audio too.
pub struct StopSpotify(Arc<SpotifyClient>);

#[async_trait]
impl Skill for StopSpotify {
    fn name(&self) -> &str {
        "stop_spotify"
    }
    async fn call(&self, _args: &Value, ctx: &CallCtx) -> String {
        let spotify_was_playing = self.0.is_playing();
        if spotify_was_playing {
            self.0.stop().await;
        }
        let media_was_playing = ctx.media.as_ref().and_then(|m| m.stop()).is_some();
        if spotify_was_playing || media_was_playing {
            "Stopped.".to_string()
        } else {
            "Nothing's playing.".to_string()
        }
    }
}

pub struct WhatsPlaying(Arc<SpotifyClient>);

#[async_trait]
impl Skill for WhatsPlaying {
    fn name(&self) -> &str {
        "whats_playing"
    }
    async fn call(&self, _args: &Value, _ctx: &CallCtx) -> String {
        match self.0.now_playing().await {
            Some(text) => format!("This is {text}."),
            None => "Nothing's playing on Spotify.".to_string(),
        }
    }
}
