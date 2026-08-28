//! PoC-owned browser playground.
//!
//! FlowCat's upstream page deliberately stays minimal.  Babel's validation page
//! adds microphone selection and a pre-call level meter so permission, device
//! routing, and WebRTC failures can be distinguished without OS-specific tools.

use axum::http::header::CACHE_CONTROL;
use axum::response::{Html, IntoResponse};

const PLAYGROUND_HTML: &str = include_str!("playground.html");

pub async fn page() -> impl IntoResponse {
    ([(CACHE_CONTROL, "no-store")], Html(PLAYGROUND_HTML))
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use axum::http::{header::CACHE_CONTROL, header::CONTENT_TYPE, StatusCode};
    use axum::response::IntoResponse;

    use super::{page, PLAYGROUND_HTML};

    #[tokio::test]
    async fn serves_the_embedded_html_page() {
        let response = page().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "text/html; charset=utf-8");
        assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        assert_eq!(body.as_ref(), PLAYGROUND_HTML.as_bytes());
    }

    #[test]
    fn playground_keeps_the_flowcat_wire_contract() {
        for expected in [
            "fetch(\"/webrtc/offer\"",
            "/webrtc/events/${pcId}",
            "new RTCPeerConnection",
            "rtf-user-transcription",
            "rtf-bot-text",
        ] {
            assert!(
                PLAYGROUND_HTML.contains(expected),
                "playground is missing {expected:?}"
            );
        }
    }

    #[test]
    fn playground_exposes_microphone_diagnostics() {
        for expected in [
            "id=\"microphone\"",
            "id=\"refresh-microphones\"",
            "id=\"test-microphone\"",
            "role=\"meter\"",
            "enumerateDevices()",
            "getUserMedia",
            "createAnalyser()",
            "replaceTrack",
            "devicechange",
            "getSettings()",
            "Stop microphone test",
            "primeAudioContext()",
            "new AbortController()",
            "assertCurrent(generation)",
            "pagehide",
            "audioContext.close()",
            "track.stop()",
        ] {
            assert!(
                PLAYGROUND_HTML.contains(expected),
                "playground is missing {expected:?}"
            );
        }
    }

    #[test]
    fn loopback_negotiation_is_local_bounded_and_observable() {
        assert!(PLAYGROUND_HTML.contains("new RTCPeerConnection({ iceServers: [] })"));
        assert!(!PLAYGROUND_HTML.contains("stun:"));
        assert!(PLAYGROUND_HTML.contains("const ICE_GATHERING_TIMEOUT_MS = 1000"));
        assert!(PLAYGROUND_HTML.contains("timeout = setTimeout(() =>"));
        assert!(PLAYGROUND_HTML.contains("resolve(false)"));

        for phase in [
            "creating local media offer…",
            "finding local media route…",
            "starting voice pipeline…",
            "applying media answer…",
            "connecting media…",
        ] {
            assert!(
                PLAYGROUND_HTML.contains(phase),
                "playground is missing negotiation phase {phase:?}"
            );
        }
    }
}
