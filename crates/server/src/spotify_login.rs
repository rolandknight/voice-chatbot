//! `voice-chatbot-server spotify-login [--headless]` — the one-time Spotify
//! PKCE authorisation, writing the token where the skills (and the legacy
//! `scripts/spotify.py`) expect it. No client secret involved.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

use base64::Engine;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::skills::spotify_client::{config_dir, Token, SCOPE};

const AUTH_URL: &str = "https://accounts.spotify.com/authorize";
const TOKEN_URL: &str = "https://accounts.spotify.com/api/token";

fn b64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn code_verifier() -> String {
    let mut bytes = [0u8; 64];
    rand::fill(&mut bytes);
    b64url(&bytes)
}

fn code_challenge(verifier: &str) -> String {
    b64url(&Sha256::digest(verifier.as_bytes()))
}

/// The `code` query parameter of the redirect URL Spotify sends the browser to.
fn code_from_redirect(url_or_path: &str) -> Option<String> {
    let query = url_or_path.split_once('?')?.1;
    let query = query.split(['#', ' ']).next().unwrap_or("");
    query
        .split('&')
        .find_map(|kv| kv.strip_prefix("code="))
        .map(str::to_string)
        .filter(|c| !c.is_empty())
}

/// Serve the redirect once on the URI's host:port; return the auth code.
fn wait_for_redirect(redirect_uri: &str) -> Result<String, String> {
    let url = reqwest::Url::parse(redirect_uri).map_err(|e| format!("redirect uri: {e}"))?;
    let host = url.host_str().unwrap_or("127.0.0.1");
    let port = url.port().unwrap_or(80);
    let listener =
        TcpListener::bind((host, port)).map_err(|e| format!("listen on {host}:{port}: {e}"))?;
    loop {
        let (mut stream, _) = listener.accept().map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .map_err(|e| e.to_string())?;
        let path = request_line.split_whitespace().nth(1).unwrap_or("");
        let code = code_from_redirect(path);
        let body = if code.is_some() {
            "Spotify authorised. You can close this tab."
        } else {
            "No code in the redirect; try again."
        };
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        if let Some(code) = code {
            return Ok(code);
        }
    }
}

pub async fn run(client_id: &str, redirect_uri: &str, headless: bool) -> Result<(), String> {
    if client_id.trim().is_empty() {
        return Err(
            "set SPOTIPY_CLIENT_ID in .env first (https://developer.spotify.com/dashboard)".into(),
        );
    }
    let token_path = config_dir().join("spotify_token.json");
    if let Ok(t) = Token::load(&token_path) {
        if !t.refresh_token.is_empty() {
            println!(
                "Already authorised (token at {}). Delete it to re-authorise.",
                token_path.display()
            );
            return Ok(());
        }
    }
    let verifier = code_verifier();
    let url = reqwest::Url::parse_with_params(
        AUTH_URL,
        &[
            ("client_id", client_id),
            ("response_type", "code"),
            ("redirect_uri", redirect_uri),
            ("code_challenge_method", "S256"),
            ("code_challenge", &code_challenge(&verifier)),
            ("scope", SCOPE),
        ],
    )
    .map_err(|e| e.to_string())?;

    let code = if headless {
        println!(
            "Open this URL in a browser on ANY device, approve, then paste the FULL URL it\n\
             redirects to (a 'can't connect' page at {redirect_uri} is fine):\n\n{url}\n"
        );
        print!("Redirected URL: ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        code_from_redirect(line.trim()).ok_or("no code= in that URL")?
    } else {
        println!("Opening Spotify authorisation in your browser (or open this URL):\n\n{url}\n");
        let _ = std::process::Command::new(if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        })
        .arg(url.as_str())
        .status();
        let redirect = redirect_uri.to_string();
        tokio::task::spawn_blocking(move || wait_for_redirect(&redirect))
            .await
            .map_err(|e| e.to_string())??
    };

    let resp: Value = reqwest::Client::new()
        .post(TOKEN_URL)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code.as_str()),
            ("redirect_uri", redirect_uri),
            ("client_id", client_id),
            ("code_verifier", verifier.as_str()),
        ])
        .send()
        .await
        .map_err(|e| format!("token exchange: {e}"))?
        .error_for_status()
        .map_err(|e| format!("token exchange: {e}"))?
        .json()
        .await
        .map_err(|e| format!("token exchange body: {e}"))?;
    let mut token = Token {
        access_token: String::new(),
        token_type: "Bearer".into(),
        expires_in: 0,
        refresh_token: String::new(),
        scope: SCOPE.into(),
        expires_at: 0,
    };
    token.apply_response(&resp, chrono::Utc::now().timestamp())?;
    if token.refresh_token.is_empty() {
        return Err("token response had no refresh_token".into());
    }
    token.save(&token_path)?;
    println!("Authorised. Token cached at {}", token_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_is_s256_of_verifier() {
        // RFC 7636 appendix B.
        assert_eq!(
            code_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        assert_eq!(code_verifier().len(), 86);
    }

    #[test]
    fn extracts_code_from_redirects() {
        assert_eq!(
            code_from_redirect("/callback?code=abc&state=x").as_deref(),
            Some("abc")
        );
        assert_eq!(
            code_from_redirect("http://127.0.0.1:8765/callback?state=x&code=abc HTTP/1.1")
                .as_deref(),
            Some("abc")
        );
        assert_eq!(code_from_redirect("/callback?error=denied"), None);
        assert_eq!(code_from_redirect("/favicon.ico"), None);
    }
}
