//! FlowCat's HTTP offer/answer and companion event-stream URLs.

use anyhow::{bail, Context, Result};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct OfferRequest<'a> {
    sdp: &'a str,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct OfferResponse {
    pub sdp: String,
    pub pc_id: String,
}

#[derive(Clone, Debug)]
pub struct ServerEndpoints {
    health: Url,
    offer: Url,
    events_base: Url,
}

impl ServerEndpoints {
    pub fn new(server_url: &str) -> Result<Self> {
        let mut base = Url::parse(server_url).context("parse --server-url URL")?;
        match base.scheme() {
            "http" | "https" => {}
            scheme => bail!("server URL must use http or https, got {scheme:?}"),
        }
        if base.host().is_none() {
            bail!("server URL must include a host");
        }
        base.set_query(None);
        base.set_fragment(None);

        let mut health = base.clone();
        health.set_path("/healthz");
        let mut offer = base.clone();
        offer.set_path("/webrtc/offer");
        let mut events_base = base;
        events_base.set_path("/");
        events_base
            .set_scheme(if events_base.scheme() == "https" {
                "wss"
            } else {
                "ws"
            })
            .map_err(|_| anyhow::anyhow!("cannot derive WebSocket URL"))?;

        Ok(Self {
            health,
            offer,
            events_base,
        })
    }

    /// Host and port of the server (for choosing the interface that reaches it).
    pub fn host_port(&self) -> Result<(String, u16)> {
        let host = self
            .health
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("server URL has no host"))?
            .trim_matches(|c| c == '[' || c == ']')
            .to_string();
        let port = self
            .health
            .port_or_known_default()
            .ok_or_else(|| anyhow::anyhow!("server URL has no port"))?;
        Ok((host, port))
    }

    pub fn health_url(&self) -> &Url {
        &self.health
    }

    pub fn offer_url(&self) -> &Url {
        &self.offer
    }

    pub fn events_url(&self, pc_id: &str) -> Result<Url> {
        if pc_id.trim().is_empty() {
            bail!("FlowCat returned an empty pc_id");
        }
        let mut url = self.events_base.clone();
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("event URL cannot contain path segments"))?
            .extend(["webrtc", "events", pc_id]);
        Ok(url)
    }
}

pub async fn require_healthy(client: &Client, endpoints: &ServerEndpoints) -> Result<()> {
    client
        .get(endpoints.health_url().clone())
        .send()
        .await
        .context("connect to FlowCat health endpoint")?
        .error_for_status()
        .context("FlowCat is not healthy")?;
    Ok(())
}

pub async fn exchange_offer(
    client: &Client,
    endpoints: &ServerEndpoints,
    sdp: &str,
) -> Result<OfferResponse> {
    if sdp.trim().is_empty() {
        bail!("refusing to send an empty SDP offer");
    }
    let response = client
        .post(endpoints.offer_url().clone())
        .json(&OfferRequest { sdp })
        .send()
        .await
        .context("send WebRTC offer to FlowCat")?
        .error_for_status()
        .context("FlowCat rejected the WebRTC offer")?
        .json::<OfferResponse>()
        .await
        .context("decode FlowCat offer response")?;
    if response.sdp.trim().is_empty() {
        bail!("FlowCat returned an empty SDP answer");
    }
    if response.pc_id.trim().is_empty() {
        bail!("FlowCat returned an empty pc_id");
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_flowcat_urls_and_discards_input_path() {
        let endpoints = ServerEndpoints::new("http://127.0.0.1:6210/old?q=1#frag").unwrap();
        assert_eq!(
            endpoints.health_url().as_str(),
            "http://127.0.0.1:6210/healthz"
        );
        assert_eq!(
            endpoints.offer_url().as_str(),
            "http://127.0.0.1:6210/webrtc/offer"
        );
        assert_eq!(
            endpoints.events_url("pc-17").unwrap().as_str(),
            "ws://127.0.0.1:6210/webrtc/events/pc-17"
        );
    }

    #[test]
    fn secure_and_ipv6_urls_map_to_wss() {
        let endpoints = ServerEndpoints::new("https://[::1]:6210").unwrap();
        assert_eq!(
            endpoints.events_url("pc a").unwrap().as_str(),
            "wss://[::1]:6210/webrtc/events/pc%20a"
        );
    }

    #[test]
    fn invalid_urls_and_empty_ids_are_rejected() {
        assert!(ServerEndpoints::new("ftp://localhost:6210").is_err());
        assert!(ServerEndpoints::new("not a URL").is_err());
        assert!(ServerEndpoints::new("http://localhost")
            .unwrap()
            .events_url(" ")
            .is_err());
    }

    #[test]
    fn wire_response_requires_both_fields() {
        let parsed: OfferResponse =
            serde_json::from_str(r#"{"sdp":"v=0\\r\\n","pc_id":"pc-1"}"#).unwrap();
        assert_eq!(parsed.pc_id, "pc-1");
        assert!(serde_json::from_str::<OfferResponse>(r#"{"sdp":"v=0"}"#).is_err());
    }
}
