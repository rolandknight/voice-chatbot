//! `web_search` — Brave by default, DuckDuckGo or Tavily on request.
//!
//! Advertised to the **local** model only. Claude gets Anthropic's server-side
//! web search instead, which carries the same tool name (see `llm_claude.rs`).

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{arg_str, CallCtx, Skill};

const HTTP_TIMEOUT: Duration = Duration::from_secs(6);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Provider {
    DuckDuckGo,
    Brave,
    Tavily,
}

impl Provider {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "brave" => Ok(Self::Brave),
            "duckduckgo" => Ok(Self::DuckDuckGo),
            "tavily" => Ok(Self::Tavily),
            other => Err(format!(
                "unsupported WEB_SEARCH_PROVIDER {other:?} (expected brave, duckduckgo, or tavily)"
            )),
        }
    }
}

pub struct WebSearch {
    http: reqwest::Client,
    provider: Provider,
    brave_key: String,
    tavily_key: String,
    /// ISO 3166-1 alpha-2 from `SEARCH_LOCATION`, passed to Brave so results
    /// are local. `None` sends no country and lets Brave guess from the egress
    /// IP, which on a home server is usually right but not always.
    country: Option<String>,
}

fn text_field<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// DuckDuckGo: the abstract if there is one, else up to three related-topic texts.
fn extract_duckduckgo(data: &Value) -> String {
    if let Some(abs) = text_field(data, "AbstractText") {
        return abs.to_string();
    }
    let mut snippets: Vec<&str> = Vec::new();
    for item in data
        .get("RelatedTopics")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(t) = text_field(item, "Text") {
            snippets.push(t);
        } else if let Some(subs) = item.get("Topics").and_then(Value::as_array) {
            snippets.extend(subs.iter().filter_map(|s| text_field(s, "Text")));
        }
        if snippets.len() >= 3 {
            break;
        }
    }
    snippets.truncate(3);
    snippets.join(" ").trim().to_string()
}

/// Brave: up to three results as `Title: description`. The title carries the
/// venue or publication, which is often the part that answers the question.
fn extract_brave(data: &Value) -> String {
    data.pointer("/web/results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(3)
        .filter_map(|r| {
            let desc = text_field(r, "description")?;
            Some(match text_field(r, "title") {
                Some(title) => format!("{title}: {desc}"),
                None => desc.to_string(),
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn extract_tavily(data: &Value) -> String {
    if let Some(a) = text_field(data, "answer") {
        return a.to_string();
    }
    data.get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .take(3)
        .filter_map(|r| text_field(r, "content"))
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

impl WebSearch {
    pub fn new(
        provider: Provider,
        brave_key: String,
        tavily_key: String,
        country: Option<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(HTTP_TIMEOUT)
                .build()
                .expect("reqwest client"),
            provider,
            brave_key,
            tavily_key,
            country,
        }
    }

    async fn search(&self, query: &str) -> Result<String, reqwest::Error> {
        Ok(match self.provider {
            Provider::DuckDuckGo => {
                let data: Value = self
                    .http
                    .get("https://api.duckduckgo.com/")
                    .query(&[
                        ("q", query),
                        ("format", "json"),
                        ("no_html", "1"),
                        ("skip_disambig", "1"),
                    ])
                    .header("User-Agent", "babel-voice-bot/1.0")
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                extract_duckduckgo(&data)
            }
            Provider::Brave => {
                let key = self.brave_key.trim();
                if key.is_empty() {
                    return Ok(
                        "Brave search isn't configured. Add a BRAVE_API_KEY to .env to enable it."
                            .to_string(),
                    );
                }
                let mut params: Vec<(&str, &str)> = vec![("q", query), ("count", "3")];
                if let Some(c) = &self.country {
                    params.push(("country", c));
                }
                let data: Value = self
                    .http
                    .get("https://api.search.brave.com/res/v1/web/search")
                    .query(&params)
                    .header("Accept", "application/json")
                    .header("X-Subscription-Token", key)
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                extract_brave(&data)
            }
            Provider::Tavily => {
                let key = self.tavily_key.trim();
                if key.is_empty() {
                    return Ok("Tavily search isn't configured. Add a TAVILY_API_KEY to .env to enable it.".to_string());
                }
                let data: Value = self
                    .http
                    .post("https://api.tavily.com/search")
                    .json(&json!({
                        "api_key": key,
                        "query": query,
                        "max_results": 3,
                        "include_answer": true,
                        "search_depth": "basic",
                    }))
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                extract_tavily(&data)
            }
        })
    }
}

#[async_trait]
impl Skill for WebSearch {
    fn name(&self) -> &str {
        "web_search"
    }

    async fn call(&self, args: &Value, _ctx: &CallCtx) -> String {
        let query = arg_str(args, "query");
        if query.is_empty() {
            return "I need a search query to look something up.".to_string();
        }
        match self.search(query).await {
            Ok(text) if text.is_empty() => {
                format!("I searched for {query} but didn't get useful results.")
            }
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(provider = ?self.provider, error = %e, "web search failed");
                "I couldn't reach the web right now.".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duckduckgo_prefers_abstract_then_related_topics() {
        assert_eq!(
            extract_duckduckgo(&json!({"AbstractText": " Abstract. "})),
            "Abstract."
        );
        let data = json!({"AbstractText": "", "RelatedTopics": [
            {"Text": "one"},
            {"Topics": [{"Text": "two"}, {"Text": "three"}, {"Text": "four"}]},
            {"Text": "five"}
        ]});
        assert_eq!(extract_duckduckgo(&data), "one two three");
        assert_eq!(extract_duckduckgo(&json!({})), "");
    }

    #[test]
    fn brave_and_tavily_extraction() {
        let brave = json!({"web": {"results": [{"description": "a"}, {"description": ""}, {"description": "b"}, {"description": "c"}]}});
        assert_eq!(extract_brave(&brave), "a b");
        assert_eq!(extract_tavily(&json!({"answer": "42"})), "42");
        let tav = json!({"answer": "", "results": [{"content": "x"}, {"content": "y"}]});
        assert_eq!(extract_tavily(&tav), "x y");
    }

    #[test]
    fn provider_parsing_defaults_to_brave() {
        assert_eq!(Provider::parse("").unwrap(), Provider::Brave);
        assert_eq!(Provider::parse(" Brave ").unwrap(), Provider::Brave);
        assert_eq!(Provider::parse("duckduckgo").unwrap(), Provider::DuckDuckGo);
        assert_eq!(Provider::parse("tavily").unwrap(), Provider::Tavily);
        assert!(Provider::parse("bing").is_err());
    }

    #[test]
    fn brave_extraction_keeps_titles() {
        let brave = json!({"web": {"results": [
            {"title": "Showtimes", "description": "a"},
            {"title": "", "description": ""},
            {"description": "b"},
            {"title": "T", "description": "c"}
        ]}});
        assert_eq!(extract_brave(&brave), "Showtimes: a b");
    }
}

#[cfg(test)]
mod network_tests {
    //! Real DuckDuckGo call: `cargo test -p voice-chatbot-server -- --ignored network`.
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn network_duckduckgo_instant_answer() {
        let s = WebSearch::new(
            Provider::DuckDuckGo,
            String::new(),
            String::new(),
            Some("CA".to_string()),
        );
        let out = s
            .call(&json!({"query": "Eiffel Tower"}), &CallCtx::detached(0))
            .await;
        assert!(!out.starts_with("I couldn't reach"), "{out}");
        assert!(out.len() > 20, "{out}");
    }
}
