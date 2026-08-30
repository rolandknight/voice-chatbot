//! `SEARCH_LOCATION` — where the household is, for search that depends on it.
//!
//! One setting feeds two consumers: Claude's server-side web search takes it as
//! a structured `user_location` object, and Brave takes its `country`. Without
//! it "what's on at Cineplex Etobicoke" is a query with no place attached, which
//! is how the DuckDuckGo era failed. `WEATHER_DEFAULT_LOCATION` is
//! deliberately untouched: it is free text with its own geolocation fallback.

use serde_json::{json, Value};

/// `city,region,country,timezone`. Country is ISO 3166-1 alpha-2; timezone is
/// an IANA id. An empty setting means "send no location".
pub const DEFAULT: &str = "Toronto,Ontario,CA,America/Toronto";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchLocation {
    pub city: String,
    pub region: String,
    /// ISO 3166-1 alpha-2, uppercase. The API rejects anything else with a 400.
    pub country: String,
    /// IANA timezone id.
    pub timezone: String,
}

impl SearchLocation {
    /// Parsed once at startup, like every other config value here, so a typo is
    /// a boot failure and not a puzzling search result months later.
    pub fn parse(value: &str) -> Result<Option<Self>, String> {
        let value = value.trim();
        if value.is_empty() {
            return Ok(None);
        }
        let parts: Vec<&str> = value.split(',').map(str::trim).collect();
        let [city, region, country, timezone] = parts.as_slice() else {
            return Err(format!(
                "SEARCH_LOCATION must be city,region,country,timezone (got {} field(s) in {value:?}; e.g. {DEFAULT})",
                parts.len()
            ));
        };
        if country.len() != 2 || !country.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(format!(
                "SEARCH_LOCATION country must be a two-letter ISO 3166-1 alpha-2 code (got {country:?}; Canada is CA)"
            ));
        }
        Ok(Some(Self {
            city: city.to_string(),
            region: region.to_string(),
            country: country.to_ascii_uppercase(),
            timezone: timezone.to_string(),
        }))
    }

    /// The Messages API `user_location` object for the web search tool.
    pub fn user_location(&self) -> Value {
        json!({
            "type": "approximate",
            "city": self.city,
            "region": self.region,
            "country": self.country,
            "timezone": self.timezone,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_four_fields() {
        let l = SearchLocation::parse("Toronto,Ontario,CA,America/Toronto")
            .unwrap()
            .unwrap();
        assert_eq!(l.city, "Toronto");
        assert_eq!(l.region, "Ontario");
        assert_eq!(l.country, "CA");
        assert_eq!(l.timezone, "America/Toronto");
    }

    #[test]
    fn the_default_parses() {
        assert!(SearchLocation::parse(DEFAULT).unwrap().is_some());
    }

    #[test]
    fn trims_fields_and_uppercases_the_country() {
        let l = SearchLocation::parse(" Toronto , Ontario , ca , America/Toronto ")
            .unwrap()
            .unwrap();
        assert_eq!(l.city, "Toronto");
        assert_eq!(l.country, "CA");
    }

    #[test]
    fn empty_means_no_location() {
        assert_eq!(SearchLocation::parse("").unwrap(), None);
        assert_eq!(SearchLocation::parse("   ").unwrap(), None);
    }

    #[test]
    fn rejects_the_wrong_field_count_and_a_bad_country() {
        assert!(SearchLocation::parse("Toronto,Ontario,CA").is_err());
        assert!(SearchLocation::parse("Toronto,Ontario,CA,America/Toronto,extra").is_err());
        assert!(SearchLocation::parse("Toronto,Ontario,Canada,America/Toronto").is_err());
    }

    #[test]
    fn renders_the_anthropic_user_location_object() {
        let l = SearchLocation::parse(DEFAULT).unwrap().unwrap();
        assert_eq!(
            l.user_location(),
            serde_json::json!({
                "type": "approximate",
                "city": "Toronto",
                "region": "Ontario",
                "country": "CA",
                "timezone": "America/Toronto"
            })
        );
    }
}
