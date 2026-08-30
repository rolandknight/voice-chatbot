//! Lenient `.env` loading. The repo-root `.env` is shared with the Python
//! chatbot (python-dotenv), whose grammar is looser than dotenvy's — a single
//! odd line there made dotenvy reject the whole file, silently dropping every
//! secret. Here each `KEY=VALUE` line stands alone: unparsable lines are
//! skipped with a debug log, and variables already set are never overridden.

use std::path::Path;

/// `Some((key, value))` for a `KEY=VALUE` line; `None` for blanks, comments
/// and anything else. Surrounding single/double quotes are stripped; an
/// unquoted value ends at ` #`.
pub fn parse_line(line: &str) -> Option<(String, String)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let value = value.trim();
    // A quoted value ends at its closing quote, so a `#` inside is data and a
    // trailing comment outside is not. Only an unquoted value ends at ` #`.
    let quoted = value
        .strip_prefix('"')
        .and_then(|rest| rest.split_once('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|rest| rest.split_once('\''))
        });
    let value = match quoted {
        Some((inner, _after)) => inner.to_string(),
        None => value.split(" #").next().unwrap_or("").trim().to_string(),
    };
    Some((key.to_string(), value))
}

/// Set every `KEY=VALUE` in `path` that isn't already in the environment.
/// A missing file is fine (returns 0).
pub fn load_if_unset(path: &Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    let mut loaded = 0;
    for (n, line) in text.lines().enumerate() {
        match parse_line(line) {
            Some((key, value)) => {
                if std::env::var_os(&key).is_none() {
                    std::env::set_var(&key, value);
                    loaded += 1;
                }
            }
            None if !line.trim().is_empty() && !line.trim_start().starts_with('#') => {
                tracing::debug!(path = %path.display(), line = n + 1, "env file: skipped unparsable line");
            }
            None => {}
        }
    }
    loaded
}

/// Names dropped when the `POC_` prefix was retired. A stale `.env` would
/// otherwise be read as "unset" and silently take the defaults — on a home
/// server that surfaces days later as the wrong STT model or a missing skill.
pub const RETIRED_PREFIX: &str = "POC_";

/// Every `POC_*` name found in the environment, so startup can refuse to run.
pub fn retired_names<I: Iterator<Item = String>>(keys: I) -> Vec<String> {
    names_with_prefix(keys, RETIRED_PREFIX)
}

/// Every name in `keys` starting with `prefix`, sorted. Each binary owns its
/// own retired prefix and its own rename hint: the server's is `POC_`, the
/// native client's is `FLOWCAT_`.
pub fn names_with_prefix<I: Iterator<Item = String>>(keys: I, prefix: &str) -> Vec<String> {
    let mut found: Vec<String> = keys.filter(|k| k.starts_with(prefix)).collect();
    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_python_dotenv_style_lines() {
        assert_eq!(parse_line("A=1"), Some(("A".into(), "1".into())));
        assert_eq!(
            parse_line("  export B = two words  "),
            Some(("B".into(), "two words".into()))
        );
        assert_eq!(
            parse_line("C=\"quoted # not comment\""),
            Some(("C".into(), "quoted # not comment".into()))
        );
        assert_eq!(
            parse_line("D='single'"),
            Some(("D".into(), "single".into()))
        );
        assert_eq!(
            parse_line("E=value # comment"),
            Some(("E".into(), "value".into()))
        );
        assert_eq!(parse_line("F="), Some(("F".into(), String::new())));
        // A quoted value may still be followed by a comment. The closing quote
        // ends the value; python-dotenv reads it the same way.
        assert_eq!(
            parse_line("G = \"http://127.0.0.1:6210\"   # dead port"),
            Some(("G".into(), "http://127.0.0.1:6210".into()))
        );
        assert_eq!(
            parse_line("H='two words' # trailing"),
            Some(("H".into(), "two words".into()))
        );
        assert_eq!(parse_line("# comment"), None);
        assert_eq!(parse_line(""), None);
        assert_eq!(parse_line("hey babel,hey babe,hey baby"), None);
        assert_eq!(parse_line("bad key=1"), None);
    }
}

#[cfg(test)]
mod retired_tests {
    use super::*;

    #[test]
    fn flags_only_the_retired_prefix() {
        let keys = [
            "POC_STT_BACKEND",
            "SERVER_URL",
            "POC_LLM_MODEL",
            "BRAVE_API_KEY",
        ]
        .into_iter()
        .map(String::from);
        assert_eq!(
            retired_names(keys),
            vec!["POC_LLM_MODEL".to_string(), "POC_STT_BACKEND".to_string()]
        );
    }

    #[test]
    fn an_environment_without_them_is_clean() {
        assert!(retired_names(["SERVER_URL".to_string()].into_iter()).is_empty());
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    #[test]
    fn finds_every_name_under_an_arbitrary_prefix_sorted() {
        let keys = ["FLOWCAT_URL", "SERVER_URL", "FLOWCAT_NO_WAKE", "PATH"]
            .into_iter()
            .map(String::from);
        assert_eq!(
            names_with_prefix(keys, "FLOWCAT_"),
            vec!["FLOWCAT_NO_WAKE".to_string(), "FLOWCAT_URL".to_string()]
        );
    }

    #[test]
    fn a_prefix_nothing_matches_is_empty() {
        let keys = ["SERVER_URL".to_string()].into_iter();
        assert!(names_with_prefix(keys, "FLOWCAT_").is_empty());
    }
}
