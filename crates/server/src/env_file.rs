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
    let value = if let Some(inner) = value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        inner.to_string()
    } else if let Some(inner) = value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')) {
        inner.to_string()
    } else {
        value.split(" #").next().unwrap_or("").trim().to_string()
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
        assert_eq!(parse_line("# comment"), None);
        assert_eq!(parse_line(""), None);
        assert_eq!(parse_line("hey babel,hey babe,hey baby"), None);
        assert_eq!(parse_line("bad key=1"), None);
    }
}
