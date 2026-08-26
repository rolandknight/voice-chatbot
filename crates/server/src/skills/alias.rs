//! Spoken-alias matching shared by BBC radio and shows (port of
//! `build_alias_table` / `match_alias` in scripts/radio.py).
//!
//! Longest alias first, so "five sports extra" beats "five live" when both
//! appear; matching is on a punctuation-stripped, single-spaced lower form so
//! the recognizer's stray commas/periods don't break it.

/// Aliases with their item index, longest alias first.
pub struct AliasTable {
    entries: Vec<(String, usize)>,
}

impl AliasTable {
    pub fn new<'a>(aliases: impl IntoIterator<Item = (usize, &'a [&'a str])>) -> Self {
        let mut entries: Vec<(String, usize)> = aliases
            .into_iter()
            .flat_map(|(idx, list)| list.iter().map(move |a| (normalise(a), idx)))
            .collect();
        // Stable sort keeps table order among equal lengths, like Python's sorted().
        entries.sort_by_key(|(alias, _)| std::cmp::Reverse(alias.len()));
        Self { entries }
    }

    /// Index of the item whose alias occurs in `text` as whole words.
    pub fn find(&self, text: &str) -> Option<usize> {
        let haystack = format!(" {} ", normalise(text));
        self.entries
            .iter()
            .find(|(alias, _)| haystack.contains(&format!(" {alias} ")))
            .map(|(_, idx)| *idx)
    }
}

/// Lower-case, non-word characters → space, whitespace collapsed.
pub fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.extend(c.to_lowercase());
        } else {
            pending_space = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalises_like_python() {
        assert_eq!(normalise("  Radio 4, please! "), "radio 4 please");
        assert_eq!(normalise("6-Music"), "6 music");
    }

    #[test]
    fn longest_alias_wins_and_words_are_whole() {
        let table = AliasTable::new([
            (0, ["five live", "5 live"].as_slice()),
            (1, ["five sports extra", "5 sports extra"].as_slice()),
            (2, ["radio 4"].as_slice()),
            (3, ["radio 4 extra"].as_slice()),
        ]);
        assert_eq!(table.find("put on five sports extra"), Some(1));
        assert_eq!(table.find("five live please"), Some(0));
        assert_eq!(table.find("Radio 4 Extra."), Some(3));
        assert_eq!(table.find("radio 4"), Some(2));
        assert_eq!(table.find("radio 45"), None);
        assert_eq!(table.find("nothing here"), None);
    }
}
