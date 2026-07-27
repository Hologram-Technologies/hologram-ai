//! Minimal JSON support for the FFI/CLI envelopes — no serde dependency.
//!
//! Scope (deliberate): flat objects with string/number/bool/null values on
//! input; string-escaped output on emit. This covers the binding envelope
//! (`session_invoke_json`) and CLI `--json` payloads. It is not a general
//! JSON library.

use hologram_ai_core::{AiError, AiResult};

/// A parsed flat JSON object value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

/// Parse a flat JSON object `{ "key": value, … }` (values: string, number,
/// bool, null). Rejects nesting, arrays, trailing garbage, and anything
/// over `MAX_INPUT` bytes.
pub fn parse_flat_object(input: &str) -> AiResult<Vec<(String, JsonValue)>> {
    const MAX_INPUT: usize = 1 << 20;
    if input.len() > MAX_INPUT {
        return Err(AiError::invalid_argument("json envelope too large"));
    }
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let out = p.object()?;
    p.ws();
    if p.pos != p.bytes.len() {
        return Err(bad("trailing bytes after json object"));
    }
    Ok(out)
}

fn bad(msg: &str) -> AiError {
    AiError::invalid_argument(format!("json: {msg}"))
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, b: u8) -> AiResult<()> {
        self.ws();
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(bad(&format!("expected '{}'", b as char)))
        }
    }

    fn object(&mut self) -> AiResult<Vec<(String, JsonValue)>> {
        self.expect(b'{')?;
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(out);
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.expect(b':')?;
            let value = self.value()?;
            out.push((key, value));
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(out);
                }
                _ => return Err(bad("expected ',' or '}'")),
            }
        }
    }

    fn value(&mut self) -> AiResult<JsonValue> {
        self.ws();
        match self.peek() {
            Some(b'"') => Ok(JsonValue::Str(self.string()?)),
            Some(b't') => self.literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.literal("false", JsonValue::Bool(false)),
            Some(b'n') => self.literal("null", JsonValue::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(bad("unsupported value (flat objects only)")),
        }
    }

    fn literal(&mut self, lit: &str, value: JsonValue) -> AiResult<JsonValue> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(value)
        } else {
            Err(bad("invalid literal"))
        }
    }

    fn number(&mut self) -> AiResult<JsonValue> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| bad("utf-8"))?;
        text.parse::<f64>()
            .map(JsonValue::Num)
            .map_err(|_| bad("invalid number"))
    }

    fn string(&mut self) -> AiResult<String> {
        if self.peek() != Some(b'"') {
            return Err(bad("expected string"));
        }
        self.pos += 1;
        let mut out = String::new();
        loop {
            let c = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| bad("unterminated string"))?;
            self.pos += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => out.push(self.escape()?),
                c if c < 0x20 => return Err(bad("control character in string")),
                c => {
                    // Collect the full UTF-8 sequence.
                    let len = utf8_len(c);
                    let start = self.pos - 1;
                    let end = start + len;
                    let slice = self
                        .bytes
                        .get(start..end)
                        .ok_or_else(|| bad("truncated utf-8"))?;
                    let s = std::str::from_utf8(slice).map_err(|_| bad("utf-8"))?;
                    out.push_str(s);
                    self.pos = end;
                }
            }
        }
    }

    fn escape(&mut self) -> AiResult<char> {
        let c = *self.bytes.get(self.pos).ok_or_else(|| bad("bad escape"))?;
        self.pos += 1;
        Ok(match c {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let hex = self
                    .bytes
                    .get(self.pos..self.pos + 4)
                    .ok_or_else(|| bad("bad \\u escape"))?;
                let s = std::str::from_utf8(hex).map_err(|_| bad("utf-8"))?;
                let code = u32::from_str_radix(s, 16).map_err(|_| bad("bad \\u escape"))?;
                self.pos += 4;
                char::from_u32(code).ok_or_else(|| bad("bad \\u escape"))?
            }
            _ => return Err(bad("unknown escape")),
        })
    }
}

fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

/// Append a JSON-escaped string (with quotes) to `out`.
pub fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_envelope() {
        let fields = parse_flat_object(
            r#"{ "operation": "generate", "prompt": "hi \"there\"", "maxOutputTokens": 128, "stream": true }"#,
        )
        .unwrap();
        assert_eq!(fields.len(), 4);
        assert_eq!(
            fields[0],
            ("operation".into(), JsonValue::Str("generate".into()))
        );
        assert_eq!(
            fields[1],
            ("prompt".into(), JsonValue::Str("hi \"there\"".into()))
        );
        assert_eq!(fields[2], ("maxOutputTokens".into(), JsonValue::Num(128.0)));
        assert_eq!(fields[3], ("stream".into(), JsonValue::Bool(true)));
    }

    #[test]
    fn rejects_nesting_arrays_and_trailing() {
        assert!(parse_flat_object(r#"{"a": {"b": 1}}"#).is_err());
        assert!(parse_flat_object(r#"{"a": [1]}"#).is_err());
        assert!(parse_flat_object(r#"{"a": 1} x"#).is_err());
        assert!(parse_flat_object(r#"{"a": 1"#).is_err());
    }

    #[test]
    fn string_output_is_escaped() {
        let mut s = String::new();
        push_json_string(&mut s, "a\"b\nc");
        assert_eq!(s, "\"a\\\"b\\nc\"");
    }
}
