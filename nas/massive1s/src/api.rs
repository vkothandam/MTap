//! The Massive custom-bars call: GET /v2/aggs/ticker/{T}/range/1/second/{d0}/{d1}.
//!
//! The endpoint has no time-of-day filter: every day comes back 04:00:00-19:59:59 New
//! York (probed 2026-10-19), so each bar is kept only if its start falls in its session
//! window. Up to 50k bars a page, then `next_url` (the key is re-added). Responses are
//! asked for gzip and decoded here, so the wire bytes can be counted.

use std::io::Read;
use std::thread::sleep;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_ENCODING};
use serde::Deserialize;
use url::Url;

use crate::bars::{Bar, Cols};
use crate::inputs::Windows;

pub const DEFAULT_BASE: &str = "https://api.massive.com";
const MAX_PAGES: u64 = 100_000; // guards against a next_url loop

#[derive(Deserialize)]
struct Page {
    results: Option<Vec<Bar>>,
    next_url: Option<String>,
}

/// One ticker's bars inside the windows, plus what it cost.
pub struct Fetched {
    pub cols: Cols,
    pub stats: Stats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub calls: u64,
    pub wire_bytes: u64,
    pub bars: u64,
    pub dropped: u64,
}

pub struct Api {
    client: Client,
    base: Url,
    key: String,
    /// Waits before each retry of a 429, 5xx, transport or bad-body error.
    pub retry_delays: Vec<Duration>,
}

enum Attempt {
    Retry(anyhow::Error),
    Fatal(anyhow::Error),
}

impl Api {
    pub fn new(base: &str, key: &str) -> Result<Self> {
        if key.trim().is_empty() {
            bail!("MASSIVE_API_KEY is empty");
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(180))
            .user_agent("mtap-massive1s")
            .build()?;
        Ok(Api {
            client,
            base: Url::parse(base).with_context(|| format!("bad base URL {base:?}"))?,
            key: key.trim().to_string(),
            retry_delays: [1, 2, 4].map(Duration::from_secs).to_vec(),
        })
    }

    /// Hide the key in a message that may quote a URL.
    pub fn mask(&self, s: &str) -> String {
        s.replace(&self.key, "***")
    }

    pub fn first_url(&self, ticker: &str, d0: &str, d1: &str) -> Result<Url> {
        let mut u = self.base.clone();
        u.path_segments_mut()
            .map_err(|_| anyhow!("base URL cannot take a path"))?
            .pop_if_empty()
            .extend([
                "v2", "aggs", "ticker", ticker, "range", "1", "second", d0, d1,
            ]);
        u.query_pairs_mut()
            .append_pair("adjusted", "true")
            .append_pair("sort", "asc")
            .append_pair("limit", "50000")
            .append_pair("apiKey", &self.key);
        Ok(u)
    }

    /// `next_url` with its cursor kept and exactly one apiKey (ours).
    pub fn with_key(&self, next: &str) -> Result<Url> {
        let mut u =
            Url::parse(next).with_context(|| format!("bad next_url {:?}", self.mask(next)))?;
        let pairs: Vec<(String, String)> = u
            .query_pairs()
            .filter(|(k, _)| k != "apiKey")
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        u.query_pairs_mut()
            .clear()
            .extend_pairs(pairs)
            .append_pair("apiKey", &self.key);
        Ok(u)
    }

    fn try_page(&self, url: &Url) -> Result<(Page, u64), Attempt> {
        let resp = self
            .client
            .get(url.clone())
            .header(ACCEPT_ENCODING, "gzip")
            .send()
            .map_err(|e| Attempt::Retry(anyhow!(e.without_url())))?;
        let status = resp.status();
        let gzip = resp
            .headers()
            .get(CONTENT_ENCODING)
            .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"gzip"));
        let body = resp
            .bytes()
            .map_err(|e| Attempt::Retry(anyhow!(e.without_url())))?;
        let wire = body.len() as u64;
        let text = if gzip {
            let mut out = Vec::with_capacity(body.len() * 5);
            flate2::read::MultiGzDecoder::new(&body[..])
                .read_to_end(&mut out)
                .map_err(|e| Attempt::Retry(anyhow!("bad gzip body: {e}")))?;
            out
        } else {
            body.to_vec()
        };
        if !status.is_success() {
            let snippet: String = String::from_utf8_lossy(&text).chars().take(300).collect();
            let err = anyhow!("HTTP {}: {}", status.as_u16(), self.mask(&snippet));
            return Err(match status.as_u16() {
                429 | 500..=599 => Attempt::Retry(err),
                403 => Attempt::Fatal(
                    err.context("forbidden (a date before the rolling 5-year edge?)"),
                ),
                _ => Attempt::Fatal(err),
            });
        }
        let page = serde_json::from_slice(&text)
            .map_err(|e| Attempt::Retry(anyhow!("bad JSON body: {e}")))?;
        Ok((page, wire))
    }

    fn get_page(&self, url: &Url) -> Result<(Page, u64)> {
        let mut last = None;
        for attempt in 0..=self.retry_delays.len() {
            if attempt > 0 {
                sleep(self.retry_delays[attempt - 1]);
            }
            match self.try_page(url) {
                Ok(p) => return Ok(p),
                Err(Attempt::Fatal(e)) => return Err(e),
                Err(Attempt::Retry(e)) => last = Some(e),
            }
        }
        Err(last.expect("at least one attempt").context(format!(
            "gave up after {} attempts",
            self.retry_delays.len() + 1
        )))
    }

    /// Every page of one ticker over the windows' date span, cut to the windows.
    pub fn fetch_ticker(&self, ticker: &str, win: &Windows) -> Result<Fetched> {
        let (d0, d1) = (&win.dates[0], &win.dates[win.len() - 1]);
        let mut url = self.first_url(ticker, d0, d1)?;
        let mut cols = Cols::default();
        let mut stats = Stats::default();
        loop {
            if stats.calls >= MAX_PAGES {
                bail!("{ticker}: more than {MAX_PAGES} pages");
            }
            let (page, wire) = self.get_page(&url)?;
            stats.calls += 1;
            stats.wire_bytes += wire;
            for bar in page.results.unwrap_or_default() {
                if win.index(bar.t).is_some() {
                    cols.push(&bar);
                    stats.bars += 1;
                } else {
                    stats.dropped += 1;
                }
            }
            match page.next_url {
                Some(next) if !next.is_empty() => url = self.with_key(&next)?,
                _ => break,
            }
        }
        cols.sort();
        Ok(Fetched { cols, stats })
    }
}
