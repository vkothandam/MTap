//! The two CSV inputs: the stock list and the per-session time windows.

use std::path::Path;

use anyhow::{Context, Result, bail};
use chrono::{Datelike, NaiveDate};
use serde::Deserialize;

/// One [start, end) UTC-millisecond window per session, in date order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Windows {
    pub dates: Vec<String>,
    pub start: Vec<i64>,
    pub end: Vec<i64>,
}

impl Windows {
    pub fn from_rows(rows: Vec<(String, i64, i64)>) -> Result<Self> {
        let mut w = Windows::default();
        for (d, s, e) in rows {
            NaiveDate::parse_from_str(&d, "%Y-%m-%d").with_context(|| format!("bad date {d:?}"))?;
            if s >= e {
                bail!("{d}: window start {s} is not before end {e}");
            }
            if let (Some(pd), Some(&pe)) = (w.dates.last(), w.end.last())
                && (d <= *pd || s < pe)
            {
                bail!("{d}: windows must be in date order and not overlap ({pd} before it)");
            }
            w.dates.push(d);
            w.start.push(s);
            w.end.push(e);
        }
        Ok(w)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut rdr =
            csv::Reader::from_path(path).with_context(|| format!("reading {}", path.display()))?;
        let rows = rdr
            .deserialize()
            .collect::<Result<Vec<(String, i64, i64)>, _>>()?;
        Self::from_rows(rows).with_context(|| format!("in {}", path.display()))
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let mut w = csv::Writer::from_writer(vec![]);
        w.write_record(["date", "start_utc_ms", "end_utc_ms"])?;
        for i in 0..self.len() {
            w.serialize((&self.dates[i], self.start[i], self.end[i]))?;
        }
        crate::write_atomic(path, &w.into_inner()?)?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.dates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }

    /// The session whose window holds `t` (ms UTC), if any.
    pub fn index(&self, t: i64) -> Option<usize> {
        let i = self.start.partition_point(|&s| s <= t);
        (i > 0 && t < self.end[i - 1]).then(|| i - 1)
    }

    /// The sessions with from <= date <= to (ISO dates compare as strings).
    pub fn between(&self, from: Option<&str>, to: Option<&str>) -> Windows {
        self.select(|d| from.is_none_or(|f| d >= f) && to.is_none_or(|t| d <= t))
    }

    /// The sessions whose date passes `keep`.
    pub fn select(&self, keep: impl Fn(&str) -> bool) -> Windows {
        let mut w = Windows::default();
        for i in (0..self.len()).filter(|&i| keep(&self.dates[i])) {
            w.dates.push(self.dates[i].clone());
            w.start.push(self.start[i]);
            w.end.push(self.end[i]);
        }
        w
    }

    /// Split into calendar quarters (the jobs), in order.
    pub fn quarters(&self) -> Vec<Windows> {
        let quarter = |d: &str| {
            let d = NaiveDate::parse_from_str(d, "%Y-%m-%d").expect("validated on load");
            (d.year(), d.month0() / 3)
        };
        let mut keys: Vec<(i32, u32)> = self.dates.iter().map(|d| quarter(d)).collect();
        keys.dedup();
        keys.into_iter()
            .map(|k| self.select(|d| quarter(d) == k))
            .collect()
    }

    /// `<first>_<last>`, the job folder name.
    pub fn name(&self) -> String {
        format!("{}_{}", self.dates[0], self.dates[self.len() - 1])
    }
}

#[derive(Deserialize)]
struct UniverseRow {
    ticker: String,
    kept: u8,
}

/// The kept tickers of universe.csv, sorted and unique.
pub fn load_universe(path: &Path) -> Result<Vec<String>> {
    let mut rdr =
        csv::Reader::from_path(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = vec![];
    for row in rdr.deserialize() {
        let r: UniverseRow = row.with_context(|| format!("in {}", path.display()))?;
        if r.kept == 1 {
            out.push(r.ticker);
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}
