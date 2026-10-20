//! Massive REST 1-second bars -> the agg_1s Arrow schema, and the Parquet writer settings.
//!
//! AGG_1S_SCHEMA (python/sourcing_py/massive_ws/rows.py) unchanged. From REST:
//!     c -> price, o/h/l -> open_i/high_i/low_i, v -> volume_i and volume_i_dec,
//!     vw -> vwap_i, round(v / n) -> avg_trade_size, otc = false.
//! volume_cum, volume_cum_dec, vwap_session and day_open stay null: in the WebSocket feed
//! they are Massive's running day totals, which include prints that are in no second
//! bar, so they cannot be rebuilt from these bars.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::Result;
use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, TimestampMillisecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use serde::Deserialize;

pub const SOURCE: &str = "massive_rest";
pub const SCHEMA_VERSION: &str = "2";
pub const ROW_GROUP: usize = 1_000_000;

/// The agg_1s schema with source=massive_rest metadata.
pub fn schema() -> SchemaRef {
    static S: OnceLock<SchemaRef> = OnceLock::new();
    S.get_or_init(|| {
        let f = |name: &str, t: DataType| Field::new(name, t, true);
        let fields = vec![
            f("symbol", DataType::Utf8),
            f(
                "ts_utc",
                DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            ),
            f("price", DataType::Float64),
            f("open_i", DataType::Float64),
            f("high_i", DataType::Float64),
            f("low_i", DataType::Float64),
            f("volume_i", DataType::Int64),
            f("volume_cum", DataType::Int64),
            f("volume_i_dec", DataType::Float64),
            f("volume_cum_dec", DataType::Float64),
            f("vwap_i", DataType::Float64),
            f("vwap_session", DataType::Float64),
            f("day_open", DataType::Float64),
            f("avg_trade_size", DataType::Int64),
            f("otc", DataType::Boolean),
        ];
        let meta = HashMap::from([
            (
                "agg_1s_schema_version".to_string(),
                SCHEMA_VERSION.to_string(),
            ),
            ("source".to_string(), SOURCE.to_string()),
        ]);
        Arc::new(Schema::new_with_metadata(fields, meta))
    })
    .clone()
}

/// One bar as Massive returns it (`t` is the bar start, ms UTC).
#[derive(Debug, Deserialize)]
pub struct Bar {
    pub t: i64,
    pub o: Option<f64>,
    pub h: Option<f64>,
    pub l: Option<f64>,
    pub c: Option<f64>,
    pub v: Option<f64>,
    pub vw: Option<f64>,
    pub n: Option<f64>,
}

/// One ticker's bars, column-wise.
#[derive(Default)]
pub struct Cols {
    ts: Vec<i64>,
    price: Vec<Option<f64>>,
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    volume: Vec<Option<i64>>,
    volume_dec: Vec<Option<f64>>,
    vwap: Vec<Option<f64>>,
    avg_size: Vec<Option<i64>>,
}

impl Cols {
    pub fn push(&mut self, b: &Bar) {
        self.ts.push(b.t);
        self.price.push(b.c);
        self.open.push(b.o);
        self.high.push(b.h);
        self.low.push(b.l);
        self.volume.push(b.v.map(|v| v.round() as i64));
        self.volume_dec.push(b.v);
        self.vwap.push(b.vw);
        self.avg_size.push(match (b.v, b.n) {
            (Some(v), Some(n)) if n > 0.0 => Some((v / n).round() as i64),
            _ => None,
        });
    }

    pub fn len(&self) -> usize {
        self.ts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }

    /// Put the bars in time order (Massive sends them sorted; this is a safeguard).
    pub fn sort(&mut self) {
        if self.ts.is_sorted() {
            return;
        }
        let mut idx: Vec<usize> = (0..self.len()).collect();
        idx.sort_by_key(|&i| self.ts[i]);
        fn take<T: Copy>(v: &mut Vec<T>, idx: &[usize]) {
            *v = idx.iter().map(|&i| v[i]).collect();
        }
        take(&mut self.ts, &idx);
        take(&mut self.price, &idx);
        take(&mut self.open, &idx);
        take(&mut self.high, &idx);
        take(&mut self.low, &idx);
        take(&mut self.volume, &idx);
        take(&mut self.volume_dec, &idx);
        take(&mut self.vwap, &idx);
        take(&mut self.avg_size, &idx);
    }

    pub fn into_batch(self, symbol: &str) -> Result<RecordBatch> {
        let n = self.len();
        let f64s = |v: Vec<Option<f64>>| Arc::new(Float64Array::from(v)) as ArrayRef;
        let cols: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![symbol; n])),
            Arc::new(TimestampMillisecondArray::from(self.ts).with_timezone("UTC")),
            f64s(self.price),
            f64s(self.open),
            f64s(self.high),
            f64s(self.low),
            Arc::new(Int64Array::from(self.volume)),
            Arc::new(Int64Array::new_null(n)),
            f64s(self.volume_dec),
            Arc::new(Float64Array::new_null(n)),
            f64s(self.vwap),
            Arc::new(Float64Array::new_null(n)),
            Arc::new(Float64Array::new_null(n)),
            Arc::new(Int64Array::from(self.avg_size)),
            Arc::new(BooleanArray::from(vec![false; n])),
        ];
        Ok(RecordBatch::try_new(schema(), cols)?)
    }
}

/// zstd at `level`, 1M-row groups, and the source/version as Parquet key-value metadata.
pub fn writer_props(level: i32) -> Result<WriterProperties> {
    let kv = vec![
        KeyValue::new(
            "agg_1s_schema_version".to_string(),
            SCHEMA_VERSION.to_string(),
        ),
        KeyValue::new("source".to_string(), SOURCE.to_string()),
    ];
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(level)?))
        .set_max_row_group_row_count(Some(ROW_GROUP))
        .set_key_value_metadata(Some(kv))
        .build())
}

/// Write one ticker's staged file (zstd-3, temp file then rename). Empty is fine: it
/// marks the ticker as fetched.
pub fn write_staged(path: &Path, batch: &RecordBatch) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("staged");
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    let mut w = ArrowWriter::try_new(File::create(&tmp)?, schema(), Some(writer_props(3)?))?;
    if batch.num_rows() > 0 {
        w.write(batch)?;
    }
    w.close()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Staged file name for a ticker (`/` and `%` escaped).
pub fn staged_name(ticker: &str) -> String {
    format!("{}.parquet", ticker.replace('%', "%25").replace('/', "%2F"))
}

/// The ticker of a staged file name, or None for anything else (temp files).
pub fn staged_ticker(file_name: &str) -> Option<String> {
    if file_name.starts_with('.') {
        return None;
    }
    let t = file_name.strip_suffix(".parquet")?;
    Some(t.replace("%2F", "/").replace("%25", "%"))
}
