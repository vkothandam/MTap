//! Staged per-ticker files -> one sorted day file per session.
//!
//! Pass 1 reads the staged files in ticker order and appends each ticker-day slice to a
//! per-date Arrow IPC bucket (LZ4), so every bucket is already sorted by (symbol, ts_utc)
//! and no global sort is needed. Pass 2 turns each bucket into
//! OUT/date=D/source=massive_rest/day-D.parquet (zstd, 1M-row groups), several dates in
//! parallel: temp file, row count checked against pass 1, fsync, rename. Then the job's
//! staging and buckets are removed and `done` is written.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use arrow::array::{AsArray, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::TimestampMillisecondType;
use arrow::ipc::CompressionType;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::bars::{SOURCE, schema, staged_name, writer_props};
use crate::inputs::Windows;
use crate::job::{gb, job_name, set_phase, staged};
use crate::log;

const BUCKET_FLUSH_ROWS: usize = 262_144;

pub fn day_path(out: &Path, date: &str) -> PathBuf {
    out.join(format!("date={date}"))
        .join(format!("source={SOURCE}"))
        .join(format!("day-{date}.parquet"))
}

struct Bucket {
    writer: Option<FileWriter<BufWriter<File>>>,
    pending: Vec<RecordBatch>,
    pending_rows: usize,
    rows: u64,
}

impl Bucket {
    fn flush(&mut self, path: &Path) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        if self.writer.is_none() {
            let opts = IpcWriteOptions::default()
                .try_with_compression(Some(CompressionType::LZ4_FRAME))?;
            let f = BufWriter::new(
                File::create(path).with_context(|| format!("creating {}", path.display()))?,
            );
            self.writer = Some(FileWriter::try_new_with_options(f, &schema(), opts)?);
        }
        let batch = concat_batches(&schema(), &self.pending)?;
        self.writer.as_mut().expect("opened above").write(&batch)?;
        self.pending.clear();
        self.pending_rows = 0;
        Ok(())
    }
}

/// Pass 1: route staged rows to per-date buckets. Returns rows per date.
fn bucket(staging: &Path, buckets: &Path, win: &Windows) -> Result<Vec<u64>> {
    let path = |i: usize| buckets.join(format!("{}.arrow", win.dates[i]));
    let mut bs: Vec<Bucket> = (0..win.len())
        .map(|_| Bucket {
            writer: None,
            pending: vec![],
            pending_rows: 0,
            rows: 0,
        })
        .collect();
    for ticker in staged(staging)? {
        let file = staging.join(staged_name(&ticker));
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&file)?)?
            .with_batch_size(65_536)
            .build()?;
        for batch in reader {
            let batch = RecordBatch::try_new(schema(), batch?.columns().to_vec())
                .with_context(|| format!("{} does not have the agg_1s schema", file.display()))?;
            let ts = batch
                .column(1)
                .as_primitive::<TimestampMillisecondType>()
                .values();
            let mut start = 0;
            while start < ts.len() {
                let Some(d) = win.index(ts[start]) else {
                    bail!(
                        "{}: bar at {} ms is outside every window",
                        file.display(),
                        ts[start]
                    );
                };
                let end = start + ts[start..].partition_point(|&t| t < win.end[d]);
                let b = &mut bs[d];
                b.pending.push(batch.slice(start, end - start));
                b.pending_rows += end - start;
                b.rows += (end - start) as u64;
                if b.pending_rows >= BUCKET_FLUSH_ROWS {
                    b.flush(&path(d))?;
                }
                start = end;
            }
        }
    }
    let mut rows = vec![];
    for (i, mut b) in bs.into_iter().enumerate() {
        b.flush(&path(i))?;
        if let Some(mut w) = b.writer.take() {
            w.finish()?;
        }
        rows.push(b.rows);
    }
    Ok(rows)
}

/// Pass 2 for one date: bucket -> checked day file. Returns its size in bytes.
fn write_day(bucket: &Path, dest: &Path, rows: u64, zstd_level: i32) -> Result<u64> {
    std::fs::create_dir_all(dest.parent().expect("day path has a parent"))?;
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .expect("day file name");
    let tmp = dest.with_file_name(format!(".{name}.tmp"));
    let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut w = ArrowWriter::try_new(file.try_clone()?, schema(), Some(writer_props(zstd_level)?))?;
    for batch in FileReader::try_new(BufReader::new(File::open(bucket)?), None)? {
        w.write(&batch?)?;
    }
    w.close()?;
    file.sync_all()?;
    let got = ParquetRecordBatchReaderBuilder::try_new(File::open(&tmp)?)?
        .metadata()
        .file_metadata()
        .num_rows() as u64;
    if got != rows {
        bail!(
            "{}: wrote {got} rows, expected {rows}; left as {}",
            dest.display(),
            tmp.display()
        );
    }
    std::fs::rename(&tmp, dest)?;
    Ok(std::fs::metadata(dest)?.len())
}

/// Turn a fully staged job into day files under `out`, then mark it done.
pub fn finalize(
    dir: &Path,
    win: &Windows,
    out: &Path,
    zstd_level: i32,
    threads: usize,
) -> Result<()> {
    let t0 = Instant::now();
    let name = job_name(dir);
    let staging = dir.join("staging");
    let buckets = dir.join("buckets");
    let _ = std::fs::remove_dir_all(&buckets); // left by an interrupted finalize
    std::fs::create_dir_all(&buckets)?;
    set_phase(dir, "finalize");
    log!("finalize {name}: pass 1 (per-date buckets)");
    let rows = bucket(&staging, &buckets, win)?;
    log!(
        "finalize {name}: {:.1}M bars in {} s; pass 2 (zstd-{zstd_level}, {threads} threads)",
        rows.iter().sum::<u64>() as f64 / 1e6,
        t0.elapsed().as_secs()
    );

    let todo: Vec<usize> = (0..win.len()).filter(|&i| rows[i] > 0).collect();
    for i in (0..win.len()).filter(|&i| rows[i] == 0) {
        log!("  {}: no bars at all, no file written", win.dates[i]);
    }
    let next = AtomicUsize::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(vec![]);
    let bytes = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, todo.len().max(1)) {
            s.spawn(|| {
                while let Some(&i) = todo.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let (d, n) = (&win.dates[i], rows[i]);
                    let bucket = buckets.join(format!("{d}.arrow"));
                    match write_day(&bucket, &day_path(out, d), n, zstd_level) {
                        Ok(size) => {
                            bytes.fetch_add(size as usize, Ordering::Relaxed);
                            let _ = std::fs::remove_file(&bucket);
                            log!(
                                "  {d}: {n} bars, {:.1} MB ({:.1} B/bar)",
                                size as f64 / 1e6,
                                size as f64 / n as f64
                            );
                        }
                        Err(e) => errors.lock().unwrap().push(format!("{d}: {e:#}")),
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        set_phase(dir, "finalize-failed");
        bail!(
            "finalize {name}: {} day file(s) failed: {}",
            errors.len(),
            errors.join("; ")
        );
    }
    let total: u64 = rows.iter().sum();
    let size = bytes.into_inner() as u64;
    let done = serde_json::json!({
        "job": name, "day_files": todo.len(), "bars": total, "bytes": size,
        "finished_utc": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    });
    crate::write_atomic(
        &dir.join("done"),
        serde_json::to_string_pretty(&done)?.as_bytes(),
    )?;
    set_phase(dir, "done");
    std::fs::remove_dir_all(&staging)?;
    std::fs::remove_dir_all(&buckets)?;
    log!(
        "finalize {name}: {} day files, {:.1}M bars, {} ({:.1} B/bar) in {} s; {}",
        todo.len(),
        total as f64 / 1e6,
        gb(size),
        size as f64 / total.max(1) as f64,
        t0.elapsed().as_secs(),
        crate::resources()
    );
    Ok(())
}
