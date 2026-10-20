//! Calendar-quarter jobs: fetch every ticker into staging/, then finalize.
//!
//! ```text
//! jobs/<d0>_<d1>/windows.csv      this job's sessions (finalize reads only this)
//! jobs/<d0>_<d1>/staging/<T>.parquet
//! jobs/<d0>_<d1>/progress.json    rewritten every 100 tickers
//! jobs/<d0>_<d1>/failures.jsonl   tickers that failed this run (key masked)
//! jobs/<d0>_<d1>/done             written by finalize; a done job is skipped
//! ```
//!
//! A rerun fetches only the tickers with no staged file, so failures are retried and an
//! interrupted run resumes. Sessions that already have a day file when a job starts are
//! left out of it. A job is finalized only when every ticker is staged; the
//! finalize of one quarter runs while the next quarter is fetched.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::api::{Api, Stats};
use crate::bars::{staged_name, staged_ticker, write_staged};
use crate::inputs::Windows;
use crate::{finalize, log};

/// Where jobs and day files go. A `--tickers` run is a sample: it gets its own folders so
/// it can never replace a full day file with a partial one.
pub struct Layout {
    pub jobs: PathBuf,
    pub out: PathBuf,
}

impl Layout {
    pub fn full(data: &Path) -> Self {
        Layout {
            jobs: data.join("massive_1s/jobs"),
            out: data.join("agg_1s"),
        }
    }

    pub fn sample(data: &Path) -> Self {
        Layout {
            jobs: data.join("massive_1s/sample/jobs"),
            out: data.join("massive_1s/sample/agg_1s"),
        }
    }
}

pub struct Opts {
    pub workers: usize,
    pub zstd_level: i32,
    pub finalize_threads: usize,
}

#[derive(Serialize)]
struct Progress {
    job: String,
    phase: &'static str,
    tickers: usize,
    staged_before: usize,
    done: usize,
    failed: usize,
    calls: u64,
    bars: u64,
    dropped_outside_window: u64,
    wire_bytes: u64,
    started_utc: String,
    updated_utc: String,
    elapsed_s: u64,
}

impl Progress {
    fn add(&mut self, s: Stats) {
        self.calls += s.calls;
        self.bars += s.bars;
        self.dropped_outside_window += s.dropped;
        self.wire_bytes += s.wire_bytes;
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn write_progress(dir: &Path, p: &mut Progress, t0: Instant) {
    p.updated_utc = now();
    p.elapsed_s = t0.elapsed().as_secs();
    let body = serde_json::to_vec_pretty(p).expect("progress serializes");
    if let Err(e) = crate::write_atomic(&dir.join("progress.json"), &body) {
        log!("warning: could not write progress.json: {e}");
    }
}

/// Set `phase` in an existing progress.json (finalize's steps).
pub fn set_phase(dir: &Path, phase: &str) {
    let path = dir.join("progress.json");
    let Ok(mut v) = std::fs::read(&path)
        .map_err(drop)
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).map_err(drop))
    else {
        return;
    };
    v["phase"] = phase.into();
    v["updated_utc"] = now().into();
    let _ = crate::write_atomic(&path, &serde_json::to_vec_pretty(&v).expect("json"));
}

pub fn staged(staging: &Path) -> Result<Vec<String>> {
    let mut out = vec![];
    for e in std::fs::read_dir(staging)? {
        if let Some(t) = e?.file_name().to_str().and_then(staged_ticker) {
            out.push(t);
        }
    }
    out.sort();
    Ok(out)
}

/// The job folder's name (`<d0>_<d1>` of its whole quarter within the run's range).
pub fn job_name(dir: &Path) -> String {
    dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

pub fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

/// Fetch the tickers of one job not yet staged. Returns how many failed.
pub fn fetch_job(
    api: &Api,
    dir: &Path,
    win: &Windows,
    tickers: &[String],
    workers: usize,
) -> Result<usize> {
    let staging = dir.join("staging");
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;
    win.write(&dir.join("windows.csv"))?;
    let name = job_name(dir);
    let have: HashSet<String> = staged(&staging)?.into_iter().collect();
    let todo: Vec<&String> = tickers.iter().filter(|t| !have.contains(*t)).collect();
    let staged_before = tickers.len() - todo.len();
    log!(
        "job {name}: {} sessions, {} tickers, {} already staged, {} to fetch",
        win.len(),
        tickers.len(),
        staged_before,
        todo.len()
    );

    let t0 = Instant::now();
    let progress = Mutex::new(Progress {
        job: name.clone(),
        phase: "fetch",
        tickers: tickers.len(),
        staged_before,
        done: staged_before,
        failed: 0,
        calls: 0,
        bars: 0,
        dropped_outside_window: 0,
        wire_bytes: 0,
        started_utc: now(),
        updated_utc: now(),
        elapsed_s: 0,
    });
    let failures: Mutex<Vec<(String, String)>> = Mutex::new(vec![]);
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..workers.clamp(1, todo.len().max(1)) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&t) = todo.get(i) else { break };
                    let res = api.fetch_ticker(t, win).and_then(|f| {
                        write_staged(&staging.join(staged_name(t)), &f.cols.into_batch(t)?)?;
                        Ok(f.stats)
                    });
                    let mut p = progress.lock().unwrap();
                    match res {
                        Ok(st) => {
                            p.add(st);
                            p.done += 1;
                        }
                        Err(e) => {
                            p.failed += 1;
                            failures
                                .lock()
                                .unwrap()
                                .push((t.clone(), api.mask(&format!("{e:#}"))));
                        }
                    }
                    let n = p.done + p.failed - staged_before;
                    if n.is_multiple_of(100) {
                        write_progress(dir, &mut p, t0);
                    }
                    if n.is_multiple_of(500) {
                        let secs = t0.elapsed().as_secs_f64().max(1.0);
                        log!(
                            "  {n}/{} fetched, {} failed, {} calls, {:.1}M bars, {} ({:.1} MB/s)",
                            todo.len(),
                            p.failed,
                            p.calls,
                            p.bars as f64 / 1e6,
                            gb(p.wire_bytes),
                            p.wire_bytes as f64 / 1e6 / secs
                        );
                    }
                }
            });
        }
    });

    let mut p = progress.into_inner().unwrap();
    let failures = failures.into_inner().unwrap();
    if !failures.is_empty() {
        p.phase = "fetch-failed";
    }
    write_progress(dir, &mut p, t0);
    let lines: String = failures
        .iter()
        .map(|(t, e)| serde_json::json!({"ticker": t, "error": e}).to_string() + "\n")
        .collect();
    crate::write_atomic(&dir.join("failures.jsonl"), lines.as_bytes())?;
    log!(
        "job {name}: fetched {} in {} s: {} calls, {:.1}M bars kept, {:.1}M outside the window, {} downloaded; {}",
        todo.len() - failures.len(),
        p.elapsed_s,
        p.calls,
        p.bars as f64 / 1e6,
        p.dropped_outside_window as f64 / 1e6,
        gb(p.wire_bytes),
        crate::resources()
    );
    for (t, e) in failures.iter().take(5) {
        log!("  failed {t}: {}", e.chars().take(300).collect::<String>());
    }
    Ok(failures.len())
}

/// Every quarter of `win`: fetch, then finalize in the background. True when all done.
pub fn run(
    api: &Api,
    layout: &Layout,
    win: &Windows,
    tickers: &[String],
    opts: &Opts,
) -> Result<bool> {
    let jobs = win.quarters();
    log!(
        "massive1s: {} tickers, {} sessions ({} .. {}), {} job(s), {} workers, zstd-{}",
        tickers.len(),
        win.len(),
        win.dates[0],
        win.dates[win.len() - 1],
        jobs.len(),
        opts.workers,
        opts.zstd_level
    );
    let mut all_ok = true;
    std::thread::scope(|s| -> Result<()> {
        let mut pending: Option<std::thread::ScopedJoinHandle<'_, Result<()>>> = None;
        for q in jobs {
            let name = q.name();
            let dir = layout.jobs.join(&name);
            if dir.join("done").exists() {
                log!("job {name}: already done, skipped");
                continue;
            }
            // Sessions that already have a day file (an earlier run, e.g. the one-month
            // test) are left out. A started job keeps its windows.csv, so a resume
            // fetches the same sessions even after some of its day files were written.
            let started = dir.join("windows.csv");
            let jw = if started.exists() {
                Windows::load(&started)?
            } else {
                q.select(|d| !finalize::day_path(&layout.out, d).exists())
            };
            if jw.is_empty() {
                log!(
                    "job {name}: all {} sessions already have a day file, skipped",
                    q.len()
                );
                continue;
            }
            if jw.len() < q.len() {
                log!(
                    "job {name}: {} of {} sessions already have a day file; fetching the other {}",
                    q.len() - jw.len(),
                    q.len(),
                    jw.len()
                );
            }
            let failed = fetch_job(api, &dir, &jw, tickers, opts.workers)?;
            if failed > 0 {
                all_ok = false;
                log!(
                    "job {name}: {failed} ticker(s) failed; not finalized. Run again to retry them."
                );
                continue;
            }
            if let Some(h) = pending.take() {
                h.join().expect("finalize thread panicked")?;
            }
            let out = &layout.out;
            pending = Some(s.spawn(move || {
                finalize::finalize(&dir, &jw, out, opts.zstd_level, opts.finalize_threads)
            }));
        }
        if let Some(h) = pending.take() {
            h.join().expect("finalize thread panicked")?;
        }
        Ok(())
    })?;
    Ok(all_ok)
}
