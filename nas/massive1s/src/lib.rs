//! Massive 1-second bars per common stock -> `agg_1s` day files, run on the NAS.
//!
//! Inputs (written on the Mac by `sourcing-py massive-1s prepare`):
//!     DATA/massive_1s/inputs/universe.csv   ticker,name,exchange,kept,reason
//!     DATA/massive_1s/inputs/windows.csv    date,start_utc_ms,end_utc_ms
//!
//! The range is split into calendar-quarter jobs under DATA/massive_1s/jobs/<d0>_<d1>/.
//! Each ticker is one range call (paged by `next_url`); its bars inside the day windows
//! are staged as staging/<T>.parquet, so a rerun resumes. Once every ticker of a job is
//! staged, finalize writes DATA/agg_1s/date=D/source=massive_rest/day-D.parquet (the
//! AGG_1S_SCHEMA of python/sourcing_py/massive_ws/rows.py) and marks the job `done`.
//! See nas/README.md.

pub mod api;
pub mod bars;
pub mod finalize;
pub mod inputs;
pub mod job;

/// A UTC-stamped line on stdout (run.sh sends it to the log file).
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {
        println!("{} {}", chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ"), format!($($arg)*))
    };
}

/// Write `bytes` to `path` via a hidden temp file and a rename.
pub fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Peak memory and CPU time of this process so far (Linux /proc; "n/a" elsewhere).
pub fn resources() -> String {
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        });
    let cpu = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| {
            // fields after the ")" that closes the command name: utime is the 12th, stime the 13th
            let rest: Vec<&str> = s.rsplit_once(')')?.1.split_whitespace().collect();
            Some((rest.get(11)?.parse::<u64>().ok()? + rest.get(12)?.parse::<u64>().ok()?) / 100)
        });
    match (rss, cpu) {
        (Some(kb), Some(s)) => format!("peak RSS {} MB, CPU {s} s", kb / 1024),
        _ => "resources n/a".to_string(),
    }
}
