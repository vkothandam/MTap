//! End-to-end against a local mock of the Massive endpoint: window cut, field mapping,
//! paging, retries, resume after a failure, and the finalized day files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::{Float64Type, Int64Type, TimestampMillisecondType};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use massive1s::api::Api;
use massive1s::bars::{Bar, Cols, schema, staged_name, staged_ticker};
use massive1s::finalize::day_path;
use massive1s::inputs::{Windows, load_universe};
use massive1s::job::{Layout, Opts, fetch_job, run};

// 2025-11-26 (14:00-21:07 UTC) and the 2025-11-28 half day (14:00-18:07 UTC)
const D1: &str = "2025-11-26";
const D2: &str = "2025-11-28";
const S1: i64 = 1_764_165_600_000;
const E1: i64 = 1_764_191_220_000;
const S2: i64 = 1_764_338_400_000;
const E2: i64 = 1_764_353_220_000;

fn windows() -> Windows {
    Windows::from_rows(vec![(D1.into(), S1, E1), (D2.into(), S2, E2)]).unwrap()
}

fn bar(t: i64, c: f64, v: f64, n: f64) -> String {
    format!(r#"{{"v":{v},"vw":{c},"o":{c},"c":{c},"h":{c},"l":{c},"t":{t},"n":{n}}}"#)
}

type Handler = dyn Fn(&str, &str) -> (u16, Vec<u8>, bool) + Send + Sync;

/// Serves `handler(base, target)` -> (status, body, gzip); records every request target.
fn serve(handler: Arc<Handler>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let log = Arc::new(Mutex::new(vec![]));
    let (b, l) = (base.clone(), log.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let (b, l, h) = (b.clone(), l.clone(), handler.clone());
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                loop {
                    let mut hdr = String::new();
                    reader.read_line(&mut hdr).unwrap();
                    if hdr.trim().is_empty() {
                        break;
                    }
                }
                let target = line.split_whitespace().nth(1).unwrap_or("").to_string();
                l.lock().unwrap().push(target.clone());
                let (status, body, gzip) = h(&b, &target);
                let body = if gzip {
                    let mut e =
                        flate2::write::GzEncoder::new(vec![], flate2::Compression::default());
                    e.write_all(&body).unwrap();
                    e.finish().unwrap()
                } else {
                    body
                };
                let enc = if gzip {
                    "Content-Encoding: gzip\r\n"
                } else {
                    ""
                };
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n{enc}Content-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(head.as_bytes()).unwrap();
                stream.write_all(&body).unwrap();
            });
        }
    });
    (base, log)
}

fn fast(mut api: Api) -> Api {
    api.retry_delays = vec![Duration::from_millis(5); 3];
    api
}

fn read_day(path: &Path) -> RecordBatch {
    let b = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
    let schema = b.schema().clone();
    let batches: Vec<RecordBatch> = b.build().unwrap().map(Result::unwrap).collect();
    concat_batches(&schema, &batches).unwrap()
}

#[test]
fn window_edges() {
    let w = windows();
    assert_eq!(w.index(S1 - 1), None); // 08:59:59.999
    assert_eq!(w.index(S1), Some(0)); // 09:00:00
    assert_eq!(w.index(E1 - 1000), Some(0)); // 16:06:59
    assert_eq!(w.index(E1), None); // 16:07:00
    assert_eq!(w.index(E2 - 1000), Some(1)); // 13:06:59 on the half day
    assert_eq!(w.index(E2), None);
    assert_eq!(w.index(i64::MIN), None);
}

#[test]
fn windows_validate_and_split_by_quarter() {
    assert!(Windows::from_rows(vec![(D2.into(), S2, E2), (D1.into(), S1, E1)]).is_err());
    assert!(Windows::from_rows(vec![(D1.into(), E1, S1)]).is_err());
    let w = Windows::from_rows(vec![
        ("2026-09-29".into(), 10, 20),
        ("2026-09-30".into(), 30, 40),
        ("2026-10-01".into(), 50, 60),
    ])
    .unwrap();
    let q: Vec<String> = w.quarters().iter().map(Windows::name).collect();
    assert_eq!(q, ["2026-09-29_2026-09-30", "2026-10-01_2026-10-01"]);
    assert_eq!(
        w.between(Some("2026-09-30"), None).dates,
        ["2026-09-30", "2026-10-01"]
    );
}

#[test]
fn universe_keeps_only_kept_rows() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("universe.csv");
    std::fs::write(&p, "ticker,name,exchange,kept,reason\nZZ,Z Corp,XNYS,1,\nAA,\"A, Inc\",XNAS,1,\nAB,A Notes due 2030,XNYS,0,note\n").unwrap();
    assert_eq!(load_universe(&p).unwrap(), ["AA", "ZZ"]);
}

#[test]
fn staged_names_round_trip() {
    for t in ["AAPL", "BRK.B", "A/B", "X%Y"] {
        assert_eq!(staged_ticker(&staged_name(t)).as_deref(), Some(t));
    }
    assert_eq!(staged_ticker(".AAPL.parquet.tmp"), None);
}

#[test]
fn bar_mapping_and_null_running_fields() {
    let mut c = Cols::default();
    let b: Bar = serde_json::from_str(&bar(S1 + 1000, 10.5, 100.0, 3.0)).unwrap();
    c.push(&b);
    let b: Bar = serde_json::from_str(&bar(S1, 10.0, 7.0, 0.0)).unwrap();
    c.push(&b);
    c.sort();
    let batch = c.into_batch("AAA").unwrap();
    assert_eq!(batch.schema(), schema());
    assert_eq!(batch.schema().metadata()["source"], "massive_rest");
    let ts = batch.column(1).as_primitive::<TimestampMillisecondType>();
    assert_eq!(ts.values().to_vec(), [S1, S1 + 1000]);
    assert_eq!(batch.column(2).as_primitive::<Float64Type>().value(1), 10.5);
    assert_eq!(batch.column(6).as_primitive::<Int64Type>().value(1), 100);
    assert_eq!(
        batch.column(8).as_primitive::<Float64Type>().value(1),
        100.0
    );
    let ats = batch.column(13).as_primitive::<Int64Type>();
    assert_eq!((ats.is_null(0), ats.value(1)), (true, 33)); // n = 0 -> null; round(100 / 3)
    for i in [7, 9, 11, 12] {
        assert_eq!(batch.column(i).null_count(), 2, "column {i} must be null");
    }
    assert_eq!(batch.column(14).as_boolean().true_count(), 0);
}

#[test]
fn pages_retries_and_window_cut() {
    let calls = Arc::new(Mutex::new(0));
    let c = calls.clone();
    let (base, log) = serve(Arc::new(move |base: &str, target: &str| {
        let mut n = c.lock().unwrap();
        *n += 1;
        if *n == 1 {
            return (503, b"busy".to_vec(), false);
        }
        if !target.contains("cursor=") {
            let body = format!(
                r#"{{"results":[{},{},{}],"next_url":"{base}/v2/aggs/ticker/AAA/range/1/second/x/y?cursor=abc&apiKey=OLD"}}"#,
                bar(S1 - 1000, 1.0, 1.0, 1.0), // 08:59:59: dropped
                bar(S1, 2.0, 10.0, 2.0),
                bar(E1 - 1000, 3.0, 10.0, 2.0),
            );
            (200, body.into_bytes(), true)
        } else {
            let body = format!(
                r#"{{"results":[{},{}]}}"#,
                bar(E1, 4.0, 1.0, 1.0),
                bar(S2, 5.0, 1.0, 1.0)
            );
            (200, body.into_bytes(), false)
        }
    }));
    let api = fast(Api::new(&base, "KEY123").unwrap());
    let f = api.fetch_ticker("AAA", &windows()).unwrap();
    assert_eq!((f.stats.calls, f.stats.bars, f.stats.dropped), (2, 3, 2));
    assert!(f.stats.wire_bytes > 0);
    let log = log.lock().unwrap();
    assert_eq!(log.len(), 3); // 503, page 1, page 2
    assert!(log[1].starts_with(&format!("/v2/aggs/ticker/AAA/range/1/second/{D1}/{D2}?adjusted=true&sort=asc&limit=50000&apiKey=KEY123")));
    assert_eq!(
        log[2],
        "/v2/aggs/ticker/AAA/range/1/second/x/y?cursor=abc&apiKey=KEY123"
    );
}

#[test]
fn forbidden_fails_fast_and_masks_the_key() {
    let (base, log) = serve(Arc::new(|_: &str, _: &str| {
        (
            403,
            br#"{"status":"NOT_AUTHORIZED","apiKey":"KEY123"}"#.to_vec(),
            false,
        )
    }));
    let api = fast(Api::new(&base, "KEY123").unwrap());
    let err = format!("{:#}", api.fetch_ticker("AAA", &windows()).err().unwrap());
    assert!(err.contains("5-year edge") && err.contains("403"), "{err}");
    assert!(!err.contains("KEY123"), "{err}");
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[test]
fn job_resumes_after_a_failure_and_writes_sorted_day_files() {
    let fail_bbb = Arc::new(Mutex::new(true));
    let f = fail_bbb.clone();
    let (base, log) = serve(Arc::new(move |_: &str, target: &str| {
        let t = target.split('/').nth(4).unwrap_or("");
        let bars = match t {
            "AAA" => vec![
                bar(S1 + 5000, 1.0, 5.0, 1.0),
                bar(S2, 1.1, 6.0, 2.0),
                bar(E2 + 1000, 9.0, 1.0, 1.0),
            ],
            "BBB" if *f.lock().unwrap() => return (400, b"bad".to_vec(), false),
            "BBB" => vec![bar(S1, 2.0, 7.0, 1.0), bar(S1 + 1000, 2.1, 8.0, 1.0)],
            _ => vec![], // CCC: no bars in the windows
        };
        (
            200,
            format!(r#"{{"results":[{}]}}"#, bars.join(",")).into_bytes(),
            false,
        )
    }));
    let api = fast(Api::new(&base, "K").unwrap());
    let data = tempfile::tempdir().unwrap();
    let layout = Layout::full(data.path());
    let opts = Opts {
        workers: 4,
        zstd_level: 3,
        finalize_threads: 2,
    };
    let tickers: Vec<String> = ["AAA", "BBB", "CCC"].map(String::from).to_vec();

    assert!(!run(&api, &layout, &windows(), &tickers, &opts).unwrap());
    let job = layout.jobs.join(format!("{D1}_{D2}"));
    let failures = std::fs::read_to_string(job.join("failures.jsonl")).unwrap();
    assert!(
        failures.contains("BBB") && !job.join("done").exists(),
        "{failures}"
    );
    assert!(!day_path(&layout.out, D1).exists());

    *fail_bbb.lock().unwrap() = false;
    log.lock().unwrap().clear();
    assert!(run(&api, &layout, &windows(), &tickers, &opts).unwrap());
    assert_eq!(log.lock().unwrap().len(), 1, "only BBB is fetched again");
    assert!(
        job.join("done").exists() && !job.join("staging").exists() && !job.join("buckets").exists()
    );

    let d1 = read_day(&day_path(&layout.out, D1));
    let syms: Vec<&str> = d1
        .column(0)
        .as_string::<i32>()
        .iter()
        .map(Option::unwrap)
        .collect();
    assert_eq!(syms, ["AAA", "BBB", "BBB"]);
    let ts = d1
        .column(1)
        .as_primitive::<TimestampMillisecondType>()
        .values()
        .to_vec();
    assert_eq!(ts, [S1 + 5000, S1, S1 + 1000]); // sorted by (symbol, ts_utc)
    assert_eq!(
        d1.schema().metadata().get("source").map(String::as_str),
        Some("massive_rest")
    );
    let d2 = read_day(&day_path(&layout.out, D2));
    assert_eq!(d2.num_rows(), 1); // AAA's 13:07:01 bar is outside the half-day window

    log.lock().unwrap().clear();
    assert!(run(&api, &layout, &windows(), &tickers, &opts).unwrap());
    assert!(log.lock().unwrap().is_empty(), "a done job is skipped");

    // the contract fixture for python/tests/test_massive_1s.py
    if let Ok(out) = std::env::var("MASSIVE1S_FIXTURE_OUT") {
        std::fs::copy(day_path(&layout.out, D1), out).unwrap();
    }
}

#[test]
fn staged_job_counts_failures() {
    let (base, _) = serve(Arc::new(|_: &str, _: &str| (500, b"down".to_vec(), false)));
    let api = fast(Api::new(&base, "K").unwrap());
    let data = tempfile::tempdir().unwrap();
    let failed = fetch_job(&api, data.path(), &windows(), &["AAA".to_string()], 2).unwrap();
    assert_eq!(failed, 1);
    let p: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&std::fs::read(data.path().join("progress.json")).unwrap()).unwrap();
    assert_eq!(p["phase"], "fetch-failed");
    assert_eq!(p["failed"], 1);
}

#[test]
fn sessions_with_a_day_file_are_left_out_and_a_started_job_keeps_its_sessions() {
    let fail = Arc::new(Mutex::new(true));
    let f = fail.clone();
    let (base, log) = serve(Arc::new(move |_: &str, _: &str| {
        if *f.lock().unwrap() {
            return (400, b"bad".to_vec(), false);
        }
        (
            200,
            format!(r#"{{"results":[{}]}}"#, bar(S2, 1.0, 1.0, 1.0)).into_bytes(),
            false,
        )
    }));
    let api = fast(Api::new(&base, "K").unwrap());
    let data = tempfile::tempdir().unwrap();
    let layout = Layout::full(data.path());
    let opts = Opts {
        workers: 1,
        zstd_level: 3,
        finalize_threads: 1,
    };
    let tickers = vec!["AAA".to_string()];
    let d1 = day_path(&layout.out, D1);
    std::fs::create_dir_all(d1.parent().unwrap()).unwrap();
    std::fs::write(&d1, b"from an earlier run").unwrap();

    assert!(!run(&api, &layout, &windows(), &tickers, &opts).unwrap());
    let job = layout.jobs.join(format!("{D1}_{D2}")); // still named after the whole quarter
    assert_eq!(Windows::load(&job.join("windows.csv")).unwrap().dates, [D2]);
    // the D1 file goes away before the resume: the job still fetches only D2
    std::fs::remove_file(&d1).unwrap();
    *fail.lock().unwrap() = false;
    log.lock().unwrap().clear();
    assert!(run(&api, &layout, &windows(), &tickers, &opts).unwrap());
    let targets = log.lock().unwrap().clone();
    assert_eq!(targets.len(), 1);
    assert!(
        targets[0].contains(&format!("/{D2}/{D2}?")),
        "{}",
        targets[0]
    );
    assert!(!d1.exists() && read_day(&day_path(&layout.out, D2)).num_rows() == 1);

    // every session already on disk: nothing to fetch
    let data = tempfile::tempdir().unwrap();
    let layout = Layout::full(data.path());
    for d in [D1, D2] {
        let p = day_path(&layout.out, d);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"x").unwrap();
    }
    log.lock().unwrap().clear();
    assert!(run(&api, &layout, &windows(), &tickers, &opts).unwrap());
    assert!(log.lock().unwrap().is_empty() && !layout.jobs.join(format!("{D1}_{D2}")).exists());
}
