//! massive1s fetch | finalize | status. See nas/README.md.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};

use massive1s::inputs::{Windows, load_universe};
use massive1s::job::{self, Layout, Opts};
use massive1s::{api, finalize, log};

#[derive(Parser)]
#[command(
    version,
    about = "Massive 1-second bars -> agg_1s day files (source=massive_rest)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct FinalizeOpts {
    /// zstd level of the day files (the live feed's compacted files use 19)
    #[arg(long, default_value_t = 19)]
    zstd_level: i32,
    /// Day files written in parallel
    #[arg(long, default_value_t = 3)]
    finalize_threads: usize,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fetch every stock of universe.csv over windows.csv (or --from/--to within it)
    Fetch {
        /// The MTap-Data folder
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        from: Option<String>,
        #[arg(long)]
        to: Option<String>,
        /// Concurrent requests
        #[arg(long, default_value_t = 32)]
        workers: usize,
        /// Only these tickers (a sample: written under massive_1s/sample/, never agg_1s/)
        #[arg(long, value_delimiter = ',')]
        tickers: Vec<String>,
        /// The folder with universe.csv and windows.csv (default DATA/massive_1s/inputs)
        #[arg(long)]
        inputs: Option<PathBuf>,
        /// Add the universe's stocks missing from existing day files as day-D.extra.parquet
        #[arg(long)]
        extra: bool,
        #[command(flatten)]
        fin: FinalizeOpts,
    },
    /// Finalize a job even if some tickers failed (they will be missing from its days)
    Finalize {
        #[arg(long)]
        data: PathBuf,
        /// The job folder name, e.g. 2026-07-01_2026-09-30
        job: String,
        /// A job of an --extra run (jobs-extra/, writes day-D.extra.parquet)
        #[arg(long)]
        extra: bool,
        #[command(flatten)]
        fin: FinalizeOpts,
    },
    /// Jobs and their state, and the massive_rest day files present
    Status {
        #[arg(long)]
        data: PathBuf,
    },
}

fn inputs(data: &Path) -> PathBuf {
    data.join("massive_1s/inputs")
}

struct Run {
    from: Option<String>,
    to: Option<String>,
    workers: usize,
    tickers: Vec<String>,
    inputs: Option<PathBuf>,
    extra: bool,
}

fn fetch(data: &Path, run: Run, fin: FinalizeOpts) -> Result<bool> {
    let Run {
        from,
        to,
        workers,
        tickers,
        inputs: dir,
        extra,
    } = run;
    if extra && !tickers.is_empty() {
        bail!("--extra and --tickers cannot be combined");
    }
    let dir = dir.unwrap_or_else(|| inputs(data));
    let key = std::env::var("MASSIVE_API_KEY")
        .context("MASSIVE_API_KEY is not set (see nas/README.md)")?;
    let base = std::env::var("MASSIVE_BASE_URL").unwrap_or_else(|_| api::DEFAULT_BASE.to_string());
    let api = api::Api::new(&base, &key)?;
    let win = Windows::load(&dir.join("windows.csv"))?.between(from.as_deref(), to.as_deref());
    if win.is_empty() {
        bail!(
            "no sessions in windows.csv for {from:?} .. {to:?}; run nas/deploy.sh on the Mac first"
        );
    }
    let universe = load_universe(&dir.join("universe.csv"))?;
    let (layout, tickers) = if tickers.is_empty() {
        let layout = if extra {
            Layout::extra(data)
        } else {
            Layout::full(data)
        };
        (layout, universe)
    } else {
        let layout = Layout::sample(data);
        for t in tickers
            .iter()
            .filter(|t| universe.binary_search(t).is_err())
        {
            log!("note: {t} is not a kept stock in universe.csv");
        }
        let _ = std::fs::remove_dir_all(&layout.jobs); // a sample always starts fresh
        let mut t = tickers;
        t.sort();
        t.dedup();
        (layout, t)
    };
    let opts = Opts {
        workers,
        zstd_level: fin.zstd_level,
        finalize_threads: fin.finalize_threads,
    };
    job::run(&api, &layout, &win, &tickers, &opts)
}

fn status(data: &Path) -> Result<()> {
    let mut names: Vec<String> = vec![];
    for kind in ["jobs", "jobs-extra"] {
        if let Ok(rd) = std::fs::read_dir(data.join("massive_1s").join(kind)) {
            let mut n: Vec<String> = rd
                .filter_map(|e| {
                    Some(format!(
                        "{kind}/{}",
                        e.ok()?.file_name().into_string().ok()?
                    ))
                })
                .collect();
            n.sort();
            names.extend(n);
        }
    }
    for name in &names {
        let dir = data.join("massive_1s").join(name);
        let state = if dir.join("done").exists() {
            std::fs::read_to_string(dir.join("done"))
                .unwrap_or_default()
                .replace('\n', " ")
        } else {
            std::fs::read_to_string(dir.join("progress.json"))
                .unwrap_or_else(|_| "no progress yet".into())
                .replace('\n', " ")
        };
        println!("{name}: {state}");
    }
    let mut days = 0;
    if let Ok(rd) = std::fs::read_dir(data.join("agg_1s")) {
        for e in rd.flatten() {
            let Some(d) = e
                .file_name()
                .to_str()
                .and_then(|n| n.strip_prefix("date="))
                .map(str::to_string)
            else {
                continue;
            };
            days += usize::from(finalize::day_path(&data.join("agg_1s"), &d).exists());
        }
    }
    println!(
        "{} job(s); {days} massive_rest day file(s) under {}",
        names.len(),
        data.join("agg_1s").display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let res = match Cli::parse().cmd {
        Cmd::Fetch {
            data,
            from,
            to,
            workers,
            tickers,
            inputs,
            extra,
            fin,
        } => fetch(
            &data,
            Run {
                from,
                to,
                workers,
                tickers,
                inputs,
                extra,
            },
            fin,
        ),
        Cmd::Finalize {
            data,
            job,
            extra,
            fin,
        } => {
            let (kind, tag) = if extra {
                ("jobs-extra", ".extra")
            } else {
                ("jobs", "")
            };
            let dir = data.join("massive_1s").join(kind).join(&job);
            Windows::load(&dir.join("windows.csv"))
                .and_then(|win| {
                    finalize::finalize(
                        &dir,
                        &win,
                        &data.join("agg_1s"),
                        tag,
                        fin.zstd_level,
                        fin.finalize_threads,
                    )
                })
                .map(|_| true)
        }
        Cmd::Status { data } => status(&data).map(|_| true),
    };
    match res {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            log!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
