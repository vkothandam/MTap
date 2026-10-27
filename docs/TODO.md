# TODO

Running list of follow-ups that were decided but deferred. Newest first; tick and date an item when done. What runs today, and where: [STATUS.md](STATUS.md).

- [ ] **Daily model, phase 2 (after the R0–R8 series).** Added 2026-10-27 (ml/MODEL-STRATEGY.md Q12, Q15).
  - **Cross-model bake-off:** AutoGluon-TimeSeries, DeepAR, PatchTST, Chronos, fitted fold by fold on the same 16 folds. Needs a separate environment and hours per fold; done once the inputs are settled.
  - **Recency weighting** of training samples.
  - **Recycled tickers:** a ticker reused by a different company (BNY) beyond the series breaks.
  - **GNN / latent-vector ideas** from docs/training-options-daily.md (cross-stock links, the TFT's latent state as an input).
  - **Intraday re-rank of the watch list after 09:31** from the watchlist's live prices.
  - **XGBoost or LightGBM for the screener**, only if R8 shows HGB clearly behind.
  - **Live Massive prices** replace ChartIQ/watchlist (`[premarket] live_source`); the 09:45 full step can then move to 09:30.

- [ ] **A second copy of MTap-Data (contingency, low priority).** Added 2026-10-25.
  - **Why:** the NAS holds the only copy of the data, and since the cutover it is also
    what writes to it.
  - **Why not done:** volume2 is ext4, and DSM's Snapshot Replication needs Btrfs.
  - **Benefits of a copy:**
    - a bad job (or a bug) that overwrites or deletes files can be undone;
    - a mistaken delete by hand (in Finder over SMB, or over ssh) can be undone;
    - ransomware on a Mac that mounts the share cannot reach a read-only copy;
    - a failed disk or volume does not lose the 5-year history (refetching it costs hours
      of Massive REST), and lost E*TRADE watchlist and live massive-ws days cannot be
      refetched at all.
  - **Options, cheapest first:**
    - Hyper Backup (works on ext4), daily and versioned, to a USB disk or another volume;
      or to a cloud target (C2, B2, S3) for an off-site copy;
    - a Btrfs volume for MTap-Data, then snapshots (needs a volume migration).

- [x] **E*TRADE collector: exit when the session recovery fails.** Added 2026-10-24
  (trial step 9, LEARNINGS.md `tag-sourcing-host`). Done 2026-10-25:
  `handleSessionExpired` exits 1 after `CRITICAL: Session recovery FAILED`, and the restart
  loop (`run-etrade.sh` on the NAS, `mtap-schedule.sh supervise` on the iMac) starts a fresh
  collector with backoff.

- [ ] **Move the sourcing to the NAS.** Added 2026-10-24. Trial 10-24/25, cutover Sunday 10-25, live from the open on 10-26. Design: [nas-migration.md](nas-migration.md); runbook: [deploy/synology/README.md](../deploy/synology/README.md). After a stable week, decide whether the iMac target (`scripts/schedule`) stays as the fallback.

- [ ] **1-second history: after the one-month measurement.** Added 2026-10-19. `nas/massive1s` fetches Massive REST 1-second bars on the NAS (nas/README.md).
  - Phase 0: `./nas/deploy.sh 2026-09-01 2026-09-30`, Run on the NAS; report bars, calls, time, GB, CPU/RAM, B/bar; DuckDB size comparison; REST vs WS on 14–16 Oct (prices, `v` vs `v`/`dv`); then choose the storage.
  - Then the 2-year run (`2024-10-21 .. 2026-10-16`), and later 5 years (re-probe the rolling edge first; it moves daily).
  - A panel builder that reads `agg_1s` from both sources (`massive_rest` history + `massive_ws` live).
  - Optional: a small daily incremental, so REST also covers days the live feed missed.
- [ ] **Schedule this Mac's daily load.** Added 2026-10-15. The iMac fetches at 06:40 PT. The Mac that owns the DuckDB should then run `daily-sourcing.sh` with `MTAP_DAILY_INGEST_ONLY=1` (no Massive calls, reads the NAS) at a later time. `setup-schedule.sh` has one fixed `DAILY_H:DAILY_M` (06:40) and installs the collector agents too, so it needs a separate "ingest only, at HH:MM" option. Until then, run it by hand.
- [ ] **Clean up failed `massive-ext gaps` jobs.** Added 2026-10-15. A gaps run with a failed stock leaves `state/massive_ext/jobs/<from>_<to>/` behind, and the next day's run uses a new range. The sessions are retried, because their file is still missing, but the old folder stays (a few MB). Remove job folders older than 14 days at the start of `gaps`.
- [x] **Feed pre-market / after-hours into the model.** Added 2026-10-15. Decided 2026-10-27 (ml/MODEL-STRATEGY.md Q14, Q16): after-hours of day t goes in with the always-available groups (R1); pre-market in two masked groups, prices at 09:30 and full at 09:45 from Massive. Built on `feat/daily-5y-inputs`; results in LEARNINGS `tag-ml-daily`.
- [ ] **Stop storing the data twice (NAS files + local DuckDB copy).** Decided 2026-10-15: keep the duplicate for now, revisit later.
  - **Today.** The raw daily jsonl on the NAS is 1.8 GB for 2 years. `state/etrade/fundamentals.duckdb` holds the same 6.2M bars again, and is 1.0 GB in total including fundamentals, EDGAR and splits. The 5-year backfill and the pre-market/after-hours columns grow both copies by about 2.5×.
  - **(a) Parquet on the NAS as the only store, with view-only DuckDBs.**
    - Each machine keeps a DuckDB that holds only views, e.g. `CREATE VIEW daily_bars AS SELECT * FROM read_parquet('<NAS>/db/daily_bars/*.parquet')`.
    - The raw jsonl is compacted into monthly parquet, the same idea as `massive-ws compact`.
    - Derived columns (`split_factor`, pct changes) are computed in views instead of being stored.
    - One writer per dataset. It writes locally, then renames into place on the NAS.
    - Daily bars move first, then fundamentals and EDGAR.
  - **(b) DuckLake.** Parquet on the NAS plus a catalog in a Postgres container on the Synology. This allows more than one writer. A catalog *file* on SMB fails just like a DuckDB file does, because SMB has no file locks.
  - **Trade-offs to check:**
    - query speed over SMB, wired vs Wi-Fi
    - what happens while the NAS is offline (keep a local cache?)
    - size of the refactor (ingest-eod, derive-features, export-tft)
  - **When.** After the 5-year backfill lands and its real sizes are known.
