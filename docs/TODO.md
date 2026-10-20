# TODO

Running list of follow-ups that were decided but deferred. Newest first; tick and date an item when done.

- [ ] **1-second history: after the one-month measurement.** Added 2026-10-19. `nas/massive1s` fetches Massive REST 1-second bars on the NAS (nas/README.md).
  - Phase 0: `./nas/deploy.sh 2026-09-01 2026-09-30`, Run on the NAS; report bars, calls, time, GB, CPU/RAM, B/bar; DuckDB size comparison; REST vs WS on 14–16 Oct (prices, `v` vs `v`/`dv`); then choose the storage.
  - Then the 2-year run (`2024-10-21 .. 2026-10-16`), and later 5 years (re-probe the rolling edge first; it moves daily).
  - A panel builder that reads `agg_1s` from both sources (`massive_rest` history + `massive_ws` live).
  - Optional: a small daily incremental, so REST also covers days the live feed missed.
- [ ] **Schedule this Mac's daily load.** Added 2026-10-15. The iMac fetches at 06:40 PT. The Mac that owns the DuckDB should then run `daily-sourcing.sh` with `MTAP_DAILY_INGEST_ONLY=1` (no Massive calls, reads the NAS) at a later time. `setup-schedule.sh` has one fixed `DAILY_H:DAILY_M` (06:40) and installs the collector agents too, so it needs a separate "ingest only, at HH:MM" option. Until then, run it by hand.
- [ ] **Clean up failed `massive-ext gaps` jobs.** Added 2026-10-15. A gaps run with a failed stock leaves `state/massive_ext/jobs/<from>_<to>/` behind, and the next day's run uses a new range. The sessions are retried, because their file is still missing, but the old folder stays (a few MB). Remove job folders older than 14 days at the start of `gaps`.
- [ ] **Feed pre-market / after-hours into the model.** Added 2026-10-15. The columns are in `daily_bars` (ml/FEATURES.md section 11) but not in the panel. Deciding which go in, and on which step (after-hours of day t is known by 20:00 ET; pre-market of t+1 only that morning), is a model decision for ml/MODEL-STRATEGY.md.
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
