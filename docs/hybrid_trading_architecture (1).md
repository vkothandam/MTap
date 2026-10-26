# Hybrid Trading Architecture Research Specification
## Multi-Horizon Daily TFT and Real-Time Intraday XGBoost Framework

This research specification outlines the data structures, column classifications, and performance engineering requirements for a multi-tier quant execution framework. It details how a Daily Macro Transformer and an Intraday Tree-Based Execution Layer augment each other using a streaming feature bridge without introducing look-ahead data leakage or execution latency.

---

## 1. System Architecture Layout

The system splits cognitive processing into an asynchronous, deep-learning **Macro Layer** (running once daily pre-market) and a synchronous, high-speed **Micro Layer** (running continuously on live websocket streaming data).

```
   [Historical Parquet Files] 
               │
               ▼
┌──────────────────────────────────────────────┐
│  DAILY MACRO LAYER: PyTorch TFT + JEV         │ 🕒 Runs once per day (Pre-Open)
│  Inputs: Daily Historical Bars + Fundamentals │
│  Outputs: 10th, 50th, 90th Daily Quantiles   │
└──────────────────────────────────────────────┘
               │
               ▼ (Cache Static Quantile Anchors in In-Memory DuckDB)
   [Today's Static Quantile Cache] 
               │
               ▼ (Augments Real-Time Stream Rows)
┌──────────────────────────────────────────────┐
│  INTRADAY MICRO LAYER: XGBoost Execution     │ ⚡ Runs Every 1 to 5 Seconds
│  Consumes: Massive.com Real-Time WebSockets  │
└──────────────────────────────────────────────┘
               │
               ▼
   [Top 20/30 High-Velocity Trend Slots]
```

---

## 2. Strict Feature Segregation & Column Classifications

To ensure structural integrity and maximum predictive accuracy, data fields are strictly ring-fenced between layers. Intraday loops do not ingest raw daily fundamental values, and daily prediction tasks are kept isolated from sub-minute market noise.

### A. Daily Macro Layer (TFT Inputs)
*Processed once per day. Input sequences are daily historical bars across a multi-year lookback window.*

*   **Static Categoricals (JEV):** `symbol` (Mapped to 16D Entity Embedding), `industry_code` (Mapped to 4D revenue-weighted Multi-Hot Embedding), `exchange`, `issue_type`.
*   **Temporal Observed (Daily Variable Metrics):** Daily `open`, `high`, `low`, `close`, `volume`, `vwap`, joined daily news sentiment metrics (`mean_score_agg`), and 30 continuous corporate fundamental line items (Balance Sheet/Income Statement metrics).
*   **Targets (Future-Looking Daily Anchors):** `vwap_nx_1d` (Optimized via 10th, 50th, and 90th Quantile Loss functions).

### B. Intraday Micro Layer (XGBoost Inputs)
*Processed continuously on live feeds. Every row represents a 1-minute rolling bar window or tick aggregation. No corporate fundamentals or daily price levels are passed raw.*

```csv
Feature_Column_Name,Classification_Category,Data_Type,Feature_Source_Description
tft_daily_q10,Augmented Static Context,Continuous,Pre-calculated 10th percentile daily VWAP target boundary.
tft_daily_q50,Augmented Static Context,Continuous,Pre-calculated 50th percentile (median) daily VWAP target boundary.
tft_daily_q90,Augmented Static Context,Continuous,Pre-calculated 90th percentile daily VWAP target boundary.
distance_to_q90,Engineered Momentum,Continuous,Calculated: (tft_daily_q90 - current_tick_price) / current_tick_price.
symbol_id,Static Categorical,Integer,Label-encoded integer matching the symbol index table (0-8500).
industry_id_vector,Static Categorical,Array,Pre-cached array mapping the asset's primary sub-industry codes.
minute_of_day,Temporal Known Clock,Integer,Linear tracker of intraday time (e.g., 1 to 390 for standard US sessions).
time_to_close,Temporal Known Clock,Integer,Minutes remaining until the closing bell (tracks structural volume smiles).
rolling_vwap_1m,Temporal Observed Micro,Continuous,Volume Weighted Average Price calculated over the last 60 seconds.
vwap_velocity_3m,Temporal Observed Micro,Continuous,Percentage change between current rolling VWAP and the 3-minute prior window.
intraday_spread_pct,Temporal Observed Micro,Continuous,Calculated: (rolling_1m_high - rolling_1m_low) / rolling_vwap_1m.
volume_velocity,Temporal Observed Micro,Continuous,Current 1-minute volume divided by the moving average of the last 20 minutes.
trade_size_intensity,Temporal Observed Micro,Continuous,Calculated: 1-minute rolling volume divided by 1-minute transaction counts.
```

---

## 3. Performance & Computational Scaling Strategy

To execute Point-in-Time (PIT) slotting across a massive 8,000+ asset universe on constrained hardware (like an Intel iMac or minimal Cloud vCPU instances), you must deploy an asynchronous pipeline filter.

### The Dynamic Universe Downscaling Protocol
1. **The Ingestion Queue (Real-Time):** The Massive.com WebSocket consumer loop updates an in-memory state matrix inside DuckDB. It performs **no predictions** and incurs minimal CPU overhead.
2. **The 5-Second Linear Slotting Filter:** Every 5 seconds, an optimized DuckDB query scans the state matrix to evaluate an architectural gating metric:
   ```sql
   SELECT symbol FROM live_state_matrix 
   WHERE volume_velocity > 1.5 OR abs(vwap_velocity_3m) > 0.003 
   LIMIT 400;
   ```
3. **The Microsecond Inference Batch:** Only the ~400 stocks that pass through this "velvet rope" sieve are batched into a flat array and passed to XGBoost. 
4. **Hardware Performance Metrics:** On an Intel CPU, inferencing 400 rows through an XGBoost model takes under **15 milliseconds**, safely preserving your 5-second deadline while keeping total CPU usage beneath 25%.

---

## 4. Advanced Research Horizons

*   **Latent Context Vector Extraction:** Instead of passing raw scalar quantile numbers (`q10`, `q50`, `q90`) from the Daily TFT to the Intraday XGBoost model, explore extracting the final **hidden state context weights** from the TFT's Static Covariate Encoder. Passing this dense vector to XGBoost transfers deep structural peer relationships directly into the tree splits.
*   **Zero-Copy Memory Streaming (Apache Arrow):** Research using PyArrow to share data buffers between DuckDB's aggregation outputs and XGBoost's internal DMatrix memory addresses. This bypasses slow serialization/deserialization routines entirely, cutting intraday feature prep overhead down to zero.
*   **Graph Neural Networks (GNN) for Real-Time Spillover:** Investigate constructing a parallel intraday graph where nodes are stocks and edges represent shared sub-industries. When a top-slotted stock undergoes a major breakout, the GNN can propagate a momentum warning vector to connected peer nodes before their raw ticker data spikes.
