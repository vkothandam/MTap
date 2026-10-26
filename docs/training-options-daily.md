# Hybrid Trading Architecture Research Specification

## Multi-Horizon Daily TFT, Real-Time Intraday XGBoost, and Pre-Market Overlay Framework

This research specification establishes the comprehensive data structures, feature classification pipelines, hardware optimization strategies, and multi-model configuration choices developed across this session for an institutional-grade quantitative execution engine tracking 8,000+ symbols.

## 1. System Architecture Layout & Co-Augmentation Logic

The core framework splits system cognitive load into an asynchronous Macro Layer (running once per day prior to the market open) and a synchronous, high-frequency Micro Layer (running continuously on live streaming data).

```
                      [Historical Parquet Files (On Disk)]
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────────┐
          │  DAILY MACRO LAYER: PyTorch TFT + Entity Embeddings     │ 🕒 Runs once per day
          │  Outputs: 10th, 50th, 90th Daily Volatility Quantiles  │    (Pre-Market 06:00 AM)
          └────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                     [In-Memory DuckDB Quantile Cache]
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼ (Augments Stream Rows)                          ▼ (Context Vector Extraction)
┌──────────────────────────────────────────────┐   ┌──────────────────────────────────┐
│  INTRADAY MICRO LAYER: XGBoost Execution     │   │   ALPHA OVERLAY PIPELINE (9:25)  │
│  Consumes: Massive.com Real-Time WebSockets  │   │   Integrates Pre-Market Data with   │
│  Frequency: Sub-5 Second Evaluation Loops     │   │   TFT Latent Historical Memory       │
└──────────────────────────────────────────────┘   └──────────────────────────────────┘
              │                                                 │
              ▼                                                 ▼
   [Top 20/30 High-Velocity Trend Slots]             [Refined High-Alpha Breakout Scoring]
```

### The Augmentation Principle

- **Daily Macro Layer (TFT + JEV)**: Processes 30 long-horizon fundamental variables, daily price action, and massive entity embeddings. It establishes today's structural volatility boundaries (the 10th, 50th, and 90th percentile predicted session VWAPs).
- **Intraday Micro Layer (XGBoost)**: Consumes raw tick streams, transforming them into 1-minute rolling indicators. It joins today's cached static daily quantiles to every incoming tick row.
- **The Synergistic Effect**: Tree-based classifiers do not try to parse complex multi-year sequence dependencies. Instead, they evaluate immediate order-book velocity relative to the transformer's macro volatility nodes. This provides microsecond prediction cycles backed by global transformer context.

## 2. Definitive Feature Classification Matrices

### A. Daily Macro Layer (TFT Inputs)

Input sequences are daily historical bars across a multi-year lookback window.

- **Static Categoricals (JEV Layers)**:
  - `symbol`: Mapped to a 16-Dimensional Entity Embedding
  - `industry_code`: Mapped via a 4-Dimensional revenue-weighted Multi-Hot Embedding matrix
  - `exchange`: Listing node tracking
  - `issue_type`: Security classification structural anchor

- **Temporal Observed (Daily Time-Varying continuous variables)**:
  - Daily open, high, low, close, volume, vwap
  - Joined news sentiment metrics (`mean_score_agg`)
  - 30 continuous corporate fundamental line items (Balance Sheet/Income Statement metrics)

- **Targets (The Optimization Node)**:
  - `vwap_nx_1d`: Optimized using Quantile Loss at 0.10, 0.50, and 0.90 nodes

### B. Intraday Micro Layer (XGBoost Inputs)

Every row represents a 1-minute rolling bar window or sub-minute tick aggregation. Raw daily price figures and un-normalized corporate fundamentals are strictly barred from this matrix to preserve scale invariance.

| Feature Column Name | Classification Category | Data Type | Feature Source & Mathematical Transformation |
|---|---|---|---|
| `tft_daily_q10` | Augmented Static Context | Continuous | Pre-calculated 10th percentile daily VWAP target boundary from morning run |
| `tft_daily_q50` | Augmented Static Context | Continuous | Pre-calculated 50th percentile (median) daily VWAP target boundary from morning run |
| `tft_daily_q90` | Augmented Static Context | Continuous | Pre-calculated 90th percentile daily VWAP target boundary from morning run |
| `distance_to_q90` | Engineered Momentum | Continuous | Calculated on the fly: `(tft_daily_q90 - current_tick_price) / current_tick_price` |
| `symbol_id` | Static Categorical | Integer | Label-encoded integer mapping the raw string ticker to an integer index (0 to 8500) |
| `industry_id_vector` | Static Categorical | Array | Pre-cached index list mapping the asset's active multi-industry mappings |
| `minute_of_day` | Temporal Known Clock | Integer | Linear tracker of intraday time elapsed (e.g., 1 to 390 for standard US equity sessions) |
| `time_to_close` | Temporal Known Clock | Integer | Minutes remaining until the closing bell (captures structural end-of-day volume smiles) |
| `rolling_vwap_1m` | Temporal Observed Micro | Continuous | Volume Weighted Average Price calculated strictly over the last 60 seconds of streaming ticks |
| `vwap_velocity_3m` | Temporal Observed Micro | Continuous | Percentage change between the current `rolling_vwap_1m` and the 3-minute prior window block |
| `intraday_spread_pct` | Temporal Observed Micro | Continuous | Volatility scale metric: `(rolling_1m_high - rolling_1m_low) / rolling_vwap_1m` |
| `volume_velocity` | Temporal Observed Micro | Continuous | Current 1-minute volume divided by the moving average volume of the past 20 minutes |
| `trade_size_intensity` | Temporal Observed Micro | Continuous | Liquidity granularity: 1-minute rolling volume divided by 1-minute transaction prints |
| `daily_sentiment` | Temporal Observed Micro | Continuous | Rolling sentiment baseline (`mean_score_agg`) joined from the daily news store |
| `sentiment_volume` | Temporal Observed Micro | Integer | Number of articles (`n_articles`) published within the rolling tracking window |
| `is_10k` / `is_10q` | Temporal Observed Micro | Binary | Categorical flags marking if an active corporate earnings report dropped during the session |

## 3. High-Alpha Pre-Market Modelling Analysis & Trade-Offs

Pre-market metrics (early volume surges, morning order-book imbalances) possess immense alpha but present a core dilemma: they are highly sparse and operationally prone to transmission delays. If a single model is trained from scratch with these inputs, it becomes overly reliant on them; if a feed drops during live execution, the missing values cause the model's internal attention layers to collapse.

To resolve this conditional information constraint, four modeling methods are mapped below:

### Comparative Deep-Dive of Proposed Overlays

#### Method 1: TFT Base Model + XGBoost Alpha Overlay (🏆 Industry Standard Recommendation)

**Execution Logic**: Train a Base TFT model purely on clean, non-sparse historical daily data. Freeze its layers. Pass historical data through it to extract the final Latent Context Vectors (the dense internal hidden states). Append the sparse morning pre-market data to these vectors and train an XGBoost regressor to predict the final adjusted quantile adjustments.

**Pros**:
- Maximum predictive stability. Trees excel at conditional "If/Then" logic (e.g., `IF pre_market_volume == 0 THEN ignore and rely entirely on TFT Latent Vector`).
- Lightning-fast update loops. Training the XGBoost overlay takes seconds to minutes compared to hours for neural networks.
- Flawless operational fault tolerance. If pre-market feeds fail, the system drops the overlay and defaults cleanly to the Base TFT boundaries.

**Cons**: Requires managing a two-stage hybrid pipeline architecture (PyTorch extraction to XGBoost matrix).

#### Method 2: Weight Cloning, Architecture Expansion & Fine-Tuning

**Execution Logic**: Train the Base TFT on daily data. Copy its entire weight checkpoint structure into a new model instance that features an extra variable input node designed specifically for pre-market parameter matrices. Run a brief fine-tuning phase (e.g., 3 to 10 epochs) using a highly conservative learning rate (1e-5).

**Pros**: Preserves a uniform, pure deep learning pipeline without mixing model frameworks. Keeps the model's baseline understanding of historical time-series paths intact.

**Cons**: Highly susceptible to catastrophic forgetting or over-fitting to morning noise. The brief training duration can warp the pre-trained JEV symbol embeddings, distorting long-term trend tracking capabilities.

#### Method 3: Delta-Residual Target Modelling

**Execution Logic**: The Base TFT runs asynchronously to generate standard daily predictions. A separate, shallow secondary model (either a light neural network or tree structure) is trained exclusively on pre-market parameters. Its target variable is not price, but the mathematical error vector (the delta residual) of the Base model (`Realized_Price - Base_Prediction`).

**Pros**: Extremely fast training time and easy validation. If the pre-market data contains zero signal for a ticker, the residual prediction evaluates to zero, leaving the Base TFT prediction completely unchanged.

**Cons**: Imposes a rigid, strictly additive relationship between pre-market movements and historical trend patterns. Fails to capture non-linear, multiplicative cross-interactions (e.g., how morning volume velocity shifts meaning depending on a stock's underlying fundamental debt ratios).

#### Method 4: Single TFT Model Trained Completely From Scratch (With Pre-Market Included)

**Execution Logic**: Incorporate pre-market columns directly into the primary 3D training tensor alongside all daily parameters. Train the complete model from zero over 100+ epochs.

**Pros**: Yields highly cohesive, mathematically elegant path projections because the self-attention matrices blend morning velocity and historical sequences simultaneously within the same network.

**Cons**: Critical operational fragility. If a data packet lags at 09:30:00 AM, you must impute a placeholder value (like 0.0). Because the deep transformer's internal Variable Selection Networks (VSNs) were not trained to handle missing blocks at run-time, the entire model's attention allocation goes haywire, degrading prediction accuracy across the entire 8,000-asset universe.

## 4. Hardware Optimization & Computational Scaling Strategy

Executing real-time Point-in-Time (PIT) slotting across 8,000+ streaming websocket symbols every 5 seconds is impossible via a raw brute-force approach on an Intel iMac or a minimal cloud vCPU instance. The quadratic complexity O(T²) of the Transformer's attention mechanism on a 6000x30x40 tensor will cause immediate Out-Of-Memory (OOM) failures or execution lags.

To solve this on standard hardware, you must decouple data ingestion from data inference using a Two-Tier Filtering Framework:

```
[6,000+ Symbols Streaming Raw WebSocket Packets]
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  TIER 1: In-Memory Ingestion & State Matrix  │ 🕒 Ingests 24/7 with zero model lag.
│  Engine: DuckDB + PyArrow Memory Buffers     │    Appends Today's Static TFT Quantiles.
└──────────────────────────────────────────────┘
                       │
                       ▼ (Every 5 Seconds: Execute Velocity Sieve)
┌──────────────────────────────────────────────┐
│  DUCKDB LINEAR VELVET-ROPE FILTER            │ ⚡ Slices data scope by over 90%.
│  Drops ~5,500 completely flat/quiet symbols  │    Execution Time: ~50 milliseconds.
└──────────────────────────────────────────────┘
                       │
                       ▼
       [~500 High-Activity Symbols Only]
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ TIER 2A: Intraday TFT     │   │ TIER 2B: XGBoost Closer   │ ⚡ Runs microsecond loops
│ Evaluates 15-minute paths │   │ Processes tick anomalies  │    to slot the final Top
│ Runtime: ~300-600ms       │   │ Runtime: ~10-15ms         │    20/30 targets.
└───────────────────────────┘   └───────────────────────────┘
```

### Step 1: The DuckDB In-Memory Sieve

Your WebSocket consumer loop does nothing except digest incoming packets and maintain a running state matrix in DuckDB. Every 5 seconds, a highly optimized SQL indexing query filters out symbols that are locked in quiet sideways consolidation, focusing the model's attention exclusively on active liquidity expansions:

```sql
-- Slices the computation tensor from 6,000 down to the top 500 active tickers
CREATE OR REPLACE TEMP TABLE active_inference_universe AS
SELECT symbol 
FROM live_state_matrix 
WHERE volume_velocity > 1.5 OR abs(vwap_velocity_3m) > 0.003 
LIMIT 500;
```

### Step 2: High-Performance Batch Inference

By passing only these 500 volatile candidates to your model structures, you compress your computing requirements exponentially. Inferencing a batch of 500 rows through your XGBoost overlay and path configurations takes under 15 milliseconds on an older Intel CPU, keeping your execution loop responsive and well ahead of your 5-second deadline.

## 5. Advanced Quantitative Research Horizons

To iterate on this baseline setup, explore these three enterprise development vectors:

1. **Zero-Copy Memory Stream Serialization**: Eliminate slow serialization/deserialization penalties completely by utilizing Apache Arrow formats. Apache Arrow maps memory tables natively, allowing your DuckDB feature tables to stream data records directly into XGBoost's DMatrix and PyTorch tensor allocations without executing any system-level data copies.

2. **Latent Context Vector Extraction**: Instead of passing raw, scalar quantile numbers (q10, q50, q90) from your morning TFT to your intraday tree layers, write a PyTorch hook to strip out the final hidden states of the TFT's Static Covariate Encoder. Passing this dense, high-dimensional vector to XGBoost explicitly embeds deep temporal history and cross-symbol asset relationships directly into your real-time tree splits.

3. **Graph Neural Networks (GNN) for Real-Time Spillover Matrix Analysis**: Construct an intraday spatial network where stock symbols act as nodes and edges represent shared sub-industry structures or corporate supply-chain dependencies. When an individual stock enters your Top-20 execution slot due to an extreme breakout, the GNN can propagate a real-time momentum warning vector to connected peer nodes, allowing your XGBoost engine to intercept and trade highly predictable industry spillovers before the target symbols experience price anomalies.

### Next Steps

To advance this research into code execution, choose which module to build out first:

- The PyTorch architecture snippet required to slice open your base model and extract the Latent Context Tensors (Method 1)?
- The DuckDB streaming aggregation loop to handle the live 1-minute windowing parameters directly on incoming tick packets?
- Or the XGBoost Quantile Loss objective function (check loss) configuration to handle the overlay adjustments?
