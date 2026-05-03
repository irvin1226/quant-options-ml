# SPY Options Dataset - Data Dictionary

**Dataset:** Processed SPY Options Data (2018-2025)  
**Total Observations:** 1,590,954,634 rows  
**Total Files:** 96 monthly parquet files  
**Coverage:** January 2018 - December 2025  
**Source:** ThetaData via custom ETL pipeline  

---

## Column Overview

**Total Columns:** 28  
**Data Types:** 6 datetime64[ns], 1 object (string), 1 int16, 3 int32, 17 float64

---

## Column Definitions

### 1. Identifiers (6 columns)

#### `symbol` 
- **Type:** object (string)
- **Source:** ThetaData greeks_all CSV
- **Description:** Ticker symbol for the underlying asset
- **Values:** "SPY" (constant for this dataset)
- **Example:** `"SPY"`

#### `timestamp`
- **Type:** datetime64[ns]
- **Source:** ThetaData greeks_all CSV
- **Description:** Date and time of the observation in US/Eastern timezone
- **Format:** `YYYY-MM-DD HH:MM:SS`
- **Range:** Market hours (09:30:00 - 16:00:00 ET)
- **Example:** `2021-03-15 14:30:00`
- **Note:** Timezone-naive but known to be US/Eastern

#### `date`
- **Type:** datetime64[ns]
- **Source:** **Derived from folder structure**
- **Description:** Trading date extracted from data file path
- **Calculation:** Parsed from `/YYYY/MM/DD/` folder structure
- **Format:** `YYYY-MM-DD 00:00:00`
- **Example:** `2021-03-15 00:00:00`
- **Note:** Used for monthly aggregation and date-based filtering

#### `expiration`
- **Type:** datetime64[ns]
- **Source:** ThetaData greeks_all CSV
- **Description:** Expiration date of the options contract
- **Format:** `YYYY-MM-DD`
- **Example:** `2021-03-19 00:00:00`
- **Note:** Contracts expire at market close on expiration date

#### `strike`
- **Type:** float64
- **Source:** ThetaData greeks_all CSV
- **Description:** Strike price of the options contract
- **Unit:** US Dollars ($)
- **Example:** `390.0`
- **Note:** SPY strikes typically in $1 or $0.50 increments

#### `right`
- **Type:** object (string)
- **Source:** ThetaData greeks_all CSV
- **Description:** Option type identifier
- **Values:** `"CALL"` or `"PUT"`
- **Example:** `"CALL"`

---

### 2. Pricing Data (11 columns)

#### `bid`
- **Type:** float64
- **Source:** ThetaData greeks_all CSV
- **Description:** Best bid price (highest price buyers are willing to pay)
- **Unit:** US Dollars ($)
- **Example:** `5.25`
- **Filter Applied:** Removed all rows where `bid <= 0` during ETL
- **Usage:** Used as execution price for selling options

#### `ask`
- **Type:** float64
- **Source:** ThetaData greeks_all CSV
- **Description:** Best ask price (lowest price sellers are willing to accept)
- **Unit:** US Dollars ($)
- **Example:** `5.35`
- **Filter Applied:** Removed all rows where `ask <= bid` during ETL
- **Usage:** Used as execution price for buying options

#### `mid`
- **Type:** float64
- **Source:** **Derived**
- **Description:** Midpoint between bid and ask prices
- **Calculation:** `mid = (bid + ask) / 2`
- **Unit:** US Dollars ($)
- **Example:** `5.30`
- **Usage:** Reference price for fair value

#### `spread`
- **Type:** float64
- **Source:** **Derived**
- **Description:** Absolute bid-ask spread
- **Calculation:** `spread = ask - bid`
- **Unit:** US Dollars ($)
- **Example:** `0.10`
- **Usage:** Trading cost indicator, liquidity measure

#### `spread_pct`
- **Type:** float64
- **Source:** **Derived**
- **Description:** Bid-ask spread as percentage of mid price
- **Calculation:** `spread_pct = spread / mid`
- **Unit:** Decimal (0.10 = 10%)
- **Example:** `0.0189` (1.89%)
- **Usage:** Normalized trading cost, quality filter

#### `open`
- **Type:** float64
- **Source:** ThetaData OHLC CSV
- **Description:** Opening price for the 1-minute bar
- **Unit:** US Dollars ($)
- **Example:** `5.28`
- **Note:** First trade price in the minute

#### `high`
- **Type:** float64
- **Source:** ThetaData OHLC CSV
- **Description:** Highest price during the 1-minute bar
- **Unit:** US Dollars ($)
- **Example:** `5.35`

#### `low`
- **Type:** float64
- **Source:** ThetaData OHLC CSV
- **Description:** Lowest price during the 1-minute bar
- **Unit:** US Dollars ($)
- **Example:** `5.20`

#### `close`
- **Type:** float64
- **Source:** ThetaData OHLC CSV
- **Description:** Closing price for the 1-minute bar
- **Unit:** US Dollars ($)
- **Example:** `5.30`
- **Note:** Last trade price in the minute

#### `vwap`
- **Type:** float64
- **Source:** ThetaData OHLC CSV
- **Description:** Volume-weighted average price for the 1-minute bar
- **Unit:** US Dollars ($)
- **Example:** `5.29`
- **Note:** Average price weighted by trade size

#### `underlying_price`
- **Type:** float64
- **Source:** ThetaData greeks_all CSV
- **Description:** Price of the underlying SPY ETF at the timestamp
- **Unit:** US Dollars ($)
- **Example:** `393.45`
- **Usage:** Reference for moneyness calculation, delta validation

---

### 3. Volume Data (2 columns)

#### `volume`
- **Type:** int64
- **Source:** ThetaData OHLC CSV
- **Description:** Total number of contracts traded during the 1-minute bar
- **Unit:** Contracts
- **Example:** `150`
- **Note:** `volume = 0` indicates no trades (quote-only data)
- **Important:** ~91% of rows have volume=0; filter to `volume > 0` for tradeable opportunities

#### `count`
- **Type:** int64
- **Source:** ThetaData OHLC CSV
- **Description:** Number of individual trades during the 1-minute bar
- **Unit:** Trades
- **Example:** `5`
- **Note:** `count = 0` when `volume = 0`

---

### 4. Liquidity Data (1 column)

#### `open_interest`
- **Type:** float64
- **Source:** ThetaData open_interest.csv (LEFT JOIN)
- **Description:** Total number of outstanding contracts at start of trading day (reported at 6:30 AM ET from previous day's close)
- **Unit:** Contracts
- **Example:** `15420.0`
- **Missing Data:** ~5-7% of rows have NaN (contracts not in OI file)
- **Coverage:** 96.8% average across all years
- **Note:** OI updates once daily; same value for all timestamps on a given date

---

### 5. Greeks (6 columns)

All greeks sourced from ThetaData greeks_all CSV.

#### `implied_vol`
- **Type:** float64
- **Description:** Implied volatility derived from market prices using Black-Scholes model
- **Unit:** Decimal (0.20 = 20% annualized volatility)
- **Example:** `0.1847` (18.47%)
- **Usage:** Volatility forecasting, relative value analysis

#### `delta`
- **Type:** float64
- **Description:** Rate of change of option price with respect to $1 change in underlying
- **Unit:** Decimal
- **Range:** -1.0 to 1.0 (calls: 0 to 1, puts: -1 to 0)
- **Example:** `0.5247` (call), `-0.3156` (put)
- **Usage:** Directional exposure, probability of expiring ITM (approximation)

#### `gamma`
- **Type:** float64
- **Description:** Rate of change of delta with respect to $1 change in underlying
- **Unit:** Delta change per $1 underlying move
- **Range:** 0.0 to ~3.0 (always positive for long options)
- **Example:** `0.0234`
- **Usage:** Delta hedging frequency, convexity exposure

#### `theta`
- **Type:** float64
- **Description:** Rate of change of option price with respect to 1-day passage of time
- **Unit:** US Dollars per day
- **Range:** Negative for long options (time decay)
- **Example:** `-0.15` (loses $0.15 per day)
- **Usage:** Time decay measurement, carry analysis

#### `vega`
- **Type:** float64
- **Description:** Rate of change of option price with respect to 1% change in implied volatility
- **Unit:** US Dollars per 1% IV change
- **Range:** Always positive for long options
- **Example:** `0.45` (gains $0.45 if IV increases by 1%)
- **Usage:** Volatility exposure, vega hedging

#### `rho`
- **Type:** float64
- **Description:** Rate of change of option price with respect to 1% change in interest rate
- **Unit:** US Dollars per 1% rate change
- **Range:** Positive for calls, negative for puts
- **Example:** `0.23`
- **Usage:** Interest rate sensitivity (typically minor for short-dated options)

---

### 6. Derived Features (2 columns)

#### `dte`
- **Type:** int64
- **Source:** **Derived**
- **Description:** Days to expiration (calendar days)
- **Calculation:** `dte = (expiration - date).days`
- **Unit:** Days
- **Range:** 0 to 60 (filtered during data download)
- **Example:** `4` (4 days until expiration)
- **Note:** DTE=0 is expiration day

#### `moneyness`
- **Type:** float64
- **Source:** **Derived**
- **Description:** Ratio of strike price to underlying price
- **Calculation:** `moneyness = strike / underlying_price`
- **Unit:** Ratio (dimensionless)
- **Example:** `1.0127` (strike 1.27% above spot)
- **Interpretation:**
  - `moneyness = 1.0`: At-the-money (ATM)
  - `moneyness > 1.0`: Out-of-the-money for calls, in-the-money for puts
  - `moneyness < 1.0`: In-the-money for calls, out-of-the-money for puts

---

## Data Quality Filters Applied During ETL

The following filters were applied to remove low-quality data:

1. **`bid > 0`** - Removed invalid/missing bid prices
2. **`ask > bid`** - Removed crossed/invalid spreads

**Result:** ~8% of raw observations filtered out, retaining only high-quality, tradeable data.

---

## Data Merging Strategy

Data from three sources merged as follows:

1. **INNER JOIN:** greeks_all + OHLC  
   - Join keys: `symbol, timestamp, expiration, strike, right`
   - Both sources must have matching record

2. **LEFT JOIN:** merged_data + open_interest  
   - Join keys: `symbol, expiration, strike, right`
   - OI optional (creates NaN if missing)

3. **Duplicate handling:** Keep last occurrence (applied to 2021-02 only)

---

## Important Notes

### Temporal Integrity
- All data has proper temporal ordering (no look-ahead bias)
- `open_interest` reported at 6:30 AM from previous day's close
- Greeks and prices are contemporaneous (within milliseconds)

### Execution Assumptions
- **Entry (buying options):** Execute at `ask` price
- **Exit (selling options):** Execute at `bid` price
- This represents realistic, conservative execution costs

### Volume Characteristics
- **~91% of rows have `volume = 0`** (theoretical quotes, not traded)
- **~9% of rows have `volume > 0`** (actual tradeable opportunities)
- Filter to `volume > 0` during ML training for realistic backtests

### Missing Data
- **Open Interest:** 5-7% of rows have NaN (contracts not in OI file)
- **All other columns:** Zero missing values (guaranteed by filters)


## File Naming Convention

**Format:** `YYYY_MM.parquet`  
**Example:** `2021_03.parquet` = March 2021

Each file contains all observations for one calendar month.

---

## Metadata Files

Each parquet file has a corresponding `YYYY_MM_metadata.json` file containing:
- Processing date
- Row counts
- Date range
- Filters applied
- OI coverage statistics
- Validation results

---

## Version Information

- **Created:** February 2026
- **Data Source:** ThetaData
- **Processing Tool:** Custom Python ETL (pandas, pyarrow)
- **Compression:** Snappy (parquet default)

---

## Contact & Documentation

For questions about this dataset or the ETL pipeline, refer to:
- ETL Script: `process_options_etl.py`
- Validation Script: `validate_all_parquets.py`

---