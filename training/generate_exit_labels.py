import gc
import math
import numpy as np
import pandas as pd
from pathlib import Path
from data_utils import ParquetFileManager
from utils import ConfigManager

EXIT_COLUMNS = [
    'unrealized_return',
    'days_held',
    'dte_remaining',
    'daily_return',
    'delta',
    'gamma',
    'vega',
    'theta',
    'implied_vol',
    'spread_pct',
    'spy_5m_return',
    'spy_15m_return',
    'spy_30m_return',
    'spy_60m_return',
    'iv_rank',
    'net_gex',
    'ts_slope_7_30',
    'moneyness',
]

LABELED_DIR   = "data_labeled"
PROCESSED_DIR = "data_processed"
EXIT_DIR      = "data_exit_labeled"
LOG_PATH      = "logs/generate_exit_labels.log"


def _log(msg):
    from datetime import datetime
    entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(entry, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(entry + "\n")


def _process_month(labeled_path, processed_lookup, max_holding_days):
    yr  = int(labeled_path.name[:4])
    mo  = int(labeled_path.name[5:7])
    output_path = Path(EXIT_DIR) / labeled_path.name

    if output_path.exists():
        _log(f"Skipping {labeled_path.name} (already exists)")
        return

    _log(f"Processing {labeled_path.name}...")

    labeled_df = pd.read_parquet(labeled_path)
    winners = labeled_df[labeled_df['label'] == 1.0].copy()
    winners = winners.dropna(subset=['moneyness', 'underlying_price', 'dte', 'ask', 'right']).reset_index(drop=True)
    del labeled_df

    if len(winners) == 0:
        _log(f"No winners in {labeled_path.name}, skipping")
        return

    _log(f"{labeled_path.name}: {len(winners):,} winning trades found")

    proc_path = processed_lookup.get((yr, mo))
    if proc_path is None:
        _log(f"No processed data for {yr}-{mo:02d}, skipping")
        return

    proc_df = pd.read_parquet(proc_path)

    next_mo = mo + 1
    next_yr = yr
    if next_mo > 12:
        next_mo = 1
        next_yr = yr + 1

    next_proc_path = processed_lookup.get((next_yr, next_mo))
    if next_proc_path is not None:
        next_proc_df = pd.read_parquet(next_proc_path)
        combined = pd.concat([proc_df, next_proc_df], ignore_index=True)
        del proc_df, next_proc_df
    else:
        combined = proc_df
        del proc_df

    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined['date_only'] = combined['timestamp'].dt.normalize()
    combined['strike_r']  = (combined['strike'] * 2).round() / 2

    # Build grouped index - O(1) per-winner contract lookup
    contract_groups = {}
    for (exp, sr, rt), grp in combined.groupby(['expiration', 'strike_r', 'right'], sort=False):
        k = (pd.Timestamp(exp).strftime('%Y-%m-%d'), round(float(sr) * 2) / 2, round(float(rt)))
        contract_groups[k] = grp.sort_values('timestamp').reset_index(drop=True)

    _log(f"{labeled_path.name}: Indexed {len(contract_groups):,} unique contracts")
    del combined
    gc.collect()

    # Feature columns to extract as arrays for fast access
    feat_cols = ['delta', 'gamma', 'vega', 'theta', 'implied_vol', 'spread_pct',
                 'spy_5m_return', 'spy_15m_return', 'spy_30m_return', 'spy_60m_return',
                 'iv_rank', 'net_gex', 'ts_slope_7_30', 'moneyness', 'dte']

    all_rows = []

    # itertuples is ~10x faster than iterrows for Python-level iteration
    for winner in winners.itertuples(index=False):
        try:
            m = float(winner.moneyness)
            u = float(winner.underlying_price)
            d = float(winner.dte)
            a = float(winner.ask)
            r = float(winner.right)
            if not all(math.isfinite(x) for x in [m, u, d, a, r]):
                continue
            if m <= 0 or u <= 0 or a <= 0:
                continue
        except (TypeError, ValueError):
            continue

        entry_ts   = pd.Timestamp(winner.timestamp)
        entry_ask  = a
        entry_date = entry_ts.normalize()
        window_end = entry_date + pd.Timedelta(days=max_holding_days)

        expiration = entry_date + pd.Timedelta(days=int(d))
        strike     = round(m * u * 2) / 2
        right      = round(r)

        k   = (expiration.strftime('%Y-%m-%d'), strike, right)
        grp = contract_groups.get(k)
        if grp is None:
            continue

        # Filter to post-entry ticks within window
        time_mask = (grp['timestamp'] > entry_ts) & (grp['date_only'] <= window_end)
        ticks = grp[time_mask]
        if len(ticks) == 0:
            continue

        # Take first tick per trading day - groupby is faster than manual set loop
        daily_ticks = ticks.groupby('date_only', sort=True).first().reset_index()
        n = len(daily_ticks)
        if n == 0:
            continue

        # Extract all needed arrays upfront - avoids iloc per row
        bids       = daily_ticks['bid'].values.astype(np.float64)
        timestamps = daily_ticks['timestamp'].values
        dates      = daily_ticks['date_only'].values

        valid = bids > 0
        if not np.any(valid):
            continue

        # Vectorized derived columns
        unrealized = (bids - entry_ask) / entry_ask
        prev_bids  = np.empty(n)
        prev_bids[0] = entry_ask
        prev_bids[1:] = bids[:-1]
        daily_ret = (bids - prev_bids) / entry_ask

        entry_date_np = np.datetime64(entry_date, 'D')
        days_held = (dates.astype('datetime64[D]') - entry_date_np).astype(int)

        suffix_max = np.zeros(n)
        running = -np.inf
        for j in range(n - 1, -1, -1):
            suffix_max[j] = running
            if bids[j] > running:
                running = bids[j]
        # Where suffix_max is still -inf (last position), remaining upside is 0
        suffix_max = np.where(suffix_max == -np.inf, 0.0, suffix_max)
        remaining_upside = np.clip((suffix_max - bids) / entry_ask, 0.0, 5.0)

        # Extract feature arrays - fillna already applied in data_utils but be safe
        feat_arrays = {}
        for col in feat_cols:
            if col in daily_ticks.columns:
                arr = daily_ticks[col].values.astype(np.float64)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                feat_arrays[col] = arr
            else:
                feat_arrays[col] = np.zeros(n)

        # Build rows - single pass over n daily samples
        for j in range(n):
            if not valid[j]:
                continue
            all_rows.append({
                'unrealized_return': float(unrealized[j]),
                'days_held':         int(days_held[j]),
                'dte_remaining':     int(feat_arrays['dte'][j]),
                'daily_return':      float(daily_ret[j]),
                'delta':             float(feat_arrays['delta'][j]),
                'gamma':             float(feat_arrays['gamma'][j]),
                'vega':              float(feat_arrays['vega'][j]),
                'theta':             float(feat_arrays['theta'][j]),
                'implied_vol':       float(feat_arrays['implied_vol'][j]),
                'spread_pct':        float(feat_arrays['spread_pct'][j]),
                'spy_5m_return':     float(feat_arrays['spy_5m_return'][j]),
                'spy_15m_return':    float(feat_arrays['spy_15m_return'][j]),
                'spy_30m_return':    float(feat_arrays['spy_30m_return'][j]),
                'spy_60m_return':    float(feat_arrays['spy_60m_return'][j]),
                'iv_rank':           float(feat_arrays['iv_rank'][j]),
                'net_gex':           float(feat_arrays['net_gex'][j]),
                'ts_slope_7_30':     float(feat_arrays['ts_slope_7_30'][j]),
                'moneyness':         float(feat_arrays['moneyness'][j]),
                'remaining_upside':  float(remaining_upside[j]),
                'entry_timestamp':   entry_ts,
                'current_timestamp': timestamps[j],
                'entry_ask':         entry_ask,
            })

    del contract_groups
    gc.collect()

    if len(all_rows) == 0:
        _log(f"No sequences generated for {labeled_path.name}")
        return

    exit_df = pd.DataFrame(all_rows)
    exit_df.to_parquet(output_path, index=False)

    pos_pct    = (exit_df['remaining_upside'] > 0).mean() * 100
    mean_up    = exit_df['remaining_upside'].mean()
    mean_unrel = exit_df['unrealized_return'].mean()
    _log(
        f"Saved {output_path.name}: {len(exit_df):,} rows | "
        f"mean_remaining={mean_up:.3f} | {pos_pct:.1f}% have upside | "
        f"mean_unrealized={mean_unrel:.3f}"
    )
    del exit_df
    gc.collect()


if __name__ == '__main__':
    _log("Starting exit label generation pipeline")

    config           = ConfigManager()
    max_holding_days = config.get('max_holding_days')

    Path(EXIT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_PATH).touch(exist_ok=True)

    labeled_manager   = ParquetFileManager(LABELED_DIR)
    processed_manager = ParquetFileManager(PROCESSED_DIR)

    labeled_files   = labeled_manager.get_files_between(2018, 1, 2026, 12)
    processed_files = processed_manager.get_files_between(2018, 1, 2026, 12)

    processed_lookup = {}
    for pf in processed_files:
        yr = int(pf.name[:4])
        mo = int(pf.name[5:7])
        processed_lookup[(yr, mo)] = pf

    _log(f"Labeled files:   {len(labeled_files)}")
    _log(f"Processed files: {len(processed_files)}")
    _log("Running sequentially - inner vectorized ops use full CPU via numpy")

    for labeled_path in labeled_files:
        try:
            _process_month(labeled_path, processed_lookup, max_holding_days)
        except Exception as e:
            import traceback
            _log(f"ERROR in {labeled_path.name}: {e}")
            _log(traceback.format_exc())

    _log("Exit label generation complete.")
    _log(f"Files saved to: {EXIT_DIR}/")