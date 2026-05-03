import numpy as np
import pandas as pd
from pathlib import Path
from data_utils import ParquetFileManager
from utils import Logger, ConfigManager

PROCESSED_COLUMNS = [
    'timestamp',
    'ask',
    'bid',
    'expiration',
    'strike',
    'right',
    'date',
    'dte',
    'moneyness',
    'delta',
    'gamma',
    'theta',
    'vega',
    'implied_vol',
    'spread_pct',
    'volume',
    'open_interest',
    'underlying_price',
    'close',
    'vwap',
    'count',
    'spy_5m_return',
    'spy_15m_return',
    'spy_30m_return',
    'spy_45m_return',
    'spy_60m_return',
    'iv_rank',
    'realized_vol',
    'intraday_churn_ratio',
    'theta_pressure',
    'short_dte_volume_share',
    'put_call_volume_ratio',
    'net_gex',
    'ts_slope_7_30',
    'iv_hump_ratio',
    'put_skew_25',
    'forward_IV_spike_ratio',
    'prev_day_realized_vol_rank',
    'event_vega_concentration_index',
    'intraday_path_churn_ratio_live',
]

RAW_DIR = "data"
PROCESSED_DIR = "data_processed"
LOG_PATH = "logs/preprocess.log"

def _compute_fwd_spike(group):
    g = group.sort_values('dte')
    Ts = g['T'].values
    tvs = g['total_var'].values
    if len(tvs) < 2:
        return 1.0
    fwd = []
    for j in range(len(tvs) - 1):
        dt = Ts[j + 1] - Ts[j]
        diff = tvs[j + 1] - tvs[j]
        if dt > 1e-6 and diff > 0:
            fwd.append(np.sqrt(diff / dt))
    if len(fwd) < 2:
        return 1.0
    med = np.median(fwd)
    if med < 1e-8:
        return 1.0
    return float(max(fwd) / med)

if __name__ == '__main__':

    logger = Logger(LOG_PATH)
    logger.log("Starting preprocessing pipeline")

    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    file_manager = ParquetFileManager(RAW_DIR)
    all_files = file_manager.get_files_between(2018, 1, 2026, 12)

    logger.log(f"Total files found: {len(all_files)}")


    # PASS 1: Build daily IV rank across all files.
    # Rolling 252-day percentile rank of daily ATM implied vol.
    # Strictly backward-looking - no look-ahead bias.

    logger.log("Computing daily IV rank across all files...")

    daily_iv_frames = []
    for path in all_files:
        df = pd.read_parquet(path, columns=['date', 'moneyness', 'implied_vol'])
        atm = df[(df['moneyness'] >= 0.99) & (df['moneyness'] <= 1.01)]
        daily = atm.groupby('date')['implied_vol'].median().reset_index()
        daily_iv_frames.append(daily)
        del df, atm, daily

    daily_iv = pd.concat(daily_iv_frames, ignore_index=True)
    daily_iv = daily_iv.groupby('date')['implied_vol'].median().reset_index()
    daily_iv = daily_iv.sort_values('date').reset_index(drop=True)
    daily_iv['date'] = pd.to_datetime(daily_iv['date'])
    del daily_iv_frames

    daily_iv['iv_rank'] = daily_iv['implied_vol'].rolling(252, min_periods=30).rank(pct=True)
    daily_iv = daily_iv[['date', 'iv_rank']]

    logger.log("IV rank computed.")

    # PASS 2: Build daily realized volatility across all files.
    # 20-day rolling std of daily returns, annualized. Strictly backward-looking.

    logger.log("Computing daily realized volatility across all files...")

    daily_price_frames = []
    for path in all_files:
        df = pd.read_parquet(path, columns=['date', 'underlying_price'])
        daily_close = df.groupby('date')['underlying_price'].last().reset_index()
        daily_price_frames.append(daily_close)
        del df, daily_close

    daily_prices = pd.concat(daily_price_frames, ignore_index=True)
    daily_prices = daily_prices.groupby('date')['underlying_price'].last().reset_index()
    daily_prices = daily_prices.sort_values('date').reset_index(drop=True)
    daily_prices['date'] = pd.to_datetime(daily_prices['date'])
    del daily_price_frames

    daily_prices['daily_return'] = daily_prices['underlying_price'].pct_change()
    daily_prices['realized_vol'] = (
        daily_prices['daily_return']
        .rolling(20, min_periods=10)
        .std()
        * (252 ** 0.5)
    )
    daily_prices = daily_prices[['date', 'realized_vol']]

    logger.log("Realized volatility computed.")

    # PASS 3: Build daily intraday churn ratio across all files.
    # Uses previous day's value - avoids any look-ahead bias.
    # High churn with low net move signals gamma-pinned microstructure.

    logger.log("Computing daily intraday churn ratio across all files...")

    daily_churn_frames = []
    for path in all_files:
        df = pd.read_parquet(path, columns=['date', 'timestamp', 'underlying_price'])
        spy_ts = df[['timestamp', 'date', 'underlying_price']].drop_duplicates().sort_values('timestamp').copy()
        spy_ts['return_5m'] = spy_ts.groupby('date')['underlying_price'].pct_change(5).fillna(0.0)
        daily = spy_ts.groupby('date').agg(
            realized_abs_sum=('return_5m', lambda x: x.abs().sum()),
            daily_rv=('return_5m', lambda x: (x ** 2).sum()),
            first_price=('underlying_price', 'first'),
            last_price=('underlying_price', 'last')
        ).reset_index()
        daily['net_move'] = (daily['last_price'] - daily['first_price']).abs() / daily['first_price'].clip(lower=1e-8)
        daily_churn_frames.append(daily[['date', 'realized_abs_sum', 'net_move', 'daily_rv']])
        del df, spy_ts, daily

    daily_churn = pd.concat(daily_churn_frames, ignore_index=True)
    daily_churn = daily_churn.sort_values('date').reset_index(drop=True)
    daily_churn['date'] = pd.to_datetime(daily_churn['date'])
    daily_churn['intraday_churn_ratio'] = (
        daily_churn['realized_abs_sum'] / daily_churn['net_move'].clip(lower=1e-8)
    ).shift(1).fillna(0.0)

    # prev_day_realized_vol_rank: rolling 252-day percentile rank of previous day's
    # realized variance (sum of squared 5m returns). Strictly backward-looking -
    # shift(1) ensures we use yesterday's RV, ranked against the prior 252 days.
    # High rank = realized vol is elevated vs recent history = dangerous for tight stops.
    daily_churn['prev_day_rv'] = daily_churn['daily_rv'].shift(1)
    daily_churn['prev_day_realized_vol_rank'] = (
        daily_churn['prev_day_rv']
        .rolling(252, min_periods=30)
        .rank(pct=True)
        .fillna(0.0)
    )

    daily_churn = daily_churn[['date', 'intraday_churn_ratio', 'prev_day_realized_vol_rank']]
    del daily_churn_frames

    logger.log("Intraday churn ratio and prev_day_realized_vol_rank computed.")

    # PASS 4: Process each file.
    # Merges daily features, computes SPY momentum, and computes
    # timestamp-level chain aggregates from the cross-sectional options chain.

    for i in range(len(all_files)):
        raw_path = all_files[i]
        output_path = Path(PROCESSED_DIR) / raw_path.name

        if output_path.exists():
            logger.log(f"Skipping {raw_path.name} (already processed)")
            continue

        logger.log(f"Processing {raw_path.name} ({i + 1}/{len(all_files)})...")

        current_df = pd.read_parquet(raw_path)

        # SPY MOMENTUM
        spy_history = current_df[['timestamp', 'date', 'underlying_price']].drop_duplicates().sort_values('timestamp').copy()

        timeframes = [5, 15, 30, 45, 60]
        merge_cols = ['timestamp']

        for minutes in timeframes:
            col_name = f'spy_{minutes}m_return'
            spy_history[col_name] = spy_history.groupby('date')['underlying_price'].pct_change(periods=minutes)
            spy_history[col_name] = spy_history[col_name].fillna(0.0)
            merge_cols.append(col_name)

        # intraday_path_churn_ratio_live: running (sum |1m returns|) / |net 1m return cumsum|
        # from open to each timestamp within the day. Captures same-day whipsaws in real time.
        # High ratio = price traveling far but going nowhere = dangerous for tight stop/target.
        spy_history['return_1m'] = spy_history.groupby('date')['underlying_price'].pct_change(1).fillna(0.0)
        spy_history = spy_history.sort_values(['date', 'timestamp'])
        spy_history['cum_abs_1m'] = spy_history.groupby('date')['return_1m'].transform(lambda x: x.abs().cumsum())
        spy_history['cum_net_1m'] = spy_history.groupby('date')['return_1m'].transform('cumsum')
        spy_history['intraday_path_churn_ratio_live'] = spy_history['cum_abs_1m'] / (spy_history['cum_net_1m'].abs() + 1e-8)
        merge_cols.append('intraday_path_churn_ratio_live')

        current_df = current_df.merge(spy_history[merge_cols], on='timestamp', how='left')
        current_df['intraday_path_churn_ratio_live'] = current_df['intraday_path_churn_ratio_live'].fillna(0.0)
        del spy_history

        # MERGE DAILY FEATURES
        current_df['date'] = pd.to_datetime(current_df['date'])
        current_df = current_df.merge(daily_iv, on='date', how='left')
        current_df['iv_rank'] = current_df['iv_rank'].fillna(0.0)

        current_df = current_df.merge(daily_prices, on='date', how='left')
        current_df['realized_vol'] = current_df['realized_vol'].fillna(0.0)

        current_df = current_df.merge(daily_churn, on='date', how='left')
        current_df['intraday_churn_ratio'] = current_df['intraday_churn_ratio'].fillna(0.0)
        current_df['prev_day_realized_vol_rank'] = current_df['prev_day_realized_vol_rank'].fillna(0.0)

        # CHAIN-LEVEL FEATURES (cross-sectional per timestamp, no look-ahead)
        ts = current_df.copy()

        if pd.api.types.is_object_dtype(ts['right']) or pd.api.types.is_string_dtype(ts['right']):
            is_call = ts['right'].isin(['CALL', 'C', 'Call'])
            is_put = ts['right'].isin(['PUT', 'P', 'Put'])
        else:
            is_call = ts['right'] == 1.0
            is_put = ts['right'] == 0.0

        near_atm = (ts['moneyness'] >= 0.97) & (ts['moneyness'] <= 1.03)

        # theta_pressure: time decay pressure relative to vega for tradeable near-ATM contracts
        tp_rows = ts[near_atm & ts['dte'].between(3, 10)].copy()
        tp_rows['theta_w'] = tp_rows['theta'].abs() * tp_rows['open_interest']
        tp_rows['vega_w'] = tp_rows['vega'] * tp_rows['open_interest']
        theta_sum = tp_rows.groupby('timestamp')['theta_w'].sum()
        vega_sum = tp_rows.groupby('timestamp')['vega_w'].sum()
        theta_pressure = (theta_sum / vega_sum.clip(lower=1e-8)).rename('theta_pressure')
        del tp_rows, theta_sum, vega_sum

        # short_dte_volume_share: fraction of near-expiry activity
        vol_short = ts[ts['dte'] <= 2].groupby('timestamp')['volume'].sum()
        vol_med = ts[ts['dte'] <= 10].groupby('timestamp')['volume'].sum()
        short_dte_volume_share = (vol_short / vol_med.clip(lower=1)).rename('short_dte_volume_share')
        del vol_short, vol_med

        # put_call_volume_ratio: directional flow asymmetry
        vol_put = ts[is_put].groupby('timestamp')['volume'].sum()
        vol_call = ts[is_call].groupby('timestamp')['volume'].sum()
        put_call_volume_ratio = (vol_put / vol_call.clip(lower=1)).rename('put_call_volume_ratio')
        del vol_put, vol_call

        # net_gex: dealer gamma positioning proxy
        ts['gamma_notional'] = ts['gamma'] * ts['open_interest'] * ts['underlying_price'] ** 2
        gex_call = ts[is_call].groupby('timestamp')['gamma_notional'].sum()
        gex_put = ts[is_put].groupby('timestamp')['gamma_notional'].sum()
        spot_sq = ts.groupby('timestamp')['underlying_price'].first() ** 2
        net_gex = ((gex_call - gex_put) / spot_sq.clip(lower=1e-8)).rename('net_gex')
        del gex_call, gex_put, spot_sq

        # IV term structure features
        atm_ts = ts[near_atm]

        iv_atm_7 = atm_ts[atm_ts['dte'].between(4, 10)].groupby('timestamp')['implied_vol'].median()
        iv_atm_30 = atm_ts[atm_ts['dte'].between(20, 40)].groupby('timestamp')['implied_vol'].median()
        ts_slope_7_30 = (iv_atm_7 - iv_atm_30).rename('ts_slope_7_30')

        iv_horizon = atm_ts[atm_ts['dte'].between(3, 7)].groupby('timestamp')['implied_vol'].median()
        iv_max_all = atm_ts.groupby('timestamp')['implied_vol'].max()
        iv_hump_ratio = (iv_horizon / iv_max_all.clip(lower=1e-8)).rename('iv_hump_ratio')

        # put_skew_25: put tail demand relative to ATM
        put_25d = ts[is_put & ts['delta'].abs().between(0.20, 0.30)]
        iv_put_25 = put_25d.groupby('timestamp')['implied_vol'].median()
        put_skew_25 = (iv_put_25 - iv_atm_30).rename('put_skew_25')
        del put_25d, iv_put_25, iv_atm_30

        # forward_IV_spike_ratio: detects event premium concentrated at a specific expiration
        atm_exp = atm_ts.groupby(['timestamp', 'dte'])['implied_vol'].median().reset_index()
        atm_exp = atm_exp.sort_values(['timestamp', 'dte'])
        atm_exp['T'] = atm_exp['dte'] / 365.0
        atm_exp['total_var'] = atm_exp['implied_vol'] ** 2 * atm_exp['T']
        forward_IV_spike_ratio = atm_exp.groupby('timestamp').apply(_compute_fwd_spike).rename('forward_IV_spike_ratio')
        del atm_ts, atm_exp

        # event_vega_concentration_index: Herfindahl-Hirschman Index of OI-weighted vega
        # across expiries. High HHI = vega concentrated in one expiry = binary event premium.
        # When most of the chain's vega sits in one expiration date, IV crush is imminent
        # after that event resolves. 
        ts['vega_oi'] = ts['vega'] * ts['open_interest'].clip(lower=0)
        vega_by_exp = ts.groupby(['timestamp', 'expiration'])['vega_oi'].sum().reset_index()
        vega_total_ts = vega_by_exp.groupby('timestamp')['vega_oi'].sum().rename('vega_total')
        vega_by_exp = vega_by_exp.join(vega_total_ts, on='timestamp')
        vega_by_exp['share'] = vega_by_exp['vega_oi'] / vega_by_exp['vega_total'].clip(lower=1e-8)
        vega_by_exp['share_sq'] = vega_by_exp['share'] ** 2
        event_vega_concentration_index = vega_by_exp.groupby('timestamp')['share_sq'].sum().rename('event_vega_concentration_index')
        del vega_by_exp, vega_total_ts

        # Merge all chain-level features back onto current_df
        chain_features = pd.DataFrame({
            'theta_pressure': theta_pressure,
            'short_dte_volume_share': short_dte_volume_share,
            'put_call_volume_ratio': put_call_volume_ratio,
            'net_gex': net_gex,
            'ts_slope_7_30': ts_slope_7_30,
            'iv_hump_ratio': iv_hump_ratio,
            'put_skew_25': put_skew_25,
            'forward_IV_spike_ratio': forward_IV_spike_ratio,
            'event_vega_concentration_index': event_vega_concentration_index,
        })
        chain_features.index.name = 'timestamp'
        chain_features = chain_features.reset_index()

        current_df = current_df.merge(chain_features, on='timestamp', how='left')
        del ts, chain_features
        del theta_pressure, short_dte_volume_share, put_call_volume_ratio
        del net_gex, ts_slope_7_30, iv_hump_ratio, put_skew_25, forward_IV_spike_ratio
        del event_vega_concentration_index

        current_df = current_df[PROCESSED_COLUMNS]

        current_df['right'] = current_df['right'].map({'CALL': 1.0, 'PUT': 0.0, 'C': 1.0, 'P': 0.0})
        current_df['open_interest'] = current_df['open_interest'].fillna(0.0)

        current_df.to_parquet(output_path, index=False)
        del current_df

        logger.log(f"Saved {output_path.name}")

    logger.log("Preprocessing complete.")
    logger.log(f"Processed files saved to: {PROCESSED_DIR}/")