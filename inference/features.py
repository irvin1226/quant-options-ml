import numpy as np
import pandas as pd
from pathlib import Path

ROLLING_WINDOW = 252
MIN_PERIODS = 30


def _compute_fwd_spike(atmExp):
    g = atmExp.sort_values('dte')
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


def _add_spy_momentum(df, minuteBars):
    momentumWindows = [5, 15, 30, 45, 60]

    if minuteBars.empty:
        for minutes in momentumWindows:
            df[f'spy_{minutes}m_return'] = 0.0
        df['intraday_path_churn_ratio_live'] = 0.0
        return df

    bars = minuteBars.sort_values('timestamp').copy()

    for minutes in momentumWindows:
        bars[f'spy_{minutes}m_return'] = bars['close'].pct_change(periods=minutes).fillna(0.0)

    bars['return_1m'] = bars['close'].pct_change(1).fillna(0.0)
    bars['cum_abs_1m'] = bars['return_1m'].abs().cumsum()
    bars['cum_net_1m'] = bars['return_1m'].cumsum()
    bars['intraday_path_churn_ratio_live'] = bars['cum_abs_1m'] / (bars['cum_net_1m'].abs() + 1e-8)

    latestBar = bars.iloc[-1]

    for minutes in momentumWindows:
        df[f'spy_{minutes}m_return'] = float(latestBar[f'spy_{minutes}m_return'])

    df['intraday_path_churn_ratio_live'] = float(latestBar['intraday_path_churn_ratio_live'])

    return df


def _add_daily_features(df, historyPath, atmIv):
    historyFile = Path(historyPath)
    historyIsMissing = not historyFile.exists() or historyFile.stat().st_size == 0

    if historyIsMissing:
        df['iv_rank'] = 0.0
        df['intraday_churn_ratio'] = 0.0
        df['prev_day_realized_vol_rank'] = 0.0
        return df

    history = pd.read_csv(historyPath, parse_dates=['date'])
    history = history.sort_values('date').reset_index(drop=True)

    todayAtmIv = atmIv if atmIv is not None else np.nan
    ivSeries = pd.concat([history['atm_iv'], pd.Series([todayAtmIv])], ignore_index=True)
    ivRankSeries = ivSeries.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).rank(pct=True)
    todayIvRank = ivRankSeries.iloc[-1]
    df['iv_rank'] = float(todayIvRank) if not np.isnan(todayIvRank) else 0.0

    history['churn_ratio'] = history['realized_abs_sum'] / history['net_move'].clip(lower=1e-8)
    yesterdayChurn = history['churn_ratio'].iloc[-1] if len(history) > 0 else 0.0
    df['intraday_churn_ratio'] = float(yesterdayChurn) if not np.isnan(yesterdayChurn) else 0.0

    history['prev_day_rv'] = history['daily_rv'].shift(1)
    rvRankSeries = (
        history['prev_day_rv']
        .rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS)
        .rank(pct=True)
        .fillna(0.0)
    )
    df['prev_day_realized_vol_rank'] = float(rvRankSeries.iloc[-1])

    return df


def _add_chain_features(df):
    isCall = df['right'] == 1.0
    isPut = df['right'] == 0.0
    nearAtm = (df['moneyness'] >= 0.97) & (df['moneyness'] <= 1.03)

    tpRows = df[nearAtm & df['dte'].between(3, 10)].copy()
    if not tpRows.empty:
        thetaWeighted = (tpRows['theta'].abs() * tpRows['open_interest']).sum()
        vegaWeighted = (tpRows['vega'] * tpRows['open_interest']).sum()
        thetaPressure = thetaWeighted / max(vegaWeighted, 1e-8)
    else:
        thetaPressure = 0.0
    df['theta_pressure'] = thetaPressure

    volShort = df[df['dte'] <= 2]['volume'].sum()
    volMed = df[df['dte'] <= 10]['volume'].sum()
    df['short_dte_volume_share'] = volShort / max(volMed, 1)

    volPut = df[isPut]['volume'].sum()
    volCall = df[isCall]['volume'].sum()
    df['put_call_volume_ratio'] = volPut / max(volCall, 1)

    df['gamma_notional'] = df['gamma'] * df['open_interest'] * df['underlying_price'] ** 2
    gexCall = df[isCall]['gamma_notional'].sum()
    gexPut = df[isPut]['gamma_notional'].sum()
    spotSq = float(df['underlying_price'].iloc[0]) ** 2 if not df.empty else 1.0
    df['net_gex'] = (gexCall - gexPut) / max(spotSq, 1e-8)

    atmDf = df[nearAtm]
    ivAtm7 = atmDf[atmDf['dte'].between(4, 10)]['implied_vol'].median()
    ivAtm30 = atmDf[atmDf['dte'].between(20, 40)]['implied_vol'].median()

    ivAtm7IsMissing = pd.isna(ivAtm7)
    ivAtm30IsMissing = pd.isna(ivAtm30)
    df['ts_slope_7_30'] = 0.0 if (ivAtm7IsMissing or ivAtm30IsMissing) else float(ivAtm7 - ivAtm30)

    ivHorizon = atmDf[atmDf['dte'].between(3, 7)]['implied_vol'].median()
    ivMaxAll = atmDf['implied_vol'].max()
    ivHumpRatio = float(ivHorizon) / max(float(ivMaxAll), 1e-8)
    df['iv_hump_ratio'] = 0.0 if pd.isna(ivHorizon) else ivHumpRatio

    put25d = df[isPut & df['delta'].abs().between(0.20, 0.30)]
    ivPut25 = put25d['implied_vol'].median()
    df['put_skew_25'] = 0.0 if (pd.isna(ivPut25) or ivAtm30IsMissing) else float(ivPut25 - ivAtm30)

    atmExp = atmDf.groupby('dte')['implied_vol'].median().reset_index()
    atmExp['T'] = atmExp['dte'] / 365.0
    atmExp['total_var'] = atmExp['implied_vol'] ** 2 * atmExp['T']
    df['forward_IV_spike_ratio'] = _compute_fwd_spike(atmExp)

    df['vega_oi'] = df['vega'] * df['open_interest'].clip(lower=0)
    vegaByExp = df.groupby('expiration')['vega_oi'].sum()
    vegaTotal = vegaByExp.sum()
    if vegaTotal > 1e-8:
        shares = vegaByExp / vegaTotal
        df['event_vega_concentration_index'] = float((shares ** 2).sum())
    else:
        df['event_vega_concentration_index'] = 0.0

    return df


def _add_contract_features(df):
    df['theta_ratio'] = df['theta'] / df['ask']
    df['delta_dte'] = df['delta'] * df['dte']

    dteClipped = df['dte'].clip(lower=1)
    isCall = df['right'] == 1.0
    strike = df['moneyness'] * df['underlying_price']
    intrinsicValue = np.where(
        isCall,
        np.maximum(0, df['underlying_price'] - strike),
        np.maximum(0, strike - df['underlying_price'])
    )
    df['time_value_per_dte'] = (df['ask'] - intrinsicValue) / dteClipped

    thetaAbs = df['theta'].abs().clip(lower=1e-8)
    df['vega_to_theta'] = df['vega'] / thetaAbs

    return df


def build_features(chainData, minuteBars, historyPath, atmIv):
    df = chainData.copy()

    df['right'] = df['right'].map({'CALL': 1.0, 'PUT': 0.0})
    df['open_interest'] = df['open_interest'].fillna(0.0)
    df['count'] = df['count'].fillna(0.0)

    # Snapshot endpoints return volume=0 before the first trade of the day.
    # The training filter requires volume >= 1 (active bar filter).
    # At inference we care about liquidity (open_interest), not intraday volume.
    # Clip to 1 so the filter passes - open_interest handles the liquidity gate.
    df['volume'] = df['volume'].clip(lower=1)

    df = _add_chain_features(df)
    df = _add_spy_momentum(df, minuteBars)
    df = _add_daily_features(df, historyPath, atmIv)
    df = _add_contract_features(df)

    return df


def update_history(historyPath, atmIv, minuteBars):
    historyFile = Path(historyPath)
    today = pd.Timestamp.now().normalize()

    if not minuteBars.empty:
        bars = minuteBars.sort_values('timestamp').copy()
        bars['return_5m'] = bars['close'].pct_change(5).fillna(0.0)
        dailyRv = float((bars['return_5m'] ** 2).sum())
        realizedAbsSum = float(bars['return_5m'].abs().sum())
        firstPrice = float(bars['close'].iloc[0])
        lastPrice = float(bars['close'].iloc[-1])
        netMove = abs(lastPrice - firstPrice) / max(firstPrice, 1e-8)
    else:
        dailyRv = 0.0
        realizedAbsSum = 0.0
        netMove = 1e-8

    newRow = pd.DataFrame([{
        'date':             today,
        'atm_iv':           atmIv if atmIv is not None else np.nan,
        'daily_rv':         dailyRv,
        'realized_abs_sum': realizedAbsSum,
        'net_move':         netMove,
    }])

    if historyFile.exists() and historyFile.stat().st_size > 0:
        history = pd.read_csv(historyPath, parse_dates=['date'])
        alreadyHasToday = (history['date'] == today).any()
        if not alreadyHasToday:
            history = pd.concat([history, newRow], ignore_index=True)
    else:
        history = newRow

    history.to_csv(historyPath, index=False)