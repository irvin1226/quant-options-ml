import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class ParquetFileManager:

    def __init__(self, data_dir: str):
        self._all_files = self._discover_files(data_dir)

    def _discover_files(self, data_dir: str):
        directory = Path(data_dir)
        parquet_files = []

        for path in directory.iterdir():
            if path.is_file() and path.suffix == ".parquet":
                parquet_files.append(path)

        parquet_files.sort()
        return parquet_files

    def _extract_year_month(self, file_path: Path):
        name = file_path.name
        year = int(name[:4])
        month = int(name[5:7])
        return (year, month)

    def get_files_between(self, start_year: int, start_month: int, end_year: int, end_month: int):
        start = (start_year, start_month)
        end = (end_year, end_month)

        result = []
        for file_path in self._all_files:
            year_month = self._extract_year_month(file_path)
            if start <= year_month <= end:
                result.append(file_path)
        return result

class Features:
    COLUMNS = [
        'dte', 'moneyness', 'right', 'delta', 'gamma',
        'theta', 'vega', 'implied_vol',
        'spread_pct', 'volume', 'open_interest',
        'underlying_price',
        'spy_5m_return', 'spy_15m_return', 'spy_30m_return',
        'spy_45m_return', 'spy_60m_return',
        'theta_ratio', 'delta_dte',
        'time_value_per_dte', 'vega_to_theta', 'count',
        'iv_rank',
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
        # 'momentum_acceleration',
    ]

    def __init__(self, min_volume: int = 1, max_spread_pct: float = 0.15, min_dte: int = 1, max_dte: int = 60, min_open_interest: int = 100):
        self._min_volume = min_volume
        self._max_spread_pct = max_spread_pct
        self._min_dte = min_dte
        self._max_dte = max_dte
        self._min_open_interest = min_open_interest
        self._scaler = StandardScaler()
        self._fitted = False

    def _filter(self, df: pd.DataFrame):
        mask = (
            (df['volume'] >= self._min_volume) &
            (df['spread_pct'] <= self._max_spread_pct) &
            (df['dte'] >= self._min_dte) &
            (df['dte'] <= self._max_dte) &
            (df['open_interest'] >= self._min_open_interest)
        )
        return df[mask]

    def _prepare(self, df: pd.DataFrame):
        filtered = self._filter(df).copy()

        if pd.api.types.is_object_dtype(filtered['right']) or pd.api.types.is_string_dtype(filtered['right']):
            filtered['right'] = filtered['right'].map({'CALL': 1.0, 'PUT': 0.0, 'C': 1.0, 'P': 0.0})

        filtered['open_interest'] = filtered['open_interest'].fillna(0.0)

        filtered['theta_ratio'] = filtered['theta'] / filtered['ask']

        filtered['delta_dte'] = filtered['delta'] * filtered['dte']

        dte_clipped = filtered['dte'].clip(lower=1)
        is_call = filtered['right'] == 1.0
        strike = filtered['moneyness'] * filtered['underlying_price']
        intrinsic_value = np.where(
            is_call,
            np.maximum(0, filtered['underlying_price'] - strike),
            np.maximum(0, strike - filtered['underlying_price'])
        )
        filtered['time_value_per_dte'] = (filtered['ask'] - intrinsic_value) / dte_clipped

        theta_abs = filtered['theta'].abs()
        theta_abs = theta_abs.where(theta_abs > 1e-8, 1e-8)
        filtered['vega_to_theta'] = filtered['vega'] / theta_abs

        filtered['count'] = filtered['count'].fillna(0.0)

        filtered['iv_rank'] = filtered['iv_rank'].fillna(0.0)

        # Chain-level features computed in preprocess.py - just fill NaN here
        chain_cols = [
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
        for col in chain_cols:
            filtered[col] = filtered[col].fillna(0.0)

        filtered[self.COLUMNS] = filtered[self.COLUMNS].replace([np.inf, -np.inf], np.nan)
        filtered[self.COLUMNS] = filtered[self.COLUMNS].fillna(0.0)

        return filtered

    def fit(self, df: pd.DataFrame):
        prepared = self._prepare(df)
        self._scaler.fit(prepared[self.COLUMNS].values)
        self._fitted = True

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        prepared = self._prepare(df)
        X = self._scaler.transform(prepared[self.COLUMNS].values).astype(np.float32)
        return X, prepared

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)