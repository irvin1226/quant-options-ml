import pandas as pd
from pathlib import Path

from data_utils import ParquetFileManager
from labels import LabelGenerator
from utils import Logger, ConfigManager

LABELED_COLUMNS = [
    'timestamp',
    'ask',
    'dte',
    'moneyness',
    'right',
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
    'label',
    'realized_return',
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

PROCESSED_DIR = "data_processed"
LABELED_DIR = "data_labeled"
LOG_PATH = "logs/generate_labels.log"


if __name__ == '__main__':

    logger = Logger(LOG_PATH)
    logger.log("Starting label generation pipeline")

    config = ConfigManager()

    target_return = config.get('target_return')
    stop_loss = config.get('stop_loss')
    max_holding_days = config.get('max_holding_days')
    min_volume = config.get('min_volume')
    max_spread_pct = config.get('max_spread_pct')
    min_dte = config.get('min_dte')
    max_dte = config.get('max_dte')
    min_open_interest = config.get('min_open_interest')

    logger.log(f"target_return={target_return}  stop_loss={stop_loss}  max_holding_days={max_holding_days}")
    logger.log(f"min_volume={min_volume}  max_spread_pct={max_spread_pct}  min_dte={min_dte}  max_dte={max_dte}  min_open_interest={min_open_interest}")

    Path(LABELED_DIR).mkdir(parents=True, exist_ok=True)

    file_manager = ParquetFileManager(PROCESSED_DIR)
    all_files = file_manager.get_files_between(2018, 1, 2026, 12)

    logger.log(f"Total processed files found: {len(all_files)}")

    label_generator = LabelGenerator(
        target_return = target_return,
        stop_loss = stop_loss,
        max_holding_days = max_holding_days,
    )

    for i in range(len(all_files)):
        processed_path = all_files[i]
        output_path = Path(LABELED_DIR) / processed_path.name

        if output_path.exists():
            logger.log(f"Skipping {processed_path.name} (already labeled)")
            continue

        logger.log(f"Labeling {processed_path.name} ({i + 1}/{len(all_files)})...")

        current_df = pd.read_parquet(processed_path)

        mask = (
            (current_df['volume'] >= min_volume) &
            (current_df['spread_pct'] <= max_spread_pct) &
            (current_df['dte'] >= min_dte) &
            (current_df['dte'] <= max_dte) &
            (current_df['open_interest'] >= min_open_interest)
        )
        current_df['eval_row'] = mask

        if i + 1 < len(all_files):
            next_df = pd.read_parquet(all_files[i + 1])
            next_df['eval_row'] = False
        else:
            next_df = pd.DataFrame(columns=current_df.columns)
            next_df['eval_row'] = False

        labeled_df = label_generator.generate(current_df, next_df)
        del current_df, next_df

        filtered_df = labeled_df[labeled_df['eval_row']].copy()
        del labeled_df

        pos = int(filtered_df['label'].sum())
        total = len(filtered_df)
        if total > 0:
            mean_realized = filtered_df['realized_return'].mean()
            mean_win_realized = filtered_df.loc[filtered_df['label'] == 1.0, 'realized_return'].mean() if pos > 0 else 0.0
            logger.log(f"{total:,} rows | {pos:,} positive ({pos / total * 100:.2f}%) | mean_realized={mean_realized:.4f} | mean_win_realized={mean_win_realized:.4f}")
        else:
            logger.log("0 rows after filtering")

        filtered_df = filtered_df[LABELED_COLUMNS]
        filtered_df.to_parquet(output_path, index=False)
        del filtered_df

        logger.log(f"Saved {output_path.name}")

    logger.log("Label generation complete.")
    logger.log(f"Labeled files saved to: {LABELED_DIR}/")