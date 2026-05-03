import argparse
import gc
import json
import joblib
import random
import numpy as np
import pandas as pd
from data_utils import ParquetFileManager, Features
from model_gbt import ModelGBT
from evaluation import Metrics, Backtester
from calibration import BetaCalibrator, find_threshold
from utils import Logger, ConfigManager

# Configuration
config = ConfigManager()

seed = config.get('random_seed')
if seed is None:
    seed = 42

DATA_DIR = "data_labeled"
LOG_PATH = "models/gbt/train_gbt.log"

TARGET_PRECISION = config.get('target_precision')
HC_TARGET_PRECISION = config.get('high_confidence_target_precision')

BREAKEVEN_FLOOR = config.get('stop_loss') / (config.get('target_return') + config.get('stop_loss'))

BLEND_ALPHA = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('--test-year', type=int, default=None)
parser.add_argument('--walk-forward', action='store_true')
parser.add_argument('--group-dro', action='store_true')
parser.add_argument('--mode', choices=['fixed', 'dynamic'], default='fixed',
                    help='fixed: exit at +7%%/-5%% | dynamic: hold to window end with post-target stop check')
args = parser.parse_args()

if args.walk_forward:
    test_years = [2022, 2023, 2024, 2025, 2026]
else:
    test_years = [args.test_year if args.test_year else 2026]

logger = Logger(LOG_PATH)
logger.log("Starting Gradient Boosted Tree training pipeline")
logger.log(f"target_return={config.get('target_return')}  stop_loss={config.get('stop_loss')}  target_precision={TARGET_PRECISION}  scale_pos_weight={config.get('scale_pos_weight')}")
logger.log(f"n_estimators={config.get('gbt_n_estimators')}  learning_rate={config.get('gbt_learning_rate')}  max_depth={config.get('gbt_max_depth')}")
logger.log(f"min_dte={config.get('min_dte')}  max_dte={config.get('max_dte')}  min_open_interest={config.get('min_open_interest')}  max_spread_pct={config.get('max_spread_pct')}")
logger.log(f"breakeven_floor={BREAKEVEN_FLOOR:.4f}  blend_alpha={BLEND_ALPHA}")
logger.log(f"group_dro={args.group_dro}  mode={args.mode}")

if args.walk_forward:
    logger.log(f"Walk-forward mode: test years {test_years}")

file_manager = ParquetFileManager(DATA_DIR)

for test_year in test_years:

    train_start = (2018, 1)
    train_end   = (test_year - 2, 12)
    val_start   = (test_year - 1, 1)
    val_end     = (test_year - 1, 12)
    test_start  = (test_year, 1)
    test_end    = (test_year, 12)

    logger.log(f"")
    logger.log(f"=== Test Year: {test_year} ===")
    logger.log(f"Train: {train_start[0]}-{train_end[0]}  Val: {val_start[0]}  Test: {test_year}")

    logger.log("Discovering files...")

    train_files = file_manager.get_files_between(train_start[0], train_start[1], train_end[0], train_end[1])
    val_files = file_manager.get_files_between(val_start[0], val_start[1], val_end[0], val_end[1])
    test_files = file_manager.get_files_between(test_start[0], test_start[1], test_end[0], test_end[1])

    logger.log(f"Train files: {len(train_files)}")
    logger.log(f"Validation files: {len(val_files)}")
    logger.log(f"Test files: {len(test_files)}")

    logger.log("Fitting feature scaler on training data...")

    features = Features(
        min_volume = config.get('min_volume'),
        max_spread_pct = config.get('max_spread_pct'),
        min_dte = config.get('min_dte'),
        max_dte = config.get('max_dte'),
        min_open_interest = config.get('min_open_interest'),
    )

    random.seed(seed)

    sample_files = random.sample(train_files, min(12, len(train_files)))
    sample_frames = []
    for f in sample_files:
        sample_frames.append(pd.read_parquet(f))
    train_sample = pd.concat(sample_frames, ignore_index=True)
    features.fit(train_sample)
    del sample_frames, train_sample

    logger.log("Scaler fitted.")

    joblib.dump(features, f"models/gbt/gbt_features_{test_year}.pkl")
    logger.log(f"Features scaler saved to models/gbt/gbt_features_{test_year}.pkl")

    logger.log("Loading training data...")

    X_parts = []
    y_parts = []
    for path in train_files:
        df = pd.read_parquet(path)
        X_np, cleaned = features.transform(df)
        y_np = cleaned['label'].values
        del df, cleaned
        X_parts.append(X_np)
        y_parts.append(y_np)
        del X_np, y_np

    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    del X_parts, y_parts

    X_train_df = pd.DataFrame(X_train, columns=Features.COLUMNS)
    logger.log(f"Train rows: {len(y_train):,}")

    group_labels = None
    if args.group_dro:
        iv_rank_vals = X_train_df['iv_rank'].values
        quantile_bins = np.nanpercentile(iv_rank_vals, [25, 50, 75])
        group_labels = np.digitize(iv_rank_vals, quantile_bins)
        group_counts = np.bincount(group_labels)
        logger.log(f"Group DRO: iv_rank quantile bins {quantile_bins}")
        logger.log(f"Group DRO: group sizes {group_counts} (groups 0-3 = low->high iv_rank)")

    logger.log("Training Gradient Boosted Tree...")

    gbt_model = ModelGBT(
        n_estimators = config.get('gbt_n_estimators'),
        learning_rate = config.get('gbt_learning_rate'),
        max_depth = config.get('gbt_max_depth'),
        scale_pos_weight = config.get('scale_pos_weight'),
        random_state = seed,
    )

    if args.group_dro:
        gbt_model.fit_group_dro(X_train_df, y_train, group_labels)
    else:
        gbt_model.fit(X_train_df, y_train)

    del X_train, X_train_df, y_train
    gc.collect()

    logger.log("Gradient Boosted Tree training complete.")

    joblib.dump(gbt_model, f"models/gbt/gbt_model_{test_year}.pkl")
    logger.log(f"GBT model saved to models/gbt/gbt_model_{test_year}.pkl")

    importance_df = pd.DataFrame({
        'feature': Features.COLUMNS,
        'importance': gbt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.log(f"Feature importance (test year {test_year}):\n{importance_df.to_string()}")

    logger.log("Loading validation data for calibration...")

    X_val_parts = []
    y_val_parts = []
    for path in val_files:
        df = pd.read_parquet(path)
        X_np, cleaned = features.transform(df)
        y_val_parts.append(cleaned['label'].values)
        del df, cleaned
        X_val_parts.append(X_np)
        del X_np

    X_val = np.concatenate(X_val_parts)
    y_val = np.concatenate(y_val_parts)
    del X_val_parts, y_val_parts

    X_val_df = pd.DataFrame(X_val, columns=Features.COLUMNS)
    val_raw = gbt_model.predict(X_val_df)
    del X_val, X_val_df

    calibrator = BetaCalibrator()
    calibrator.fit(val_raw, y_val)
    val_calibrated = calibrator.transform(val_raw)

    val_blended = BLEND_ALPHA * val_calibrated + (1 - BLEND_ALPHA) * val_raw

    logger.log("Beta calibration fitted on validation set.")

    joblib.dump(calibrator, f"models/gbt/gbt_calibrator_{test_year}.pkl")
    logger.log(f"GBT calibrator saved to models/gbt/gbt_calibrator_{test_year}.pkl")

    threshold = find_threshold(val_blended, y_val, TARGET_PRECISION, fallback_percentile=99)
    hc_threshold = find_threshold(val_blended, y_val, HC_TARGET_PRECISION, fallback_percentile=99)

    threshold = max(threshold, BREAKEVEN_FLOOR)

    if hc_threshold <= threshold:
        hc_threshold = threshold + 0.05

    logger.log(f"Derived threshold={threshold:.4f} (target_precision={TARGET_PRECISION}, floor={BREAKEVEN_FLOOR:.4f})")
    logger.log(f"Derived HC threshold={hc_threshold:.4f} (target_precision={HC_TARGET_PRECISION})")

    thresholds = {
        'threshold': threshold,
        'hc_threshold': hc_threshold,
        'breakeven_floor': BREAKEVEN_FLOOR,
        'blend_alpha': BLEND_ALPHA,
    }
    with open(f"models/gbt/gbt_thresholds_{test_year}.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    logger.log(f"Thresholds saved to models/gbt/gbt_thresholds_{test_year}.json")

    del val_raw, val_calibrated, val_blended, y_val
    gc.collect()

    logger.log("Loading test data...")

    X_test_parts = []
    y_test_parts = []
    ask_parts = []
    timestamp_parts = []
    realized_return_parts = []
    for path in test_files:
        df = pd.read_parquet(path)
        X_np, cleaned = features.transform(df)
        y_test_parts.append(cleaned['label'].values)
        ask_parts.append(cleaned['ask'].values)
        timestamp_parts.append(cleaned['timestamp'].values)
        if args.mode == 'dynamic':
            realized_return_parts.append(df.loc[cleaned.index, 'realized_return'].values)
        del df, cleaned
        X_test_parts.append(X_np)
        del X_np

    X_test = np.concatenate(X_test_parts)
    y_test = np.concatenate(y_test_parts)
    entry_prices = np.concatenate(ask_parts)
    timestamps = np.concatenate(timestamp_parts)
    realized_returns = np.concatenate(realized_return_parts) if args.mode == 'dynamic' else None
    del X_test_parts, y_test_parts, ask_parts, timestamp_parts, realized_return_parts

    X_test_df = pd.DataFrame(X_test, columns=Features.COLUMNS)

    logger.log(f"Test rows: {len(y_test):,}")
    logger.log("Evaluating Gradient Boosted Tree on test set...")

    test_raw = gbt_model.predict(X_test_df)
    test_calibrated = calibrator.transform(test_raw)
    gbt_predictions = BLEND_ALPHA * test_calibrated + (1 - BLEND_ALPHA) * test_raw
    del X_test, X_test_df, test_raw, test_calibrated
    gc.collect()

    logger.log("Gradient Boosted Tree Metrics:")
    gbt_metrics = Metrics(gbt_predictions, y_test, threshold=threshold)
    gbt_metrics.summary()

    logger.log("Sorting test data chronologically...")
    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]
    entry_prices = entry_prices[sort_idx]
    y_test = y_test[sort_idx]
    gbt_predictions = gbt_predictions[sort_idx]
    if realized_returns is not None:
        realized_returns = realized_returns[sort_idx]

    backtester = Backtester(
        threshold = threshold,
        high_confidence_threshold = hc_threshold,
        starting_capital = 100000.0,
        target_return = config.get('target_return'),
        stop_loss = config.get('stop_loss'),
        position_size = 0.02,
        max_positions = config.get('max_positions'),
        max_high_confidence_positions = config.get('max_high_confidence_positions'),
        commission_per_contract = config.get('commission_per_contract'),
        breakeven_floor = BREAKEVEN_FLOOR,
        max_holding_days = config.get('max_holding_days'),
    )

    logger.log(f"Backtesting Gradient Boosted Tree ({args.mode} mode)...")
    gbt_final_capital, gbt_trade_log = backtester.run(gbt_predictions, y_test, entry_prices, timestamps, realized_returns)
    logger.log(f"Gradient Boosted Tree Backtest Results (Test Year: {test_year}) - {args.mode}:")
    backtester.summary(gbt_final_capital, gbt_trade_log)

    gbt_trades_df = pd.DataFrame(gbt_trade_log)
    gbt_csv_path = f"models/gbt/trades/gbt_trades_{test_year}.csv"
    gbt_trades_df.to_csv(gbt_csv_path, index=False)
    logger.log(f"Trade log saved to {gbt_csv_path}")

logger.log("Pipeline complete.")