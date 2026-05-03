import argparse
import json
import joblib
import torch
import random
import numpy as np
import pandas as pd
from data_utils import ParquetFileManager, Features
from dataset import VRAMDataLoader
from model_nn import ModelNN, Trainer
from evaluation import Metrics, Backtester
from calibration import PlattCalibrator, find_threshold
from utils import Logger, ConfigManager

# Device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Configuration
config = ConfigManager()

seed = config.get('random_seed')
if seed is None:
    seed = 42

DATA_DIR = "data_labeled"
LOG_PATH = "models/nn/train_nn.log"

BATCH_SIZE = 32768
EPOCHS = config.get('epochs')
POS_WEIGHT = config.get('pos_weight')
TARGET_PRECISION = config.get('target_precision')
HC_TARGET_PRECISION = config.get('high_confidence_target_precision')

BREAKEVEN_FLOOR = config.get('stop_loss') / (config.get('target_return') + config.get('stop_loss'))

MC_DROPOUT_PASSES = 5

parser = argparse.ArgumentParser()
parser.add_argument('--test-year', type=int, default=None)
parser.add_argument('--walk-forward', action='store_true')
parser.add_argument('--mode', choices=['fixed', 'dynamic'], default='dynamic',
                    help='fixed: exit at +7%%/-5%% | dynamic: hold to window end with post-target stop check')
parser.add_argument('--standard', action='store_true', help='Run standard backtester')
parser.add_argument('--exit-model', action='store_true', help='Run exit model backtester (dynamic mode only)')
args = parser.parse_args()

if not args.standard and not args.exit_model:
    args.standard = True

if args.exit_model and args.mode == 'fixed':
    print("Warning: --exit-model requires --mode dynamic. Ignoring --exit-model.")
    args.exit_model = False

if args.walk_forward:
    test_years = [2022, 2023, 2024, 2025, 2026]
else:
    test_years = [args.test_year if args.test_year else 2026]

logger = Logger(LOG_PATH)
logger.log(f"Device: {device}")
logger.log("Starting Neural Network training pipeline")
logger.log(f"target_return={config.get('target_return')}  stop_loss={config.get('stop_loss')}  target_precision={TARGET_PRECISION}  pos_weight={POS_WEIGHT}")
logger.log(f"epochs={EPOCHS}  learning_rate={config.get('nn_learning_rate')}  batch_size={BATCH_SIZE}")
logger.log(f"min_dte={config.get('min_dte')}  max_dte={config.get('max_dte')}  min_open_interest={config.get('min_open_interest')}  max_spread_pct={config.get('max_spread_pct')}")
logger.log(f"breakeven_floor={BREAKEVEN_FLOOR:.4f}")
logger.log(f"mode={args.mode}  standard={args.standard}  exit_model={args.exit_model}")

if args.walk_forward:
    logger.log(f"Walk-forward mode: test years {test_years}")

file_manager = ParquetFileManager(DATA_DIR)

for test_year in test_years:

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    joblib.dump(features, f"models/nn/nn_features_{test_year}.pkl")
    logger.log(f"Features scaler saved to models/nn/nn_features_{test_year}.pkl")

    logger.log("Loading datasets into VRAM...")

    train_loader = VRAMDataLoader(train_files, features, batch_size=BATCH_SIZE, shuffle=True, device=device)
    val_loader = VRAMDataLoader(val_files, features, batch_size=BATCH_SIZE, shuffle=False, device=device)
    test_loader = VRAMDataLoader(test_files, features, batch_size=BATCH_SIZE, shuffle=False, device=device)

    logger.log(f"Train rows: {train_loader.dataset_len:,}")
    logger.log(f"Validation rows: {val_loader.dataset_len:,}")
    logger.log(f"Test rows: {test_loader.dataset_len:,}")

    logger.log("Training Neural Network...")

    input_size = len(Features.COLUMNS)
    nn_model = ModelNN(input_size=input_size)
    trainer = Trainer(nn_model, device=device, learning_rate=config.get('nn_learning_rate'), pos_weight=POS_WEIGHT)

    checkpoint_path = f"models/nn/best_nn_{test_year}.pt"

    train_losses, val_losses = trainer.fit(
        train_loader,
        val_loader,
        epochs = EPOCHS,
        patience = config.get('early_stopping_patience'),
        checkpoint_path = checkpoint_path,
    )

    logger.log("Neural Network training complete.")
    logger.log(f"NN model saved to {checkpoint_path}")

    logger.log("Generating validation predictions for calibration (blend)...")

    nn_model.eval()
    val_logits_parts = []
    val_labels_parts = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            single_logits = nn_model(X_batch)

            for m in nn_model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            mc_passes = []
            for _ in range(MC_DROPOUT_PASSES):
                mc_passes.append(nn_model(X_batch))
            nn_model.eval()
            mc_mean = torch.stack(mc_passes, dim=0).mean(dim=0)

            blended = 0.6 * single_logits + 0.4 * mc_mean
            val_logits_parts.append(blended.cpu().numpy())
            val_labels_parts.append(y_batch.cpu().numpy())

    val_logits = np.concatenate(val_logits_parts).flatten()
    val_labels = np.concatenate(val_labels_parts).flatten()
    del val_logits_parts, val_labels_parts

    calibrator = PlattCalibrator()
    calibrator.fit(val_logits, val_labels)
    val_calibrated = calibrator.transform(val_logits)

    logger.log("Platt scaling calibrator fitted on validation set.")

    joblib.dump(calibrator, f"models/nn/nn_calibrator_{test_year}.pkl")
    logger.log(f"NN calibrator saved to models/nn/nn_calibrator_{test_year}.pkl")

    threshold = find_threshold(val_calibrated, val_labels, TARGET_PRECISION, fallback_percentile=99)
    threshold = max(threshold, BREAKEVEN_FLOOR)

    hc_threshold = find_threshold(val_calibrated, val_labels, HC_TARGET_PRECISION, fallback_percentile=99)

    if hc_threshold <= threshold:
        hc_threshold = threshold + 0.05

    logger.log(f"Derived threshold={threshold:.4f} (target_precision={TARGET_PRECISION}, floor={BREAKEVEN_FLOOR:.4f})")
    logger.log(f"Derived HC threshold={hc_threshold:.4f} (target_precision={HC_TARGET_PRECISION})")

    thresholds = {
        'threshold': threshold,
        'hc_threshold': hc_threshold,
        'breakeven_floor': BREAKEVEN_FLOOR,
        'mc_dropout_passes': MC_DROPOUT_PASSES,
        'blend_alpha': 0.6,
    }
    with open(f"models/nn/nn_thresholds_{test_year}.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    logger.log(f"Thresholds saved to models/nn/nn_thresholds_{test_year}.json")

    logger.log("Evaluating Neural Network on test set (blend)...")

    nn_model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            single_logits = nn_model(X_batch)

            for m in nn_model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
            mc_passes = []
            for _ in range(MC_DROPOUT_PASSES):
                mc_passes.append(nn_model(X_batch))
            nn_model.eval()
            mc_mean = torch.stack(mc_passes, dim=0).mean(dim=0)

            blended = 0.6 * single_logits + 0.4 * mc_mean
            all_logits.append(blended.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    test_logits = np.concatenate(all_logits).flatten()
    nn_true_labels = np.concatenate(all_labels).flatten()
    del all_logits, all_labels

    nn_predictions = calibrator.transform(test_logits)

    print(f"Min:             {np.min(nn_predictions):.4f}")
    print(f"25th Percentile: {np.percentile(nn_predictions, 25):.4f}")
    print(f"50th Percentile: {np.percentile(nn_predictions, 50):.4f}")
    print(f"75th Percentile: {np.percentile(nn_predictions, 75):.4f}")
    print(f"95th Percentile: {np.percentile(nn_predictions, 95):.4f}")
    print(f"99th Percentile: {np.percentile(nn_predictions, 99):.4f}")
    print(f"Max:             {np.max(nn_predictions):.4f}")

    logger.log("Neural Network Metrics:")
    nn_metrics = Metrics(nn_predictions, nn_true_labels, threshold=threshold)
    nn_metrics.summary()

    logger.log("Loading test data for backtester...")

    ask_parts = []
    timestamp_parts = []
    realized_return_parts = []
    for path in test_files:
        df = pd.read_parquet(path)
        _, cleaned = features.transform(df)
        ask_parts.append(cleaned['ask'].values)
        timestamp_parts.append(cleaned['timestamp'].values)
        if args.mode == 'dynamic':
            realized_return_parts.append(df.loc[cleaned.index, 'realized_return'].values)
        del df, cleaned

    entry_prices = np.concatenate(ask_parts)
    timestamps = np.concatenate(timestamp_parts)
    realized_returns = np.concatenate(realized_return_parts) if args.mode == 'dynamic' else None
    del ask_parts, timestamp_parts, realized_return_parts

    if args.exit_model:
        spy_return_parts = []
        for path in test_files:
            df = pd.read_parquet(path)
            _, cleaned = features.transform(df)
            spy_return_parts.append(cleaned['spy_5m_return'].values)
            del df, cleaned
        spy_returns = np.concatenate(spy_return_parts)
        del spy_return_parts
    else:
        spy_returns = None

    logger.log("Sorting test data chronologically...")
    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]
    entry_prices = entry_prices[sort_idx]
    nn_true_labels = nn_true_labels[sort_idx]
    nn_predictions = nn_predictions[sort_idx]
    if realized_returns is not None:
        realized_returns = realized_returns[sort_idx]
    if spy_returns is not None:
        spy_returns = spy_returns[sort_idx]

    np.save(f"models/nn/nn_predictions_{test_year}.npy", nn_predictions)
    np.save(f"models/nn/nn_true_labels_{test_year}.npy", nn_true_labels)

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

    nn_trade_log = None

    if args.standard:
        logger.log(f"Backtesting Neural Network (standard, {args.mode} mode)...")
        nn_final_capital, nn_trade_log = backtester.run(nn_predictions, nn_true_labels, entry_prices, timestamps, realized_returns)
        logger.log(f"Neural Network Backtest Results (Test Year: {test_year}) - Standard ({args.mode}):")
        backtester.summary(nn_final_capital, nn_trade_log)

    if args.exit_model:
        exit_pred_path = f"models/nn/exit_predictions_{test_year}.parquet"
        logger.log("Backtesting Neural Network (with exit model)...")
        nn_exit_capital, nn_exit_log = backtester.run_with_exit(
            nn_predictions, nn_true_labels, entry_prices, timestamps,
            realized_returns, spy_returns, exit_pred_path,
        )
        logger.log(f"Neural Network Backtest Results (Test Year: {test_year}) - With Exit Model:")
        backtester.summary(nn_exit_capital, nn_exit_log)
        if nn_trade_log is None:
            nn_trade_log = nn_exit_log

    if nn_trade_log is not None:
        nn_trades_df = pd.DataFrame(nn_trade_log)
        nn_csv_path = f"models/nn/trades/nn_trades_{test_year}.csv"
        nn_trades_df.to_csv(nn_csv_path, index=False)
        logger.log(f"Trade log saved to {nn_csv_path}")

logger.log("Pipeline complete.")