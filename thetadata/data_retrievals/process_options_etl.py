import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("/home/irvin-oc/quant-options-ml/thetadata/data_retrievals/theta_data_responses")
OUTPUT_DIR = Path("/home/irvin-oc/quant-options-ml/thetadata/processed_parquets")
YEAR = "2026"

# Garbage filters (applied during ETL)
GARBAGE_FILTERS = {
    'bid_gt_zero': True,        # bid > 0
    'ask_gt_bid': True,         # ask > bid
    'iv_error_max': 100,        # iv_error <= 100
}

# Expected columns from source CSVs
GREEKS_COLUMNS = [
    'symbol', 'expiration', 'strike', 'right', 'timestamp',
    'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'rho',
    'implied_vol', 'iv_error', 'underlying_timestamp', 'underlying_price'
]

OHLC_COLUMNS = [
    'symbol', 'expiration', 'strike', 'right', 'timestamp',
    'open', 'high', 'low', 'close', 'volume', 'count', 'vwap'
]

OI_COLUMNS = [
    'symbol', 'expiration', 'strike', 'right', 'timestamp', 'open_interest'
]

# Final schema (27 columns)
FINAL_COLUMNS = [
    # Identifiers
    'symbol', 'timestamp', 'date', 'expiration', 'strike', 'right',
    # Pricing
    'bid', 'ask', 'mid', 'spread', 'spread_pct',
    'open', 'high', 'low', 'close', 'vwap', 'underlying_price',
    # Volume
    'volume', 'count',
    # Liquidity
    'open_interest',
    # Greeks
    'implied_vol', 'delta', 'gamma', 'theta', 'vega', 'rho',
    # Derived
    'dte', 'moneyness'
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log(message: str, level: str = "INFO"):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def log_error(message: str):
    """Print error message"""
    log(message, "ERROR")


def log_warning(message: str):
    """Print warning message"""
    log(message, "WARN")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_greeks_all(day_path: Path) -> Optional[pd.DataFrame]:
    """Load and concatenate all greeks_all files for a trading day"""
    greeks_dir = day_path / "greeks_all"
    
    if not greeks_dir.exists():
        return None
    
    greeks_files = list(greeks_dir.glob("*.csv"))
    if not greeks_files:
        return None
    
    dfs = []
    for file in greeks_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'])
            dfs.append(df)
        except Exception as e:
            log_warning(f"Failed to read {file.name}: {e}")
            continue
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)


def load_ohlc(day_path: Path) -> Optional[pd.DataFrame]:
    """Load and concatenate all OHLC files for a trading day"""
    ohlc_dir = day_path / "ohlc"
    
    if not ohlc_dir.exists():
        return None
    
    ohlc_files = list(ohlc_dir.glob("*.csv"))
    if not ohlc_files:
        return None
    
    dfs = []
    for file in ohlc_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'])
            dfs.append(df)
        except Exception as e:
            log_warning(f"Failed to read {file.name}: {e}")
            continue
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)


def load_open_interest(day_path: Path) -> Optional[pd.DataFrame]:
    """Load open interest file for a trading day"""
    oi_file = day_path / "open_interest.csv"
    
    if not oi_file.exists():
        return None
    
    try:
        df = pd.read_csv(oi_file)
        # We only need these columns for merging
        return df[['symbol', 'expiration', 'strike', 'right', 'open_interest']]
    except Exception as e:
        log_warning(f"Failed to read open_interest.csv: {e}")
        return None


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_trading_day(year: str, month: str, day: str) -> Optional[pd.DataFrame]:
    """
    Process a single trading day
    
    Returns:
        DataFrame with processed data, or None if processing failed
    """
    day_path = BASE_DIR / year / month / day
    
    if not day_path.exists():
        log_warning(f"Day path not found: {day_path}")
        return None
    
    # Load data
    greeks_df = load_greeks_all(day_path)
    ohlc_df = load_ohlc(day_path)
    oi_df = load_open_interest(day_path)
    
    # Check required data
    if greeks_df is None:
        log_error(f"{year}-{month}-{day}: No greeks data")
        return None
    
    if ohlc_df is None:
        log_error(f"{year}-{month}-{day}: No OHLC data")
        return None
    
    if oi_df is None:
        log_warning(f"{year}-{month}-{day}: No OI data (continuing without it)")
    
    # Track row counts
    greeks_rows = len(greeks_df)
    ohlc_rows = len(ohlc_df)
    
    # INNER JOIN: greeks + ohlc (both must exist)
    merged = greeks_df.merge(
        ohlc_df,
        on=['symbol', 'timestamp', 'expiration', 'strike', 'right'],
        how='inner',
        suffixes=('_greeks', '_ohlc')
    )
    
    if len(merged) == 0:
        log_error(f"{year}-{month}-{day}: Merge resulted in 0 rows")
        return None
    
    # LEFT JOIN: add open interest (optional)
    if oi_df is not None:
        merged = merged.merge(
            oi_df,
            on=['symbol', 'expiration', 'strike', 'right'],
            how='left'
        )
    else:
        merged['open_interest'] = pd.NA
    
    rows_after_merge = len(merged)
    
    # ========================================================================
    # FILTER GARBAGE DATA
    # ========================================================================
    
    # Filter 1: bid > 0
    if GARBAGE_FILTERS['bid_gt_zero']:
        merged = merged[merged['bid'] > 0]
    
    # Filter 2: ask > bid
    if GARBAGE_FILTERS['ask_gt_bid']:
        merged = merged[merged['ask'] > merged['bid']]
    
    # Filter 3: iv_error <= 100
    if GARBAGE_FILTERS['iv_error_max']:
        merged = merged[merged['iv_error'] <= GARBAGE_FILTERS['iv_error_max']]
    
    rows_after_filters = len(merged)
    
    if len(merged) == 0:
        log_warning(f"{year}-{month}-{day}: All rows filtered out")
        return None
    
    # ========================================================================
    # COMPUTE DERIVED FEATURES
    # ========================================================================
    
    # Trading date from folder path
    merged['date'] = pd.to_datetime(f"{year}-{month}-{day}")
    
    # Mid price
    merged['mid'] = (merged['bid'] + merged['ask']) / 2.0
    
    # Spread
    merged['spread'] = merged['ask'] - merged['bid']
    
    # Spread percentage
    merged['spread_pct'] = merged['spread'] / merged['mid']
    
    # Days to expiration
    merged['expiration'] = pd.to_datetime(merged['expiration'])
    merged['dte'] = (merged['expiration'] - merged['date']).dt.days
    
    # Moneyness
    merged['moneyness'] = merged['strike'] / merged['underlying_price']
    
    # ========================================================================
    # SELECT AND ORDER COLUMNS
    # ========================================================================
    
    # Ensure all expected columns exist
    for col in FINAL_COLUMNS:
        if col not in merged.columns:
            log_warning(f"Missing column: {col}")
            merged[col] = pd.NA
    
    # Select only the columns we want, in the correct order
    result = merged[FINAL_COLUMNS].copy()
    
    # Log stats
    oi_coverage = result['open_interest'].notna().sum() / len(result) * 100
    log(f"{year}-{month}-{day}: "
        f"greeks={greeks_rows:,} | ohlc={ohlc_rows:,} | "
        f"merged={rows_after_merge:,} | filtered={rows_after_filters:,} | "
        f"OI coverage={oi_coverage:.1f}%")
    
    return result


def process_month(year: str, month: str) -> Dict:
    """
    Process all trading days in a month
    
    Returns:
        Dictionary with results and metadata
    """
    log(f"\n{'='*70}")
    log(f"PROCESSING {year}-{month}")
    log(f"{'='*70}\n")
    
    month_path = BASE_DIR / year / month
    
    if not month_path.exists():
        log_error(f"Month path not found: {month_path}")
        return {'success': False, 'error': 'Month path not found'}
    
    # Get all day folders
    day_folders = sorted([d for d in month_path.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not day_folders:
        log_error(f"No trading day folders found in {month_path}")
        return {'success': False, 'error': 'No trading days found'}
    
    log(f"Found {len(day_folders)} trading days\n")
    
    # Process each day
    monthly_data = []
    successful_days = 0
    failed_days = 0
    
    for day_folder in day_folders:
        day = day_folder.name
        
        try:
            day_df = process_trading_day(year, month, day)
            
            if day_df is not None:
                monthly_data.append(day_df)
                successful_days += 1
            else:
                failed_days += 1
        
        except Exception as e:
            log_error(f"{year}-{month}-{day}: Unexpected error: {e}")
            failed_days += 1
            continue
    
    if not monthly_data:
        log_error(f"No data processed for {year}-{month}")
        return {'success': False, 'error': 'No valid data'}
    
    # Concatenate all days
    log(f"\nConcatenating {len(monthly_data)} days...")
    month_df = pd.concat(monthly_data, ignore_index=True)
    
    # Sort by timestamp
    month_df = month_df.sort_values(['timestamp', 'expiration', 'strike', 'right'])
    month_df = month_df.reset_index(drop=True)
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    log(f"\nValidating data...")
    
    validation = {
        'total_rows': len(month_df),
        'null_bids': month_df['bid'].isna().sum(),
        'null_asks': month_df['ask'].isna().sum(),
        'null_deltas': month_df['delta'].isna().sum(),
        'invalid_spreads': (month_df['ask'] < month_df['bid']).sum(),
        'date_range': (str(month_df['date'].min()), str(month_df['date'].max())),
        'dte_range': (int(month_df['dte'].min()), int(month_df['dte'].max())),
        'oi_coverage': month_df['open_interest'].notna().sum() / len(month_df)
    }
    
    # Check for issues
    has_issues = False
    if validation['null_bids'] > 0:
        log_error(f"Found {validation['null_bids']} null bids!")
        has_issues = True
    
    if validation['invalid_spreads'] > 0:
        log_error(f"Found {validation['invalid_spreads']} invalid spreads (ask < bid)!")
        has_issues = True
    
    if has_issues:
        log_error("Data validation failed!")
        return {'success': False, 'error': 'Validation failed'}
    
    # ========================================================================
    # SAVE PARQUET
    # ========================================================================
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_file = OUTPUT_DIR / f"{year}_{month}.parquet"
    log(f"\nSaving to: {parquet_file}")
    
    month_df.to_parquet(
        parquet_file,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
    
    # ========================================================================
    # SAVE METADATA
    # ========================================================================
    
    # Convert numpy types to native Python types for JSON serialization
    validation_json = {
        'total_rows': int(validation['total_rows']),
        'null_bids': int(validation['null_bids']),
        'null_asks': int(validation['null_asks']),
        'null_deltas': int(validation['null_deltas']),
        'invalid_spreads': int(validation['invalid_spreads']),
        'date_range': validation['date_range'],
        'dte_range': (int(validation['dte_range'][0]), int(validation['dte_range'][1])),
        'oi_coverage': float(validation['oi_coverage'])
    }
    
    metadata = {
        'month': f"{year}-{month}",
        'processing_date': datetime.now().isoformat(),
        'trading_days': int(len(day_folders)),
        'successful_days': int(successful_days),
        'failed_days': int(failed_days),
        'total_rows': int(len(month_df)),
        'file_size_mb': round(file_size_mb, 2),
        'filters_applied': GARBAGE_FILTERS,
        'validation': validation_json,
        'columns': list(month_df.columns)
    }
    
    metadata_file = OUTPUT_DIR / f"{year}_{month}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    log(f"\n{'='*70}")
    log(f"COMPLETED {year}-{month}")
    log(f"{'='*70}")
    log(f" Trading days processed: {successful_days}/{len(day_folders)}")
    log(f" Total rows: {len(month_df):,}")
    log(f" Date range: {validation['date_range'][0]} to {validation['date_range'][1]}")
    log(f" DTE range: {validation['dte_range'][0]} to {validation['dte_range'][1]}")
    log(f" OI coverage: {validation['oi_coverage']*100:.1f}%")
    log(f" File size: {file_size_mb:.1f} MB")
    log(f" Saved: {parquet_file}")
    log(f" Metadata: {metadata_file}")
    log(f"{'='*70}\n")
    
    return {
        'success': True,
        'metadata': metadata,
        'parquet_file': str(parquet_file)
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process ThetaData options CSVs into monthly Parquet files'
    )
    parser.add_argument(
        '--month',
        type=str,
        help='Process specific month (e.g., 01 for January)'
    )
    parser.add_argument(
        '--months',
        type=str,
        help='Process multiple months (e.g., 01,02,03)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all 12 months'
    )
    
    args = parser.parse_args()
    
    # Determine which months to process
    if args.all:
        months_to_process = [f"{i:02d}" for i in range(1, 13)]
    elif args.months:
        months_to_process = args.months.split(',')
    elif args.month:
        months_to_process = [args.month]
    else:
        print("Error: Must specify --month, --months, or --all")
        parser.print_help()
        sys.exit(1)
    
    # Validate month format
    for month in months_to_process:
        if not (month.isdigit() and 1 <= int(month) <= 12 and len(month) == 2):
            print(f"Error: Invalid month format: {month}")
            print("Use 2-digit format: 01, 02, ..., 12")
            sys.exit(1)
    
    # Start processing
    log(f"Starting ETL for {YEAR}")
    log(f"Months to process: {', '.join(months_to_process)}")
    log(f"Base directory: {BASE_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")
    
    start_time = datetime.now()
    
    results = []
    for month in months_to_process:
        result = process_month(YEAR, month)
        results.append({
            'month': month,
            'success': result['success']
        })
    
    # Final summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    
    log(f"\n{'='*70}")
    log(f"ETL COMPLETE")
    log(f"{'='*70}")
    log(f"Processed: {len(results)} months")
    log(f"Successful: {successful}")
    log(f"Failed: {failed}")
    log(f"Time elapsed: {elapsed/60:.1f} minutes")
    log(f"{'='*70}\n")
    
    if failed > 0:
        log_error(f"{failed} months failed - check logs above")
        sys.exit(1)
    
    log("All months processed successfully!")


if __name__ == "__main__":
    main()