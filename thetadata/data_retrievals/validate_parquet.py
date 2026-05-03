import sys
from pathlib import Path
import pandas as pd
import json


def validate_parquet(parquet_file: str):
    """Validate a parquet file and show summary statistics"""
    
    file_path = Path(parquet_file)
    
    if not file_path.exists():
        print(f"Error: File not found: {parquet_file}")
        sys.exit(1)
    
    print("="*70)
    print(f"VALIDATING: {file_path.name}")
    print("="*70)
    
    # Load parquet
    print("\nLoading parquet...")
    df = pd.read_parquet(file_path)
    
    # Basic info
    print(f"\n Total rows: {len(df):,}")
    print(f" Total columns: {len(df.columns)}")
    print(f" File size: {file_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024*1024)
    print(f" Memory usage: {memory_mb:.1f} MB")
    
    # Column list
    print(f"\n Columns ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        print(f"- {col:20s} {str(dtype):15s} nulls={null_count:8,} ({null_pct:5.1f}%)")
    
    # Date range
    print(f"\n Date range:")
    print(f"From: {df['date'].min()}")
    print(f"To:   {df['date'].max()}")
    print(f"Days: {df['date'].nunique()}")
    
    # Timestamp range
    print(f"\n Timestamp range:")
    print(f"From: {df['timestamp'].min()}")
    print(f"To:   {df['timestamp'].max()}")
    
    # DTE range
    print(f"\n DTE (Days to Expiration):")
    print(f"Min: {df['dte'].min()}")
    print(f"Max: {df['dte'].max()}")
    print(f"Mean: {df['dte'].mean():.1f}")
    
    # Price stats
    print(f"\n Pricing:")
    print(f"Bid range: ${df['bid'].min():.2f} - ${df['bid'].max():.2f}")
    print(f"Ask range: ${df['ask'].min():.2f} - ${df['ask'].max():.2f}")
    print(f"Median spread: ${df['spread'].median():.2f}")
    print(f"Median spread %: {df['spread_pct'].median()*100:.1f}%")
    
    # Volume stats
    print(f"\n Volume:")
    print(f"Total volume: {df['volume'].sum():,}")
    print(f"Median volume: {df['volume'].median():.0f}")
    print(f"Rows with volume=0: {(df['volume']==0).sum():,}")
    print(f"Rows with volume>0: {(df['volume']>0).sum():,}")
    
    # Open Interest
    oi_coverage = df['open_interest'].notna().sum() / len(df)
    print(f"\n Open Interest:")
    print(f"Coverage: {oi_coverage*100:.1f}%")
    print(f"Median OI: {df['open_interest'].median():.0f}")
    
    # Greeks stats
    print(f"\n Greeks:")
    print(f"Delta range: {df['delta'].min():.3f} to {df['delta'].max():.3f}")
    print(f"Gamma range: {df['gamma'].min():.6f} to {df['gamma'].max():.6f}")
    print(f"IV range: {df['implied_vol'].min():.3f} to {df['implied_vol'].max():.3f}")
    
    # Data quality checks
    print(f"\n Data Quality Checks:")
    issues = []
    
    # Check for invalid spreads
    invalid_spreads = (df['ask'] < df['bid']).sum()
    if invalid_spreads > 0:
        issues.append(f"Found {invalid_spreads:,} rows where ask < bid")
    else:
        print(f" No invalid spreads (ask < bid)")
    
    # Check for null critical fields
    critical_fields = ['bid', 'ask', 'delta', 'gamma', 'theta', 'vega']
    for field in critical_fields:
        null_count = df[field].isna().sum()
        if null_count > 0:
            issues.append(f"Found {null_count:,} nulls in {field}")
        else:
            print(f" No nulls in {field}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['timestamp', 'expiration', 'strike', 'right']).sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates:,} duplicate rows")
    else:
        print(f" No duplicate rows")
    
    # Sample data
    print(f"\n Sample rows (first 5):")
    print(df[['date', 'timestamp', 'strike', 'right', 'bid', 'ask', 'volume', 'delta', 'dte']].head().to_string())
    
    # Final verdict
    print("\n" + "="*70)
    if issues:
        print("VALIDATION FAILED")
        for issue in issues:
            print(f"{issue}")
    else:
        print("VALIDATION PASSED")
        print("All data quality checks passed successfully!")
    print("="*70)
    
    # Load metadata if exists
    metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
    if metadata_file.exists():
        print(f"\nMetadata file found: {metadata_file.name}")
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_parquet.py <parquet_file>")
        print("Example: python validate_parquet.py 2018_01.parquet")
        sys.exit(1)
    
    validate_parquet(sys.argv[1])