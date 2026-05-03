import pandas as pd
from pathlib import Path

def deduplicate_file(filepath: str):
    """Remove duplicates from a parquet file"""
    
    file_path = Path(filepath)
    
    print("="*70)
    print(f"DEDUPLICATING: {file_path.name}")
    print("="*70)
    
    # Load the file
    print("\n1. Loading parquet file...")
    df = pd.read_parquet(file_path)
    
    initial_rows = len(df)
    print(f" Initial rows: {initial_rows:,}")
    
    # Check for duplicates
    print("\n2. Checking for duplicates...")
    duplicate_mask = df.duplicated(subset=['timestamp', 'expiration', 'strike', 'right'], keep=False)
    duplicate_count = duplicate_mask.sum()
    print(f" Duplicate rows found: {duplicate_count:,}")
    
    if duplicate_count == 0:
        print("\n No duplicates found - file is already clean!")
        return
    
    # Show example duplicates
    print("\n3. Example duplicate (first 3 rows):")
    duplicates_sample = df[duplicate_mask].head(6)
    print(duplicates_sample[['timestamp', 'expiration', 'strike', 'right', 'bid', 'ask', 'volume']])
    
    # Remove duplicates
    print("\n4. Removing duplicates (keeping last occurrence)...")
    df_clean = df.drop_duplicates(
        subset=['timestamp', 'expiration', 'strike', 'right'],
        keep='last'
    )
    
    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows
    
    print(f" Rows after deduplication: {final_rows:,}")
    print(f" Rows removed: {removed_rows:,}")
    
    # Create backup
    backup_path = file_path.parent / f"{file_path.stem}_backup.parquet"
    print(f"\n5. Creating backup: {backup_path.name}")
    df.to_parquet(backup_path, compression='snappy', index=False)
    
    # Save cleaned file
    print(f"\n6. Saving cleaned file: {file_path.name}")
    df_clean.to_parquet(file_path, compression='snappy', index=False)
    
    # Verify
    print("\n7. Verifying cleaned file...")
    df_verify = pd.read_parquet(file_path)
    verify_duplicates = df_verify.duplicated(subset=['timestamp', 'expiration', 'strike', 'right']).sum()
    
    if verify_duplicates == 0:
        print(f"  Verification passed: 0 duplicates")
    else:
        print(f"  WARNING: {verify_duplicates:,} duplicates still found!")
    
    print("\n" + "="*70)
    print("DEDUPLICATION COMPLETE")
    print("="*70)
    print(f" Original file: {initial_rows:,} rows -> {final_rows:,} rows")
    print(f" Removed: {removed_rows:,} duplicate rows")
    print(f" Backup saved: {backup_path.name}")
    print("="*70)


if __name__ == "__main__":
    # Deduplicate the file
    deduplicate_file("2021_02.parquet")
