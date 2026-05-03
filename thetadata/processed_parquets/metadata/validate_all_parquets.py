import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


def validate_parquet_quick(parquet_file: Path) -> dict:
    """Quick validation of a parquet file"""
    
    try:
        # Load parquet
        df = pd.read_parquet(parquet_file)
        
        # Run checks
        issues = []
        
        # Check for critical nulls
        if df['bid'].isna().sum() > 0:
            issues.append(f"Null bids: {df['bid'].isna().sum():,}")
        
        if df['ask'].isna().sum() > 0:
            issues.append(f"Null asks: {df['ask'].isna().sum():,}")
        
        # Check for invalid spreads
        invalid_spreads = (df['ask'] < df['bid']).sum()
        if invalid_spreads > 0:
            issues.append(f"Invalid spreads: {invalid_spreads:,}")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['timestamp', 'expiration', 'strike', 'right']).sum()
        if duplicates > 0:
            issues.append(f"Duplicates: {duplicates:,}")
        
        # Calculate stats
        oi_coverage = df['open_interest'].notna().sum() / len(df)
        volume_rows = (df['volume'] > 0).sum()
        
        return {
            'file': parquet_file.name,
            'success': True,
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': parquet_file.stat().st_size / (1024*1024),
            'date_range': (str(df['date'].min()), str(df['date'].max())),
            'dte_range': (int(df['dte'].min()), int(df['dte'].max())),
            'oi_coverage': oi_coverage,
            'volume_gt_0': volume_rows,
            'issues': issues,
            'status': 'PASS' if not issues else ' ISSUES FOUND'
        }
    
    except Exception as e:
        return {
            'file': parquet_file.name,
            'success': False,
            'error': str(e),
            'status': 'FAILED'
        }


def main():
    # Get current directory
    parquet_dir = Path.cwd()
    
    # Find all parquet files
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("Error: No parquet files found in current directory")
        print(f"Current directory: {parquet_dir}")
        sys.exit(1)
    
    print("="*80)
    print("BATCH VALIDATION - ALL PARQUET FILES")
    print("="*80)
    print(f"Found {len(parquet_files)} parquet files\n")
    
    # Validate each file
    results = []
    total_rows = 0
    total_size = 0
    
    for parquet_file in parquet_files:
        print(f"Validating {parquet_file.name}...", end=" ")
        result = validate_parquet_quick(parquet_file)
        results.append(result)
        
        if result['success']:
            print(result['status'])
            total_rows += result['rows']
            total_size += result['size_mb']
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")
    
    # Print summary table
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"{'File':<20} {'Rows':>12} {'Size (MB)':>10} {'OI %':>8} {'Vol>0':>12} {'Status':<15}")
    print("-"*80)
    
    for result in results:
        if result['success']:
            print(f"{result['file']:<20} {result['rows']:>12,} {result['size_mb']:>10.1f} "
                  f"{result['oi_coverage']*100:>7.1f}% {result['volume_gt_0']:>12,} {result['status']:<15}")
        else:
            print(f"{result['file']:<20} {'ERROR':>12} {'':>10} {'':>8} {'':>12} {result['status']:<15}")
    
    print("-"*80)
    print(f"{'TOTAL':<20} {total_rows:>12,} {total_size:>10.1f} MB")
    print("="*80)
    
    # Print issues if any
    issues_found = [r for r in results if r['success'] and r['issues']]
    if issues_found:
        print("\n ISSUES DETECTED:")
        for result in issues_found:
            print(f"\n{result['file']}:")
            for issue in result['issues']:
                print(f"- {issue}")
    
    # Print failures if any
    failures = [r for r in results if not r['success']]
    if failures:
        print("\nFAILED FILES:")
        for result in failures:
            print(f"- {result['file']}: {result.get('error', 'Unknown error')}")
    
    # Overall status
    print("\n" + "="*80)
    successful = sum(1 for r in results if r['success'] and not r['issues'])
    with_issues = sum(1 for r in results if r['success'] and r['issues'])
    failed = sum(1 for r in results if not r['success'])
    
    print(f"Files validated: {len(results)}")
    print(f"Passed: {successful}")
    if with_issues > 0:
        print(f" With issues: {with_issues}")
    if failed > 0:
        print(f"Failed: {failed}")
    
    if failed == 0 and with_issues == 0:
        print("\nALL FILES VALIDATED SUCCESSFULLY!")
    elif failed > 0:
        print("\nSOME FILES FAILED VALIDATION")
        sys.exit(1)
    else:
        print("\n VALIDATION COMPLETE WITH WARNINGS")
    
    print("="*80)
    
    # Save detailed report
    report_file = parquet_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            'validation_date': datetime.now().isoformat(),
            'total_files': len(results),
            'total_rows': total_rows,
            'total_size_mb': round(total_size, 2),
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file.name}\n")


if __name__ == "__main__":
    main()
