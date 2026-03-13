import pandas as pd
import numpy as np
import scipy.stats

def data_profiler(df: pd.DataFrame) -> dict:
    """Takes any DataFrame as input and generates a complete data profile dict."""
    print("=" * 50)
    print("           AUTOMATED DATA PROFILER")
    print("=" * 50)
    
    total_rows = len(df)
    profile = {
        'dataset_summary': {
            'total_rows': total_rows,
            'total_columns': len(df.columns),
            'total_memory_mb': float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        },
        'columns': {},
        'potential_issues': []
    }
    
    print(f"Rows: {total_rows} | Columns: {len(df.columns)}")
    print(f"Memory: {profile['dataset_summary']['total_memory_mb']:.4f} MB\n")
    
    for col in df.columns:
        series = df[col]
        
        # Base stats common to all types
        col_type = str(series.dtype)
        unique_cnt = series.nunique(dropna=False)
        null_cnt = series.isna().sum()
        null_pct = (null_cnt / total_rows) * 100 if total_rows > 0 else 0
        top_5 = series.value_counts(dropna=False).head(5).to_dict()
        
        col_profile = {
            'dtype': col_type,
            'unique_count': int(unique_cnt),
            'null_count': int(null_cnt),
            'null_percentage': float(null_pct),
            'top_5_frequent': {str(k): int(v) for k, v in top_5.items()}
        }
        
        # Check potential issues (Single Value columns)
        if unique_cnt == 1:
            profile['potential_issues'].append(f"Column '{col}' is a single-value constant.")
            
        if pd.api.types.is_numeric_dtype(series):
            # Numeric extended stats
            valid_nums = series.dropna()
            if len(valid_nums) > 0:
                mean_val = valid_nums.mean()
                std_val = valid_nums.std()
                
                # Check for suspicious outliers (> 3 std dev from mean)
                outliers = valid_nums[np.abs(valid_nums - mean_val) > 3 * std_val]
                if len(outliers) > 0:
                     profile['potential_issues'].append(f"Column '{col}' has {len(outliers)} outliers (> 3 std dev).")

                col_profile.update({
                    'min': float(valid_nums.min()),
                    'max': float(valid_nums.max()),
                    'mean': float(mean_val),
                    'median': float(valid_nums.median()),
                    'std': float(std_val) if pd.notna(std_val) else 0.0,
                    'skewness': float(scipy.stats.skew(valid_nums))
                })
                
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # String extended stats
            valid_strs = series.dropna().astype(str)
            if len(valid_strs) > 0:
                lens = valid_strs.str.len()
                col_profile.update({
                    'avg_length': float(lens.mean()),
                    'min_length': int(lens.min()),
                    'max_length': int(lens.max())
                })
                
                # High cardinality check (>90% uniqueness indicating likely random IDs/UUIDs)
                if unique_cnt > (total_rows * 0.9) and total_rows > 10:
                    profile['potential_issues'].append(f"Column '{col}' has high cardinality (possibly random ID field).")

        profile['columns'][col] = col_profile
        
        print(f"-> Column: [{col}] | Type: {col_type}")
        print(f"   Nulls: {null_pct:.2f}% | Uniques: {unique_cnt}")
        
    print("\n--- Potential Issues Identified ---")
    for issue in profile['potential_issues']:
        print(f" [!] {issue}")
        
    print("=" * 50)
    return profile

if __name__ == '__main__':
    # Test dataset mapping out different rules natively
    df_test = pd.DataFrame({
        'ids': range(100), # High cardinality string simulation mapping
        'constant': ['Same'] * 100, # Single value column
        'values': np.append(np.random.normal(0, 1, 98), [10, -10]), # Force outliers (>3std dev)
        'names': ['Alice']*40 + ['Bob']*40 + [np.nan]*20
    })
    
    prof_dict = data_profiler(df_test)
