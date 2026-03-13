import pandas as pd
import numpy as np

# Part D: AI-Augmented Task

"""
Documented prompt sent to AI:
"Write a Python function that takes a Pandas DataFrame and generates an automated data quality report including: shape, dtypes, missing values percentage, duplicate rows, unique value counts per column, and basic stats. Return the report as a dict and also print a formatted summary."
"""
def generate_data_quality_report(df: pd.DataFrame) -> dict:
    print("=======================================")
    print("      DATA QUALITY AUTOMATED REPORT    ")
    print("=======================================")
    
    shape = df.shape
    print(f"\n1. Shape: {shape[0]} rows, {shape[1]} columns")
    
    duplicates = int(df.duplicated().sum())
    print(f"2. Duplicate Rows: {duplicates}")
    
    print("\n3. Data Types:")
    dtypes_dict = df.dtypes.astype(str).to_dict()
    for col, dtype in dtypes_dict.items():
        print(f"   - {col}: {dtype}")
        
    print("\n4. Missing Values Percentage:")
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict() if len(df) > 0 else {}
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"   - {col}: {pct:.2f}%")
        else:
            print(f"   - {col}: 0%")
            
    print("\n5. Unique Value Counts:")
    uniques_dict = df.nunique().to_dict()
    for col, count in uniques_dict.items():
        print(f"   - {col}: {count} unique values")
        
    print("\n6. Basic Stats (Numerical):")
    if not df.select_dtypes(include=[np.number]).empty:
        stats = df.describe().to_dict()
        print(df.describe())
    else:
        stats = {}
        print("   No numerical columns found.")
        
    print("=======================================\n")
    
    report = {
        "shape": shape,
        "duplicate_rows": duplicates,
        "dtypes": dtypes_dict,
        "missing_percentage": missing_pct,
        "unique_counts": uniques_dict,
        "basic_stats": stats
    }
    return report

if __name__ == "__main__":
    clean_df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["A", "B", "C", "D"],
        "price": [10.0, 20.0, 30.0, 40.0]
    })
    
    messy_df = pd.DataFrame({
        "id": [1, 1, np.nan, 4],
        "name": ["A", "A", None, "D"],
        "price": [10.0, 10.0, np.nan, 40.0],
        "category": ["X", "X", "X", "X"] 
    })
    
    print("--- Testing Clean DataFrame ---")
    generate_data_quality_report(clean_df)
    
    print("--- Testing Messy DataFrame ---")
    generate_data_quality_report(messy_df)

"""
Critical Evaluation (200 words):
The AI-generated function handles basic reporting well but misses some edge cases. If provided with an empty DataFrame (no rows or columns), the `missing_pct` calculation would potentially break. The code correctly handles calculating percentages only if the dataframe length is not 0.

However, the code does not use `df.memory_usage()` as required for optimal performance evaluation on large datasets. While it successfully calculates `.nunique()` for unique value counts, the code doesn't explicitly flag features with single unique values (like the 'category' column in messy_df), which are essentially useless features for modeling. If all values in a column are null, they are identified only in the missing percentage section but not explicitly flagged as useless.

To improve it, I would modify the function to:
1. Explicitly identify and warn about columns having only 1 unique value (zero variance).
2. Incorporate memory usage using `df.memory_usage(deep=True).sum()`.
3. Highlight columns where >= 90% values are null, so analysts can easily drop them.
4. Catch division by zero explicitly to gracefully handle completely empty DataFrames without runtime errors.
"""
