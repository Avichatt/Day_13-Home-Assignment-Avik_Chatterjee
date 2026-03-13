import pandas as pd
import io

# Q1 (Conceptual): Explain the difference between .loc[] and .iloc[]
"""
Answer to Q1:
.loc[] is label-based indexing. It selects rows and columns based on their index/column names.
When slicing with .loc[], the start AND end labels are INCLUSIVE.

.iloc[] is integer-position based indexing. It selects rows and columns based on their integer position (0 to n-1).
When slicing with .iloc[], the start is INCLUSIVE and the end position is EXCLUSIVE (like standard Python lists).

Example:
df.loc[0:3] vs df.iloc[0:3] when index is [0, 1, 2, 3, 4]:
- df.loc[0:3] returns rows with index LABELS 0, 1, 2, 3 (4 rows total because '3' is included).
- df.iloc[0:3] returns the rows at positions 0, 1, 2 (3 rows total because position 3 is excluded).

If index is ['a', 'b', 'c', 'd', 'e']:
- df.loc[0:3] will throw a TypeError/KeyError because 0 and 3 are integers, not the string labels 'a', 'b'...
- df.iloc[0:3] will STILL return the first 3 rows (at positions 0, 1, and 2) regardless of the string index labels.
"""

# Q2 (Coding): analyze_csv function
def analyze_csv(filepath: str) -> dict:
    """Loads a CSV, prints First 5 Minutes checklist, returns summary dict."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
        
    print("--- First 5 Minutes Checklist ---")
    print("\n1. Head:\n", df.head())
    print("\n2. Info:")
    df.info()
    print("\n3. Describe:\n", df.describe())
    print("\n4. Missing Values:\n", df.isnull().sum())
    print("-" * 30)
    
    result = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "numeric_cols": list(df.select_dtypes(include=['number']).columns),
        "categorical_cols": list(df.select_dtypes(include=['object', 'category']).columns),
        "null_counts": df.isnull().sum().to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
    }
    return result

# Q3 (Debug): Fixed Code
def run_debug_fix():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000, 60000, 70000]
    })
    
    # Bug 1 Fixed: Use standard pandas bitwise operators and wrap conditions in parentheses
    high_earners = df[(df["age"] > 25) & (df["salary"] > 55000)]
    print("\nHigh Earners:\n", high_earners)
    
    # Bug 2 Fixed: Use .loc[] to avoid chained indexing warning and safely update values
    df.loc[0, "age"] = 26
    print("\nUpdated DataFrame age:\n", df)
    
    # Bug 3 Fixed: .iloc[] is exclusive of the end index, so to expect 3 rows, we slice [0:3]
    first_three = df.iloc[0:3]  # Expecting 3 rows (0, 1, 2)
    print(f"\nGot {len(first_three)} rows, expected 3")

if __name__ == "__main__":
    csv_data = "col1,col2,col3\n1,A,2.5\n2,B,\n3,C,4.0"
    df_temp = pd.read_csv(io.StringIO(csv_data))
    df_temp.to_csv("test_dummy.csv", index=False)
    
    print("Testing analyze_csv:")
    stats = analyze_csv("test_dummy.csv")
    print("\nReturned Dict:\n", stats)
    
    print("\nTesting Debug Fix:")
    run_debug_fix()
