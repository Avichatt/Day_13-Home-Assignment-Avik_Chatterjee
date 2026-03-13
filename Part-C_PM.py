import pandas as pd
import re

# Q1 (Conceptual): Handling 40% missing 'income' on a 1M row DataFrame
"""
Decision-Making Process for Missing Data:

1. Analyze the Mechanism of Missingness:
   - Is it Missing completely at random (MCAR)? If dropped, the dataset isn't biased.
   - Is it Missing at random (MAR)? Or Missing Not At Random (MNAR) - e.g. wealthy users refuse to report income.
   
2. Drop vs Fill Decision Boundary:
   - Drop Columns: Typically considered if > 60-70% data is missing AND the feature provides little informational gain. At 40% missing, dropping the column destroys too much potential value.
   - Drop Rows: Dropping 400,000 rows directly creates massive sampling bias and drastically reduces training context. I would NEVER drop rows if 40% are null.
   - Fill (Imputation): This is the overwhelmingly preferred route.

3. Selected Fill Strategy for Income:
   - I would use MEDIAN imputation rather than Mean imputation. Income is almost universally right-skewed (few billionaires, many middle class), meaning the average (mean) is falsely inflated upwards. Median robustly captures the center distribution.
   - If predictive modeling applies, I could also add a binary flag column `income_is_missing` (0/1) so the algorithm can detect the pattern of lack of disclosure itself (often statistically significant).
"""

# Q2 (Coding): standardize_column
def standardize_column(series: pd.Series) -> pd.Series:
    """Takes messy text Pandas Series: strips whitespace, lowercases, deduplicates space, removes special chars."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # 1. Convert to string strictly
    clean_series = series.astype(str)
    
    # 2. Lowercase and strip leading/trailing edges
    clean_series = clean_series.str.strip().str.lower()
    
    # 3. Remove special characters using regex (keep only alphanumeric and spaces)
    clean_series = clean_series.str.replace(r'[^a-z0-9\s]', '', regex=True)
    
    # 4. Replace multiple internal spaces with a single space
    clean_series = clean_series.str.replace(r'\s+', ' ', regex=True)
    
    return clean_series

# Q3 (Debug): Fix the script containing 4 bugs
def debug_fixed_script():
    df = pd.DataFrame({
        "price": ["1,500", "2000", "N/A", "3,200", "abc"],
        "category": ["  Electronics ", "CLOTHING", "electronics", " Books", ""],
        "date": ["15/03/2024", "2024-07-01", "22-Nov-2024", "01/10/2024", None],
    })
    
    print("--- Original Messy DF ---")
    print(df)
    
    # Bug 1 Fixed: Replace obvious string comma formats and hidden NaNs before parsing
    df['price'] = df['price'].str.replace(',', '')
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    
    # Bug 2 Fixed: Use Pandas bitwise '&' operator instead of python natively scoped 'and'
    clean = df[(df["price"] > 1000) & (df["category"] != "")]
    
    # Bug 3 Fixed: When operating on string containment checks, add `na=False` strictly to prevent propagating boolean nulls
    df['category'] = df['category'].str.strip().str.lower() # Pre-standardize
    electronics = df[df["category"].str.contains("electronics", na=False)]
    
    # Bug 4 Fixed: Since dates are heavily mixed format, explicitly enforce coercions over explicit formatting params
    # and utilize 'mixed' format if using pandas >= 2.0 or let coerce dictate fallbacks silently.
    df["date"] = pd.to_datetime(df["date"], errors='coerce') 
    
    print("\n--- Fixed Clean DF ---")
    print(df)
    print("\n--- Electronics Filter View ---")
    print(electronics)

if __name__ == "__main__":
    # Test Q2
    test_list = ['  Hello  World!! ', '  NEW YORK  ', 'san--francisco', '   MUMBAI   ']
    print("Q2 Standardization Output:")
    print(standardize_column(pd.Series(test_list)).tolist())
    print("\n-------------------------")
    # Test Q3
    debug_fixed_script()
