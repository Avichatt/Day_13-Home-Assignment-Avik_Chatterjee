import pandas as pd
import numpy as np
import json

def generate_messy_survey_data() -> pd.DataFrame:
    """Generates 50+ rows of synthetic messy survey data."""
    np.random.seed(42)
    data = {
        'id': range(1, 56),  # 55 rows
        'age': np.random.randint(15, 65, 55).astype(str),
        'income': np.random.randint(20000, 150000, 55).astype(str),
        'satisfaction_score': np.random.randint(1, 6, 55).astype(float),
        'feedback': ['Great service!'] * 20 + [''] * 15 + ['   Terrible     '] * 10 + [None] * 5 + ['OK'] * 5,
        'country': ['USA', 'usa ', '  U.s.a', 'Canada', 'CANADA', 'uk', 'UK'] * 7 + ['None', 'N/A', '', 'USA', 'USA', 'USA'],
        'signup_date': ['2023-01-15', '15/01/2023', 'Jan 15th, 2023', 'error', ''] * 11,
        'is_active': ['True', 'False', 'true', 'false', '1', '0'] * 9 + ['yes']
    }
    df = pd.DataFrame(data)
    
    # Introduce explicit messy data quality issues
    df.loc[5, 'age'] = '-5'           # Invalid value
    df.loc[10, 'age'] = '200'         # Invalid value
    df.loc[15, 'income'] = 'Not Disclosed' # non-convertible type string
    df.loc[12, 'satisfaction_score'] = np.nan # basic nan marking
    
    # Duplicates append
    df = pd.concat([df, df.iloc[0:3]])
    
    return df

def detect_issues(df: pd.DataFrame) -> dict:
    """Returns a comprehensive data quality report dict."""
    total_rows = len(df)
    total_missing = df.isna().sum().sum()
    missing_per_column = df.isna().sum().to_dict()
    duplicate_count = df.duplicated().sum()
    
    wrong_types = {col: str(dtype) for col, dtype in df.dtypes.items() if dtype == 'object'}
    invalid_values = 0 # Dummy count without strict schemas for the report dict
    
    # Basic detect logic check
    for col in df.columns:
        if df[col].dtype == 'object':
             invalid_values += df[col].astype(str).str.contains(r'^\s*$|-5|200|error', regex=True, na=False).sum()
             
    report = {
        'total_rows': total_rows,
        'total_missing': int(total_missing),
        'missing_per_column': {k: int(v) for k, v in missing_per_column.items()},
        'duplicate_count': int(duplicate_count),
        'wrong_types': wrong_types,
        'invalid_values_approx': int(invalid_values)
    }
    return report

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes types, removes invalid ranges, and applies fill strategies."""
    df_clean = df.copy()
    
    # 1. Standardize hidden NaNs (replace empty strings and "None"/"N/A")
    df_clean.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df_clean.replace(['None', 'N/A', 'error'], np.nan, inplace=True)
    
    # 2. Fix Types with coercion where errors occur
    # Income string "Not Disclosed" becomes NaN.
    df_clean['income'] = pd.to_numeric(df_clean['income'], errors='coerce')
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    df_clean['satisfaction_score'] = pd.to_numeric(df_clean['satisfaction_score'], errors='coerce')
    
    # 3. Handle invalid numeric values mapping to boundaries
    # Keep only ages between 0 and 120, else mark NaN
    df_clean.loc[(df_clean['age'] < 0) | (df_clean['age'] > 120), 'age'] = np.nan
    
    # 4. Standardizing text fields via .str accessor
    df_clean['country'] = df_clean['country'].str.strip().str.upper().str.replace('.', '')
    df_clean['feedback'] = df_clean['feedback'].str.strip()
    df_clean['is_active'] = df_clean['is_active'].str.lower().str.strip()
    # Coerce to strict booleans loosely mapping typical textual equivalents
    active_map = {'true': True, '1': True, 'yes': True, 'false': False, '0': False}
    df_clean['is_active'] = df_clean['is_active'].map(active_map)
    
    # 5. Handle missing value imputation (State strategy justifications below)
    """
    FILL STRATEGIES:
    - 'income': Fills with median. Income implies high right-skew distributions where outliers inflate mean bounds.
    - 'age': Fills with median. Age should be integer and resilient against extreme entries (if any sneak past bounds).
    - 'satisfaction_score': Filled with mean rounded. Survey scores perform decently under normalized means assuming gaussian distribution.
    - 'feedback': Fill with 'No Comment' as qualitative datasets shouldn't fabricate opinions.
    - 'country': Fill with 'UNKNOWN'. A categorical value can't be guessed safely natively.
    """
    df_clean['income'] = df_clean['income'].fillna(df_clean['income'].median())
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    val = round(df_clean['satisfaction_score'].mean() if not pd.isna(df_clean['satisfaction_score'].mean()) else 3)
    df_clean['satisfaction_score'] = df_clean['satisfaction_score'].fillna(val)
    df_clean['feedback'] = df_clean['feedback'].fillna('No Comment')
    df_clean['country'] = df_clean['country'].fillna('UNKNOWN')
    
    # Optional DateTime cast for signup
    df_clean['signup_date'] = pd.to_datetime(df_clean['signup_date'], errors='coerce')
    
    # 6. Drop Rows (Drop entirely missing rows if somehow created)
    df_clean.dropna(how='all', inplace=True)
    
    # 7. Remove Duplicates
    df_clean.drop_duplicates(inplace=True)
    
    return df_clean

def run_pipeline():
    messy_df = generate_messy_survey_data()
    print("--- BEFORE CLEANING ---")
    print(f"Rows: {len(messy_df)}")
    print(f"Total Nulls (raw): {messy_df.isna().sum().sum()}")
    print("Memory Usage:", messy_df.memory_usage(deep=True).sum(), "bytes")
    
    report_dict = detect_issues(messy_df)
    
    clean_df = clean_data(messy_df)
    print("\n--- AFTER CLEANING ---")
    print(f"Rows: {len(clean_df)}")
    print(f"Total Nulls (Remaining): {clean_df.isna().sum().sum()}")
    print("Memory Usage:", clean_df.memory_usage(deep=True).sum(), "bytes")
    
    with open('data_quality_report.json', 'w') as f:
        json.dump(report_dict, f, indent=4)
        print("\nExported: data_quality_report.json")
        
    clean_df.to_csv('cleaned_survey.csv', index=False)
    print("Exported: cleaned_survey.csv")

if __name__ == '__main__':
    run_pipeline()
