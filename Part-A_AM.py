import pandas as pd

def create_product_data() -> pd.DataFrame:
    """Creates a DataFrame with 20+ products."""
    data = {
        "name": [
            "Laptop", "Smartphone", "Tablet", "Monitor", "Keyboard",
            "T-Shirt", "Jeans", "Jacket", "Sneakers", "Hat",
            "Python Book", "Data Science 101", "Cookbook", "Sci-Fi Novel", "History Book",
            "Sofa", "Desk", "Chair", "Lamp", "Bed",
            "Headphones", "Mouse"
        ],
        "category": [
            "Electronics", "Electronics", "Electronics", "Electronics", "Electronics",
            "Clothing", "Clothing", "Clothing", "Clothing", "Clothing",
            "Books", "Books", "Books", "Books", "Books",
            "Home", "Home", "Home", "Home", "Home",
            "Electronics", "Electronics"
        ],
        "price": [
            55000.0, 30000.0, 15000.0, 8000.0, 2000.0,
            500.0, 1200.0, 3000.0, 2500.0, 300.0,
            800.0, 1200.0, 600.0, 400.0, 500.0,
            12000.0, 4500.0, 1500.0, 800.0, 18000.0,
            4000.0, 1000.0
        ],
        "stock": [
            10, 50, 30, 20, 100,
            200, 150, 40, 60, 300,
            50, 45, 20, 100, 80,
            5, 10, 25, 40, 8,
            80, 120
        ],
        "rating": [
            4.5, 4.2, 3.8, 4.6, 4.1,
            3.5, 4.0, 4.7, 4.3, 3.9,
            4.8, 4.9, 4.0, 4.5, 3.8,
            4.2, 4.1, 3.9, 4.0, 4.6,
            4.4, 4.2
        ],
        "num_reviews": [
            150, 800, 45, 120, 300,
            50, 200, 80, 150, 90,
            250, 180, 40, 300, 60,
            20, 45, 30, 80, 25,
            400, 220
        ]
    }
    return pd.DataFrame(data)

def run_first_5_minutes(df: pd.DataFrame) -> None:
    """Runs the 'First 5 Minutes' checklist on the DataFrame."""
    print("--- First 5 Minutes Checklist ---")
    print("\n1. DataFrame Head:")
    print(df.head())
    print("\n2. DataFrame Tail:")
    print(df.tail())
    print("\n3. Info:")
    df.info()
    print("\n4. Describe (Numerical Stats):")
    print(df.describe())
    print("\n5. Shape:")
    print(df.shape)
    print("\n6. Missing Values Check:")
    print(df.isnull().sum())
    print("-" * 30)

def analyze_products() -> None:
    """Runs the full product analyzer requirements."""
    df = create_product_data()
    
    # Run the 'First 5 Minutes' checklist
    run_first_5_minutes(df)
    
    print("\n--- .loc[] Operations ---")
    # (a) select all Electronics
    electronics_df = df.loc[df['category'] == 'Electronics']
    print("\n(a) Electronics:")
    print(electronics_df.head())
    
    # (b) select products rated > 4.0 with price < 5000
    top_rated_affordable = df.loc[(df['rating'] > 4.0) & (df['price'] < 5000)]
    print("\n(b) Rated > 4.0 and Price < 5000:")
    print(top_rated_affordable)
    
    # (c) update stock for a specific product
    laptop_idx = df.loc[df['name'] == 'Laptop'].index[0]
    df.loc[laptop_idx, 'stock'] = 15
    print("\n(c) Updated Laptop stock to 15:")
    print(df.loc[df['name'] == 'Laptop'])
    
    print("\n--- .iloc[] Operations ---")
    # (a) get first 5 and last 5 products
    first_5 = df.iloc[:5]
    last_5 = df.iloc[-5:]
    print("\n(a) First 5 products:")
    print(first_5)
    print("\n(a) Last 5 products:")
    print(last_5)
    
    # (b) select every other row
    every_other_row = df.iloc[::2]
    print("\n(b) Every other row:")
    print(every_other_row.head())
    
    # (c) get rows 10-15 with columns 0-3
    rows_10_15_cols_0_3 = df.iloc[10:16, 0:4] 
    print("\n(c) Rows 10-15 with columns 0-3:")
    print(rows_10_15_cols_0_3)
    
    print("\n--- Filtering and Exporting ---")
    budget_products = df.loc[df['price'] < 1000]
    premium_products = df.loc[df['price'] > 10000]
    popular_products = df.loc[(df['num_reviews'] > 100) & (df['rating'] > 4.0)]
    
    dfs_to_export = {
        'budget_products.csv': budget_products,
        'premium_products.csv': premium_products,
        'popular_products.csv': popular_products
    }
    
    for filename, dataframe in dfs_to_export.items():
        dataframe.to_csv(filename, index=False)
        print(f"Exported {len(dataframe)} rows to {filename}")

if __name__ == "__main__":
    analyze_products()
