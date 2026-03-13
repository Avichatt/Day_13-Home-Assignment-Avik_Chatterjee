import pandas as pd

def create_sales_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates 3 DataFrames representing sales data from 3 different months."""
    month1 = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "product": ["Laptop", "Mouse", "Keyboard", "Laptop", "Monitor"],
        "price": [50000, 1000, 2000, 50000, 8000],
        "quantity": [1, 5, 2, 1, 1]
    })
    month1['revenue'] = month1['price'] * month1['quantity']
    
    month2 = pd.DataFrame({
        "order_id": [6, 7, 8, 9, 10],
        "product": ["Smartphone", "Mouse", "Smartphone", "Tablet", "Mouse"],
        "price": [30000, 1000, 30000, 15000, 1000],
        "quantity": [2, 10, 1, 1, 5]
    })
    month2['revenue'] = month2['price'] * month2['quantity']
    
    month3 = pd.DataFrame({
        "order_id": [11, 12, 13, 14, 15],
        "product": ["Monitor", "Desk", "Chair", "Monitor", "Keyboard"],
        "price": [8000, 5000, 1500, 8000, 2000],
        "quantity": [2, 1, 4, 1, 3]
    })
    month3['revenue'] = month3['price'] * month3['quantity']
    
    return month1, month2, month3

def analyze_monthly_sales(df: pd.DataFrame, month_name: str) -> dict:
    """Calculates metrics for a single month."""
    total_rev = df['revenue'].sum()
    avg_order_val = df['revenue'].mean()
    
    product_group = df.groupby('product')['revenue'].sum()
    top_product = product_group.idxmax()
    
    return {
        "Month": month_name,
        "Total Revenue": total_rev,
        "Average Order Value": avg_order_val,
        "Top Selling Product": top_product
    }

def generate_comparison_report() -> None:
    """Runs the Multi-DataFrame Comparison Report logic."""
    m1, m2, m3 = create_sales_data()
    
    metrics = [
        analyze_monthly_sales(m1, "January"),
        analyze_monthly_sales(m2, "February"),
        analyze_monthly_sales(m3, "March")
    ]
    
    summary_df = pd.DataFrame(metrics).set_index("Month")
    print("--- Summary Comparison Report ---")
    print(summary_df)
    
    all_months = pd.concat([m1, m2, m3], keys=["Jan", "Feb", "Mar"]).reset_index(level=0).rename(columns={'level_0': 'month'})
    
    print("\n--- .query() Filtering (High revenue orders) ---")
    high_revenue_orders = all_months.query("revenue > 20000")
    print(high_revenue_orders)
    
    print("\n--- Outliers using .nlargest() and .nsmallest() ---")
    top_3_largest_orders = all_months.nlargest(3, 'revenue')
    print("Top 3 Largest Orders (Outliers - High):")
    print(top_3_largest_orders[['month', 'order_id', 'product', 'revenue']])
    
    bottom_3_smallest_orders = all_months.nsmallest(3, 'revenue')
    print("\nBottom 3 Smallest Orders (Outliers - Low):")
    print(bottom_3_smallest_orders[['month', 'order_id', 'product', 'revenue']])

if __name__ == "__main__":
    generate_comparison_report()
