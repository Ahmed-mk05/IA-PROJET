import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Number of months
num_months = 36

# Generate data
months = [f"Month_{i}" for i in range(1, num_months + 1)]
marketing_budget = np.random.randint(5000, 20000, size=num_months)
social_media_ads = np.random.randint(2000, 10000, size=num_months)

# Generate Monthly_Revenue with positive correlation
# Revenue = base + factor1 * budget + factor2 * ads + some noise
base_revenue = 10000
monthly_revenue = (base_revenue 
                   + 2.5 * marketing_budget 
                   + 4.0 * social_media_ads 
                   + np.random.normal(0, 5000, size=num_months))

# Ensure revenue is positive and format to 2 decimal places
monthly_revenue = np.maximum(monthly_revenue, 0).round(2)

# Create DataFrame
df = pd.DataFrame({
    'Month': months,
    'Marketing_Budget': marketing_budget,
    'Social_Media_Ads': social_media_ads,
    'Monthly_Revenue': monthly_revenue
})

# Save to CSV in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'revenue_data.csv')

df.to_csv(csv_path, index=False)

print(f"Dataset successfully created and saved to {csv_path}")
