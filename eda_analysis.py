import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the data using Pandas
# Build the path to the CSV file automatically
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'revenue_data.csv')

print(f"Loading data from: {csv_path}\n")
df = pd.read_csv(csv_path)

# Show the first 5 rows
print("--- First 5 rows of the dataset ---")
print(df.head())
print("\n")

# Show basic statistics for the numerical columns
print("--- Basic Statistics ---")
print(df.describe())
print("\n")

# Set a visually appealing style for Seaborn
sns.set_theme(style="whitegrid")

# ==========================================
# 2. Line Chart of 'Monthly_Revenue' over time
# ==========================================
plt.figure(figsize=(12, 5)) # Set the size of the figure

# Plot the revenue over the months
plt.plot(df['Month'], df['Monthly_Revenue'], marker='o', color='Teal', linestyle='-', linewidth=2)

plt.title('Monthly Revenue Trend (3 Years)', fontsize=16, pad=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)

# Rotate month names so they are readable, and only show every 3rd month to avoid crowding
plt.xticks(rotation=45) 
plt.tight_layout() # Adjust the layout so labels aren't cut off

# Save the line chart image
line_chart_path = os.path.join(script_dir, 'revenue_trend.png')
plt.savefig(line_chart_path)
print(f"Line chart successfully saved to: {line_chart_path}")

# Show the plot
plt.show()

# ==========================================
# 3. Create a Heatmap to show the correlation
# ==========================================
# Select only numerical columns to calculate correlation
numerical_df = df[['Marketing_Budget', 'Social_Media_Ads', 'Monthly_Revenue']]

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(8, 6)) # Set a new figure for the heatmap

# Draw the heatmap with Seaborn
# annot=True shows the numbers inside the squares
# cmap='coolwarm' uses a blue-to-red color scale
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)

plt.title('Correlation Heatmap: Revenue vs. Marketing', fontsize=16, pad=15)
plt.tight_layout()

# Save the heatmap image
heatmap_path = os.path.join(script_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path)
print(f"Heatmap successfully saved to: {heatmap_path}")

# Show the heatmap
plt.show()
