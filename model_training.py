import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import os

# Build path to the CSV dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'revenue_data.csv')

# 1. Load the data
print(f"Loading data from: {csv_path}...")
df = pd.read_csv(csv_path)

# Split into Features (X) and Target (y)
X = df[['Marketing_Budget', 'Social_Media_Ads']]
y = df['Monthly_Revenue']

print("Features (X) assigned: Marketing_Budget, Social_Media_Ads")
print("Target (y) assigned: Monthly_Revenue\n")

# 2. Split the data into Training and Testing sets (80/20)
# test_size=0.2 means 20% of the data is kept to evaluate the model
# random_state=42 ensures that the random splitting is identical every time you run this script
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Data successfully split! (80% Train / 20% Test)")
print(f"Training instances: {len(X_train)}")
print(f"Testing instances: {len(X_test)}\n")

# 3. Train the Linear Regression Model
model = LinearRegression()
print("Training the Linear Regression model on training data...")
model.fit(X_train, y_train)

# Make predictions using the 20% testing data we set aside
print("Making predictions on testing data...\n")
y_pred = model.predict(X_test)

# 4. Evaluate the model mathematically
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=======================================")
print("      MODEL EVALUATION METRICS         ")
print("=======================================")
print(f"R\u00b2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print("=======================================\n")

# (Extra) View the learned formula weights
print("--- Learned Coefficients ---")
print(f"Marketing Budget Weight: {model.coef_[0]:.4f}")
print(f"Social Media Ads Weight: {model.coef_[1]:.4f}")
print(f"Base Revenue (Intercept): {model.intercept_:.2f}")
