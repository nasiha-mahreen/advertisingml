import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
advertising_data = pd.read_csv('advertising.csv')

# Define feature variables and the target variable
features = advertising_data[['TV', 'Radio', 'Newspaper']]
target = advertising_data['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict sales for the test set
y_predictions = linear_model.predict(X_test)

# Calculate evaluation metrics
mse_value = mean_squared_error(y_test, y_predictions)
r2_value = r2_score(y_test, y_predictions)

# Determine the coefficients for each feature
feature_impact = pd.DataFrame({'Medium': features.columns, 'Impact': linear_model.coef_})
feature_impact = feature_impact.sort_values('Impact', ascending=False)

# Visualizations
plt.figure(figsize=(12, 5))

# Plot Actual vs Predicted Sales
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')

# Plot Impact of Advertising Mediums on Sales
plt.subplot(1, 2, 2)
plt.bar(feature_impact['Medium'], feature_impact['Impact'])
plt.title('Advertising Medium Impact on Sales')
plt.xlabel('Medium')
plt.ylabel('Impact')

plt.tight_layout()
plt.show()

# Print results
print("\n=== Advertising Impact on Sales Analysis ===\n")

print("Model Performance:")
print(f"R-squared Score: {r2_value:.2%}")
print(f"Mean Squared Error: {mse_value:.2f}")

print("\nImpact of Advertising Mediums on Sales:")
for index, row in feature_impact.iterrows():
    print(f"{row['Medium']}: {row['Impact']:.4f}")

print("\nInterpretation:")
for index, row in feature_impact.iterrows():
    print(f"- A $1,000 increase in {row['Medium']} advertising is associated with a ${row['Impact']*1000:.2f} increase in sales.")

print("\nRecommendations:")
top_medium = feature_impact.iloc[0]['Medium']
second_medium = feature_impact.iloc[1]['Medium']
least_medium = feature_impact.iloc[2]['Medium']

print(f"1. Prioritize {top_medium} advertising for the highest impact on sales.")
print(f"2. Allocate a significant portion of the budget to {second_medium} advertising as well.")
if feature_impact.iloc[2]['Impact'] < feature_impact.iloc[1]['Impact'] / 2:
    print(f"3. Consider reducing or reallocating budget from {least_medium} advertising, as it has significantly less impact.")
else:
    print(f"3. Maintain a balanced approach with {least_medium} advertising, but prioritize the top two mediums.")

print("\nNext Steps:")
print("1. Conduct A/B testing to validate these findings in real-world scenarios.")
print("2. Analyze the cost-effectiveness of each advertising medium.")
print("3. Regularly update this model with new data to ensure ongoing accuracy.")
