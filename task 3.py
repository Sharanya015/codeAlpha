# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Step 1: Create a synthetic dataset
data = {
    'Brand_Goodwill': np.random.randint(1, 10, 500),  # Scale 1 to 10
    'Features': np.random.randint(1, 10, 500),  # Scale 1 to 10
    'Horsepower': np.random.randint(100, 400, 500),  # Horsepower range
    'Mileage': np.random.randint(10, 30, 500),  # Miles per gallon
    'Age': np.random.randint(0, 15, 500),  # Age of the car
    'Price': np.random.randint(5000, 50000, 500),  # Price range in dollars
}

df = pd.DataFrame(data)

# Step 2: Explore the dataset
print("Dataset preview:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# Visualize correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 3: Split the dataset
X = df[['Brand_Goodwill', 'Features', 'Horsepower', 'Mileage', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"R² Score: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Step 6: Visualize actual vs predicted prices
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Step 7: Save the model (optional)
joblib.dump(model, "car_price_predictor.pkl")
print("\nModel saved as 'car_price_predictor.pkl'.")

# Step 8: Load and use the saved model
loaded_model = joblib.load("car_price_predictor.pkl")
new_data = pd.DataFrame({
    'Brand_Goodwill': [8],
    'Features': [7],
    'Horsepower': [250],
    'Mileage': [18],
    'Age': [2]
})

new_prediction = loaded_model.predict(new_data)
print("\nNew data prediction:")
print("Input data:", new_data.to_dict(orient='records'))
print("Predicted Price:", new_prediction[0])
