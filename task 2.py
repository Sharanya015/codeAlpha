# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a synthetic dataset
data = {
    'Month': pd.date_range(start='2018-01', periods=60, freq='M'),
    'Unemployment_Rate': np.concatenate([
        np.random.uniform(3.5, 5.0, 24),  # Before COVID-19
        np.random.uniform(8.0, 14.0, 12),  # During COVID-19
        np.random.uniform(4.5, 6.5, 24)   # After COVID-19
    ]),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 60)  # Random regions
}

df = pd.DataFrame(data)

# Step 2: Display the dataset
print("Dataset Preview:")
print(df.head())

# Step 3: Visualize unemployment rate trends
# Line plot of unemployment rate over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Month', y='Unemployment_Rate', marker='o', label='Unemployment Rate')
plt.axvspan(pd.Timestamp('2020-03'), pd.Timestamp('2021-03'), color='red', alpha=0.2, label='COVID-19 Period')
plt.title("Unemployment Rate Over Time")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Analyze unemployment rates by region
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Region', y='Unemployment_Rate', palette='Set2')
plt.title("Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

# Step 5: Monthly Unemployment Rate Heatmap During COVID-19
covid_data = df[(df['Month'] >= '2020-03') & (df['Month'] <= '2021-03')]
pivot_table = covid_data.pivot_table(values='Unemployment_Rate', index='Month', columns='Region')

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Unemployment Rate (%)'})
plt.title("Monthly Unemployment Rates During COVID-19 (by Region)")
plt.ylabel("Month")
plt.xlabel("Region")
plt.show()

# Step 6: Key Statistics
print("\nKey Statistics:")
print("Average Unemployment Rate by Region:")
print(df.groupby('Region')['Unemployment_Rate'].mean())

print("\nMaximum Unemployment Rate by Month:")
print(df.groupby('Month')['Unemployment_Rate'].max())

# Step 7: Identify periods of high unemployment
high_unemployment = df[df['Unemployment_Rate'] > 10]
print("\nPeriods with High Unemployment (Rate > 10%):")
print(high_unemployment)

