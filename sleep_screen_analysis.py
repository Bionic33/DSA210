
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Step 1: Data Preprocessing and Cleaning
data = {
    'Sleep Duration (Hours)': [8 + 46/60, 11 + 51/60, 10 + 49/60, 10 + 3/60, 14 + 46/60, 8 + 56/60, 9 + 56/60,
                               8 + 23/60, 11 + 28/60, 9 + 48/60, 4 + 8/60, 5 + 47/60, 8 + 57/60, 3 + 48/60],
    'Screen Time (Hours)': [6 + 23/60, 6 + 35/60, 7 + 20/60, 6 + 25/60, 8 + 15/60, 8 + 30/60, 6 + 50/60,
                            7 + 32/60, 6 + 39/60, 7 + 3/60, 8 + 37/60, 9 + 23/60, 7 + 24/60, 8 + 40/60]
}

df = pd.DataFrame(data)

# Step 2: Exploratory Data Analysis
plt.figure(figsize=(8, 6))
plt.scatter(df['Screen Time (Hours)'], df['Sleep Duration (Hours)'], color='blue', alpha=0.7)
plt.title('Screen Time vs Sleep Duration')
plt.xlabel('Screen Time (Hours)')
plt.ylabel('Sleep Duration (Hours)')
plt.grid(True)
plt.show()

# Correlation Analysis
correlation = df['Screen Time (Hours)'].corr(df['Sleep Duration (Hours)'])
print(f"Correlation coefficient: {correlation:.2f}")

# Significance Testing for Correlation
n = len(df)
t_statistic = correlation * np.sqrt((n - 2) / (1 - correlation**2))
p_value = 2 * (1 - t.cdf(abs(t_statistic), df=n-2))
print(f"t-statistic: {t_statistic:.2f}, p-value: {p_value:.4f}")

# Regression Analysis
from sklearn.linear_model import LinearRegression

X = np.array(df['Screen Time (Hours)']).reshape(-1, 1)
y = np.array(df['Sleep Duration (Hours)'])

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(df['Screen Time (Hours)'], df['Sleep Duration (Hours)'], color='blue', alpha=0.7, label='Data Points')
plt.plot(df['Screen Time (Hours)'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Screen Time vs Sleep Duration with Regression Line')
plt.xlabel('Screen Time (Hours)')
plt.ylabel('Sleep Duration (Hours)')
plt.legend()
plt.grid(True)
plt.show()

slope = model.coef_[0]
intercept = model.intercept_
print(f"Regression Line Equation: Sleep Duration = {slope:.2f} * Screen Time + {intercept:.2f}")

print("\nSummary Statistics:")
print(df.describe())
