import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# -----------------------------
# 1. Load CSV
# -----------------------------
data = pd.read_csv("rui_fish_data.csv")

# Select columns
if 'age' in data.columns and 'length' in data.columns:
    X = data[['age']].values
    y = data['length'].values
elif 'week' in data.columns and 'size' in data.columns:
    X = data[['week']].values
    y = data['size'].values
else:
    raise KeyError("Expected columns 'age' & 'length' (or 'week' & 'size')")

# -----------------------------
# 2. Polynomial Transformation
# -----------------------------
degree = 3   # change to 1 or 2 if needed
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

# -----------------------------
# 3. Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_poly, y)

# Prediction (for R²)
y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)

# -----------------------------
# 4. Tabular Output
# -----------------------------
feature_names = poly.get_feature_names_out(['Age'])

table = pd.DataFrame({
    'Term': feature_names,
    'Coefficient': np.round(model.coef_, 4)
})

intercept_row = pd.DataFrame({
    'Term': ['Intercept'],
    'Coefficient': [round(model.intercept_, 4)]
})

table = pd.concat([intercept_row, table], ignore_index=True)

print("\nPolynomial Regression Results (Tabular Format):\n")
print(table)
print(f"\nR² Value: {r2:.4f}")

# -----------------------------
# 5. Display Equation
# -----------------------------
equation = f"Length = {model.intercept_:.3f}"
for coef, term in zip(model.coef_, feature_names):
    equation += f" + {coef:.3f}*{term}"

print("\nFitted Polynomial Equation:\n")
print(equation)

# -----------------------------
# 6. Plot Graph
# -----------------------------
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_curve = model.predict(X_range_poly)

plt.figure(figsize=(8,5))
plt.scatter(X, y, label="Observed Data")
plt.plot(X_range, y_curve, label=f"Polynomial Degree {degree}")
plt.xlabel("Age")
plt.ylabel("Length")
plt.title("Polynomial Regression of Rui Fish Length vs Age")
plt.legend()
plt.grid(True)
plt.show()
