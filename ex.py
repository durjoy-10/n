import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Given System
# -----------------------------
A = np.array([
    [1, 1, 1],
    [2, -1, 1],
    [1, 2, -1]
], dtype=float)

B = np.array([6, 3, 3], dtype=float)

# -----------------------------
# -----------------------------
# Gauss-Jordan Method
# -----------------------------
M = np.hstack([A, B.reshape(-1,1)])

for i in range(3):
    M[i] = M[i] / M[i][i]
    for j in range(3):
        if i != j:
            M[j] = M[j] - M[j][i] * M[i]

solution_gj = M[:, -1]

df_gj = pd.DataFrame({
    'Variable': ['x', 'y', 'z'],
    'Value': solution_gj
})

print("\nGauss-Jordan Solution")
print(df_gj)