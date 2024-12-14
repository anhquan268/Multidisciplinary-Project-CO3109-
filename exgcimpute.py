import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.helper_data import load_GSS
from gcimpute.helper_mask import mask_MCAR

# File paths
imputed_data_path = r"\\wsl.localhost\Ubuntu\home\anhquan2682001\imputed_data.csv"
smae_results_path = r"\\wsl.localhost\Ubuntu\home\anhquan2682001\smae_results.txt"

# Load GSS dataset (simulated here for demonstration)
data = load_GSS()  # e.g., includes age, income, health status, etc.

# Introduce 10% missing data randomly for testing
data_with_missing = mask_MCAR(X=data, mask_fraction=.1, seed=101)

# Convert data_with_missing (NumPy array) back to DataFrame
data_with_missing_df = pd.DataFrame(data_with_missing, columns=data.columns)

# Impute missing data using Gaussian Copula
model = GaussianCopula(verbose=1)
imputed_data = model.fit_transform(X=data_with_missing)

from gcimpute.helper_evaluation import get_smae

# Assuming data is the original dataset without missing values
# Calculate Scaled Mean Absolute Error (SMAE)
smae = get_smae(imputed_data, x_true=data, x_obs=data_with_missing)
print(f"Scaled Mean Absolute Error (SMAE): {smae.mean():.3f}")

# Save imputed data to a CSV file
imputed_data_df = pd.DataFrame(imputed_data, columns=data.columns)
imputed_data_df.to_csv(imputed_data_path, index=False)

# Save SMAE results to a text file
with open(smae_results_path, 'w') as f:
    f.write(f"Scaled Mean Absolute Error (SMAE): {smae.mean():.3f}\n")
    f.write("SMAE values per feature:\n")
    for i, col in enumerate(data.columns):
        f.write(f"{col}: {smae[i]:.3f}\n")
