import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gcimpute.gaussian_copula import GaussianCopula
from gcimpute.low_rank_gaussian_copula import LowRankGaussianCopula
from gcimpute.helper_data import load_GSS, load_movielens1m
from gcimpute.helper_mask import mask_MCAR
from gcimpute.helper_evaluation import get_smae, get_mae

print("Example 1: Basic usage")

# File paths
imputed_data_path = r"\\wsl.localhost\Ubuntu\home\anhquan2682001\imputed_data.csv"
smae_results_path = r"\\wsl.localhost\Ubuntu\home\anhquan2682001\smae_results.txt"

# Load GSS dataset (simulated here for demonstration)
data = load_GSS()  # e.g., includes age, income, health status, etc.

# Introduce 10% missing data randomly for testing
data_with_missing = mask_MCAR(X=data, mask_fraction=.1, seed=101)

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

print("\nExample 2: Accelerating datasets with many variables: low rank structure")
data_movie = load_movielens1m(num=400, min_obs=150, verbose=True)
movie_masked = mask_MCAR(X=data_movie, mask_fraction=0.1, seed=101)

a = time.time()
model_movie_lrgc = LowRankGaussianCopula(rank=10, verbose=1)
m_imp_lrgc = model_movie_lrgc.fit_transform(X=movie_masked)
print(f'LRGC runtime {(time.time()-a)/60:.2f} mins.')
a = time.time()
model_movie_gc = GaussianCopula(verbose=1)
m_imp_gc = model_movie_gc.fit_transform(X=movie_masked)
print(f'GC runtime {(time.time()-a)/60:.2f} mins.')

mae_gc = get_mae(x_imp=m_imp_gc, x_true=data_movie, x_obs=movie_masked)
mae_lrgc = get_mae(x_imp=m_imp_lrgc, x_true=data_movie, x_obs=movie_masked)
print(f'LRGC imputation MAE: {mae_lrgc:.3f}')
print(f'GC imputation MAE: {mae_gc:.3f}')

print("\nExample 3: Accelerating datasets with many samples: mini-batch training")
print("We now run min-batch training with the defaults on the GSS dataset:")
data_gss = load_GSS()
gss_masked = mask_MCAR(X=data_gss, mask_fraction=.1, seed=101)

t1=time.time()
model_minibatch = GaussianCopula(training_mode='minibatch-offline')
Ximp_batch = model_minibatch.fit_transform(X=gss_masked)
t2=time.time()
print(f'Runtime: {t2-t1:.2f} seconds')
smae_batch = get_smae(x_imp=Ximp_batch, x_true=data_gss, x_obs=gss_masked)
print(f'Imputation error: {smae_batch.mean():.3f}')

print("Let us also re-run and record the runtime of the standard training mode:")
t1=time.time()
GaussianCopula().fit_transform(X=gss_masked)
t2=time.time()
print(f'Runtime: {t2-t1:.2f} seconds')