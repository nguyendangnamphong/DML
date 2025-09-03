# EconML causal pipeline (EconML only)
# File expected at: D:/Khóa luận/data/final_data.csv

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from econml.dml import LinearDML
except Exception as e:
    print("ERROR: econml not installed. Install with: pip install econml")
    raise
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DEFAULT_INPUT = "D:/Khóa luận/data/final_data.csv"
DEFAULT_OUTPUT_DIR = "D:/Khóa luận/data"

def compute_size_deviation_pct(df, components=['Cu','Al','Cr','Mg','Ni','Si'], radii=None):
    """Compute percent atomic size deviation treatment."""
    missing = [c for c in components if c not in df.columns]
    if missing:
        raise ValueError(f"Missing component columns: {missing}")
    if radii is None:
        radii = {'Cu':128,'Al':143,'Cr':128,'Mg':160,'Ni':124,'Si':111}
    vals = df[components].astype(float).values
    row_sums = vals.sum(axis=1, keepdims=True)
    fractions = np.divide(vals, row_sums, where=row_sums!=0)
    r_array = np.array([radii[c] for c in components], dtype=float)
    r_mean = (fractions * r_array).sum(axis=1)
    diffs_sq = (r_array - r_mean[:, None])**2
    weighted = (fractions * diffs_sq).sum(axis=1)
    size_dev_pct = 100.0 * np.sqrt(weighted)
    return size_dev_pct

def first_stage_r2(df, treatment, covariates):
    """R^2 of linear regression treatment ~ covariates."""
    if not covariates:
        return np.nan
    X = df[covariates].astype(float).values
    y = df[treatment].astype(float).values
    model = LinearRegression().fit(X, y)
    return model.score(X, y)

def bootstrap_ate_econml(df, treatment, outcome, controls, n_boot=100, random_state=0):
    """Bootstrap ATE by resampling and refitting LinearDML. Returns array of ATEs."""
    rng = np.random.RandomState(random_state)
    N = df.shape[0]
    estimates = []
    for i in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        try:
            est = run_econml_fit(df_b, treatment, outcome, controls, random_state=rng.randint(1e6))
            if est is None:
                estimates.append(np.nan)
            else:
                t0 = df_b[treatment].mean()
                t1 = t0 + 1.0
                eff = est.effect(X=None, T0=t0, T1=t1)
                # effect may be array, take mean
                estimates.append(np.mean(eff))
        except Exception:
            estimates.append(np.nan)
    arr = np.array(estimates, dtype=float)
    return arr[~np.isnan(arr)]

def permutation_placebo_econml(df, treatment, outcome, controls, n_perm=200, random_state=1):
    """Permutation placebo by shuffling treatment and refitting. Returns observed ATE, permuted array, p-value."""
    est = run_econml_fit(df, treatment, outcome, controls, random_state=random_state)
    if est is None:
        raise RuntimeError('EconML estimator failed on observed data')
    t0 = df[treatment].mean()
    t1 = t0 + 1.0
    observed_eff = est.effect(X=None, T0=t0, T1=t1)
    observed_ate = float(np.mean(observed_eff))

    rng = np.random.RandomState(random_state)
    perm_ates = []
    N = df.shape[0]
    for i in range(n_perm):
        df_p = df.copy()
        df_p[treatment] = rng.permutation(df_p[treatment].values)
        try:
            est_p = run_econml_fit(df_p, treatment, outcome, controls, random_state=rng.randint(1e6))
            eff_p = est_p.effect(X=None, T0=t0, T1=t1)
            perm_ates.append(float(np.mean(eff_p)))
        except Exception:
            perm_ates.append(np.nan)
    perm = np.array(perm_ates, dtype=float)
    perm = perm[~np.isnan(perm)]
    pval = np.mean(np.abs(perm) >= abs(observed_ate)) if perm.size>0 else np.nan
    return observed_ate, perm, pval

def run_econml_fit(df, treatment, outcome, controls, random_state=0):
    """Fit LinearDML and return fitted estimator (or None on failure)."""
    # define nuisance learners
    try:
        model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=int(random_state))
        model_t = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=int(random_state)+1)
        est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=False, cv=3, random_state=int(random_state))
        Y = df[outcome].astype(float).values
        T = df[treatment].astype(float).values
        W = df[controls].astype(float).values if controls else None
        est.fit(Y, T, W=W)
        return est
    except Exception as e:
        return None

def plot_and_save(arr, observed, ci_low, ci_high, outpath):
    plt.figure(figsize=(8,5))
    plt.hist(arr, bins=30, alpha=0.8)
    plt.axvline(observed, color='red', linewidth=2, label=f'Observed ATE={observed:.4g}')
    plt.axvline(ci_low, color='black', linestyle='--', label=f'95% CI low={ci_low:.4g}')
    plt.axvline(ci_high, color='black', linestyle='--', label=f'95% CI high={ci_high:.4g}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_permutation(perm, observed, pval, outpath):
    plt.figure(figsize=(8,5))
    if perm.size>0:
        plt.hist(perm, bins=30, alpha=0.8)
    plt.axvline(observed, color='red', linewidth=2, label=f'Observed ATE={observed:.4g}')
    plt.legend()
    plt.title(f'Permutation distribution (placebo) — p={pval:.4f}')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT_DIR, n_boot=100, n_perm=200, random_state=0):
    df = pd.read_csv(input_path)
    # compute treatment
    components = [c for c in ['Cu','Al','Cr','Mg','Ni','Si'] if c in df.columns]
    if len(components)==0:
        raise ValueError('No component columns found in data')
    df['V5'] = compute_size_deviation_pct(df, components=components)

    treatment = 'V5'
    outcome = 'Electrical_Conductivity_IACS'
    if outcome not in df.columns:
        raise KeyError(f"Outcome column '{outcome}' not found in data. Columns: {list(df.columns)}")

    precision_vars = [v for v in ['Solid_Solution_Temp_K','Aging_Temp_K','Aging_Time_h'] if v in df.columns]
    controls = precision_vars + components

    # fit on observed data
    est = run_econml_fit(df, treatment, outcome, controls, random_state=random_state)
    if est is None:
        raise RuntimeError('EconML estimator failed to fit on observed data')

    t0 = df[treatment].mean()
    t1 = t0 + 1.0
    effects = est.effect(X=None, T0=t0, T1=t1)
    observed_ate = float(np.mean(effects))
    # CI from estimator
    try:
        ci_low, ci_high = est.ate_interval(X=None, T0=t0, T1=t1, alpha=0.05)
        # ate_interval may return scalars or arrays
        if np.ndim(ci_low)>0:
            ci_low = float(np.mean(ci_low))
            ci_high = float(np.mean(ci_high))
    except Exception:
        ci_low, ci_high = (np.nan, np.nan)

    print(f"ATE point estimate: {observed_ate}")

    # bootstrap (may be slow)
    boot = bootstrap_ate_econml(df, treatment, outcome, controls, n_boot=n_boot, random_state=random_state)
    boot_path = os.path.join(output_dir, 'bootstrap_ATE_CI.png')
    if boot.size>0:
        b_low = np.percentile(boot, 2.5)
        b_high = np.percentile(boot, 97.5)
        plot_and_save(boot, observed_ate, b_low, b_high, boot_path)
    else:
        # fallback to estimator CI
        plot_and_save(np.array([observed_ate]), observed_ate, ci_low, ci_high, boot_path)

    # permutation placebo
    obs, perm, pval = permutation_placebo_econml(df, treatment, outcome, controls, n_perm=n_perm, random_state=random_state)
    perm_path = os.path.join(output_dir, 'placebo_permutation.png')
    plot_permutation(perm, observed_ate, pval, perm_path)

    # first-stage R^2
    r2 = first_stage_r2(df, treatment, controls)

    print('95% CI plots saved to:')
    print('  ' + boot_path)
    print('  ' + perm_path)

if __name__ == '__main__':
    main()
