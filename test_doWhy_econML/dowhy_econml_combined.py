import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# import guards
try:
    from dowhy import CausalModel
except Exception:
    print("ERROR: 'dowhy' is required. Install via: pip install dowhy")
    raise
try:
    from econml.dml import LinearDML
except Exception:
    print("ERROR: 'econml' is required. Install via: pip install econml")
    raise

# Defaults
INPUT_PATH = "D:/Khóa luận/data/final_data.csv"
OUTPUT_DIR = "D:/Khóa luận/data"

# Function: compute treatment (size deviation percent)
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

# Function: build DoWhy causal model
def build_dowhy_model(df, treatment='V5', outcome='Electrical_Conductivity_IACS', common_causes=None):
    """Construct DoWhy CausalModel."""
    return CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=common_causes)

# Function: estimate ATE via DoWhy linear regression
def estimate_dowhy_linear(model, identified_estimand):
    """Estimate ATE via DoWhy backdoor linear regression and return numeric value."""
    est = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression", target_units=None)
    val = getattr(est, 'value', None)
    if val is None:
        try:
            val = float(str(est))
        except Exception:
            val = np.nan
    return val

# Function: DoWhy bootstrap
def dowhy_bootstrap_ate(df, treatment, outcome, common_causes, n_boot=200, random_state=0):
    """Bootstrap ATE using DoWhy OLS estimator."""
    rng = np.random.RandomState(random_state)
    N = df.shape[0]
    ests = []
    for i in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        try:
            model_b = build_dowhy_model(df_b, treatment=treatment, outcome=outcome, common_causes=common_causes)
            ident_b = model_b.identify_effect()
            val = estimate_dowhy_linear(model_b, ident_b)
            ests.append(val)
        except Exception:
            ests.append(np.nan)
    arr = np.array(ests, dtype=float)
    return arr[~np.isnan(arr)]

# Function: DoWhy permutation placebo
def dowhy_placebo_test(df, treatment, outcome, common_causes, n_perm=200, random_state=1):
    """Permutation placebo using DoWhy OLS estimator."""
    model = build_dowhy_model(df, treatment=treatment, outcome=outcome, common_causes=common_causes)
    ident = model.identify_effect()
    obs = estimate_dowhy_linear(model, ident)
    rng = np.random.RandomState(random_state)
    perm = []
    N = df.shape[0]
    for _ in range(n_perm):
        df_p = df.copy()
        df_p[treatment] = rng.permutation(df_p[treatment].values)
        try:
            m_p = build_dowhy_model(df_p, treatment=treatment, outcome=outcome, common_causes=common_causes)
            id_p = m_p.identify_effect()
            val = estimate_dowhy_linear(m_p, id_p)
            perm.append(val)
        except Exception:
            perm.append(np.nan)
    perm = np.array(perm, dtype=float)
    perm = perm[~np.isnan(perm)]
    pval = np.mean(np.abs(perm) >= abs(obs)) if perm.size>0 else np.nan
    return obs, perm, pval

# Function: fit EconML LinearDML
def fit_econml_dml(df, treatment, outcome, controls, random_state=0):
    """Fit LinearDML (EconML) and return fitted estimator or None on failure."""
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

# Function: EconML bootstrap
def econml_bootstrap_ate(df, treatment, outcome, controls, n_boot=100, random_state=0):
    """Bootstrap ATE using EconML LinearDML."""
    rng = np.random.RandomState(random_state)
    N = df.shape[0]
    ests = []
    for i in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        try:
            est = fit_econml_dml(df_b, treatment, outcome, controls, random_state=rng.randint(1e6))
            if est is None:
                ests.append(np.nan)
            else:
                t0 = df_b[treatment].mean()
                t1 = t0 + 1.0
                eff = est.effect(X=None, T0=t0, T1=t1)
                ests.append(float(np.mean(eff)))
        except Exception:
            ests.append(np.nan)
    arr = np.array(ests, dtype=float)
    return arr[~np.isnan(arr)]

# Function: EconML permutation placebo
def econml_placebo_test(df, treatment, outcome, controls, n_perm=200, random_state=1):
    """Permutation placebo test for EconML LinearDML."""
    est = fit_econml_dml(df, treatment, outcome, controls, random_state=random_state)
    if est is None:
        raise RuntimeError('EconML failed on observed data')
    t0 = df[treatment].mean(); t1 = t0 + 1.0
    observed_eff = est.effect(X=None, T0=t0, T1=t1)
    observed_ate = float(np.mean(observed_eff))
    rng = np.random.RandomState(random_state)
    perm = []
    N = df.shape[0]
    for _ in range(n_perm):
        df_p = df.copy()
        df_p[treatment] = rng.permutation(df_p[treatment].values)
        try:
            est_p = fit_econml_dml(df_p, treatment, outcome, controls, random_state=rng.randint(1e6))
            eff_p = est_p.effect(X=None, T0=t0, T1=t1)
            perm.append(float(np.mean(eff_p)))
        except Exception:
            perm.append(np.nan)
    perm = np.array(perm, dtype=float)
    perm = perm[~np.isnan(perm)]
    pval = np.mean(np.abs(perm) >= abs(observed_ate)) if perm.size>0 else np.nan
    return observed_ate, perm, pval

# Plot helpers
def plot_hist_with_ci(arr, observed, outpath, title='Bootstrap distribution with 95% CI'):
    if arr.size == 0:
        # write a simple empty plot with observed
        plt.figure(figsize=(8,5)); plt.text(0.5,0.5,'No estimates',ha='center'); plt.title(title); plt.savefig(outpath); plt.close(); return
    low = np.percentile(arr, 2.5); high = np.percentile(arr, 97.5)
    plt.figure(figsize=(8,5))
    plt.hist(arr, bins=30, alpha=0.8)
    plt.axvline(observed, color='red', linewidth=2, label=f'Observed ATE={observed:.4g}')
    plt.axvline(low, color='black', linestyle='--', label=f'95% CI low={low:.4g}')
    plt.axvline(high, color='black', linestyle='--', label=f'95% CI high={high:.4g}')
    plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_perm(arr, observed, pval, outpath, title='Permutation distribution'):
    plt.figure(figsize=(8,5))
    if arr.size>0:
        plt.hist(arr, bins=30, alpha=0.8)
    plt.axvline(observed, color='red', linewidth=2, label=f'Observed ATE={observed:.4g}')
    plt.legend(); plt.title(f"{title} — p={pval:.4f}"); plt.tight_layout(); plt.savefig(outpath); plt.close()

# first-stage R^2
def first_stage_r2(df, treatment, covariates):
    if not covariates:
        return np.nan
    X = df[covariates].astype(float).values
    y = df[treatment].astype(float).values
    return LinearRegression().fit(X,y).score(X,y)

# Main pipeline
def run_pipeline(input_path=INPUT_PATH, output_dir=OUTPUT_DIR,
                 dowhy_boot=200, econml_boot=100, n_perm=200, random_state=0):
    df = pd.read_csv(input_path)
    components = [c for c in ['Cu','Al','Cr','Mg','Ni','Si'] if c in df.columns]
    if len(components)==0:
        raise ValueError('No component columns found in data')
    df['V5'] = compute_size_deviation_pct(df, components=components)
    treatment = 'V5'
    outcome = 'Electrical_Conductivity_IACS'
    if outcome not in df.columns:
        raise KeyError(f"Outcome column '{outcome}' not found. Columns: {list(df.columns)}")
    precision_vars = [v for v in ['Solid_Solution_Temp_K','Aging_Temp_K','Aging_Time_h'] if v in df.columns]
    common_causes = precision_vars
    controls = precision_vars + components

    # DoWhy OLS
    dm = build_dowhy_model(df, treatment=treatment, outcome=outcome, common_causes=common_causes)
    ident = dm.identify_effect()
    ate_dowhy = estimate_dowhy_linear(dm, ident)

    # DoWhy bootstrap & placebo
    dowhy_boot_est = dowhy_bootstrap_ate(df, treatment, outcome, common_causes, n_boot=dowhy_boot, random_state=random_state)
    dowhy_boot_path = os.path.join(output_dir, 'dowhy_bootstrap_ATE_CI.png')
    plot_hist_with_ci(dowhy_boot_est, ate_dowhy, dowhy_boot_path, title='DoWhy bootstrap ATE distribution')
    dowhy_obs, dowhy_perm, dowhy_p = dowhy_placebo_test(df, treatment, outcome, common_causes, n_perm=n_perm, random_state=random_state)
    dowhy_perm_path = os.path.join(output_dir, 'dowhy_placebo_permutation.png')
    plot_perm(dowhy_perm, ate_dowhy, dowhy_p, dowhy_perm_path, title='DoWhy permutation (placebo)')

    # EconML DML
    est = fit_econml_dml(df, treatment, outcome, controls, random_state=random_state)
    if est is None:
        raise RuntimeError('EconML failed to fit on observed data')
    t0 = df[treatment].mean(); t1 = t0 + 1.0
    effects = est.effect(X=None, T0=t0, T1=t1)
    ate_econml = float(np.mean(effects))
    try:
        ci_low, ci_high = est.ate_interval(X=None, T0=t0, T1=t1, alpha=0.05)
        if np.ndim(ci_low)>0:
            ci_low = float(np.mean(ci_low)); ci_high = float(np.mean(ci_high))
    except Exception:
        ci_low, ci_high = (np.nan, np.nan)

    # EconML bootstrap & placebo
    econml_boot_est = econml_bootstrap_ate(df, treatment, outcome, controls, n_boot=econml_boot, random_state=random_state)
    econml_boot_path = os.path.join(output_dir, 'econml_bootstrap_ATE_CI.png')
    plot_hist_with_ci(econml_boot_est, ate_econml, econml_boot_path, title='EconML bootstrap ATE distribution')
    econ_obs, econ_perm, econ_p = econml_placebo_test(df, treatment, outcome, controls, n_perm=n_perm, random_state=random_state)
    econ_perm_path = os.path.join(output_dir, 'econml_placebo_permutation.png')
    plot_perm(econ_perm, ate_econml, econ_p, econ_perm_path, title='EconML permutation (placebo)')

    # First-stage R2
    r2 = first_stage_r2(df, treatment, controls)

    # Print concise summary
    print('--- Results summary ---')
    print(f'DoWhy (OLS) ATE: {ate_dowhy}')
    print(f'EconML (LinearDML) ATE: {ate_econml}')
    print('95% CI plots saved to:')
    print('  ' + dowhy_boot_path)
    print('  ' + dowhy_perm_path)
    print('  ' + econml_boot_path)
    print('  ' + econ_perm_path)

if __name__ == '__main__':
    run_pipeline()

