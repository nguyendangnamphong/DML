# DoWhy-only causal pipeline (file and data expected at D:/Khóa luận/data)

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
try:
    from dowhy import CausalModel
except Exception as e:
    print("ERROR: dowhy not installed. Install with: pip install dowhy")
    raise
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Defaults
DEFAULT_INPUT = "D:/Khóa luận/data/final_data.csv"
DEFAULT_OUTPUT_DIR = "D:/Khóa luận/data"


def compute_size_deviation_pct(df, components=['Cu','Al','Cr','Mg','Ni','Si'], radii=None):
    """Compute percent atomic size deviation used as treatment.
    Formula: 100 * sqrt( sum_i (fraction_i * (r_i - r_mean)^2) )
    where r_mean = sum_i fraction_i * r_i.
    radii: dict mapping element -> atomic radius (same units).
    """
    missing = [c for c in components if c not in df.columns]
    if missing:
        raise ValueError(f"Missing component columns: {missing}")
    if radii is None:
        # default atomic radii (pm) - replace with authoritative values if needed
        radii = {'Cu':128,'Al':143,'Cr':128,'Mg':160,'Ni':124,'Si':111}
    # fractions must sum to 1 per sample; if given as percent, normalize
    vals = df[components].astype(float).values
    row_sums = vals.sum(axis=1, keepdims=True)
    fractions = np.divide(vals, row_sums, where=row_sums!=0)
    r_array = np.array([radii[c] for c in components], dtype=float)
    r_mean = (fractions * r_array).sum(axis=1)
    diffs_sq = (r_array - r_mean[:, None])**2
    weighted = (fractions * diffs_sq).sum(axis=1)
    size_dev_pct = 100.0 * np.sqrt(weighted)
    return size_dev_pct


def build_causal_model(df, treatment='V5', outcome='Electrical_Conductivity_IACS', common_causes=None):
    """Construct DoWhy CausalModel."""
    return CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=common_causes)


def estimate_ate_linear(model, identified_estimand):
    """Estimate ATE using backdoor linear regression via DoWhy and return numeric value."""
    est = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression", target_units=None)
    val = getattr(est, 'value', None)
    if val is None:
        try:
            val = float(str(est))
        except Exception:
            val = np.nan
    return val


def bootstrap_ate(df, treatment, outcome, common_causes, n_boot=200, random_state=0):
    """Bootstrap ATE estimates by resampling rows."""
    rng = np.random.RandomState(random_state)
    N = df.shape[0]
    ests = []
    for _ in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        df_b = df.iloc[idx].reset_index(drop=True)
        try:
            model_b = build_causal_model(df_b, treatment=treatment, outcome=outcome, common_causes=common_causes)
            ident_b = model_b.identify_effect()
            val = estimate_ate_linear(model_b, ident_b)
            ests.append(val)
        except Exception:
            ests.append(np.nan)
    ests = np.array(ests, dtype=float)
    return ests[~np.isnan(ests)]


def placebo_permutation_test(df, treatment, outcome, common_causes, n_perm=200, random_state=1):
    """Permutation placebo test returning observed ATE, permuted estimates and p-value."""
    model = build_causal_model(df, treatment=treatment, outcome=outcome, common_causes=common_causes)
    ident = model.identify_effect()
    obs = estimate_ate_linear(model, ident)
    rng = np.random.RandomState(random_state)
    perm = []
    N = df.shape[0]
    for _ in range(n_perm):
        df_p = df.copy()
        df_p[treatment] = rng.permutation(df_p[treatment].values)
        try:
            m_p = build_causal_model(df_p, treatment=treatment, outcome=outcome, common_causes=common_causes)
            id_p = m_p.identify_effect()
            val = estimate_ate_linear(m_p, id_p)
            perm.append(val)
        except Exception:
            perm.append(np.nan)
    perm = np.array(perm, dtype=float)
    perm = perm[~np.isnan(perm)]
    pval = np.mean(np.abs(perm) >= abs(obs)) if perm.size>0 else np.nan
    return obs, perm, pval


def plot_and_save_bootstrap(boot_estimates, observed_ate, outpath):
    """Plot bootstrap distribution and 95% CI, save image."""
    if boot_estimates.size == 0:
        return np.nan, np.nan
    low = np.percentile(boot_estimates, 2.5)
    high = np.percentile(boot_estimates, 97.5)
    plt.figure(figsize=(8,5))
    plt.hist(boot_estimates, bins=30, alpha=0.8)
    plt.axvline(observed_ate, color='red', linewidth=2, label=f'Observed ATE={observed_ate:.4g}')
    plt.axvline(low, color='black', linestyle='--', label=f'95% CI low={low:.4g}')
    plt.axvline(high, color='black', linestyle='--', label=f'95% CI high={high:.4g}')
    plt.legend()
    plt.title('Bootstrap ATE distribution with 95% CI')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return low, high


def plot_and_save_permutation(perm_estimates, observed_ate, pval, outpath):
    """Plot permutation distribution and save image."""
    plt.figure(figsize=(8,5))
    if perm_estimates.size > 0:
        plt.hist(perm_estimates, bins=30, alpha=0.8)
    plt.axvline(observed_ate, color='red', linewidth=2, label=f'Observed ATE={observed_ate:.4g}')
    plt.legend()
    plt.title(f'Permutation (placebo) distribution — p={pval:.4f}')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def first_stage_r2(df, treatment, covariates):
    """Compute R^2 from linear regression treatment ~ covariates."""
    if not covariates:
        return np.nan
    X = df[covariates].astype(float).values
    y = df[treatment].astype(float).values
    model = LinearRegression().fit(X, y)
    return model.score(X, y)


def main(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT_DIR, n_boot=200, n_perm=200, random_state=0):
    # load data
    df = pd.read_csv(input_path)

    # compute treatment variable (size deviation percent) and store in column 'V5'
    components = [c for c in ['Cu','Al','Cr','Mg','Ni','Si'] if c in df.columns]
    df['V5'] = compute_size_deviation_pct(df, components=components)

    treatment = 'V5'
    outcome = 'Electrical_Conductivity_IACS'
    precision_vars = [v for v in ['Solid_Solution_Temp_K','Aging_Temp_K','Aging_Time_h'] if v in df.columns]
    common_causes = precision_vars

    # observed ATE
    model = build_causal_model(df, treatment=treatment, outcome=outcome, common_causes=common_causes)
    ident = model.identify_effect()
    observed_ate = estimate_ate_linear(model, ident)
    print(f"ATE point estimate: {observed_ate}")

    # bootstrap and plot CI
    boot = bootstrap_ate(df, treatment, outcome, common_causes, n_boot=n_boot, random_state=random_state)
    boot_path = os.path.join(output_dir, 'bootstrap_ATE_CI.png')
    ci_low, ci_high = plot_and_save_bootstrap(boot, observed_ate, boot_path)

    # placebo permutation test and plot
    obs, perm_estimates, pval = placebo_permutation_test(df, treatment, outcome, common_causes, n_perm=n_perm, random_state=random_state)
    perm_path = os.path.join(output_dir, 'placebo_permutation.png')
    plot_and_save_permutation(perm_estimates, observed_ate, pval, perm_path)

    # first-stage R^2 (V5 ~ precision + components)
    covs = precision_vars + components
    r2 = first_stage_r2(df, treatment, covs)

    # print locations of saved images
    print('95% CI plots saved to:')
    print('  ' + boot_path)
    print('  ' + perm_path)


if __name__ == '__main__':
    main()
