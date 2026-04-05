"""
Reproduction script for Figure 6 of:
  "Neuronal tuning aligns dynamically with object and texture manifolds across the visual hierarchy"
  Nature Neuroscience, 2026

Figure 6: Geometry of tuning landscapes in BigGAN latent space (Hessian analysis).
  - Panel 6C: Example bell-shaped tuning curve along a sampled class-space eigenvector axis
  - Panel 6D: Heatmap of mean neuronal responses along multiple eigenvectors (class and noise subspace)
  - Panel 6E: Distribution of tuning curve peak locations and shape types (bell-shaped vs. ramp)
              as a function of BigGAN evolution success
"""
import math
import os
import pandas as pd
from os.path import join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.proportion import proportion_confint

# --- Parula colormap (MATLAB default) ---
_parula_data = [
    [0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286], [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429], [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429], [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381], [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048], [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143], [0.3481142857, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905], [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762], [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905], [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286], [0.8185047619, 0.7327142857, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619], [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857], [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538],
]
parula = ListedColormap(_parula_data, name='parula')

# --- Paths ---
source_data_dir = join(os.path.dirname(__file__), "..", "source_data", "Fig6")


# ===========================
# ==== Fitting & Plotting Helpers ====
# ===========================

def gaussian_with_baseline(x, amplitude, mean, stddev, baseline):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + baseline


def gaussian_curve_fitting(x, y, constrained=True):
    df_tmp = pd.DataFrame({"x": x, "y": y})
    df_tmp = df_tmp.groupby("x").agg({"y": ["mean", "std", "sem"]}).reset_index()
    x_uniq = df_tmp["x"]
    y_mean = df_tmp["y"]["mean"]
    x_range = np.max(x_uniq) - np.min(x_uniq)
    initial_params = [np.max(y_mean) - np.min(y_mean), x_uniq[np.argmax(y_mean)], x_range / 4, np.min(y_mean)]
    bounds = ([0.0, np.min(x_uniq), 0.01, 0.0],
              [1.25 * np.max(y_mean), np.max(x_uniq), x_range * 2, np.max(y_mean)]) if constrained else None
    try:
        params, param_cov = curve_fit(gaussian_with_baseline, x, y, p0=initial_params, bounds=bounds)
        explained_variance = 1 - np.var(y - gaussian_with_baseline(x, *params)) / np.var(y)
    except RuntimeError:
        params, param_cov, explained_variance = None, None, None
    return {"params": params, "param_cov": param_cov, "explained_variance": explained_variance}


def linear_regression_fitting(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    explained_variance = model.score(x.reshape(-1, 1), y)
    return {"params": {"slope": model.coef_[0], "intercept": model.intercept_},
            "param_cov": None, "explained_variance": explained_variance}


def gaussian_process_regression(x_train, y_train, n_eval_points=100):
    length_scale = (x_train.max() - x_train.min()) / 10
    df_tmp = pd.DataFrame({"x": x_train, "y": y_train})
    noise_var = df_tmp.groupby("x").agg({"y": ['var']})["y"]["var"].mean()
    y_var = y_train.var()
    if np.isnan(noise_var):
        noise_var = y_var
    kernel = (ConstantKernel(y_var, (y_var * 1e-2, y_var * 1e2)) *
              RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=noise_var * 0.1, noise_level_bounds=(noise_var * 1e-2, noise_var * 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10)
    gpr.fit(x_train.reshape(-1, 1), y_train.reshape(-1))
    explained_variance = 1 - np.var(y_train - gpr.predict(x_train.reshape(-1, 1))) / np.var(y_train)
    x_eval = np.linspace(x_train.min(), x_train.max(), n_eval_points)
    y_mean, y_std = gpr.predict(x_eval.reshape(-1, 1), return_std=True)
    return {"gpr": gpr, "explained_variance": explained_variance,
            "x_eval": x_eval, "y_mean": y_mean.reshape(-1), "y_std": y_std}


def anova_test_df(df, x_col="lin_dist", y_col="pref_unit_resp"):
    try:
        model = ols(f'{y_col} ~ C({x_col})', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        F_value = anova_table.loc[f'C({x_col})', 'F']
        p_value = anova_table.loc[f'C({x_col})', 'PR(>F)']
        return {"F_value": F_value, "p_value": p_value,
                "stats_str": f"F-val: {F_value:.2f} | p-val: {p_value:.1e}",
                "anova_table": anova_table, "error": None}
    except Exception as e:
        return {"F_value": np.nan, "p_value": np.nan, "stats_str": "", "anova_table": None, "error": e}


def regression_combined_plot(x_train, y_train, gauss_fit_results, ols_fit_results, gpr_fit_results, anova_results=None, title_str="", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(x_train, y_train, 'o', label='data', alpha=0.5)
    tmp_df = pd.DataFrame({"x": x_train, "y": y_train}).groupby("x").agg({"y": ["mean", "std"]}).reset_index()
    ax.errorbar(tmp_df["x"], tmp_df["y"]["mean"], yerr=tmp_df["y"]["std"],
                label='mean ± std', marker='D', color="C1", markersize=5, capsize=5, alpha=0.5, linestyle="")
    x_eval = np.linspace(x_train.min(), x_train.max(), 100)
    gauss_params = gauss_fit_results["params"]
    ols_params = ols_fit_results["params"]
    gpr_kernel_str = str(gpr_fit_results["gpr"].kernel_).replace("length_scale", "len").replace("noise_level", "noise")
    if gauss_params is not None:
        ax.plot(x_eval, gaussian_with_baseline(x_eval, *gauss_params), label='Gaussian fit', color="k")
    ax.plot(x_eval, ols_params["slope"] * x_eval + ols_params["intercept"], label='OLS fit', color="magenta", linestyle="--")
    ax.plot(gpr_fit_results["x_eval"], gpr_fit_results["y_mean"], label='GPR mean', color="red")
    ax.fill_between(gpr_fit_results["x_eval"],
                    gpr_fit_results["y_mean"] - gpr_fit_results["y_std"],
                    gpr_fit_results["y_mean"] + gpr_fit_results["y_std"],
                    alpha=0.2, label="GPR std", color="red")
    ax.legend()
    caption = f"{title_str}\n" if title_str else ""
    if gauss_params is not None:
        caption += f"Gauss: Ampl={gauss_params[0]:.2f} Mean={gauss_params[1]:.2f} Std={gauss_params[2]:.2f} Bsl={gauss_params[3]:.2f} [R2={gauss_fit_results['explained_variance']:.2f}]\n"
    caption += f"OLS: Slope={ols_params['slope']:.2f} Int={ols_params['intercept']:.2f} [R2={ols_fit_results['explained_variance']:.2f}]\n"
    caption += f"GPR: {gpr_kernel_str} [R2={gpr_fit_results['explained_variance']:.2f}]"
    if anova_results is not None:
        caption += f"\nANOVA: {anova_results['stats_str']}"
    ax.set_title(caption, fontsize=10)
    fig.tight_layout()
    return fig


def plot_heatmap(grouped, space, ax, CLIM, annot=True, fmt=".1f"):
    space_data = grouped[grouped['space_name'] == space]
    pivot_table = space_data.pivot(index='eig_id', columns='lin_dist', values='pref_unit_resp').astype(float)
    if pivot_table.empty:
        return
    plt.sca(ax)
    sns.heatmap(pivot_table, annot=annot, fmt=fmt, cmap=parula,
                cbar_kws={'label': 'Preferred Unit Response'}, ax=ax, vmin=CLIM[0], vmax=CLIM[1])
    plt.title(f'Heatmap of Preferred Unit Response: {space} space')
    plt.xlabel('Linear Distance')
    plt.ylabel('Eigenvalue ID')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)


# ===========================
# ==== Figure 6C: Example tuning curve ====
# ===========================
print("=== Figure 6C: Example tuning curve ===")
sgtr_resp_df = pd.read_csv(join(source_data_dir, "Figure6C_src_B07092020_sgtr_resp_df.csv"))
title_str = "B-07092020-003 | Pref Chan 5B"
sgtr_resp_at_origin = sgtr_resp_df.query("lin_dist == 0.0")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
space, eig_id = "class", 0
sgtr_resp_per_axis = sgtr_resp_df.query("space_name == @space and eig_id == @eig_id")
if 0.0 not in sgtr_resp_per_axis["lin_dist"].unique():
    sgtr_resp_per_axis = pd.concat([sgtr_resp_per_axis, sgtr_resp_at_origin])
gauss_fit_results = gaussian_curve_fitting(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values)
ols_fit_results = linear_regression_fitting(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values)
gpr_fit_results = gaussian_process_regression(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values)
anova_results = anova_test_df(sgtr_resp_per_axis)
regression_combined_plot(sgtr_resp_per_axis["lin_dist"].values, sgtr_resp_per_axis["pref_unit_resp"].values,
                          gauss_fit_results, ols_fit_results, gpr_fit_results,
                          anova_results=anova_results, title_str=f"{title_str} | {space} {eig_id} axis", ax=ax)
plt.xlabel("Linear Distance")
plt.ylabel("Response (events/s)")
plt.tight_layout()
plt.show()

# ===========================
# ==== Figure 6D: Heatmap of tuning landscape ====
# ===========================
print("=== Figure 6D: Heatmap of tuning landscape ===")
avgresp_df = pd.read_csv(join(source_data_dir, "Figure6D_src_prefchan_avgresp_df.csv"))
ephysFN = avgresp_df.iloc[0]["ephysFN"]
prefchan_str = avgresp_df.iloc[0]["prefchan_str"]
CLIM = np.quantile(avgresp_df['pref_unit_resp'], [0.02, 0.98])
figh, axs = plt.subplots(1, 2, figsize=(13, 6))
for ax, space in zip(axs, ['class', 'noise']):
    plot_heatmap(avgresp_df, space, ax, CLIM)
    ax.set_title(f'{space} space')
axs[-1].figure.axes[-1].set_ylabel("Response (events/s)")
plt.suptitle(f'Preferred Unit Response\n{ephysFN} | Pref Channel {prefchan_str}')
plt.tight_layout()
plt.show()

# Non-successful evolution example
avgresp_df_fail = pd.read_csv(join(source_data_dir, "Figure6D_src_prefchan_avgresp_df_failevol.csv"))
ephysFN = avgresp_df_fail.ephysFN.iloc[0]
prefchan_str = avgresp_df_fail.prefchan_str.iloc[0]
CLIM = np.quantile(avgresp_df_fail['pref_unit_resp'], [0.02, 0.98])
figh, ax = plt.subplots(1, 1, figsize=(4.5, 4))
plot_heatmap(avgresp_df_fail.query("eig_id in [0,1,2,3,6,9,13,21,30,60]"), "class", ax, CLIM, annot=False)
ax.set_title('class space (failed evolution)')
ax.figure.axes[-1].set_ylabel("Response (events/s)")
plt.suptitle(f'Preferred Unit Response\n{ephysFN} | Pref Channel {prefchan_str}')
plt.tight_layout()
plt.show()

# ===========================
# ==== Figure 6E: Tuning shape & peak location ====
# ===========================
print("=== Figure 6E: Tuning shape type barplot ===")
tuning_fitting_stats_table_sel = pd.read_csv(join(source_data_dir, "Figure6E_src_tuning_shape_fitting_stats_synopsis_selcolumn.csv"))
filtered_data = tuning_fitting_stats_table_sel.query("anova_p_value < 0.01 and is_common_axis")
melted_data = filtered_data.melt(
    id_vars='is_BigGAN_evol_success',
    value_vars=['gpr_y_is_bellshaped', 'gpr_y_is_monotonic', 'gpr_y_is_unimodal'],
    var_name='Metric', value_name='Value'
)
annotation_data = melted_data.groupby(['is_BigGAN_evol_success', 'Metric']).agg(
    True_Count=('Value', 'sum'), Total_Count=('Value', 'count'), Ratio=('Value', 'mean'),
).reset_index().sort_values(['is_BigGAN_evol_success', 'Metric'])

plt.figure(figsize=(4.5, 6))
ax = sns.barplot(data=melted_data, x='is_BigGAN_evol_success', y='Value', hue='Metric',
                 order=["True", "False"], hue_order=["gpr_y_is_bellshaped", "gpr_y_is_monotonic"],
                 errorbar=("ci", 95))
plt.ylabel("Fraction of tuning axis")
plt.xlabel("BigGAN evolution success")
for p in ax.patches:
    height = p.get_height()
    row = annotation_data[annotation_data['Ratio'] == height]
    if row.empty:
        continue
    row = row.iloc[0]
    ax.annotate(f"{row['Ratio']:.2f}\n{int(row['True_Count'])}/{int(row['Total_Count'])}",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color="white", fontweight="bold")
plt.suptitle("Tuning curve shape type as a function of BigGAN evolution success\n[signif axis, ANOVA p < 0.01]")
plt.show()

print("=== Figure 6E: Peak location distribution ===")
tuning_stats_synopsis_df = pd.read_csv(join(source_data_dir, "Figure6E_src_tuning_stats_synopsis_selcolumn.csv"))
fs = 10
pval_threshold = 0.01
tuning_stats_synopsis_df['sig'] = tuning_stats_synopsis_df['p_value'] < pval_threshold

def plot_peak_location_panel(ax, df, title, show_ylabel=False):
    sns.countplot(data=df, x='max_resp_lin_dist_bin', stat='proportion', ax=ax, color='k')
    ax.set_title(f'{title} [N={len(df)}]', fontsize=fs)
    ax.set_xlabel('Peak Location on Axis', fontsize=fs)
    ax.set_ylabel('Fraction' if show_ylabel else '', fontsize=fs)
    counts = df['max_resp_lin_dist_bin'].value_counts().sort_index()
    N = counts.sum()
    props = counts / N
    ci_low, ci_high = proportion_confint(counts.values, N, alpha=0.05, method='wilson')
    bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
    ax.errorbar(bar_positions, props.values, yerr=[props.values - ci_low, ci_high - props.values],
                fmt='none', capsize=3, linewidth=1.5, color='red')
    for xpos, count in zip(bar_positions, counts.values):
        ax.text(xpos, 0, str(count), ha='center', va='bottom', fontsize=fs - 1, color='white')

fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)
plot_peak_location_panel(ax[0], tuning_stats_synopsis_df.query("is_BigGAN_evol_success and sig"), "Successful", show_ylabel=True)
plot_peak_location_panel(ax[1], tuning_stats_synopsis_df.query("not is_BigGAN_evol_success and sig"), "Non-successful")
for axi in ax:
    for label in axi.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    axi.tick_params(axis='x', labelsize=fs)
fig.suptitle(f'Peak Response Location for successful vs non-successful evolution\n[Tuning Axis Signif. p < {pval_threshold}]', fontsize=fs)
plt.tight_layout()
plt.show()
