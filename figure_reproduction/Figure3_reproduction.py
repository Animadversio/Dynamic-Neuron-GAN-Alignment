"""
Reproduction script for Figure 3 of:
  "Neuronal tuning aligns dynamically with object and texture manifolds across the visual hierarchy"
  Nature Neuroscience, 2026

Figure 3: Optimized images showed local feature similarity.
  - Panel 3G: Image similarity (cosine distance in ResNet embedding / LPIPS) inversely correlated
    with PSTH differences, suggesting neurons respond to shared local motifs rather than global form.
"""
import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from scipy.stats import pearsonr, spearmanr

# --- Paths ---
source_data_dir = join(os.path.dirname(__file__), "..", "source_data", "Fig3")


def scatter_corr(df, x, y, ax=None, corrtype="pearson", **kwargs):
    """Scatter plot with Pearson/Spearman correlation annotated in title."""
    if ax is None:
        ax = plt.gca()
    sns.scatterplot(data=df, x=x, y=y, ax=ax, **kwargs)
    validmsk = np.logical_and(np.isfinite(df[x]), np.isfinite(df[y]))
    if corrtype.lower() == "pearson":
        rho, pval = pearsonr(df[x][validmsk], df[y][validmsk])
    elif corrtype.lower() == "spearman":
        rho, pval = spearmanr(df[x][validmsk], df[y][validmsk])
    else:
        raise NotImplementedError
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs. {y}\ncorr={rho:.3f} p={pval:.1e} n={validmsk.sum()}")
    return ax, rho, pval


# --- Load data ---
imgdist_df = pd.read_csv(join(source_data_dir, "Fig3G_proto_imdist_psth_covstr_sim_df.csv"))
meta_act_df = pd.read_csv(join(source_data_dir, "Fig3_meta_df.csv"))

validmsk = meta_act_df["valid"]
bothsucmsk = (meta_act_df.p_maxinit_0 < 0.05) & (meta_act_df.p_maxinit_1 < 0.05)

# --- Figure 3G: Image similarity vs PSTH difference ---
# ResNet embedding cosine distance vs PSTH mean absolute error
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'cosine_reevol_resnet_avgpool',
                   'psth_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()

# LPIPS perceptual similarity vs PSTH mean absolute error
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'reevol_pix_RNrobust_L4focus',
                   'psth_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()

# --- Control: image similarity vs activation difference (instead of PSTH) ---
fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'reevol_pix_RNrobust_L4focus',
                   'act_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=[4, 4])
ax, rho, pval = scatter_corr(imgdist_df[validmsk & bothsucmsk],
                   'cosine_reevol_resnet_avgpool',
                   'act_MAE', s=25, alpha=0.5, hue='visual_area')
plt.tight_layout()
plt.show()
