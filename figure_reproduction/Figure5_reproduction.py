"""
Reproduction script for Figure 5 of:
  "Neuronal tuning aligns dynamically with object and texture manifolds across the visual hierarchy"
  Nature Neuroscience, 2026

Figure 5: Object space preferentially activated late responses in PIT neurons.
  - Panel 5A: Population-averaged PSTHs from V1, V4, PIT at first (block 0) and last (block 55) generation
  - Panel 5C: Evolution trajectories for representative 10 ms time bins, showing texture dominance early
    and object dominance late in PIT temporal responses
"""
import os
import pandas as pd
from os.path import join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# --- Paths ---
source_data_dir = join(os.path.dirname(__file__), "..", "source_data", "Fig5")


def plot_maxnorm_psth_per_block(norm_psth_extrap_arr_per_block, block, commonmsk=True, titlestr="",
                                mask_col=[], mask_labels=[]):
    figh, axs = plt.subplots(1, 3, figsize=(9, 3.25), squeeze=False)
    for colj, (msk_minor, lable_minor) in enumerate(zip(mask_col, mask_labels)):
        msk = msk_minor & commonmsk
        axs[0, colj].plot(norm_psth_extrap_arr_per_block[msk, 0, :].T, color="blue", alpha=0.2, lw=0.7)
        axs[0, colj].plot(norm_psth_extrap_arr_per_block[msk, 1, :].T, color="red", alpha=0.2, lw=0.7)
        mean_psth_FC = norm_psth_extrap_arr_per_block[msk, 0, :].mean(axis=0)
        sem_psth_FC = norm_psth_extrap_arr_per_block[msk, 0, :].std(axis=0) / np.sqrt(msk.sum())
        mean_psth_BG = norm_psth_extrap_arr_per_block[msk, 1, :].mean(axis=0)
        sem_psth_BG = norm_psth_extrap_arr_per_block[msk, 1, :].std(axis=0) / np.sqrt(msk.sum())
        axs[0, colj].plot(mean_psth_FC, color="blue", lw=3, label="DeePSim")
        axs[0, colj].fill_between(np.arange(len(mean_psth_FC)),
                                    mean_psth_FC - sem_psth_FC,
                                    mean_psth_FC + sem_psth_FC,
                                    color="blue", alpha=0.25)
        axs[0, colj].plot(mean_psth_BG, color="red", lw=3, label="BigGAN")
        axs[0, colj].fill_between(np.arange(len(mean_psth_BG)),
                                    mean_psth_BG - sem_psth_BG,
                                    mean_psth_BG + sem_psth_BG,
                                    color="red", alpha=0.25)
        axs[0, colj].set_title(f"{lable_minor} (N={msk.sum()})")
    for ax in axs.ravel():
        ax.set_ylim([0, 2.0])
    axs[0, 0].legend(loc="lower right", frameon=False)
    plt.suptitle(f"Max Normalized PSTH block {block} [{titlestr} Sessions]")
    plt.tight_layout()
    plt.show()
    return figh


def load_norm_psth_extrap_arr_block(source_data_dir, block, slice_names=["DeePSim_mean", "BigGAN_mean"]):
    arrs = [
        pd.read_csv(
            join(source_data_dir, f"Figure5A_src_all_exps_maxnorm_mean_psth_per_block{block:02d}_{slice_name}.csv")
        ).values
        for slice_name in slice_names
    ]
    return np.stack(arrs, axis=1)


def plot_normalized_response_trajectories(normresp_extrap_arr_univ_col, rsp_wdws, area_masks, area_labels,
                                          commonmsk, signif_test=False, signif_alpha=0.05,
                                          plot_individual_exp=True,
                                          mcc_corrections=["nomcc", "fdr", "bonf"],
                                          panel_width=3, panel_height=3):
    figh, axs = plt.subplots(len(area_masks), len(rsp_wdws),
                              figsize=(panel_width * len(rsp_wdws), panel_height * len(area_masks) + .5),
                              sharey="row")
    for colj, (rsp_wdw, normresp_extrap_arr_univ) in enumerate(normresp_extrap_arr_univ_col.items()):
        for rowi, (msk_major, label_major) in enumerate(zip(area_masks, area_labels)):
            msk = msk_major & commonmsk
            if plot_individual_exp:
                axs[rowi, colj].plot(normresp_extrap_arr_univ[msk, :, 0].T, color="blue", alpha=0.2, lw=0.7)
                axs[rowi, colj].plot(normresp_extrap_arr_univ[msk, :, 1].T, color="red", alpha=0.2, lw=0.7)

            mean_trace_FC = normresp_extrap_arr_univ[msk, :, 0].mean(axis=0)
            sem_trace_FC = normresp_extrap_arr_univ[msk, :, 0].std(axis=0) / np.sqrt(msk.sum())
            mean_trace_BG = normresp_extrap_arr_univ[msk, :, 1].mean(axis=0)
            sem_trace_BG = normresp_extrap_arr_univ[msk, :, 1].std(axis=0) / np.sqrt(msk.sum())
            axs[rowi, colj].plot(mean_trace_FC, color="blue", lw=3, label="DeePSim")
            axs[rowi, colj].fill_between(np.arange(len(mean_trace_FC)),
                                         mean_trace_FC - sem_trace_FC,
                                         mean_trace_FC + sem_trace_FC,
                                         color="blue", alpha=0.25)
            axs[rowi, colj].plot(mean_trace_BG, color="red", lw=3, label="BigGAN")
            axs[rowi, colj].fill_between(np.arange(len(mean_trace_BG)),
                                         mean_trace_BG - sem_trace_BG,
                                         mean_trace_BG + sem_trace_BG,
                                         color="red", alpha=0.25)
            axs[rowi, colj].set_title(f"[{rsp_wdw.start}, {rsp_wdw.stop}ms]")
            if colj == 0:
                axs[rowi, 0].set_ylabel(f"Max Normalized Response\n{label_major} (N={msk.sum()})")

            if signif_test:
                FC_arr = normresp_extrap_arr_univ[msk, :, 0]
                BG_arr = normresp_extrap_arr_univ[msk, :, 1]
                pvals, tstat_signs = [], []
                for t in range(FC_arr.shape[1]):
                    tstat, p = stats.ttest_rel(FC_arr[:, t], BG_arr[:, t])
                    pvals.append(p)
                    tstat_signs.append(tstat > 0)
                pvals = np.array(pvals)
                tstat_signs = np.array(tstat_signs)
                sig_mask_nomcc = pvals < signif_alpha
                reject, pvals_fdr, _, _ = multipletests(pvals, alpha=signif_alpha, method='fdr_bh')
                sig_mask_fdr = reject
                pvals_bonf = pvals * len(pvals)
                sig_mask_bonf = pvals_bonf < signif_alpha

                for mcc_correction in mcc_corrections:
                    height_map = {"nomcc": 1.3, "fdr": 1.35, "bonf": 1.40}
                    label_map = {"nomcc": f'p<{signif_alpha}', "fdr": f'FDR p<{signif_alpha}', "bonf": f'Bonf p<{signif_alpha}'}
                    sig_map = {"nomcc": sig_mask_nomcc, "fdr": sig_mask_fdr, "bonf": sig_mask_bonf}
                    sig = sig_map[mcc_correction]
                    h = height_map[mcc_correction]
                    lbl = label_map[mcc_correction] if colj == 0 and rowi == 0 else ""
                    axs[rowi, colj].plot(np.where(sig & tstat_signs)[0], np.ones(np.sum(sig & tstat_signs)) * h,
                                'b.', markersize=4, label=lbl)
                    axs[rowi, colj].plot(np.where(sig & ~tstat_signs)[0], np.ones(np.sum(sig & ~tstat_signs)) * h,
                                'r.', markersize=4)

    for ax in axs.ravel():
        ax.set_xlim([-0.5, 45.5])
        ax.set_ylim([0, 1.5])
    axs[0, 0].legend(loc="upper right", frameon=False)
    return figh


# ===========================
# ==== Figure 5A: Population PSTHs ====
# ===========================
print("=== Figure 5A: Population PSTHs ===")
mask_df = pd.read_csv(join(source_data_dir, "Figure5_src_exp_masks_and_succ_labels.csv"))
V1msk = mask_df["V1msk"].values.astype(bool)
V4msk = mask_df["V4msk"].values.astype(bool)
ITmsk = mask_df["ITmsk"].values.astype(bool)
validmsk = mask_df["valid"].values.astype(bool)
sucsmsk = mask_df["anysucc005"].values.astype(bool)
commonmsk = validmsk & sucsmsk

norm_psth_extrap_arr_block0 = load_norm_psth_extrap_arr_block(source_data_dir, 0)
norm_psth_extrap_arr_block55 = load_norm_psth_extrap_arr_block(source_data_dir, 55)
plot_maxnorm_psth_per_block(norm_psth_extrap_arr_block0, 0, commonmsk=commonmsk,
                             titlestr="Valid & Succ", mask_col=[V1msk, V4msk, ITmsk], mask_labels=["V1", "V4", "IT"])
plot_maxnorm_psth_per_block(norm_psth_extrap_arr_block55, 55, commonmsk=commonmsk,
                             titlestr="Valid & Succ", mask_col=[V1msk, V4msk, ITmsk], mask_labels=["V1", "V4", "IT"])

# ===========================
# ==== Figure 5C: Time-binned trajectories ====
# ===========================
print("=== Figure 5C: Time-binned trajectories ===")
window_length = 10
rsp_wdws = [range(i * window_length, (i + 1) * window_length) for i in range(200 // window_length)]
slice_names = ["DeePSim_mean", "BigGAN_mean", "DeePSim_sem", "BigGAN_sem"]
normresp_extrap_arr_univ_col = {}
for rsp_wdw in rsp_wdws:
    slices = []
    for slice_name in slice_names:
        savepath = join(source_data_dir, f"Figure5C_src_normresp_wdw_{rsp_wdw.start}-{rsp_wdw.stop}_{slice_name}.csv")
        slices.append(pd.read_csv(savepath).values)
    normresp_extrap_arr_univ_col[rsp_wdw] = np.stack(slices, axis=-1)

bothsucsmsk = mask_df["bothsucc005"].values.astype(bool)
commonmsk_both = validmsk & bothsucsmsk
figh = plot_normalized_response_trajectories(normresp_extrap_arr_univ_col, rsp_wdws,
                            [V4msk, ITmsk], ["V4", "IT"],
                            commonmsk_both, signif_alpha=0.05, signif_test=True,
                            plot_individual_exp=False, mcc_corrections=["fdr"],
                            panel_width=1.5, panel_height=3)
figh.suptitle("Universal Max Normalized response 10ms window across blocks [Valid & Both Success Sessions]")
figh.tight_layout()
figh.show()
