"""
Reproduction script for Extended Figure 4 of:
  "Neuronal tuning aligns dynamically with object and texture manifolds across the visual hierarchy"
  Nature Neuroscience, 2026

Extended Figure 4: Time-binned response analysis.
  - Panel 4B: Activation increase attributed to different time windows (V1/V4/IT) for DeePSim vs BigGAN
  - Panel 4C: Evolution trajectories for multiple time-window sizes across visual areas
"""
import os
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection, multipletests

# --- Paths ---
source_data_dir = join(os.path.dirname(__file__), "..", "source_data", "ExtendedFig4")


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
                                         mean_trace_FC + sem_trace_FC, color="blue", alpha=0.25)
            axs[rowi, colj].plot(mean_trace_BG, color="red", lw=3, label="BigGAN")
            axs[rowi, colj].fill_between(np.arange(len(mean_trace_BG)),
                                         mean_trace_BG - sem_trace_BG,
                                         mean_trace_BG + sem_trace_BG, color="red", alpha=0.25)
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


# --- Load metadata and masks ---
meta_df = pd.read_csv(join(source_data_dir, "FigureExt4_meta_df.csv"), index_col=0)
V1msk = meta_df.visual_area == "V1"
V4msk = meta_df.visual_area == "V4"
ITmsk = meta_df.visual_area == "IT"
validmsk = meta_df.valid
thresh = 0.01
bothsucsmsk = (meta_df.p_maxinit_0 < thresh) & (meta_df.p_maxinit_1 < thresh)
FCsucsmsk = (meta_df.p_maxinit_0 < thresh)
BGsucsmsk = (meta_df.p_maxinit_1 < thresh)

# ===========================
# ==== Extended Figure 4B: Attribution to time windows ====
# ===========================
print("=== Extended Figure 4B: Temporal attribution ===")
for bin_size in [5, 10, 20, 25, 50]:
    DeePSim_df = pd.read_csv(join(source_data_dir, f"FigureExt4B_DeePSim_diff_attrib_norm_bin_tsr_{bin_size}ms.csv"), index_col=0)
    BigGAN_df = pd.read_csv(join(source_data_dir, f"FigureExt4B_BigGAN_diff_attrib_norm_bin_tsr_{bin_size}ms.csv"), index_col=0)
    diff_attrib_norm_bin_tsr = np.stack([DeePSim_df.values, BigGAN_df.values], axis=-1)
    thread_colors = ['b', 'r']
    figh, axs = plt.subplots(1, 3, figsize=[9, 3.5], sharex=True, sharey=True)
    time_ticks = np.arange(0, 200, bin_size) + bin_size / 2
    for i, (visual_area, area_mask) in enumerate(zip(["V1", "V4", "IT"], [V1msk, V4msk, ITmsk])):
        ax = axs[i]
        diff_attrib_both = np.zeros((2, diff_attrib_norm_bin_tsr.shape[1]))
        for thread, GANname, thread_sucsmsk in zip([0, 1], ["DeePSim", "BigGAN"], [FCsucsmsk, BGsucsmsk]):
            mask = area_mask & validmsk & thread_sucsmsk
            diff_attrib = diff_attrib_norm_bin_tsr[mask].mean(axis=0)
            diff_attrib_sem = diff_attrib_norm_bin_tsr[mask].std(axis=0) / np.sqrt(mask.sum())
            diff_attrib_both[thread, :] = diff_attrib[:, thread]
            ax.plot(time_ticks, diff_attrib[:, thread], color=thread_colors[thread],
                    label=f"{GANname} N={mask.sum()}")
            ax.fill_between(time_ticks,
                            diff_attrib[:, thread] - diff_attrib_sem[:, thread],
                            diff_attrib[:, thread] + diff_attrib_sem[:, thread],
                            color=thread_colors[thread], alpha=0.3)
        pvals, tvals = [], []
        for t in range(diff_attrib_norm_bin_tsr.shape[1]):
            tval, p = ttest_ind(diff_attrib_norm_bin_tsr[area_mask & validmsk & FCsucsmsk][:, t, 0],
                                diff_attrib_norm_bin_tsr[area_mask & validmsk & BGsucsmsk][:, t, 1])
            pvals.append(p)
            tvals.append(tval)
        pvals = np.array(pvals)
        tvals = np.array(tvals)
        signif_orig = pvals < 0.05
        fdr_reject, pvals_fdr = fdrcorrection(pvals, alpha=0.05)
        for signif, y_offset in zip([signif_orig, fdr_reject], [0, 0.1]):
            annot_y = diff_attrib_both.max() * (1.15 + y_offset)
            ax.plot(time_ticks[(tvals > 0) & signif], np.ones(np.sum((tvals > 0) & signif)) * annot_y, 'b.', markersize=4)
            ax.plot(time_ticks[(tvals < 0) & signif], np.ones(np.sum((tvals < 0) & signif)) * annot_y, 'r.', markersize=4)
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.set_title(visual_area)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Fraction of activation difference")
        ax.set_xlim([0, 200])
        ax.legend(loc="lower right")
    for ax in axs:
        ax.relim()
        ax.margins(y=0.15)
        ax.autoscale_view()
    plt.suptitle(f"Activation increase attributed to {bin_size}ms time windows [Evol succ p < {thresh}]")
    plt.tight_layout()
    plt.show()

# ===========================
# ==== Extended Figure 4C: Time-binned trajectory analysis ====
# ===========================
print("=== Extended Figure 4C: Time-binned trajectories ===")
slice_names = ["DeePSim_mean", "BigGAN_mean", "DeePSim_sem", "BigGAN_sem"]
for window_length in (10, 20, 25, 50):
    rsp_wdws = [range(i * window_length, (i + 1) * window_length) for i in range(200 // window_length)]
    normresp_extrap_arr_univ_col = {}
    for rsp_wdw in rsp_wdws:
        slices = []
        for slice_name in slice_names:
            savepath = join(source_data_dir, f"FigureExtended4C_src_normresp_wdw_{rsp_wdw.start}-{rsp_wdw.stop}_{slice_name}.csv")
            slices.append(pd.read_csv(savepath).values)
        normresp_extrap_arr_univ_col[rsp_wdw] = np.stack(slices, axis=-1)

    commonmsk = validmsk & bothsucsmsk
    figh = plot_normalized_response_trajectories(normresp_extrap_arr_univ_col, rsp_wdws,
                            [V4msk, ITmsk], ["V4", "IT"],
                            commonmsk, signif_alpha=0.05, signif_test=True,
                            plot_individual_exp=False, mcc_corrections=["fdr"],
                            panel_width=3, panel_height=3)
    figh.suptitle(f"Universal Max Normalized response {window_length}ms window across blocks [Valid & Both Success Sessions]")
    figh.tight_layout()
    figh.show()
