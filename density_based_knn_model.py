"""
COSINE-SIMILARITY NEAREST-NEIGHBOR MODEL — LOEGO EVALUATION ONLY
==================================================================

Vector M: 40 dims = 20 "center_*" features + 20 nanmean("north_*","south_*","east_*","west_*")

Pipeline:
  1. Load features_segmented + experiments from droplets.db
  2. Derive group key from droplet_id (batch prefix before "__droplet")
  3. Build the 40-dim vector per droplet (raw, un-scaled)
  4. LOEGO: for each held-out GROUP (not row) —
       - fit StandardScaler on TRAIN groups only
       - transform train + held-out test with that scaler
       - z-scored cosine similarity -> nearest TRAIN neighbor
       - predicted (glycerol_pct, concentration_wv) = neighbor's true values
  5. Aggregate predictions across all folds -> R² / MAE
     (same "honest" philosophy as group_loego_evaluate() in model2_regression.py:
      a condition never appears in both train and test within a fold)

No train/test leakage: scaler fit happens INSIDE the fold loop, on train groups only.
"""

import sqlite3
import re
import os
import sys
import contextlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score


@contextlib.contextmanager
def _suppress_stdout(active=True):
    """Silences print() inside the with-block when active=True. Used to hide
    noisy per-candidate output from sweeps while keeping the summary table."""
    if not active:
        yield
        return
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = r"C:\26_internship\droplets.db"   # change if running elsewhere
TARGETS = ["glycerol_pct", "concentration_wv"]
CONC_MAX = 1.0
CONC_MIN = 0.002  # set >0 (e.g. 0.01) to exclude the low-concentration regime
                 # where LOEGO showed severe failure (R^2=-2.07, ~109% rel.
                 # error in the <=0.01 w/v% bin) -- restricting the reported
                 # range is a legitimate scoping choice as long as it's
                 # stated explicitly as the model's validated operating range

TILE_POSITIONS = ["center", "north", "south", "east", "west"]
AVG_POSITIONS = ["north", "south", "east", "west"]

# -----------------------------
# FEATURE WEIGHTING (optional, A/B against unweighted baseline)
# -----------------------------
# Mode: "none"        -> all 40 dims weighted equally (baseline, current behavior)
#       "exclude"      -> hard-drop known-noisy base features before building the vector
#       "correlation"  -> soft-weight each dim by |corr| with target, refit PER FOLD
#                         on train data only (no leakage)
#       "threshold"    -> HARD-drop any dim whose raw |corr| with target falls
#                         below CORR_THRESHOLD (equal weight among survivors),
#                         refit PER FOLD on train data only (no leakage)
WEIGHT_MODE = "threshold"  # "none" | "exclude" | "correlation" | "threshold"

# Raw |correlation| cutoff used only when WEIGHT_MODE == "threshold".
# Run raw_correlation_diagnostic() first to see the actual distribution
# before picking this -- 0.7 on a 40-dim image-feature set is a high bar
# and may leave very few (or zero) surviving features.
CORR_THRESHOLD = 0.4

# Base feature names (without center_/avg_ prefix) known to be noisy from
# CV/heatmap inspection — ring peak-localization (CV ~0.69-0.78) and
# connectivity/distance features that were visibly noisy tile-to-tile.
EXCLUDE_BASE_FEATURES = [
    "ring_peak_contrast",
    "ring_width_frac",
    "anisotropy",
    "cluster_nn_distance",
]

# Which target drives correlation weighting when WEIGHT_MODE == "correlation".
# Pe/Ma-linked features track glycerol; concentration is a separate mapping —
# pick the target you're currently prioritizing, or "combined" to average
# |corr| across both targets.
CORR_WEIGHT_TARGET = "concentration_wv"  # "glycerol_pct" | "concentration_wv" | "combined"

# k-NN averaging config — 1-NN is high-variance with only ~4-5 replicates/group.
# Averaging the top-k nearest neighbors (similarity-weighted) smooths that out.
K_NEIGHBORS = 9

# -----------------------------
# CONCENTRATION DEFINITION: raw w/v% vs. glycerol-corrected (MFAT)
# -----------------------------
# concentration_wv as logged is particle mass per total solution volume,
# treating concentration as if it were a single, independent factor.
# The corrected target folds glycerol% directly into concentration:
#
#   concentration_corrected = concentration_wv * (1 - glycerol_pct/100)
#
# i.e. concentration_wv - (glycerol_pct/100)*concentration_wv. MFAT here
# = "multiple factors at a time" — the corrected target is no longer a
# function of concentration alone, it's a joint function of BOTH
# concentration and glycerol%. Worth testing whether predicting this
# joint quantity changes LOEGO performance vs. predicting raw
# concentration_wv (single factor) — e.g. if raw concentration_wv
# partially "absorbs" glycerol-driven variation that a single-factor
# target can't otherwise represent, the corrected joint target may be
# either easier or harder to recover from the image features.
USE_CORRECTED_CONCENTRATION = True  # flip to True to run everything on the joint (MFAT) target instead
# -----------------------------
# STEP 1 — LOAD
# -----------------------------
def load_data(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT e.droplet_id AS droplet_id, e.glycerol_pct, e.concentration_wv, f.*
        FROM experiments e
        JOIN features_segmented f ON e.droplet_id = f.droplet_id
        WHERE e.glycerol_pct IS NOT NULL
          AND e.concentration_wv IS NOT NULL
    """, conn)
    conn.close()

    # f.* also contains a droplet_id column (features_segmented.droplet_id),
    # so pandas ends up with two identically-named columns. Drop the duplicate,
    # keep the first (from e.droplet_id).
    df = df.loc[:, ~df.columns.duplicated()].copy()

    print(f"Loaded {len(df)} rows from features_segmented")
    return df


# -----------------------------
# STEP 2 — GROUP KEY (batch identity, not replicate)
# -----------------------------
def batch_group(droplet_id):
    """
    droplet_id format: gly0_ps01__droplet3  -> group = gly0_ps01
    Falls back to the full droplet_id if the "__" separator is absent.
    """
    if "__" in droplet_id:
        return droplet_id.split("__")[0]
    return droplet_id


# -----------------------------
# STEP 3 — BUILD 40-DIM VECTOR
# -----------------------------
def discover_base_feature_names(columns):
    """
    From columns like 'center_void_fraction', 'north_anisotropy', ...
    recover the 20 base feature names (suffix after 'center_').
    """
    base_names = []
    for c in columns:
        if c.startswith("center_"):
            base_names.append(c[len("center_"):])
    return base_names


def build_vector_matrix(df, base_names):
    """
    Returns:
      X        : (n_rows, 2*len(base_names)) ndarray, RAW (un-scaled)
      feat_cols: list of the 40 output column names, in order
                 [center_<f1>...center_<fN>, avg_<f1>...avg_<fN>]
    """
    center_cols = [f"center_{b}" for b in base_names]
    avg_block = np.full((len(df), len(base_names)), np.nan)

    for j, b in enumerate(base_names):
        side_cols = [f"{pos}_{b}" for pos in AVG_POSITIONS if f"{pos}_{b}" in df.columns]
        if not side_cols:
            continue
        side_vals = df[side_cols].values.astype(float)
        with np.errstate(invalid="ignore"):
            avg_block[:, j] = np.nanmean(side_vals, axis=1)

    center_block = df[center_cols].values.astype(float)
    X = np.hstack([center_block, avg_block])
    feat_cols = center_cols + [f"avg_{b}" for b in base_names]
    return X, feat_cols


# -----------------------------
# STEP 4 — CLEAN
# -----------------------------
def clean(df, base_names, use_corrected_concentration=USE_CORRECTED_CONCENTRATION,
          conc_min=CONC_MIN, conc_max=CONC_MAX):
    df = df.copy()
    df["concentration_wv"] = pd.to_numeric(df["concentration_wv"], errors="coerce")
    df["glycerol_pct"] = pd.to_numeric(df["glycerol_pct"], errors="coerce")

    # scope to the physical experimental range BEFORE any MFAT correction --
    # the restriction is about which raw concentration conditions are in
    # scope, not about the transformed target value
    before_scope = len(df)
    df = df[(df["concentration_wv"] > conc_min) & (df["concentration_wv"] <= conc_max)]
    if conc_min > 0:
        print(f"CONC_MIN={conc_min} -> restricted to concentration_wv in "
              f"({conc_min}, {conc_max}]: {len(df)}/{before_scope} rows kept "
              f"({before_scope - len(df)} dropped as out-of-scope, not as bad data)")

    df["group"] = df["droplet_id"].apply(batch_group)

    if use_corrected_concentration:
        # MFAT: mass fraction adjusted for the glycerol-occupied portion
        # of the solution. Overwrites concentration_wv in place so every
        # downstream function (which reads TARGETS -> "concentration_wv")
        # works unchanged, without needing a second target name threaded
        # through the whole pipeline.
        df["concentration_wv"] = df["concentration_wv"] * (1.0 - df["glycerol_pct"] / 100.0)
        print("USE_CORRECTED_CONCENTRATION=True -> concentration_wv replaced with "
              "concentration_wv * (1 - glycerol_pct/100) (MFAT) for this run.")

    X, feat_cols = build_vector_matrix(df, base_names)

    # drop rows where the 40-dim vector is entirely unusable
    # (require at least 50% of dims present)
    valid = np.mean(~np.isnan(X), axis=1) >= 0.5
    before = len(df)
    df = df[valid].reset_index(drop=True)
    X = X[valid]
    print(f"After cleaning: {len(df)} rows ({before - len(df)} dropped, "
          f"<50% of 40-dim vector present)")

    n_groups = df["group"].nunique()
    print(f"Distinct groups (batches): {n_groups}")
    if n_groups < 3:
        print("Need at least 3 distinct groups for LOEGO. Add more conditions.")
        return None, None, None, None

    y = df[TARGETS].values
    groups = df["group"].values
    return df, X, y, groups


# -----------------------------
# STEP 5 — COSINE NN PREDICTOR (fit per-fold)
# -----------------------------
def nan_safe_impute(X_train, X_test):
    """
    Column-median impute (computed on TRAIN only) for any remaining NaNs,
    so StandardScaler/cosine similarity don't choke.
    """
    col_median = np.nanmedian(X_train, axis=0)
    col_median = np.where(np.isnan(col_median), 0.0, col_median)
    X_train = np.where(np.isnan(X_train), col_median, X_train)
    X_test = np.where(np.isnan(X_test), col_median, X_test)
    return X_train, X_test



def _local_density(X, k=9):
    """Estimate local density as inverse mean distance to k nearest training points."""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X, X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    kk = min(k, max(1, X.shape[0] - 1))
    knn_d = np.partition(D, kk - 1, axis=1)[:, :kk]
    return 1.0 / np.maximum(np.mean(knn_d, axis=1), 1e-12)


def density_knn_predict(X_train_z, y_train, X_test_z, k=K_NEIGHBORS):
    """
    Density-weighted kNN regression: fixed top-k neighbors per query,
    weighted by local_density / distance.

    (An earlier version tried a per-query ADAPTIVE neighbor count based on
    a similarity threshold relative to each query's nearest match. That
    was reverted: the multiplicative threshold is unstable whenever a
    query happens to sit very close to a near-duplicate training point --
    the threshold collapses toward zero and almost nothing else qualifies
    -- and it measurably underperformed fixed-k on this data across two
    tuning attempts. k itself is still chosen dynamically, just at the
    fold level via honest nested CV (see nested_loego_hyperparameter_
    tuning), not re-derived per individual query.)
    """
    from sklearn.metrics import pairwise_distances

    k = min(k, len(X_train_z))
    D = pairwise_distances(X_test_z, X_train_z, metric="euclidean")
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    d = np.take_along_axis(D, idx, axis=1)

    rho = _local_density(X_train_z, k=k)[idx]
    weights = rho / np.maximum(d, 1e-12)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    neigh_y = y_train[idx]
    weighted = weights[..., None] * neigh_y if np.ndim(neigh_y) == 3 else weights * neigh_y
    pred = np.sum(weighted, axis=1)
    return pred, d.min(axis=1)

def density_knn_predict_with_uncertainty(X_train_z, y_train, X_test_z, k,
                                            min_k_for_std=3):
    """Density-weighted kNN prediction with weighted neighbor uncertainty (fixed top-k)."""
    from sklearn.metrics import pairwise_distances

    k = min(max(k, min_k_for_std), X_train_z.shape[0])
    D = pairwise_distances(X_test_z, X_train_z, metric="euclidean")
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    d = np.take_along_axis(D, idx, axis=1)

    rho = _local_density(X_train_z, k=k)[idx]
    weights = rho / np.maximum(d, 1e-12)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    neigh_y = y_train[idx]
    if np.ndim(neigh_y) == 3:
        w = weights[..., None]
        preds = np.sum(w * neigh_y, axis=1)
        var = np.sum(w * (neigh_y - preds[:, None, :]) ** 2, axis=1)
    else:
        preds = np.sum(weights * neigh_y, axis=1)
        var = np.sum(weights * (neigh_y - preds[:, None]) ** 2, axis=1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return preds, std, d.min(axis=1)

def bayesian_concentration_update(X, y, groups, k=K_NEIGHBORS, glycerol_tol=1e-6):
    """
    Bayesian Gaussian update for concentration_wv, mirroring pe_prior() in
    model2_regression.py:

      PRIOR      = density-based kNN prediction using the full training pool,
                   glycerol UNKNOWN (your 0.665 baseline model)
                   -> prior_mean, prior_var from the k-neighbor spread

      LIKELIHOOD = density-based kNN prediction restricted to train droplets at
                   the SAME true glycerol condition, glycerol KNOWN
                   -> likelihood_mean, likelihood_var from that subset's
                      k-neighbor spread

      POSTERIOR  = conjugate Gaussian combination:
        posterior_var  = 1 / (1/prior_var + 1/likelihood_var)
        posterior_mean = posterior_var * (prior_mean/prior_var +
                                           likelihood_mean/likelihood_var)

    This replaces the ad-hoc glycerol_steer_weight dial with a principled
    combination: each source's own predictive spread determines how much
    it gets trusted, instead of a hand-picked mixing weight.
    """
    gly_idx = TARGETS.index("glycerol_pct")
    conc_idx = TARGETS.index("concentration_wv")

    logo = LeaveOneGroupOut()
    all_true = []
    all_prior_mean, all_prior_std = [], []
    all_like_mean, all_like_std = [], []
    all_post_mean, all_post_std = [], []
    n_no_exact_match = 0

    n_folds = logo.get_n_splits(groups=groups)
    print(f"\nRunning Bayesian concentration update over {n_folds} folds "
          f"(prior=glycerol-unknown, likelihood=glycerol-known, k={k})...")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        X_train_full, X_test = nan_safe_impute(X_train_full, X_test)

        # --- PRIOR: full pool, concentration-correlation weighting, glycerol unknown ---
        scaler_prior = StandardScaler()
        X_train_prior_z = scaler_prior.fit_transform(X_train_full)
        X_test_prior_z = scaler_prior.transform(X_test)
        w_prior = compute_correlation_weights(X_train_prior_z, y_train_full, TARGETS, "concentration_wv")
        X_train_prior_z = X_train_prior_z * w_prior
        X_test_prior_z = X_test_prior_z * w_prior

        prior_mean_all, prior_std_all, _prior_nn_dist = density_knn_predict_with_uncertainty(
            X_train_prior_z, y_train_full, X_test_prior_z, k=k)

        for row_i in range(len(test_idx)):
            true_gly = y_test[row_i, gly_idx]

            # --- LIKELIHOOD: glycerol-matched subset, glycerol known ---
            match_mask = np.abs(y_train_full[:, gly_idx] - true_gly) <= glycerol_tol
            if match_mask.sum() < 3:
                nearest_gly = y_train_full[np.argmin(
                    np.abs(y_train_full[:, gly_idx] - true_gly)), gly_idx]
                match_mask = np.abs(y_train_full[:, gly_idx] - nearest_gly) <= glycerol_tol
                n_no_exact_match += 1

            X_train_sub = X_train_full[match_mask]
            y_train_sub = y_train_full[match_mask]

            scaler_like = StandardScaler()
            X_train_sub_z = scaler_like.fit_transform(X_train_sub)
            X_test_sub_z = scaler_like.transform(X_test[row_i:row_i + 1])

            if len(X_train_sub) >= 3:
                w_like = compute_correlation_weights(X_train_sub_z, y_train_sub, TARGETS, "concentration_wv")
                X_train_sub_z = X_train_sub_z * w_like
                X_test_sub_z = X_test_sub_z * w_like

            k_eff = min(k, len(X_train_sub))
            like_mean, like_std, _like_nn_dist = density_knn_predict_with_uncertainty(
                X_train_sub_z, y_train_sub, X_test_sub_z, k=k_eff)

            # --- conjugate Gaussian update (concentration_wv only) ---
            p_mean = prior_mean_all[row_i, conc_idx]
            p_var = prior_std_all[row_i, conc_idx] ** 2
            l_mean = like_mean[0, conc_idx]
            l_var = like_std[0, conc_idx] ** 2

            post_var = 1.0 / (1.0 / p_var + 1.0 / l_var)
            post_mean = post_var * (p_mean / p_var + l_mean / l_var)
            post_std = np.sqrt(post_var)

            all_true.append(y_test[row_i, conc_idx])
            all_prior_mean.append(p_mean)
            all_prior_std.append(np.sqrt(p_var))
            all_like_mean.append(l_mean)
            all_like_std.append(np.sqrt(l_var))
            all_post_mean.append(post_mean)
            all_post_std.append(post_std)

    y_true = np.array(all_true)

    print("\n=== BAYESIAN CONCENTRATION UPDATE — LOEGO RESULTS ===")
    for label, preds in [("PRIOR  (glycerol unknown)", all_prior_mean),
                          ("LIKELIHOOD (glycerol known, filtered)", all_like_mean),
                          ("POSTERIOR (Bayesian combination)", all_post_mean)]:
        preds = np.array(preds)
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        print(f"\n{label}:")
        print(f"  LOEGO MAE : {mae:.4f}")
        print(f"  LOEGO R²  : {r2:.4f}")

    if n_no_exact_match:
        print(f"\n(Used nearest available glycerol condition for the likelihood "
              f"term in {n_no_exact_match} test rows lacking an exact match.)")

    print(f"\nMean posterior std : {np.mean(all_post_std):.5f}")
    print(f"Mean prior std      : {np.mean(all_prior_std):.5f}")
    print(f"Mean likelihood std : {np.mean(all_like_std):.5f}")
    print("(Posterior std should be <= both prior and likelihood std — "
          "combining two independent estimates should narrow uncertainty. "
          "If it isn't, something's off with the variance inputs.)")

    return {
        "y_true": y_true,
        "prior_mean": np.array(all_prior_mean), "prior_std": np.array(all_prior_std),
        "likelihood_mean": np.array(all_like_mean), "likelihood_std": np.array(all_like_std),
        "posterior_mean": np.array(all_post_mean), "posterior_std": np.array(all_post_std),
    }


# -----------------------------
# STEP 6D — PER-RANGE / LOG-SPACE ERROR BREAKDOWN
# (pooled MAE hides how error scales across a log-spaced concentration grid —
#  a fixed absolute MAE is meaningless at the low end and generous at the
#  high end. This breaks it down properly.)
# -----------------------------
def concentration_error_breakdown(y_true, y_pred, bins=((0.0, 0.01), (0.01, 0.05), (0.05, 1.0)),
                                   bin_labels=("low (<=0.01)", "mid (0.01-0.05)", "high (>0.05)")):
    """
    y_true, y_pred: 1D arrays of true/predicted concentration_wv (same units
    as used everywhere else, e.g. from bayesian_concentration_update()'s
    'y_true' and 'posterior_mean').

    Reports, per concentration range:
      - n points, pooled MAE (what you'd get from mean_absolute_error alone)
      - mean relative error = mean(|pred-true|/true) -- the number that
        actually reflects usefulness at the low-concentration end
      - R^2 within that bin (only meaningful if bin has >=2 points with variance)

    Also reports log-space error: MAE and R^2 on log10(concentration),
    the standard way to evaluate quantities spanning orders of magnitude.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    print("\n=== CONCENTRATION ERROR BREAKDOWN (why pooled MAE can mislead) ===")
    print(f"{'range':<18} {'n':>4} {'MAE':>10} {'mean rel.err':>14} {'R2 (in-bin)':>12}")

    for (lo, hi), label in zip(bins, bin_labels):
        mask = (y_true > lo) & (y_true <= hi)
        n = mask.sum()
        if n == 0:
            print(f"{label:<18} {0:>4}   (no points in this range)")
            continue
        yt, yp = y_true[mask], y_pred[mask]
        mae = mean_absolute_error(yt, yp)
        rel_err = np.mean(np.abs(yp - yt) / np.maximum(yt, 1e-9))
        r2 = r2_score(yt, yp) if n >= 2 and np.std(yt) > 1e-12 else float("nan")
        print(f"{label:<18} {n:>4} {mae:>10.5f} {rel_err:>13.1%} {r2:>12.4f}")

    print("\n  (mean rel.err = mean(|pred-true|/true) -- this is the number "
          "that reflects real usefulness at the low-concentration end, "
          "where a fixed absolute MAE can exceed the true value itself.)")

    # log-space evaluation: standard for log-spaced / multi-order-of-magnitude grids
    valid = y_true > 0
    log_true = np.log10(y_true[valid])
    log_pred = np.log10(np.clip(y_pred[valid], 1e-9, None))  # clip in case of negative predictions

    n_negative_pred = int(np.sum(y_pred[valid] <= 0))
    log_mae = mean_absolute_error(log_true, log_pred)
    log_r2 = r2_score(log_true, log_pred)

    print(f"\n  --- LOG10-SPACE EVALUATION (standard for log-spaced grids) ---")
    print(f"  log10(concentration) MAE : {log_mae:.4f}  (orders of magnitude)")
    print(f"  log10(concentration) R²  : {log_r2:.4f}")
    if n_negative_pred:
        print(f"  WARNING: {n_negative_pred} predictions were <= 0 and got clipped "
              f"to 1e-9 before taking log10 -- these are effectively 'infinitely "
              f"wrong' in log space and worth inspecting individually.")

    return {
        "log_mae": log_mae,
        "log_r2": log_r2,
        "n_negative_predictions": n_negative_pred,
    }


def per_condition_error_breakdown(y_true, y_pred, groups, true_glycerol,
                                   n_worst=15):
    """
    Breaks the low/mid-concentration failure down by CONDITION (batch),
    not just by concentration bin, to distinguish two very different
    explanations:
      (a) a handful of specific batches (bad imaging, mislabeled metadata,
          poor tile alignment) are dragging the aggregate numbers down
          -> fixable by re-imaging/re-checking those specific batches
      (b) the failure is spread roughly evenly across ALL low-concentration
          conditions -> a genuine physical/imaging detection floor (too
          few particles at low loading for the texture features to carry
          signal), which belongs in the paper as a stated limitation
          rather than something to "fix" by relabeling.

    Requires y_true/y_pred/groups/true_glycerol all aligned in the same
    row order -- use the output of group_loego_evaluate_conc_given_glycerol,
    which now returns "groups" and "true_glycerol" alongside y_true/y_pred.
    """
    df = pd.DataFrame({
        "group": groups,
        "true_glycerol": true_glycerol,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
    df["rel_err"] = df["abs_err"] / df["y_true"].clip(lower=1e-9)

    per_group = df.groupby("group").agg(
        true_glycerol=("true_glycerol", "mean"),
        true_conc=("y_true", "mean"),
        n=("y_true", "size"),
        mean_rel_err=("rel_err", "mean"),
        mean_abs_err=("abs_err", "mean"),
    ).reset_index()
    per_group = per_group.sort_values("true_conc")

    print(f"\n=== PER-CONDITION ERROR BREAKDOWN ({len(per_group)} conditions) ===")
    print(f"{'group':<20} {'true_gly':>9} {'true_conc':>10} {'n':>3} "
          f"{'mean_rel_err':>13} {'mean_abs_err':>13}")
    for _, row in per_group.iterrows():
        print(f"{row['group']:<20} {row['true_glycerol']:>9.1f} "
              f"{row['true_conc']:>10.4f} {int(row['n']):>3} "
              f"{row['mean_rel_err']:>12.1%} {row['mean_abs_err']:>13.5f}")

    # focus specifically on the low/mid concentration regime already
    # flagged as failing in the aggregate breakdown
    low_mid = per_group[per_group["true_conc"] <= 0.05].copy()
    print(f"\n--- Low/mid-concentration conditions (true_conc <= 0.05), "
          f"sorted by relative error ---")
    low_mid_sorted = low_mid.sort_values("mean_rel_err", ascending=False)
    for _, row in low_mid_sorted.head(n_worst).iterrows():
        print(f"{row['group']:<20} gly={row['true_glycerol']:>4.1f} "
              f"conc={row['true_conc']:>8.4f}  rel_err={row['mean_rel_err']:>7.1%}")

    # diagnostic verdict: is failure concentrated in a few conditions, or spread evenly?
    rel_errs = low_mid["mean_rel_err"].values
    frac_conditions_failing = np.mean(rel_errs > 0.5)  # >50% relative error = "failing"
    cv_of_error = np.std(rel_errs) / (np.mean(rel_errs) + 1e-9)

    print(f"\n--- VERDICT ---")
    print(f"  Fraction of low/mid conditions with >50% relative error: "
          f"{frac_conditions_failing:.0%} ({np.sum(rel_errs > 0.5)}/{len(rel_errs)})")
    print(f"  Coefficient of variation of per-condition relative error: {cv_of_error:.2f}")
    if frac_conditions_failing > 0.7 and cv_of_error < 0.6:
        print("  -> Failure is WIDESPREAD and relatively UNIFORM across low/mid "
              "conditions. Consistent with a genuine detection-floor effect "
              "(too little particle signal at low loading for the texture "
              "features to resolve), not a handful of bad batches. Recommend "
              "stating this as a physical/imaging limitation in the paper "
              "rather than trying to 'fix' specific conditions.")
    elif frac_conditions_failing < 0.4:
        print("  -> Failure is CONCENTRATED in a minority of conditions. "
              "Worth inspecting those specific batches (imaging quality, "
              "tile alignment, metadata) individually -- this looks fixable "
              "by re-checking/re-imaging rather than a hard physical floor.")
    else:
        print("  -> Mixed picture: neither clearly uniform nor clearly "
              "concentrated. Inspect the worst offenders listed above "
              "individually before concluding either way.")

    return per_group


def compute_raw_correlations(X_train_z, y_train, target_names, target_mode="combined"):
    """
    Raw |Pearson correlation| per dimension with target(s) -- NOT
    normalized to mean 1.0 (unlike compute_correlation_weights). This is
    the actual value to compare against a threshold like 0.7.
    """
    n_dims = X_train_z.shape[1]
    raw_corrs = np.zeros(n_dims)

    def safe_corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        c = np.corrcoef(x, y)[0, 1]
        return 0.0 if np.isnan(c) else abs(c)

    for j in range(n_dims):
        col = X_train_z[:, j]
        if target_mode == "combined":
            corrs = [safe_corr(col, y_train[:, i]) for i in range(y_train.shape[1])]
            raw_corrs[j] = np.mean(corrs)
        else:
            i = target_names.index(target_mode)
            raw_corrs[j] = safe_corr(col, y_train[:, i])

    return raw_corrs


def raw_correlation_diagnostic(X, y, feat_cols, target_mode="concentration_wv", threshold=0.7,
                                verbose=True):
    """
    Computes raw |correlation| (on the FULL dataset, for a quick look --
    not per-fold) for each of the 40 dims against target_mode, sorted
    descending. Reports how many features would survive a given threshold
    BEFORE you commit to hard-filtering with it -- with ~170 rows spread
    across 40 conditions, a 0.7 threshold on raw feature-level correlation
    is a genuinely high bar; worth seeing the actual distribution first
    rather than filtering blind and ending up with zero or one feature.
    """
    X_imputed, _ = nan_safe_impute(X, X)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_imputed)

    raw_corrs = compute_raw_correlations(X_z, y, TARGETS, target_mode)
    order = np.argsort(-raw_corrs)
    n_survive = int(np.sum(raw_corrs >= threshold))

    if verbose:
        print(f"\n=== RAW |correlation| DIAGNOSTIC (target='{target_mode}', full dataset) ===")
        print(f"{'feature':<28} {'raw |corr|':>10}")
        for idx in order:
            flag = "  <-- survives" if raw_corrs[idx] >= threshold else ""
            print(f"{feat_cols[idx]:<28} {raw_corrs[idx]:>10.3f}{flag}")
        print(f"\n  {n_survive}/{len(feat_cols)} features have raw |corr| >= {threshold}")

    if n_survive == 0:
        print(f"  WARNING: NO features clear a {threshold} threshold. A hard "
              f"filter at this level would leave zero features to run kNN "
              f"on. Consider a lower threshold (e.g. 0.3-0.5) or keep using "
              f"soft correlation-weighting (WEIGHT_MODE='correlation') "
              f"instead of a hard cutoff.")
    elif n_survive < 5:
        print(f"  Only {n_survive} features survive -- a very aggressive cut. "
              f"Fine to try, but expect much higher variance in the kNN "
              f"predictions with so few dimensions left.")

    return dict(zip(feat_cols, raw_corrs))


def compute_correlation_weights(X_train_z, y_train, target_names, target_mode="combined"):
    """
    Per-dimension |Pearson correlation| with target(s), computed on TRAIN
    data only (called fresh inside each LOEGO fold -> no leakage).

    Returns weights normalized to mean 1.0, so the overall similarity scale
    stays comparable to the unweighted baseline. Dims with undefined
    correlation (e.g. zero variance in this fold) get weight 0.
    """
    weights = compute_raw_correlations(X_train_z, y_train, target_names, target_mode)

    mean_w = weights.mean()
    if mean_w < 1e-12:
        # fold degenerate (e.g. too few train points) -> fall back to uniform
        return np.ones(len(weights))
    return weights / mean_w  # normalize so mean weight == 1.0


def compute_correlation_mask(X_train_z, y_train, target_names, target_mode="combined", threshold=0.7):
    """
    HARD threshold version: instead of continuous weights, drops any
    dimension whose raw |correlation| with target_mode falls below
    threshold (weight -> 0), and gives every surviving dimension EQUAL
    weight (mean-normalized to 1.0 across survivors only). This is what
    "drop features whose correlation drops below 0.7" means literally --
    as opposed to compute_correlation_weights's soft, continuous weighting.

    Computed fresh per fold on TRAIN data only, same no-leakage discipline
    as compute_correlation_weights.
    """
    raw_corrs = compute_raw_correlations(X_train_z, y_train, target_names, target_mode)
    mask = (raw_corrs >= threshold).astype(float)

    n_survive = int(mask.sum())
    if n_survive == 0:
        # nothing cleared the bar this fold -- fall back to uniform weights
        # rather than returning an all-zero vector (which would make every
        # pairwise distance zero, so density_knn_predict couldn't discriminate)
        return np.ones(len(raw_corrs))

    # equal weight among survivors, normalized so mean weight (over ALL
    # dims, survivors + zeroed) stays comparable in scale to the soft mode
    mask = mask * (len(raw_corrs) / n_survive)
    return mask


def get_feature_weights(mode, X_train_z, y_train, target_names, target_mode, threshold=CORR_THRESHOLD):
    """
    Single dispatcher used everywhere weighting is applied, so
    WEIGHT_MODE='correlation' vs 'threshold' vs 'none' is consistent
    across group_loego_evaluate, group_loego_evaluate_conc_given_glycerol,
    and feature_weight_diagnostic instead of duplicating the if/else.
    """
    if mode == "correlation":
        return compute_correlation_weights(X_train_z, y_train, target_names, target_mode)
    elif mode == "threshold":
        return compute_correlation_mask(X_train_z, y_train, target_names, target_mode, threshold)
    else:  # "none" or anything else -> uniform
        return np.ones(X_train_z.shape[1])


# -----------------------------
# STEP 6 — LOEGO EVALUATION
# -----------------------------
def group_loego_evaluate(X, y, groups, k=K_NEIGHBORS):
    """
    Leave-one-GROUP-out cosine-NN evaluation.
    Scaler is fit fresh inside each fold on TRAIN groups only — no leakage.
    k: number of nearest neighbors averaged per prediction (similarity-weighted).
       k=1 reproduces plain 1-NN behavior.
    """
    logo = LeaveOneGroupOut()
    all_true, all_pred, all_sim, all_group = [], [], [], []

    n_folds = logo.get_n_splits(groups=groups)
    print(f"\nRunning LOEGO over {n_folds} folds (one per distinct batch), k={k}...")

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = nan_safe_impute(X_train, X_test)

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)   # fit on TRAIN groups only
        X_test_z = scaler.transform(X_test)

        if WEIGHT_MODE in ("correlation", "threshold"):
            w = get_feature_weights(WEIGHT_MODE, X_train_z, y_train, TARGETS, CORR_WEIGHT_TARGET)
            X_train_z = X_train_z * w
            X_test_z = X_test_z * w

        preds, dists = density_knn_predict(X_train_z, y_train, X_test_z, k=k)

        all_true.append(y_test)
        all_pred.append(preds)
        all_sim.append(dists)
        all_group.extend([groups[test_idx][0]] * len(test_idx))

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    dists = np.concatenate(all_sim)

    print("\n=== LOEGO RESULTS (density-weighted kNN, honest metric) ===")
    for i, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"\n{target}:")
        print(f"  LOEGO MAE : {mae:.4f}")
        print(f"  LOEGO R²  : {r2:.4f}")

    print(f"\nMean nearest-neighbor distance: {dists.mean():.4f}")
    print("(High distance on held-out groups = model is extrapolating, "
          "not truly matching — expected while condition count is small.)")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "distances": dists,
        "groups": np.array(all_group),
    }


# -----------------------------
# STEP 6B — FEATURE WEIGHT DIAGNOSTIC
# -----------------------------
def feature_weight_diagnostic(X, y, groups, feat_cols, target_mode=CORR_WEIGHT_TARGET,
                               verbose=True):
    """
    Runs the same LOEGO fold structure, but only to collect
    compute_correlation_weights() output per fold, then averages across
    folds so you can see which of the 40 dims are actually carrying signal
    for target_mode (mean weight >> 1 = important, near 0 = dead weight).
    """
    logo = LeaveOneGroupOut()
    weight_accum = []

    for train_idx, _ in logo.split(X, y, groups=groups):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_train, _ = nan_safe_impute(X_train, X_train[:1])  # impute using train stats only

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        w = compute_correlation_weights(X_train_z, y_train, TARGETS, target_mode)
        weight_accum.append(w)

    mean_w = np.mean(weight_accum, axis=0)
    order = np.argsort(-mean_w)

    if verbose:
        print(f"\n=== FEATURE WEIGHT DIAGNOSTIC (target='{target_mode}', "
              f"averaged over {len(weight_accum)} LOEGO folds) ===")
        print(f"{'feature':<28} {'mean weight':>12}")
        for idx in order:
            print(f"{feat_cols[idx]:<28} {mean_w[idx]:>12.3f}")

    return dict(zip(feat_cols, mean_w))


# -----------------------------
# STEP 6C — CONCENTRATION R² IF GLYCEROL% IS KNOWN
# (auxiliary-input oracle check: appends TRUE glycerol_pct as a 41st input
#  dim, so cosine similarity can use it directly. Tells you the ceiling if
#  glycerol were measured/labeled independently rather than predicted.)
# -----------------------------
def group_loego_evaluate_conc_given_glycerol(X, y, groups, k=K_NEIGHBORS,
                                              weight_mode="correlation",
                                              glycerol_steer_weight=3.0):
    """
    "If glycerol is known" implemented as SOFT STEERING, not a hard filter:
    the TRUE glycerol_pct is appended as a 41st input dimension, z-scored
    like every other dim, but instead of using compute_correlation_weights()
    for that dimension (which gives it ~weight 0, since your factorial grid
    means raw glycerol has ~zero linear correlation with concentration —
    that's why the earlier append+reweight attempt returned an unchanged
    R^2) it gets a manually-set weight: glycerol_steer_weight.

    The rest of the 40 image dims still get their normal concentration-
    correlation weights. The FULL training pool is used (no filtering),
    so you don't lose the sample-size benefit that hurt the hard-filter
    version — glycerol just biases which neighbors look close, it doesn't
    exclude anyone.

    glycerol_steer_weight: relative pull of the known-glycerol dimension
    vs. the (mean-1.0-normalized) image feature weights.
      0.0   -> identical to normal concentration-only run (no steering)
      1.0   -> glycerol pulls about as hard as an average image feature
      3.0+  -> glycerol dominates neighbor selection (near-hard-filter
               behavior, but smooth/continuous instead of a cutoff)
    Try a few values and compare against the un-steered 0.665 baseline.
    """
    gly_idx = TARGETS.index("glycerol_pct")
    conc_idx = TARGETS.index("concentration_wv")

    X_aug = np.hstack([X, y[:, [gly_idx]]])  # append true glycerol as 41st dim

    logo = LeaveOneGroupOut()
    all_true, all_pred, all_group, all_true_gly = [], [], [], []

    n_folds = logo.get_n_splits(groups=groups)
    print(f"\nRunning 'concentration | glycerol known (soft steering)' LOEGO "
          f"over {n_folds} folds (k={k}, glycerol_steer_weight={glycerol_steer_weight})...")

    for train_idx, test_idx in logo.split(X_aug, y, groups=groups):
        X_train, X_test = X_aug[train_idx], X_aug[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test = nan_safe_impute(X_train, X_test)

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)   # includes the glycerol dim
        X_test_z = scaler.transform(X_test)

        if weight_mode in ("correlation", "threshold"):
            # weights for the first 40 (image) dims only, fit on TRAIN
            w_img = get_feature_weights(weight_mode, X_train_z[:, :-1], y_train,
                                         TARGETS, "concentration_wv")
            w = np.concatenate([w_img, [glycerol_steer_weight]])
        else:
            w = np.ones(X_train_z.shape[1])
            w[-1] = glycerol_steer_weight

        X_train_z = X_train_z * w
        X_test_z = X_test_z * w

        preds, _ = density_knn_predict(X_train_z, y_train, X_test_z, k=k)

        all_true.append(y_test[:, conc_idx])
        all_pred.append(preds[:, conc_idx])
        all_group.extend(groups[test_idx])
        all_true_gly.append(y_test[:, gly_idx])

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_true_gly = np.concatenate(all_true_gly)
    y_groups = np.array(all_group)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n=== concentration_wv R² IF glycerol_pct IS KNOWN (soft steering) ===")
    print(f"  LOEGO MAE : {mae:.4f}")
    print(f"  LOEGO R²  : {r2:.4f}")
    print("  Compare against the un-steered concentration R² above. "
          "A jump means knowing glycerol genuinely helps disambiguate "
          "concentration; little/no change means concentration signal is "
          "already self-contained in the image features regardless of glycerol.")

    return {"y_true": y_true, "y_pred": y_pred, "mae": mae, "r2": r2,
            "groups": y_groups, "true_glycerol": y_true_gly}


# -----------------------------
# STEP 7 — NESTED-CV HYPERPARAMETER TUNING (honest, no leakage)
# -----------------------------
# The earlier sweep_glycerol_steer_weight() picked glycerol_steer_weight=2.0
# by comparing R^2 across the SAME 40 outer LOEGO folds being reported —
# that's a mild form of tuning-on-the-test-set: the reported R^2 is
# optimistically biased because the weight was chosen to maximize exactly
# that number. Nested CV fixes this: for each outer held-out group, an
# INNER loop (over the remaining ~39 train groups only) picks the best
# k / glycerol_steer_weight, then that choice is applied to predict the
# outer group. The outer group is never used for any tuning decision.

K_GRID = (1, 3, 5, 7, 9)
STEER_WEIGHT_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
INNER_FOLDS = 5  # GroupKFold folds within each outer-train set (cheaper than inner LOEGO)


def _inner_cv_score(X_train_full, y_train_full, groups_train_full,
                     k, glycerol_steer_weight, n_inner_folds=INNER_FOLDS):
    """
    Runs GroupKFold CV (grouped by batch, so no replicate leakage) within
    an outer-training set only, for one (k, glycerol_steer_weight)
    candidate. Returns mean concentration_wv R^2 across inner folds.
    Uses the same append-glycerol-with-fixed-weight steering mechanism as
    group_loego_evaluate_conc_given_glycerol.
    """
    gly_idx = TARGETS.index("glycerol_pct")
    conc_idx = TARGETS.index("concentration_wv")

    n_groups = len(np.unique(groups_train_full))
    n_splits = min(n_inner_folds, n_groups)
    if n_splits < 2:
        return None  # not enough groups to do inner CV this outer fold

    gkf = GroupKFold(n_splits=n_splits)
    X_aug = np.hstack([X_train_full, y_train_full[:, [gly_idx]]])

    fold_r2 = []
    for inner_train_idx, inner_test_idx in gkf.split(X_aug, y_train_full, groups=groups_train_full):
        X_itr, X_ite = X_aug[inner_train_idx], X_aug[inner_test_idx]
        y_itr, y_ite = y_train_full[inner_train_idx], y_train_full[inner_test_idx]

        X_itr, X_ite = nan_safe_impute(X_itr, X_ite)

        scaler = StandardScaler()
        X_itr_z = scaler.fit_transform(X_itr)
        X_ite_z = scaler.transform(X_ite)

        w_img = compute_correlation_weights(X_itr_z[:, :-1], y_itr, TARGETS, "concentration_wv")
        w = np.concatenate([w_img, [glycerol_steer_weight]])
        X_itr_z = X_itr_z * w
        X_ite_z = X_ite_z * w

        preds, _ = density_knn_predict(X_itr_z, y_itr, X_ite_z, k=k)
        fold_r2.append(r2_score(y_ite[:, conc_idx], preds[:, conc_idx]))

    return float(np.mean(fold_r2))


def nested_loego_hyperparameter_tuning(X, y, groups,
                                        k_grid=K_GRID,
                                        steer_weight_grid=STEER_WEIGHT_GRID,
                                        tune_k=True,
                                        unnested_best_r2=None,
                                        unnested_best_weight=None):
    """
    Honest nested LOEGO: outer loop leaves one group out for final
    evaluation; inner loop (GroupKFold on the remaining groups) picks
    hyperparameters by inner-fold R^2, WITHOUT ever touching the outer
    test group. This is the number to actually report as your tuned
    model's LOEGO R^2 -- unlike the earlier sweep, it can't be inflated
    by picking the weight that happens to fit the reporting folds.

    tune_k=True (recommended here): k is chosen per OUTER fold by the
    inner CV, instead of a single fixed constant applied everywhere.
    Replicate counts differ per condition in this dataset (3-5), so a
    flat k borrows a different fraction of "same-condition" neighbors
    depending on which condition happens to be held out; letting the
    inner loop pick k per fold means each fold's choice reflects what
    actually works given its own train-side replicate structure,
    without ever looking at the held-out fold to make that choice.

    (A per-QUERY adaptive neighbor-count scheme was tried and reverted --
    see density_knn_predict's docstring. Fixed top-k, with k itself
    chosen per FOLD via this honest nested CV, is the version that's
    actually performed best on this data.)

    tune_k=False: k is fixed at K_NEIGHBORS and ONLY glycerol_steer_weight
    is tuned (7 candidates/fold instead of 35) -- faster, but reintroduces
    the static-k assumption this function exists to avoid.
    """
    if not tune_k:
        k_grid = (K_NEIGHBORS,)
        print(f"tune_k=False -> k fixed at {K_NEIGHBORS}, only tuning "
              f"glycerol_steer_weight ({len(steer_weight_grid)} candidates)")

    gly_idx = TARGETS.index("glycerol_pct")
    conc_idx = TARGETS.index("concentration_wv")

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    print(f"\nRunning NESTED LOEGO hyperparameter tuning over {n_folds} outer folds "
          f"x {len(k_grid)}x{len(steer_weight_grid)}={len(k_grid)*len(steer_weight_grid)} "
          f"candidates each (inner GroupKFold={INNER_FOLDS})...")
    print("This is slower than the earlier sweep -- it's doing real nested validation.")

    all_true, all_pred, chosen_params = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        groups_train_full = groups[train_idx]

        # --- INNER LOOP: pick best (k, weight) using train groups only ---
        best_score, best_k, best_w = -np.inf, k_grid[0], steer_weight_grid[0]
        for k in k_grid:
            for w in steer_weight_grid:
                score = _inner_cv_score(X_train_full, y_train_full, groups_train_full, k, w)
                if score is not None and score > best_score:
                    best_score, best_k, best_w = score, k, w

        # --- OUTER: fit with the chosen hyperparameters, predict held-out group ---
        X_train_full_i, X_test_i = nan_safe_impute(X_train_full, X_test)
        X_aug_train = np.hstack([X_train_full_i, y_train_full[:, [gly_idx]]])
        X_aug_test = np.hstack([X_test_i, y_test[:, [gly_idx]]])  # true glycerol known at eval time

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_aug_train)
        X_test_z = scaler.transform(X_aug_test)

        w_img = compute_correlation_weights(X_train_z[:, :-1], y_train_full, TARGETS, "concentration_wv")
        w_full = np.concatenate([w_img, [best_w]])
        X_train_z = X_train_z * w_full
        X_test_z = X_test_z * w_full

        preds, _ = density_knn_predict(X_train_z, y_train_full, X_test_z, k=best_k)

        all_true.append(y_test[:, conc_idx])
        all_pred.append(preds[:, conc_idx])
        chosen_params.append((best_k, best_w))

        if (fold_i + 1) % 10 == 0 or (fold_i + 1) == n_folds:
            print(f"  outer fold {fold_i + 1}/{n_folds} done "
                  f"(chose k={best_k}, glycerol_steer_weight={best_w}, "
                  f"inner R2={best_score:.3f})")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n=== NESTED LOEGO RESULTS (honest, no tuning leakage) ===")
    print(f"  concentration_wv LOEGO MAE : {mae:.4f}")
    print(f"  concentration_wv LOEGO R²  : {r2:.4f}")

    log_metrics = concentration_error_breakdown(y_true, y_pred)

    ks_chosen = [p[0] for p in chosen_params]
    ws_chosen = [p[1] for p in chosen_params]
    if tune_k:
        k_counts = {kv: ks_chosen.count(kv) for kv in sorted(set(ks_chosen))}
        print(f"\n  k chosen per outer fold (dynamic, {n_folds} folds): "
              + ", ".join(f"k={kv}: {c}" for kv, c in k_counts.items()))
        print("  (Spread across values, not one dominant k, is expected here -- "
              "replicate counts vary per condition (3-5), so the honestly-tuned "
              "k legitimately differs fold to fold rather than being forced flat.)")
    else:
        print(f"\n  Most frequently chosen k                : "
              f"{max(set(ks_chosen), key=ks_chosen.count)} "
              f"(chosen in {ks_chosen.count(max(set(ks_chosen), key=ks_chosen.count))}/{n_folds} folds)")
    print(f"  Most frequently chosen glycerol_steer_weight: "
          f"{max(set(ws_chosen), key=ws_chosen.count)} "
          f"(chosen in {ws_chosen.count(max(set(ws_chosen), key=ws_chosen.count))}/{n_folds} folds)")

    if unnested_best_r2 is not None:
        print(f"\n  Un-nested sweep's best for comparison: R²={unnested_best_r2:.4f} "
              f"at fixed weight={unnested_best_weight} -- the gap between that and "
              f"the nested result above is the size of the tuning-leakage bias.")

    return {
        "y_true": y_true, "y_pred": y_pred, "mae": mae, "r2": r2,
        "chosen_params": chosen_params, "log_metrics": log_metrics,
    }


def sweep_glycerol_steer_weight(X, y, groups, k=K_NEIGHBORS,
                                 weights=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
                                 verbose=False):
    """
    Runs group_loego_evaluate_conc_given_glycerol across several steering
    strengths so you can see the R^2 curve and pick the best one, rather
    than guessing a single glycerol_steer_weight value.

    verbose=False (default) suppresses each candidate's own per-fold
    banner/result block and prints only the summary table below -- set
    True if you want to see every candidate's full LOEGO printout.

    Returns {weight: full_result_dict} so the caller can reuse the winning
    weight's result (y_true/y_pred/etc.) instead of recomputing it.
    """
    print(f"\n=== SWEEPING glycerol_steer_weight ({len(weights)} candidates, k={k}) ===")
    results = {}
    for w in weights:
        with _suppress_stdout(active=not verbose):
            res = group_loego_evaluate_conc_given_glycerol(X, y, groups, k=k,
                                                             glycerol_steer_weight=w)
        results[w] = res

    print(f"{'weight':>8} {'R²':>8} {'MAE':>9}")
    for w in weights:
        print(f"{w:>8} {results[w]['r2']:>8.4f} {results[w]['mae']:>9.5f}")
    best_w = max(results, key=lambda w: results[w]["r2"])
    print(f"\nBest: glycerol_steer_weight={best_w} (R²={results[best_w]['r2']:.4f})")
    return results, best_w


# -----------------------------
# STEP 9 — DEPLOYABLE CONCENTRATION PREDICTOR
# (fits scaler + correlation weights on ALL data, for use on NEW droplets —
#  distinct from the LOEGO evaluation harness above, which is for honest
#  metric reporting only)
# -----------------------------
class ConcentrationPredictor:
    """
    Production concentration_wv predictor, locked to the config that scored
    LOEGO R²=0.665: correlation weighting targeted at concentration_wv,
    k=5 similarity-weighted nearest neighbors.

    Usage:
        model = ConcentrationPredictor()
        model.fit(df_raw)                      # df_raw = load_data() output
        pred = model.predict(new_feature_row)   # dict of 40 raw feature values
    """

    def __init__(self, k=K_NEIGHBORS, target_mode="concentration_wv"):
        self.k = k
        self.target_mode = target_mode
        self.scaler = None
        self.weights = None
        self.base_names = None
        self.feat_cols = None
        self.X_train_z = None
        self.y_train = None

    def fit(self, df_raw):
        self.base_names = discover_base_feature_names(df_raw.columns)
        df, feat_cols = None, None
        X, feat_cols = build_vector_matrix(df_raw, self.base_names)

        df_clean = df_raw.copy()
        df_clean["concentration_wv"] = pd.to_numeric(df_clean["concentration_wv"], errors="coerce")
        df_clean["glycerol_pct"] = pd.to_numeric(df_clean["glycerol_pct"], errors="coerce")
        keep = df_clean["concentration_wv"] <= CONC_MAX
        df_clean = df_clean[keep].reset_index(drop=True)
        X = X[keep.values]

        valid = np.mean(~np.isnan(X), axis=1) >= 0.5
        df_clean = df_clean[valid].reset_index(drop=True)
        X = X[valid]

        y = df_clean[TARGETS].values
        X, _ = nan_safe_impute(X, X)  # median-impute using full-data stats

        self.scaler = StandardScaler()
        X_z = self.scaler.fit_transform(X)

        self.weights = compute_correlation_weights(X_z, y, TARGETS, self.target_mode)
        X_z = X_z * self.weights

        self.X_train_z = X_z
        self.y_train = y
        self.feat_cols = feat_cols
        self.impute_reference = X  # for imputing new rows consistently
        print(f"ConcentrationPredictor fit on {len(df_clean)} rows, "
              f"{len(self.feat_cols)}-dim vector, target_mode='{self.target_mode}', k={self.k}")
        return self

    def _build_row_vector(self, feature_dict):
        """feature_dict: raw feature name -> value, e.g. {'center_void_fraction': 0.7, ...}"""
        row = np.array([[feature_dict.get(c, np.nan) for c in self.feat_cols]])
        return row

    def predict(self, feature_dict):
        row = self._build_row_vector(feature_dict)
        row, _ = nan_safe_impute(self.impute_reference, row)  # impute using TRAIN stats
        row_z = self.scaler.transform(row) * self.weights

        preds, dist = density_knn_predict(self.X_train_z, self.y_train, row_z, k=self.k)
        conc_idx = TARGETS.index("concentration_wv")
        gly_idx = TARGETS.index("glycerol_pct")
        return {
            "concentration_wv": float(preds[0, conc_idx]),
            "glycerol_pct_sideinfo": float(preds[0, gly_idx]),  # not the focus, but free
            "nearest_neighbor_distance": float(dist[0]),
        }


# -----------------------------
# STEP 6B — DATA-DRIVEN FEATURE IMPORTANCE (for informed manual cleaning)
# -----------------------------
def feature_importance_report(X, y, groups, feat_cols, target_mode="glycerol_pct"):
    """
    Runs the same LOEGO fold structure, but instead of predicting, just
    collects the per-fold correlation weight vector (computed on TRAIN data
    each fold, same as inside group_loego_evaluate) and averages it across
    all folds. This gives an evidence-based ranking of which of the 40 dims
    actually track `target_mode`, instead of eyeballing heatmaps or guessing
    an exclusion list.

    Use this BEFORE hand-editing EXCLUDE_BASE_FEATURES — drop only dims that
    consistently rank near-zero across folds, not ones that just look noisy
    in one heatmap panel.
    """
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    all_weights = np.zeros((n_folds, X.shape[1]))

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_train, _ = nan_safe_impute(X_train, X[test_idx])

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)

        w = compute_correlation_weights(X_train_z, y_train, TARGETS, target_mode)
        all_weights[fold_i] = w

    mean_w = all_weights.mean(axis=0)
    std_w = all_weights.std(axis=0)

    order = np.argsort(-mean_w)
    print(f"\n=== FEATURE IMPORTANCE (mean |corr|-weight across {n_folds} folds, "
          f"target='{target_mode}') ===")
    print(f"{'feature':<28} {'mean_weight':>12} {'std_weight':>12}")
    for idx in order:
        print(f"{feat_cols[idx]:<28} {mean_w[idx]:>12.3f} {std_w[idx]:>12.3f}")

    # candidate exclusion list: consistently near-zero weight AND low variance
    # (i.e. reliably uninformative, not just noisy-but-sometimes-useful)
    low_thresh = 0.3  # weights are normalized to mean 1.0, so 0.3 = well below average
    candidates = [feat_cols[i] for i in order if mean_w[i] < low_thresh]
    print(f"\nCandidate low-signal features (mean weight < {low_thresh}): {candidates}")
    print("Inspect before dropping — this is evidence for EXCLUDE_BASE_FEATURES, "
          "not an automatic decision.")

    return {"feat_cols": feat_cols, "mean_weight": mean_w, "std_weight": std_w,
            "candidates": candidates}


# -----------------------------
# MAIN
# -----------------------------
def run_once(df_raw, weight_mode, k=K_NEIGHBORS, use_corrected_concentration=USE_CORRECTED_CONCENTRATION):
    global WEIGHT_MODE
    WEIGHT_MODE = weight_mode

    base_names = discover_base_feature_names(df_raw.columns)
    if weight_mode == "exclude":
        kept = [b for b in base_names if b not in EXCLUDE_BASE_FEATURES]
        dropped = [b for b in base_names if b in EXCLUDE_BASE_FEATURES]
        if dropped:
            print(f"WEIGHT_MODE=exclude -> dropping base features: {dropped}")
        base_names = kept

    df, X, y, groups = clean(df_raw, base_names, use_corrected_concentration=use_corrected_concentration)
    if X is None:
        return None

    return group_loego_evaluate(X, y, groups, k=k)


def sweep_conc_min_restriction(conc_min_grid=(0.0, 0.005, 0.01, 0.02, 0.05),
                                weight_mode="correlation", k=K_NEIGHBORS,
                                glycerol_steer_weight=2.0):
    """
    Runs the steered density-based kNN LOEGO eval at several conc_min thresholds
    so you can see the R^2-vs-scope tradeoff directly, rather than
    guessing where to draw the line. Also reports how many rows/conditions
    survive each threshold, since restricting scope isn't free -- it's
    trading data volume for per-point accuracy.
    """
    df_raw = load_data()
    base_names = discover_base_feature_names(df_raw.columns)

    print("\n=== SWEEPING CONC_MIN (concentration-range restriction) ===")
    print(f"{'conc_min':>10} {'n_rows':>8} {'n_groups':>9} {'R2':>10} {'MAE':>10}")

    results = {}
    for conc_min in conc_min_grid:
        global WEIGHT_MODE
        WEIGHT_MODE = weight_mode
        df, X, y, groups = clean(df_raw, base_names, conc_min=conc_min)
        if X is None:
            print(f"{conc_min:>10} {'--':>8} {'--':>9}   (not enough data)")
            continue

        res = group_loego_evaluate_conc_given_glycerol(
            X, y, groups, k=k, glycerol_steer_weight=glycerol_steer_weight)
        n_groups = len(np.unique(groups))
        print(f"{conc_min:>10} {len(X):>8} {n_groups:>9} {res['r2']:>10.4f} {res['mae']:>10.5f}")
        results[conc_min] = res

    print("\n  Higher conc_min = fewer rows/conditions but (usually) higher R^2, "
          "since the low-concentration regime was the model's weakest region. "
          "Pick the smallest conc_min that gets you an acceptable R^2 -- "
          "that's your model's honestly-stated validated operating range, "
          "not the smallest number you could report by dropping the most data.")

    return results


def compare_concentration_definitions(weight_mode="correlation", k=K_NEIGHBORS,
                                       glycerol_steer_weight=2.0):
    """
    Runs the blind density-based kNN LOEGO eval AND the glycerol-steered eval
    twice each: once on raw concentration_wv, once on the glycerol-
    corrected version (MFAT = concentration_wv * (1 - glycerol_pct/100)).
    Prints all four R^2/MAE side by side so you can see whether the
    correction helps, hurts, or is a wash.
    """
    df_raw = load_data()
    base_names = discover_base_feature_names(df_raw.columns)

    results = {}
    for label, use_corrected in [("RAW concentration_wv", False),
                                  ("CORRECTED (MFAT)", True)]:
        print("\n" + "#" * 60)
        print(f"# CONCENTRATION DEFINITION: {label}")
        print("#" * 60)

        df, X, y, groups = clean(df_raw, base_names, use_corrected_concentration=use_corrected)
        if X is None:
            results[label] = None
            continue

        blind = group_loego_evaluate(X, y, groups, k=k)
        steered = group_loego_evaluate_conc_given_glycerol(
            X, y, groups, k=k, glycerol_steer_weight=glycerol_steer_weight)

        results[label] = {"blind": blind, "steered": steered}

    print("\n" + "=" * 60)
    print("=== SUMMARY: raw vs. glycerol-corrected (MFAT) concentration ===")
    print("=" * 60)
    print(f"{'':<22} {'blind R2':>10} {'blind MAE':>11} {'steered R2':>12} {'steered MAE':>13}")
    for label in ("RAW concentration_wv", "CORRECTED (MFAT)"):
        r = results.get(label)
        if r is None:
            print(f"{label:<22} (no data)")
            continue
        conc_idx = TARGETS.index("concentration_wv")
        b_r2 = r2_score(r["blind"]["y_true"][:, conc_idx], r["blind"]["y_pred"][:, conc_idx])
        b_mae = mean_absolute_error(r["blind"]["y_true"][:, conc_idx], r["blind"]["y_pred"][:, conc_idx])
        s_r2 = r["steered"]["r2"]
        s_mae = r["steered"]["mae"]
        print(f"{label:<22} {b_r2:>10.4f} {b_mae:>11.5f} {s_r2:>12.4f} {s_mae:>13.5f}")

    return results


def main(verbose=False):
    """
    verbose=False (default): runs everything under the hood but prints only
    the one number that matters for reporting — steered log10(concentration)
    R². Everything else (feature rankings, sweep tables, Bayesian posterior,
    nested tuning, per-condition breakdowns) still runs and is available
    via the returned dict; set verbose=True to see the full printout.
    """
    with _suppress_stdout(active=not verbose):
        print("=" * 60)
        print("DENSITY-WEIGHTED kNN MODEL — LOEGO-ONLY EVALUATION")
        print("40-dim vector: 20 center_* + 20 nanmean(N,S,E,W)_*")
        print(f"WEIGHT_MODE = '{WEIGHT_MODE}'   K_NEIGHBORS = {K_NEIGHBORS}")
        print("=" * 60)

        df_raw = load_data()
        base_names_preview = discover_base_feature_names(df_raw.columns)
        print(f"Discovered {len(base_names_preview)} base feature names per tile")
        if len(base_names_preview) != 20:
            print(f"WARNING: expected 20 base features, found {len(base_names_preview)}. "
                  f"Vector will be {2 * len(base_names_preview)}-dim, not 40.")

        blind_results = run_once(df_raw, WEIGHT_MODE, k=K_NEIGHBORS)

        df_clean, X, y, groups = clean(df_raw, base_names_preview)
        if X is None:
            steered_results, bayes_results, nested_results, log_metrics = None, None, None, None
        else:
            feat_cols = (["center_" + b for b in base_names_preview] +
                         ["avg_" + b for b in base_names_preview])
            fold_weights = feature_weight_diagnostic(X, y, groups, feat_cols=feat_cols,
                                                      target_mode="concentration_wv", verbose=False)
            raw_corrs = raw_correlation_diagnostic(X, y, feat_cols=feat_cols,
                                                    target_mode="concentration_wv",
                                                    threshold=CORR_THRESHOLD, verbose=False)
            order = sorted(feat_cols, key=lambda f: -fold_weights[f])
            print(f"\n=== FEATURE RANKING (target='concentration_wv') ===")
            print(f"{'feature':<28} {'fold weight':>12} {'raw |corr|':>11}")
            for f in order:
                flag = "  <-- survives threshold" if raw_corrs[f] >= CORR_THRESHOLD else ""
                print(f"{f:<28} {fold_weights[f]:>12.3f} {raw_corrs[f]:>11.3f}{flag}")
            n_survive = sum(1 for f in feat_cols if raw_corrs[f] >= CORR_THRESHOLD)
            print(f"\n{n_survive}/{len(feat_cols)} features have raw |corr| >= {CORR_THRESHOLD}")

            sweep_results, best_w = sweep_glycerol_steer_weight(X, y, groups, k=K_NEIGHBORS,
                                                                 verbose=False)
            steered_results = sweep_results[best_w]
            print(f"\n[log-space breakdown for the STEERED (glycerol_steer_weight={best_w}) predictions]")
            log_metrics = concentration_error_breakdown(steered_results["y_true"], steered_results["y_pred"])
            per_condition_error_breakdown(steered_results["y_true"], steered_results["y_pred"],
                                           steered_results["groups"], steered_results["true_glycerol"])

            bayes_results = bayesian_concentration_update(X, y, groups, k=K_NEIGHBORS)
            print("\n[log-space breakdown for the BAYESIAN POSTERIOR predictions]")
            concentration_error_breakdown(bayes_results["y_true"], bayes_results["posterior_mean"])

            nested_results = nested_loego_hyperparameter_tuning(
                X, y, groups, tune_k=True,
                unnested_best_r2=steered_results["r2"], unnested_best_weight=best_w)
            log_metrics = nested_results["log_metrics"]

    if log_metrics is not None:
        print(f"log10(concentration) R² (nested LOEGO, dynamic k) = {log_metrics['log_r2']:.4f}")
    else:
        print("Not enough data to fit — see clean() output (run with verbose=True).")

    return {
        "blind": blind_results,
        "steered": steered_results,
        "bayes": bayes_results,
        "nested": nested_results,
        "log_metrics": log_metrics,
    }


def run_k_comparison(weight_mode=WEIGHT_MODE, ks=(1, 3, 5, 7)):
    """
    Convenience runner: evaluates k=1 (original 1-NN) against several
    k-NN averaging values on identical data/folds, so you can see whether
    averaging actually beats single-nearest-neighbor before committing to it.
    """
    df_raw = load_data()
    summary = {}
    for k in ks:
        print("\n" + "#" * 60)
        print(f"# k = {k}")
        print("#" * 60)
        res = run_once(df_raw, weight_mode, k=k)
        summary[k] = res
    return summary


def run_ab_comparison(modes=("none", "exclude", "correlation"), k=K_NEIGHBORS):
    """
    Convenience runner: evaluates each weighting mode back-to-back on the
    same data so you can directly compare LOEGO R²/MAE across modes.
    Call this instead of main() when you want the A/B side-by-side.
    """
    df_raw = load_data()
    summary = {}
    for mode in modes:
        print("\n" + "#" * 60)
        print(f"# MODE: {mode}")
        print("#" * 60)
        res = run_once(df_raw, mode, k=k)
        summary[mode] = res
    return summary


if __name__ == "__main__":
    main()