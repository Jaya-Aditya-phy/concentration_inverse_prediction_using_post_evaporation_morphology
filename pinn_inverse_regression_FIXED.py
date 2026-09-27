"""
PHYSICS-INFORMED NEURAL NETWORK — INVERSE REGRESSION
=======================================================

Predicts (glycerol_pct, concentration_wv) from the same 40-dim image
feature vector used in cosine_LOEGO.py, trained with:

    total_loss = data_loss
               + LAMBDA_PE   * pe_consistency_residual
               + LAMBDA_MASS * mass_consistency_residual

WHY A PINN HERE, AND WHAT IT BUYS YOU
--------------------------------------
The cosine-kNN model in cosine_LOEGO.py is nonparametric (no trainable
weights, no loss function) -- there is nothing to attach a physics
residual to. This file replaces the predictor with a small trainable
network so a physics-consistency term can be enforced during training.

IMPORTANT CAVEATS (read before trusting these numbers):
  1. cosine-kNN LOEGO log-R^2 ~= 0.78 is your current best, honestly
     validated result. This PINN is a genuinely different model class
     and is NOT guaranteed to beat it -- MC-dropout NNs on this exact
     feature set previously returned NEGATIVE R^2 (see model2_regression.py
     history). Report both, do not silently replace the working baseline
     until LOEGO says the PINN actually wins.
  2. The Pe/Ma physics residual uses evap_time_model.t_evap_model(), which
     is FULLY THEORETICAL (no calibration data exists yet, per Jaya,
     2026-08-29). This residual regularizes the network toward physical
     plausibility, but it is not itself validated against real time-lapse
     data. State this explicitly as a modeling assumption in the paper.
  3. Pe/Ma computation needs droplet_radius_um, contact_angle_deg,
     drop_volume_uL, rh_pct, ambient_T_C from the `experiments` table --
     these are metadata columns, NOT part of the 40-dim image feature
     vector. Rows missing this metadata cannot contribute to the physics
     term (data_loss still applies; physics term is masked to 0 for them).

PHYSICS RESIDUAL 1 -- Pe CONSISTENCY
-------------------------------------
Pe_true is computed once per row from calculate_droplet_physics()-style
relations using the TRUE glycerol_pct/metadata (this is your existing,
validated Pe formula, reproduced here so this file is self-contained).
Pe_implied is computed the same way but using the NETWORK'S predicted
glycerol_pct. Residual = (log(Pe_true) - log(Pe_implied))^2 (log-space
because Pe spans orders of magnitude, same convention as your log-space
concentration evaluation in cosine_LOEGO.py).

PHYSICS RESIDUAL 2 -- MASS CONSISTENCY
-----------------------------------------
Penalizes physically impossible (glycerol_pct, concentration_wv) pairs:
MFAT-corrected concentration must be in [0, CONC_MAX], and glycerol_pct
must be in [0, 100]. This is a bounds/plausibility penalty, not a
mechanistic law -- weakest of the two terms, cheap safety net.

LOEGO EVALUATION
-----------------
Uses the same LeaveOneGroupOut-by-batch discipline as cosine_LOEGO.py:
scaler fit on TRAIN groups only, network retrained fresh per fold (no
leakage). This is expensive (one full NN training run per held-out
group) -- expect this to take much longer than the kNN LOEGO loop.
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("WARNING: TensorFlow not installed. Run: pip install tensorflow")

from evap_time_model import (
    t_evap_model, glycerol_mole_fraction, contact_angle_model,
    M_WATER, M_GLYCEROL, R_GAS, D_VAPOR_AIR,
    ANTOINE_A, ANTOINE_B, ANTOINE_C,
)

# reuse the exact loading/cleaning/vector-building logic already
# validated in cosine_LOEGO.py -- no reason to duplicate/diverge it
from cosine_LOEGO import (
    DB_PATH, TARGETS, CONC_MAX, CONC_MIN,
    load_data, batch_group, discover_base_feature_names,
    build_vector_matrix, clean, nan_safe_impute,
    get_feature_weights,
)

# -----------------------------
# CONFIG
# -----------------------------
# NOTE on scale: glycerol_pct (0-100) and concentration_wv (0-1ish) are on
# very different raw scales, so unweighted MSE on [glycerol_pct, concentration_wv]
# lets glycerol's squared error dominate data_loss by ~2-3 orders of magnitude,
# which in turn drowns out LAMBDA_PE/LAMBDA_MASS regardless of their value
# (verified empirically: data_loss ~150-440 vs pe_loss ~0.2-0.5 on synthetic
# data with these targets -- physics term was contributing <0.03% of total).
# TARGET_SCALE divides each output dim before computing data_loss so both
# targets contribute comparably; LAMBDA_PE/LAMBDA_MASS below assume this
# rescaling is applied (see pinn_loss).
TARGET_SCALE = np.array([100.0, 1.0], dtype=np.float32)  # [glycerol_pct, concentration_wv]

LAMBDA_PE = 1.0        # weight on Pe-consistency residual, re-tuned now that
                       # data_loss is scale-normalized instead of dominated
                       # by raw glycerol_pct MSE -- start near 1.0 (comparable
                       # weight to data term) and sweep from there, not 0.05
LAMBDA_MASS = 0.1      # weight on mass-consistency bounds penalty
LAMBDA_MA = 0.002      # weight on Ma-vs-metadata consistency residual.
                       # NOTE: empirically much smaller than LAMBDA_PE (1.0) --
                       # Ma is far more sensitive to predicted glycerol_pct
                       # than Pe is, because viscosity swings ~1000x between
                       # water (8.9e-4 Pa.s) and pure glycerol (0.945 Pa.s)
                       # and Ma depends on it directly. Verified empirically
                       # (5 random seeds): raw ma_loss sits at ~300-900 vs
                       # data_loss/pe_loss at ~0.3-0.9 -- LAMBDA_MA=1.0 (same
                       # as LAMBDA_PE) would have repeated the exact scale-
                       # domination bug already caught and fixed for LAMBDA_PE
                       # before TARGET_SCALE was added. 0.002 brings all loss
                       # terms into comparable range across training (verified).
LAMBDA_MA_IMG = 0.3    # weight on Ma-vs-inner_deposit_frac RANKING residual
                       # (qualitative/monotonic constraint, not a quantitative
                       # law -- kept smaller than LAMBDA_PE/LAMBDA_MA by default)
EPOCHS = 300
BATCH_SIZE = 16
HIDDEN_DIMS = (64, 32)
DROPOUT_RATE = 0.15
LEARNING_RATE = 1e-3

# Ensemble averaging: with ~140 rows / 32 groups, a single trained network
# has real variance from random weight init + minibatch shuffle order alone.
# Averaging predictions from several independently-initialized networks per
# fold reduces that variance -- same principle as cosine_LOEGO.py's k=9
# neighbor-averaging (K_NEIGHBORS), applied here to NN init variance instead
# of neighbor selection variance. Smaller ensemble during the fast sweep
# (keeps hyperparameter search tractable), larger for the final reported
# LOEGO run (this is the number that should most benefit from variance
# reduction, since it's the one going in the paper).
ENSEMBLE_SIZE_SWEEP = 3
ENSEMBLE_SIZE_FINAL = 5

# Correlation-based feature weighting, mirroring cosine_LOEGO.py's
# WEIGHT_MODE mechanism (get_feature_weights is imported directly from
# there, not reimplemented) -- previously the PINN fed all 40 raw
# z-scored features into the network unweighted, unlike cosine_LOEGO.py
# which already down-weights/drops low-signal dims per fold. Kept as
# INDEPENDENT local constants (not `from cosine_LOEGO import WEIGHT_MODE`
# by value) so this file isn't silently stale if cosine_LOEGO.WEIGHT_MODE
# is changed later -- same class of stale-binding bug already hit twice
# with HIDDEN_DIMS/DB_PATH defaults, avoided here by always passing
# these explicitly to get_feature_weights() rather than relying on its
# own default argument.
WEIGHT_MODE = "threshold"          # "none" | "correlation" | "threshold"
CORR_THRESHOLD = 0.5               # only used when WEIGHT_MODE == "threshold"
CORR_WEIGHT_TARGET = "concentration_wv"  # "glycerol_pct" | "concentration_wv" | "combined"

RHO_WATER = 997.0
G_ACCEL = 9.81

# Real particle size, confirmed 2026-08-29: 1000nm PS microspheres, fixed
# across suspensions (not per-row in the DB, so a module-level constant
# is appropriate rather than pulling a column). The Pe residual previously
# defaulted to 200nm here -- a 5x error in particle diameter, which feeds
# directly into D ~ 1/d and therefore into every Pe value the physics
# residual is compared against. Fixed.
PARTICLE_SIZE_NM = 1000.0


# -----------------------------
# STEP 1 -- METADATA JOIN
# (Pe residual needs geometry/ambient columns not in features_segmented)
# -----------------------------
def load_data_with_metadata(db_path=None):
    """
    Same as cosine_LOEGO.load_data() but also pulls the metadata columns
    needed for the Pe residual: droplet_radius_um, contact_angle_deg,
    drop_volume_uL, rh_pct, ambient_T_C.

    db_path=None (default) reads the CURRENT module-level DB_PATH at call
    time rather than binding it at function-definition time -- a plain
    `db_path=DB_PATH` default would silently ignore any later
    `pinn_conc_glycerol.DB_PATH = ...` reassignment (same class of bug
    already fixed for HIDDEN_DIMS in train_pinn_fold).
    """
    if db_path is None:
        db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT e.droplet_id AS droplet_id, e.glycerol_pct, e.concentration_wv,
               e.droplet_radius_um, e.contact_angle_deg, e.drop_volume_uL,
               e.rh_pct, e.ambient_T_C,
               f.*
        FROM experiments e
        JOIN features_segmented f ON e.droplet_id = f.droplet_id
        WHERE e.glycerol_pct IS NOT NULL
          AND e.concentration_wv IS NOT NULL
    """, conn)
    conn.close()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    print(f"Loaded {len(df)} rows with metadata from features_segmented")
    return df


META_COLS = ["droplet_radius_um", "contact_angle_deg", "drop_volume_uL",
             "rh_pct", "ambient_T_C"]


def radius_from_volume_and_angle(drop_volume_uL, contact_angle_deg):
    """
    Spherical-cap radius from FIXED volume and contact angle -- exact
    same formula as lab_labeler.py's autocalculate_radius() GUI method,
    reproduced here (not imported, since lab_labeler.py is a tkinter
    script) so this file stays self-consistent with how the labeler
    itself derives radius from (volume, contact angle).

    For a spherical cap of volume V and contact angle theta:
        V = (pi * R^3 / 3) * (2 - 3cos(theta) + cos^3(theta)) / sin^3(theta)
    which lab_labeler.py implements equivalently via tan(theta/2) as:
        denom = pi * (3*tan(theta/2) + tan(theta/2)^3)
        R = ((6 * V) / denom)^(1/3)

    Used here because overriding contact_angle_deg (see build_meta_array)
    without also recomputing radius would leave R inconsistent with the
    new angle at the SAME droplet volume -- R and theta are geometrically
    coupled for a fixed-volume droplet, they can't be changed independently.
    """
    theta_deg = np.asarray(contact_angle_deg, dtype=float)
    V_uL = np.asarray(drop_volume_uL, dtype=float)

    theta_rad = np.radians(np.clip(theta_deg, 0.5, 179.0))
    tan_half = np.tan(theta_rad / 2.0)

    denom = np.pi * (3.0 * tan_half + tan_half ** 3)
    V_um3 = V_uL * 1e9  # uL -> um^3

    R_um = (6.0 * V_um3 / denom) ** (1.0 / 3.0)
    return R_um


def build_meta_array(df, override_contact_angle=True):
    """
    Single source of truth for assembling the (R_um, contact_angle_deg,
    drop_volume_uL, rh_pct, ambient_T_C) metadata array used everywhere
    the Pe residual needs it. Was previously duplicated inline in three
    places (pinn_loego_evaluate, sweep_pinn_hyperparameters, main) --
    centralized so the contact-angle/radius override below applies
    consistently everywhere instead of risking a spot that gets missed.

    override_contact_angle=True (default): REPLACES contact_angle_deg
    with contact_angle_model(glycerol_pct) rather than the logged
    experiments.contact_angle_deg (confirmed 2026-08-29: that column is
    a placeholder, 15 deg for every row, not a real measurement).

    CRITICAL: droplet_radius_um is ALSO recomputed here, from the fixed
    logged drop_volume_uL and the NEW contact angle, via
    radius_from_volume_and_angle() (same spherical-cap formula as
    lab_labeler.py's autocalculate_radius()). The logged
    droplet_radius_um was itself computed by that same GUI method at
    labeling time, using the OLD placeholder 15 deg angle for every row.
    Overriding contact_angle_deg alone and leaving the stale
    volume-derived radius in place would silently break the geometric
    relationship between R, theta, and V for a fixed-volume droplet --
    caught by Jaya, 2026-08-29, before this shipped. Volume is held
    fixed (it's the actual pipetted amount); radius and angle are
    recomputed together so they stay consistent with each other.

    Set override_contact_angle=False to skip BOTH overrides (contact
    angle and the radius recomputation) and use the raw logged columns
    as-is -- e.g. once real per-droplet contact angles are measured.
    """
    meta = df[META_COLS].values.astype(float)

    if override_contact_angle:
        theta_idx = META_COLS.index("contact_angle_deg")
        R_idx = META_COLS.index("droplet_radius_um")
        V_idx = META_COLS.index("drop_volume_uL")

        new_theta = contact_angle_model(df["glycerol_pct"].values)
        meta[:, theta_idx] = new_theta
        # keep volume fixed (the actual pipetted amount), recompute radius
        # to stay geometrically consistent with the new angle
        meta[:, R_idx] = radius_from_volume_and_angle(meta[:, V_idx], new_theta)

    meta_valid_mask = ~np.isnan(meta).any(axis=1)
    n_missing = int((~meta_valid_mask).sum())
    if n_missing:
        print(f"WARNING: {n_missing}/{len(df)} rows missing non-contact-angle "
              f"Pe-residual metadata ({[c for c in META_COLS if c != 'contact_angle_deg']}) "
              f"-- filling with column medians.")
        col_median = np.nanmedian(meta, axis=0)
        meta = np.where(np.isnan(meta), col_median, meta)

    return meta


def glycerol_viscosity_np(glycerol_pct, T_C=25.0):
    """
    Numpy re-implementation of lab_labeler.glycerol_viscosity() (Cheng
    2008 correlation) -- duplicated here (rather than imported) since
    lab_labeler.py is a tkinter GUI script not meant to be imported as a
    library. Keep in sync if the labeler's formula changes.
    """
    Cm = np.clip(np.asarray(glycerol_pct, dtype=float) / 100.0, 0.0, 1.0)
    a = 0.705 - 0.0017 * T_C
    b = (4.9 + 0.036 * T_C) * a ** 2.5
    alpha = 1.0 - Cm + (a * b * Cm * (1.0 - Cm)) / (a * Cm + b * (1.0 - Cm) + 1e-12)
    mu_w = 8.9e-4
    mu_g = 0.945
    return mu_w ** alpha * mu_g ** (1.0 - alpha)


def compute_pe(glycerol_pct, R_um, contact_angle_deg, drop_volume_uL,
               rh_pct, T_C, particle_size_nm=PARTICLE_SIZE_NM):
    """
    Pe = v_evap * R / D_corrected, matching lab_labeler.calculate_droplet_physics
    (wall-hindrance factor 0.5), but using the theoretical t_evap_model()
    for v_evap instead of a database evap_time_s column (none exists).
    particle_size_nm defaults to PARTICLE_SIZE_NM (confirmed 1000nm PS
    microspheres, fixed across suspensions per Jaya, 2026-08-29) -- was
    previously a wrong 200nm default, a 5x diameter error that fed
    directly into D ~ 1/d for every Pe value in the residual.
    """
    R_m = np.asarray(R_um, dtype=float) * 1e-6
    d_m = particle_size_nm * 1e-9

    t_evap = t_evap_model(glycerol_pct, R_um, contact_angle_deg,
                           drop_volume_uL, rh_pct, T_C)
    theta = np.radians(np.clip(contact_angle_deg, 0.5, 179.0))
    h0 = R_m * np.tan(theta / 2.0)
    v_evap = h0 / np.maximum(t_evap, 1e-9)

    eta = glycerol_viscosity_np(glycerol_pct, T_C)
    kT = 1.380649e-23 * (273.15 + np.asarray(T_C, dtype=float))
    D_standard = kT / (3.0 * np.pi * eta * d_m)
    D_corrected = D_standard * 0.5  # wall-hindrance factor, matches lab_labeler

    Pe = (v_evap * R_m) / np.maximum(D_corrected, 1e-30)
    return Pe


# Surface tension / diffusivity constants, matching feature_extractor.py's
# marangoni_number() exactly (NOT lab_labeler.py's cruder Ma formula --
# see compute_ma() docstring for why).
SIGMA_WATER_NM = 71.99e-3     # N/m at 25 C
SIGMA_GLYCEROL_NM = 63.4e-3   # N/m at 25 C
D_GLYCEROL_WATER = 0.95e-9    # m^2/s, glycerol diffusivity in water at 25 C


def compute_ma(glycerol_pct, R_um, contact_angle_deg, T_C=25.0):
    """
    Solutal Marangoni number, ported from feature_extractor.py's
    marangoni_number() -- NOT lab_labeler.py's Ma formula, which uses an
    arbitrary constant (delta_sigma = 0.008 * mass_fraction) and a
    generic guessed diffusivity (1e-9) rather than literature values.
    feature_extractor.py's version uses real surface tensions
    (SIGMA_WATER/SIGMA_GLYCEROL) and the real glycerol-water diffusivity
    (D_GLYCEROL_WATER, D'Errico et al. 2004), so it's used here for
    consistency with everything else in this pipeline and for a more
    defensible number in the paper.

        Ma = delta_sigma * h0 / (eta * D_glycerol)

    h0 (spherical-cap apex height) uses the SAME R_um and contact_angle_deg
    that feed Pe -- i.e. this must be called with the recomputed
    (post-override) radius from build_meta_array/radius_from_volume_and_angle,
    not a stale DB radius, for the same reason Pe needs it.
    """
    mass_frac = np.clip(np.asarray(glycerol_pct, dtype=float) / 100.0, 0.0, 1.0)

    R_m = np.asarray(R_um, dtype=float) * 1e-6
    theta = np.radians(np.clip(contact_angle_deg, 0.5, 179.0))
    h0_m = R_m * np.tan(theta / 2.0)

    sigma_mix = SIGMA_WATER_NM - (SIGMA_WATER_NM - SIGMA_GLYCEROL_NM) * mass_frac
    delta_sigma = SIGMA_WATER_NM - sigma_mix

    eta = glycerol_viscosity_np(glycerol_pct, T_C)

    Ma = delta_sigma * h0_m / (eta * D_GLYCEROL_WATER + 1e-30)
    return Ma


# -----------------------------
# STEP 1B -- TF-NATIVE (DIFFERENTIABLE) EVAPORATION/Pe CHAIN
# -----------------------------
# tf.py_function has no automatic gradient for an arbitrary numpy function
# (this is what crashed: "shape of dy was [] instead of [16]" -- TF tried
# to backprop through compute_pe() and got an undefined/wrong-shaped
# gradient). The evaporation model is pure arithmetic (Antoine equation,
# Raoult's law, Hu & Larson f(theta)) with no branching, so it can be
# ported to native tf ops directly -- gradients then flow analytically
# from the Pe residual back into the network's predicted glycerol_pct,
# which is what a genuine physics-informed loss requires.
#
# This chain is used ONLY inside the loss (needs to be differentiable
# wrt the network's predicted glycerol_pct). The fixed target log_pe_true
# is still computed once with the numpy version in evap_time_model.py /
# compute_pe() above, since it never needs a gradient.

def tf_water_vapor_pressure_pa(T_C):
    log10_P_mmHg = ANTOINE_A - (ANTOINE_B / (ANTOINE_C + T_C))
    P_mmHg = tf.pow(10.0, log10_P_mmHg)
    return P_mmHg * 133.322


def tf_water_vapor_concentration(T_C):
    T_K = T_C + 273.15
    P_sat = tf_water_vapor_pressure_pa(T_C)
    return P_sat * M_WATER / (R_GAS * T_K)


def tf_glycerol_mole_fraction(glycerol_pct):
    Cm = tf.clip_by_value(glycerol_pct / 100.0, 0.0, 1.0)
    n_gly = Cm / M_GLYCEROL
    n_water = (1.0 - Cm) / M_WATER
    return n_gly / (n_gly + n_water + 1e-30)


def tf_hu_larson_f(contact_angle_deg):
    theta_deg = tf.clip_by_value(contact_angle_deg, 0.5, 89.9)
    theta = theta_deg * (np.pi / 180.0)
    numerator = 0.27 * tf.square(theta) + 1.30
    denominator = 0.6381 - 0.2239 * tf.square(theta - np.pi / 4.0)
    return numerator / denominator


def tf_evaporation_rate_kg_s(R_m, contact_angle_deg, glycerol_pct, rh_pct, T_C):
    c_sat_water = tf_water_vapor_concentration(T_C)
    x_water = 1.0 - tf_glycerol_mole_fraction(glycerol_pct)
    c_sat_eff = x_water * c_sat_water

    RH = tf.clip_by_value(rh_pct / 100.0, 0.0, 0.999)
    c_inf = RH * c_sat_water

    f_theta = tf_hu_larson_f(contact_angle_deg)
    Q = np.pi * R_m * D_VAPOR_AIR * tf.maximum(c_sat_eff - c_inf, 1e-12) * f_theta
    return Q


def tf_t_evap_model(glycerol_pct, R_um, contact_angle_deg, drop_volume_uL, rh_pct, T_C):
    R_m = R_um * 1e-6
    V0_kg = drop_volume_uL * 1e-9 * 1000.0
    Q = tf_evaporation_rate_kg_s(R_m, contact_angle_deg, glycerol_pct, rh_pct, T_C)
    return V0_kg / tf.maximum(Q, 1e-30)


def tf_glycerol_viscosity(glycerol_pct, T_C):
    Cm = tf.clip_by_value(glycerol_pct / 100.0, 0.0, 1.0)
    a = 0.705 - 0.0017 * T_C
    b = (4.9 + 0.036 * T_C) * tf.pow(a, 2.5)
    alpha = 1.0 - Cm + (a * b * Cm * (1.0 - Cm)) / (a * Cm + b * (1.0 - Cm) + 1e-12)
    mu_w = 8.9e-4
    mu_g = 0.945
    # x**y for tensor x needs tf.pow; mu_w/mu_g are python floats (fine as base)
    return tf.pow(mu_w, alpha) * tf.pow(mu_g, 1.0 - alpha)


def tf_compute_pe(glycerol_pct, R_um, contact_angle_deg, drop_volume_uL,
                   rh_pct, T_C, particle_size_nm=PARTICLE_SIZE_NM):
    """
    Differentiable TF-native counterpart to compute_pe() -- same formula,
    ported to tensor ops so gradients flow into glycerol_pct (the only
    predicted quantity in this chain; all other args are fixed metadata
    tensors for the batch). Same PARTICLE_SIZE_NM correction as compute_pe().
    """
    R_m = R_um * 1e-6
    d_m = particle_size_nm * 1e-9

    t_evap = tf_t_evap_model(glycerol_pct, R_um, contact_angle_deg, drop_volume_uL, rh_pct, T_C)
    theta = tf.clip_by_value(contact_angle_deg, 0.5, 179.0) * (np.pi / 180.0)
    h0 = R_m * tf.tan(theta / 2.0)
    v_evap = h0 / tf.maximum(t_evap, 1e-9)

    eta = tf_glycerol_viscosity(glycerol_pct, T_C)
    kT = 1.380649e-23 * (273.15 + T_C)
    D_standard = kT / (3.0 * np.pi * eta * d_m)
    D_corrected = D_standard * 0.5

    Pe = (v_evap * R_m) / tf.maximum(D_corrected, 1e-30)
    return Pe


def tf_compute_ma(glycerol_pct, R_um, contact_angle_deg, T_C=25.0):
    """
    Differentiable TF-native counterpart to compute_ma() -- same formula
    (ported from feature_extractor.py's marangoni_number(), not
    lab_labeler.py's cruder version -- see compute_ma() docstring).
    Gradients flow into glycerol_pct, the only predicted quantity here.
    """
    mass_frac = tf.clip_by_value(glycerol_pct / 100.0, 0.0, 1.0)

    R_m = R_um * 1e-6
    theta = tf.clip_by_value(contact_angle_deg, 0.5, 179.0) * (np.pi / 180.0)
    h0_m = R_m * tf.tan(theta / 2.0)

    sigma_mix = SIGMA_WATER_NM - (SIGMA_WATER_NM - SIGMA_GLYCEROL_NM) * mass_frac
    delta_sigma = SIGMA_WATER_NM - sigma_mix

    eta = tf_glycerol_viscosity(glycerol_pct, T_C)

    Ma = delta_sigma * h0_m / (eta * D_GLYCEROL_WATER + 1e-30)
    return Ma


# -----------------------------
# STEP 2 -- PINN MODEL
# -----------------------------
def build_pinn(input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT_RATE):
    """
    REVERTED to plain linear Dense(2) output for BOTH heads. Two prior
    attempts to bound the concentration head (sigmoid*CONC_MAX, then
    softplus) each looked fine on isolated synthetic gradient/training
    tests but caused real regressions on actual data: sigmoid caused a
    vanishing-gradient collapse (glycerol R^2 -416), and softplus --
    despite passing an 800-step synthetic training check -- still
    regressed log-space R^2 from the known-good ~0.62-0.63 down to -2.65
    on real data with every other setting held identical (isolated,
    single-variable change). Given repeated real-data regressions from
    activation-function changes that don't show up in synthetic tests,
    it's not safe to keep iterating on this blind -- reverted to the
    ONLY configuration that has ever actually scored well.

    The negative-concentration-prediction problem (11/142 rows in one
    run) is real and still needs handling, but should be done as a
    non-training-affecting post-hoc step (e.g. clip only at inference,
    after the network is already trained) rather than by changing what
    the network optimizes during training -- see the WARNING block in
    pinn_loego_evaluate for where this is diagnosed; consider clipping
    predictions there specifically rather than reshaping the loss
    landscape again.
    """
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for h in hidden_dims:
        x = layers.Dense(h, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(2)(x)  # [glycerol_pct, concentration_wv], both unbounded linear
    return keras.Model(inputs, outputs)


def mass_consistency_penalty(y_pred, conc_max=CONC_MAX):
    """
    Bounds penalty: glycerol_pct in [0,100], concentration_wv in [0, conc_max].
    Penalizes predictions that fall outside physically valid ranges --
    cheap safety net, not a mechanistic law.
    """
    gly = y_pred[:, 0]
    conc = y_pred[:, 1]
    pen_gly = tf.nn.relu(-gly) + tf.nn.relu(gly - 100.0)
    pen_conc = tf.nn.relu(-conc) + tf.nn.relu(conc - conc_max)
    return tf.reduce_mean(pen_gly ** 2 + pen_conc ** 2)


def pe_consistency_penalty(y_pred, meta_batch, log_pe_true_batch):
    """
    log-space Pe residual: (log Pe_true - log Pe_implied(pred_glycerol))^2.
    meta_batch: (R_um, contact_angle_deg, drop_volume_uL, rh_pct, T_C) per
    row, passed in as a tf constant tensor (fixed metadata, not trainable).
    Only the network's predicted glycerol_pct feeds back into Pe_implied.

    Uses tf_compute_pe() (native tensor ops, see STEP 1B) so gradients
    flow analytically from this residual into the network weights --
    tf.py_function was tried first and crashed (no defined gradient for
    an arbitrary numpy black box), which is why this is a native port.
    """
    gly_pred = y_pred[:, 0]
    R_um, theta_deg, V_uL, rh, T = tf.unstack(meta_batch, axis=1)

    pe_implied = tf_compute_pe(gly_pred, R_um, theta_deg, V_uL, rh, T)
    log_pe_implied = tf.math.log(tf.maximum(pe_implied, 1e-12))

    residual = log_pe_true_batch - log_pe_implied
    return tf.reduce_mean(tf.square(residual))


def ma_consistency_penalty(y_pred, meta_batch, log_ma_true_batch):
    """
    log-space Ma residual, same structure as pe_consistency_penalty:
    (log Ma_true - log Ma_implied(pred_glycerol))^2. Ma only needs
    (R_um, contact_angle_deg, T_C) from meta_batch, not volume/RH.
    """
    gly_pred = y_pred[:, 0]
    R_um, theta_deg, V_uL, rh, T = tf.unstack(meta_batch, axis=1)

    ma_implied = tf_compute_ma(gly_pred, R_um, theta_deg, T)
    log_ma_implied = tf.math.log(tf.maximum(ma_implied, 1e-12))

    residual = log_ma_true_batch - log_ma_implied
    return tf.reduce_mean(tf.square(residual))


def marangoni_image_consistency_penalty(y_pred, meta_batch, inner_deposit_frac_batch):
    """
    RANKING residual tying predicted Ma to the REAL observed image feature
    inner_deposit_frac -- feature_extractor.py documents this as the
    "Marangoni signature": inward (Marangoni) recirculation moves deposit
    toward the interior, raising inner_deposit_frac. There's no established
    QUANTITATIVE law linking Ma to inner_deposit_frac (unlike the Pe/Ma-vs-
    metadata residuals above, which have closed-form physics), only a
    documented monotonic relationship -- so this is implemented as a
    pairwise RankNet-style soft-ranking loss: within a minibatch, if row i's
    implied Ma exceeds row j's, row i's observed inner_deposit_frac should
    also exceed row j's. Smooth/differentiable (sigmoid + BCE) rather than
    a non-differentiable sign() comparison.

    Works in LOG-Ma space (same convention as the other two Ma/Pe residuals)
    since Ma spans orders of magnitude -- pairwise differences in raw Ma
    would have wildly uneven scale and saturate the sigmoid, killing
    gradients for most pairs.

    inner_deposit_frac_batch: 1D tensor, the corresponding column already
    sliced out of the (z-scored) feature batch by the caller. Z-scoring is
    a monotonic (linear, positive-slope) transform, so pairwise ORDERING
    is preserved -- safe to use the scaled column directly, no need to
    unscale just for a ranking loss.
    """
    gly_pred = y_pred[:, 0]
    R_um, theta_deg, V_uL, rh, T = tf.unstack(meta_batch, axis=1)

    ma_implied = tf_compute_ma(gly_pred, R_um, theta_deg, T)
    log_ma_implied = tf.math.log(tf.maximum(ma_implied, 1e-12))

    ma_diff = log_ma_implied[:, None] - log_ma_implied[None, :]          # (batch, batch)
    feat_diff = inner_deposit_frac_batch[:, None] - inner_deposit_frac_batch[None, :]

    p_ij = tf.sigmoid(ma_diff)                       # model's implied P(i has higher Ma than j)
    t_ij = tf.cast(feat_diff > 0.0, tf.float32)       # observed: does i actually have higher inner_deposit_frac

    eps = 1e-7
    bce = -(t_ij * tf.math.log(p_ij + eps) + (1.0 - t_ij) * tf.math.log(1.0 - p_ij + eps))

    n = tf.shape(gly_pred)[0]
    off_diagonal_mask = 1.0 - tf.eye(n)               # exclude i==j pairs (uninformative, diff=0)
    bce = bce * off_diagonal_mask

    return tf.reduce_sum(bce) / (tf.reduce_sum(off_diagonal_mask) + 1e-8)


def pinn_loss(y_true, y_pred, meta_batch, log_pe_true_batch,
              log_ma_true_batch, inner_deposit_frac_batch,
              lambda_pe=LAMBDA_PE, lambda_mass=LAMBDA_MASS,
              lambda_ma=LAMBDA_MA, lambda_ma_img=LAMBDA_MA_IMG,
              has_metadata_mask=None):
    scale = tf.constant(TARGET_SCALE, dtype=tf.float32)
    data_loss = tf.reduce_mean(tf.square((y_true - y_pred) / scale))
    mass_loss = mass_consistency_penalty(y_pred)

    if has_metadata_mask is not None:
        # rows lacking Pe metadata contribute 0 to the physics term rather
        # than being dropped from the batch entirely (data_loss still uses them)
        pe_raw = pe_consistency_penalty(y_pred, meta_batch, log_pe_true_batch)
        pe_loss = pe_raw  # mask applied upstream by zeroing those rows' contribution
    else:
        pe_loss = pe_consistency_penalty(y_pred, meta_batch, log_pe_true_batch)

    ma_loss = ma_consistency_penalty(y_pred, meta_batch, log_ma_true_batch)
    ma_img_loss = marangoni_image_consistency_penalty(y_pred, meta_batch, inner_deposit_frac_batch)

    total = (data_loss + lambda_pe * pe_loss + lambda_mass * mass_loss
             + lambda_ma * ma_loss + lambda_ma_img * ma_img_loss)
    return total, data_loss, pe_loss, mass_loss, ma_loss, ma_img_loss


# -----------------------------
# STEP 3 -- TRAIN ONE FOLD
# -----------------------------
@tf.function
def _train_step(model, optimizer, X_batch, y_batch, meta_batch, log_pe_true_batch,
                 log_ma_true_batch, inner_deposit_frac_batch,
                 lambda_pe, lambda_mass, lambda_ma, lambda_ma_img):
    """
    Wrapped in @tf.function so TF traces this once per (fold's) unique
    batch shape and reuses the compiled graph -- eager mode was re-
    interpreting the Python loop body every single step, which is most
    of the wall-clock cost for a model this small. This should cut
    per-fold time substantially with no change in what's computed.

    inner_deposit_frac_batch is passed in as its OWN tensor (gathered by
    the caller from the PRE-correlation-weighting feature matrix), not
    sliced out of X_batch -- X_batch is now the correlation-weighted
    network input (see train_pinn_fold), and if inner_deposit_frac
    happens to correlate weakly with the WEIGHT_MODE target it could get
    threshold-zeroed there, which would collapse every row's value to 0
    and destroy the pairwise ranking signal in
    marangoni_image_consistency_penalty. Keeping it as a separate,
    unweighted tensor avoids that failure mode entirely.
    """
    with tf.GradientTape() as tape:
        y_pred = model(X_batch, training=True)
        total, d_loss, pe_loss, m_loss, ma_loss, ma_img_loss = pinn_loss(
            y_batch, y_pred, meta_batch, log_pe_true_batch,
            log_ma_true_batch, inner_deposit_frac_batch,
            lambda_pe=lambda_pe, lambda_mass=lambda_mass,
            lambda_ma=lambda_ma, lambda_ma_img=lambda_ma_img)
    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total


def find_feature_index(feat_cols, base_name, prefer_avg=True):
    """
    Locates a base feature's column index within feat_cols (the
    center_<f>...avg_<f>... list from build_vector_matrix). Prefers the
    avg_ (N/S/E/W-averaged) version over center_ since it's less
    single-tile-noise-prone, matching the convention used elsewhere in
    this pipeline (e.g. feature_weight_diagnostic in cosine_LOEGO.py).
    Raises rather than silently skipping -- if inner_deposit_frac isn't
    found, the Ma-image residual would be silently reading garbage.
    """
    candidates = [f"avg_{base_name}", f"center_{base_name}"] if prefer_avg \
        else [f"center_{base_name}", f"avg_{base_name}"]
    for c in candidates:
        if c in feat_cols:
            return feat_cols.index(c)
    raise ValueError(f"Could not find '{base_name}' in feat_cols "
                      f"(looked for {candidates}). Check feature_extractor.py "
                      f"still produces this feature name.")


def apply_feature_weighting(X_train_z, X_test_z, y_train, feat_cols):
    """
    Single source of truth for correlation-based feature weighting,
    reusing get_feature_weights() from cosine_LOEGO.py so both models are
    evaluated on features selected the SAME way -- previously the PINN
    fed all 40 raw z-scored dims into the network unweighted, unlike
    cosine_LOEGO.py's per-fold WEIGHT_MODE dropping/down-weighting of
    low-signal dims.

    Weights are computed on X_train_z/y_train ONLY (no leakage, same
    fold-local discipline as everywhere else in this file) and applied to
    both train and test.

    Returns (X_train_weighted, X_test_weighted, inner_deposit_frac_train)
    -- inner_deposit_frac_train is extracted from the PRE-weighting
    X_train_z, not the weighted output, because get_feature_weights with
    WEIGHT_MODE="threshold" can zero out a dimension entirely if its raw
    correlation with CORR_WEIGHT_TARGET falls below CORR_THRESHOLD.
    inner_deposit_frac correlates with the Marangoni/glycerol mechanism,
    not necessarily with concentration_wv (the default weighting target)
    -- if it got zeroed, every row's value would collapse to 0 and the
    Ma-image ranking residual (marangoni_image_consistency_penalty) would
    have no real pairwise signal left to learn from.
    """
    inner_idx = find_feature_index(feat_cols, "inner_deposit_frac")
    inner_deposit_frac_train = X_train_z[:, inner_idx].copy()

    w = get_feature_weights(WEIGHT_MODE, X_train_z, y_train, TARGETS,
                             CORR_WEIGHT_TARGET, threshold=CORR_THRESHOLD)
    X_train_weighted = X_train_z * w
    X_test_weighted = X_test_z * w

    return X_train_weighted, X_test_weighted, inner_deposit_frac_train


def train_pinn_fold(X_train, y_train, meta_train, X_test, inner_deposit_frac_train,
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0):
    """
    inner_deposit_frac_train: 1D array, same row order as X_train, sourced
    by the CALLER from the PRE-correlation-weighting feature matrix (see
    apply_feature_weighting()) -- not derived internally via feat_cols
    anymore, since X_train here is now the correlation-WEIGHTED network
    input and slicing inner_deposit_frac out of it directly would apply
    that same weighting to the Ma-image ranking signal too.
    """
    input_dim = X_train.shape[1]
    # NOTE: build_pinn's hidden_dims default is bound at module-load time,
    # so it would silently ignore any later `global HIDDEN_DIMS = ...`
    # reassignment (e.g. from sweep_pinn_hyperparameters) unless passed
    # explicitly here -- always read the CURRENT module-level value.
    model = build_pinn(input_dim, hidden_dims=HIDDEN_DIMS)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # Adam creates its momentum/velocity variables lazily on the FIRST
    # apply_gradients call. TF forbids creating tf.Variables inside a
    # traced @tf.function on that first call ("tf.function only supports
    # singleton tf.Variables created on the first call") -- so force the
    # optimizer to build its state eagerly, outside the trace, before the
    # training loop starts.
    optimizer.build(model.trainable_variables)

    R_um, theta_deg, V_uL, rh, T = meta_train.T
    gly_true = y_train[:, 0]
    pe_true = compute_pe(gly_true, R_um, theta_deg, V_uL, rh, T)
    log_pe_true = np.log(np.maximum(pe_true, 1e-12)).astype(np.float32)

    ma_true = compute_ma(gly_true, R_um, theta_deg, T)
    log_ma_true = np.log(np.maximum(ma_true, 1e-12)).astype(np.float32)

    X_train_tf = tf.constant(X_train, dtype=tf.float32)
    y_train_tf = tf.constant(y_train, dtype=tf.float32)
    meta_train_tf = tf.constant(meta_train, dtype=tf.float32)
    log_pe_true_tf = tf.constant(log_pe_true, dtype=tf.float32)
    log_ma_true_tf = tf.constant(log_ma_true, dtype=tf.float32)
    inner_deposit_frac_tf = tf.constant(inner_deposit_frac_train, dtype=tf.float32)
    lambda_pe_tf = tf.constant(LAMBDA_PE, dtype=tf.float32)
    lambda_mass_tf = tf.constant(LAMBDA_MASS, dtype=tf.float32)
    lambda_ma_tf = tf.constant(LAMBDA_MA, dtype=tf.float32)
    lambda_ma_img_tf = tf.constant(LAMBDA_MA_IMG, dtype=tf.float32)

    n = X_train.shape[0]
    best_loss = np.inf
    patience, bad_epochs = 30, 0

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b_idx = idx[start:start + batch_size]
            # fixed batch size keeps the @tf.function trace stable across
            # steps (ragged last batch would force a retrace) -- pad the
            # final short batch by wrapping indices instead of truncating
            if len(b_idx) < batch_size:
                pad = np.random.choice(idx, batch_size - len(b_idx), replace=True)
                b_idx = np.concatenate([b_idx, pad])
            total = _train_step(
                model, optimizer,
                tf.gather(X_train_tf, b_idx), tf.gather(y_train_tf, b_idx),
                tf.gather(meta_train_tf, b_idx), tf.gather(log_pe_true_tf, b_idx),
                tf.gather(log_ma_true_tf, b_idx), tf.gather(inner_deposit_frac_tf, b_idx),
                lambda_pe_tf, lambda_mass_tf, lambda_ma_tf, lambda_ma_img_tf,
            )
            epoch_loss += float(total) * batch_size
        epoch_loss /= n

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:>4}  loss={epoch_loss:.4f}")

    X_test_tf = tf.constant(X_test, dtype=tf.float32)
    y_pred_test = model(X_test_tf, training=False).numpy()
    return y_pred_test


def train_pinn_fold_ensemble(X_train, y_train, meta_train, X_test, inner_deposit_frac_train,
                              epochs=EPOCHS, batch_size=BATCH_SIZE,
                              ensemble_size=ENSEMBLE_SIZE_FINAL, verbose=0):
    """
    Trains `ensemble_size` independent networks on the same fold and
    averages their predictions. Each call to train_pinn_fold() naturally
    gets a different random weight initialization and minibatch shuffle
    order (Keras/numpy's global RNG state advances between calls, not
    reset), so no explicit seeding is needed for the members to differ --
    just call it multiple times and average.

    Averaging in raw target space (not log space) is consistent with
    everything else in this pipeline reporting log10(concentration) R^2
    as a post-hoc evaluation transform, not a modeling target.
    """
    preds = []
    for _ in range(ensemble_size):
        y_pred = train_pinn_fold(X_train, y_train, meta_train, X_test, inner_deposit_frac_train,
                                  epochs=epochs, batch_size=batch_size,
                                  verbose=verbose)
        preds.append(y_pred)
    return np.mean(preds, axis=0)


# -----------------------------
# STEP 3B -- FAST HYPERPARAMETER SWEEP (GroupKFold, not full LOEGO)
# -----------------------------
# Picking LAMBDA_PE by whichever value maximizes the full 32-fold LOEGO
# R^2 would be the exact tuning-on-the-test-set leakage that
# nested_loego_hyperparameter_tuning() in cosine_LOEGO.py was written to
# avoid -- the reported number would be optimistically biased because the
# hyperparameter was chosen to fit those same 32 reporting folds. This
# sweep instead uses a small GroupKFold (default 5 folds, grouped by
# batch so no replicate leakage) purely to CHOOSE a config; the winning
# config should then be plugged into LAMBDA_PE/LAMBDA_MASS/HIDDEN_DIMS
# above and evaluated with a FRESH full LOEGO run for the number you
# actually report.
#
# Also much cheaper: 5 folds x N configs, vs. 32 folds x N configs for a
# naive full-LOEGO grid search.

LAMBDA_PE_GRID = (0.1, 0.3, 1.0, 3.0)
LAMBDA_MASS_GRID = (0.1,)          # held fixed unless you want to sweep this too
ARCH_GRID = ((64, 32), (32,))       # current 2-layer vs. a simpler 1-layer net,
                                     # given ~140 rows is not much data for (64,32)
SWEEP_FOLDS = 5
SWEEP_EPOCHS = 150                  # fewer epochs than full run -- this is a
                                     # relative-ranking pass, not a final fit


def _groupkfold_score(X, y, groups, meta, feat_cols, lambda_pe, lambda_mass, hidden_dims,
                       n_folds=SWEEP_FOLDS, epochs=SWEEP_EPOCHS):
    """
    One (lambda_pe, lambda_mass, hidden_dims) candidate, scored by mean
    log10(concentration) R^2 across GroupKFold folds. Returns None if
    there aren't enough groups for n_folds.
    """
    n_groups = len(np.unique(groups))
    n_splits = min(n_folds, n_groups)
    if n_splits < 2:
        return None

    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []

    global LAMBDA_PE, LAMBDA_MASS, HIDDEN_DIMS
    old_pe, old_mass, old_hidden = LAMBDA_PE, LAMBDA_MASS, HIDDEN_DIMS
    LAMBDA_PE, LAMBDA_MASS, HIDDEN_DIMS = lambda_pe, lambda_mass, hidden_dims

    try:
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            meta_train = meta[train_idx]

            X_train, X_test = nan_safe_impute(X_train, X_test)
            scaler = StandardScaler()
            X_train_z = scaler.fit_transform(X_train)
            X_test_z = scaler.transform(X_test)

            X_train_w, X_test_w, inner_deposit_frac_train = apply_feature_weighting(
                X_train_z, X_test_z, y_train, feat_cols)

            y_pred = train_pinn_fold_ensemble(X_train_w, y_train, meta_train, X_test_w,
                                               inner_deposit_frac_train,
                                               epochs=epochs, batch_size=BATCH_SIZE,
                                               ensemble_size=ENSEMBLE_SIZE_SWEEP)

            conc_idx = TARGETS.index("concentration_wv")
            valid = y_test[:, conc_idx] > 0
            if valid.sum() < 2:
                continue
            log_true = np.log10(y_test[valid, conc_idx])
            log_pred = np.log10(np.clip(y_pred[valid, conc_idx], 1e-9, None))
            if np.std(log_true) < 1e-9:
                continue
            fold_r2.append(r2_score(log_true, log_pred))
    finally:
        LAMBDA_PE, LAMBDA_MASS, HIDDEN_DIMS = old_pe, old_mass, old_hidden

    return float(np.mean(fold_r2)) if fold_r2 else None


def sweep_pinn_hyperparameters(df, base_names,
                                lambda_pe_grid=LAMBDA_PE_GRID,
                                lambda_mass_grid=LAMBDA_MASS_GRID,
                                arch_grid=ARCH_GRID):
    """
    Fast GroupKFold sweep over (LAMBDA_PE, LAMBDA_MASS, HIDDEN_DIMS),
    scored by mean log10(concentration) R^2. Prints a ranked table and
    returns the best config -- plug it into the module-level constants
    and run pinn_loego_evaluate() fresh (full 32-fold LOEGO) to get the
    number you actually report. Do NOT report the sweep's own R^2 as
    your result -- it's a 5-fold estimate used only for model selection.
    """
    X, feat_cols = build_vector_matrix(df, base_names)
    y = df[TARGETS].values
    groups = df["group"].values

    meta = build_meta_array(df)

    n_configs = len(lambda_pe_grid) * len(lambda_mass_grid) * len(arch_grid)
    print(f"\nSweeping {n_configs} configs x {SWEEP_FOLDS} GroupKFold folds "
          f"({SWEEP_EPOCHS} epochs each, cheaper than full LOEGO)...")
    print("This picks a config -- report the FULL LOEGO run on the winner, not this number.\n")

    results = []
    for hidden_dims in arch_grid:
        for lp in lambda_pe_grid:
            for lm in lambda_mass_grid:
                score = _groupkfold_score(X, y, groups, meta, feat_cols, lp, lm, hidden_dims)
                results.append((hidden_dims, lp, lm, score))
                score_str = f"{score:.4f}" if score is not None else "N/A"
                print(f"  hidden={hidden_dims!s:<12} LAMBDA_PE={lp:<5} "
                      f"LAMBDA_MASS={lm:<5}  mean log-R2={score_str}")

    valid_results = [r for r in results if r[3] is not None]
    if not valid_results:
        print("\nNo valid configs scored -- check group/fold counts.")
        return None

    best = max(valid_results, key=lambda r: r[3])
    print(f"\nBest config: hidden_dims={best[0]}  LAMBDA_PE={best[1]}  "
          f"LAMBDA_MASS={best[2]}  (5-fold mean log-R2={best[3]:.4f})")
    print("Now set these as the module-level HIDDEN_DIMS/LAMBDA_PE/LAMBDA_MASS "
          "and rerun pinn_loego_evaluate() for the number to actually report.")

    return {"hidden_dims": best[0], "lambda_pe": best[1], "lambda_mass": best[2],
            "sweep_score": best[3], "all_results": results}


# -----------------------------
# STEP 4 -- LOEGO EVALUATION
# -----------------------------
def pinn_loego_evaluate(df, base_names, epochs=EPOCHS, batch_size=BATCH_SIZE):
    if not KERAS_AVAILABLE:
        raise RuntimeError("TensorFlow required for PINN training.")

    X, feat_cols = build_vector_matrix(df, base_names)
    y = df[TARGETS].values
    groups = df["group"].values

    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing metadata columns for Pe residual: {missing_meta}")

    meta = build_meta_array(df)

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    print(f"\nRunning PINN LOEGO over {n_folds} folds "
          f"({ENSEMBLE_SIZE_FINAL}-network ensemble per fold -- "
          f"this will take a while)...")

    all_true, all_pred, all_group = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        meta_train = meta[train_idx]

        X_train, X_test = nan_safe_impute(X_train, X_test)

        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_test_z = scaler.transform(X_test)

        X_train_w, X_test_w, inner_deposit_frac_train = apply_feature_weighting(
            X_train_z, X_test_z, y_train, feat_cols)

        y_pred = train_pinn_fold_ensemble(X_train_w, y_train, meta_train, X_test_w,
                                           inner_deposit_frac_train,
                                           epochs=epochs, batch_size=batch_size,
                                           ensemble_size=ENSEMBLE_SIZE_FINAL)

        all_true.append(y_test)
        all_pred.append(y_pred)
        all_group.extend([groups[test_idx][0]] * len(test_idx))

        if (fold_i + 1) % 5 == 0 or (fold_i + 1) == n_folds:
            print(f"  fold {fold_i + 1}/{n_folds} done")

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    print("\n=== PINN LOEGO RESULTS ===")
    for i, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"\n{target}:")
        print(f"  LOEGO MAE : {mae:.4f}")
        print(f"  LOEGO R²  : {r2:.4f}")

    # log-space concentration R^2, same convention as cosine_LOEGO.py,
    # for direct comparison against the 0.78 cosine-kNN baseline
    conc_idx = TARGETS.index("concentration_wv")
    valid = y_true[:, conc_idx] > 0
    raw_preds = y_pred[valid, conc_idx]

    n_nonpositive = int(np.sum(raw_preds <= 0))
    if n_nonpositive:
        worst_idx = np.argsort(raw_preds)[:min(5, n_nonpositive)]
        print(f"\nWARNING: {n_nonpositive}/{valid.sum()} predicted concentrations "
              f"are <= 0 -- these get clipped to 1e-9 before log10, which can "
              f"single-handedly wreck log-space R^2 (each contributes an ~9+ "
              f"order-of-magnitude log error). Raw R^2 being fine while log-R^2 "
              f"craters is the signature of exactly this.")
        print(f"  worst offenders (pred, true): "
              f"{list(zip(np.round(raw_preds[worst_idx], 6), np.round(y_true[valid, conc_idx][worst_idx], 6)))}")

    log_true = np.log10(y_true[valid, conc_idx])
    log_pred = np.log10(np.clip(raw_preds, 1e-9, None))
    log_r2 = r2_score(log_true, log_pred)
    print(f"\nlog10(concentration) LOEGO R² : {log_r2:.4f}  "
          f"(compare against cosine-kNN baseline ~0.78)")

    return {"y_true": y_true, "y_pred": y_pred, "groups": np.array(all_group),
            "log_conc_r2": log_r2}


# -----------------------------
# MAIN
# -----------------------------
def main():
    global LAMBDA_PE, LAMBDA_MASS, HIDDEN_DIMS  # must precede any use of these names below
    print("=" * 60)
    print("PINN INVERSE REGRESSION — LOEGO EVALUATION")
    print(f"Starting config: LAMBDA_PE={LAMBDA_PE}  LAMBDA_MASS={LAMBDA_MASS}  HIDDEN_DIMS={HIDDEN_DIMS}")
    print("=" * 60)

    df_raw = load_data_with_metadata()
    base_names = discover_base_feature_names(df_raw.columns)
    df_raw["group"] = df_raw["droplet_id"].apply(batch_group)

    df_clean, X, y, groups = clean(df_raw, base_names)
    if X is None:
        return None

    df_clean = df_clean.merge(
        df_raw[["droplet_id"] + META_COLS], on="droplet_id", how="left",
        suffixes=("", "_dup"))

    # --- STEP A: fast sweep to pick LAMBDA_PE/LAMBDA_MASS/HIDDEN_DIMS ---
    sweep_result = sweep_pinn_hyperparameters(df_clean, base_names)
    if sweep_result is not None:
        LAMBDA_PE = sweep_result["lambda_pe"]
        LAMBDA_MASS = sweep_result["lambda_mass"]
        HIDDEN_DIMS = sweep_result["hidden_dims"]
        print(f"\nUsing swept config for the reported LOEGO run: "
              f"LAMBDA_PE={LAMBDA_PE}  LAMBDA_MASS={LAMBDA_MASS}  HIDDEN_DIMS={HIDDEN_DIMS}")
    else:
        print("\nSweep failed/skipped -- running full LOEGO with starting config.")

    # --- STEP B: the number you actually report, on a fresh full LOEGO ---
    results = pinn_loego_evaluate(df_clean, base_names)
    return results


if __name__ == "__main__":
    main()