import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = r"C:\26_internship\synthetic_trajectories.csv"

OUTPUT_DIR = Path(r"C:\26_internship\old_csv_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set True if you want every individual condition plotted.
SAVE_INDIVIDUAL_CONDITIONS = True


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("\nLoaded CSV:")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

print("\nColumns:")
for c in df.columns:
    print("  ", c)


# ============================================================
# IDENTIFY NUMERICAL FEATURE COLUMNS
# ============================================================

metadata = [
    "t",
    "Pe",
    "Ma",
    "glycerol_pct",
    "concentration_wv",
    "particle_size_nm",
    "batch",
    "model",
    "model_name",
    "synthesizer",
    "synthesiser",
    "droplet_id",
    "condition",
    "id",
]

# Only columns that actually exist
metadata = [c for c in metadata if c in df.columns]

# Candidate columns
candidate_cols = [
    c for c in df.columns
    if c not in metadata
]

# Keep ONLY columns that pandas can actually interpret numerically
feature_cols = []

for c in candidate_cols:
    numeric = pd.to_numeric(df[c], errors="coerce")

    # A feature is considered numerical if at least 95%
    # of its values can be converted to numbers.
    valid_fraction = numeric.notna().mean()

    if valid_fraction >= 0.95:
        feature_cols.append(c)
        df[c] = numeric

print("\nDetected numerical feature columns:")
for c in feature_cols:
    print("  ", c)

print(f"\nNumber of numerical features = {len(feature_cols)}")
# ============================================================
# BASIC CHECK
# ============================================================

if "t" not in df.columns:
    raise ValueError(
        "CSV does not contain a 't' column. "
        "Change the TIME_COLUMN setting to match your CSV."
    )

if len(feature_cols) == 0:
    raise ValueError("No feature columns were detected.")


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [c for c in ["glycerol_pct", "concentration_wv", "batch", "t"]
     if c in df.columns]
).reset_index(drop=True)


# ============================================================
# 1. ALL FEATURES OVERLAID
# ============================================================

fig, axes = plt.subplots(
    len(feature_cols),
    1,
    figsize=(12, 3 * len(feature_cols)),
    squeeze=False
)

axes = axes.ravel()

for ax, feature in zip(axes, feature_cols):

    if "glycerol_pct" in df.columns and "concentration_wv" in df.columns:

        for (gly, ps), group in df.groupby(
            ["glycerol_pct", "concentration_wv"]
        ):
            group = group.sort_values("t")

            ax.plot(
                group["t"],
                group[feature],
                linewidth=1.2,
                label=f"Glycerol={gly:g}%, PS={ps:g}%"
            )

    else:

        ax.plot(
            df["t"],
            df[feature],
            linewidth=1.2
        )

    ax.set_ylabel(feature)
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("Normalized time")

handles, labels = axes[0].get_legend_handles_labels()

if handles:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=8
    )

fig.suptitle(
    "Previous Synthetic Dataset — Feature Trajectories",
    fontsize=16
)

plt.tight_layout(rect=(0, 0, 1, 0.96))

fig.savefig(
    OUTPUT_DIR / "01_all_feature_trajectories.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 2. INDIVIDUAL FEATURE PLOTS
# ============================================================

for feature in feature_cols:

    fig, ax = plt.subplots(figsize=(11, 6))

    if "glycerol_pct" in df.columns and "concentration_wv" in df.columns:

        for (gly, ps), group in df.groupby(
            ["glycerol_pct", "concentration_wv"]
        ):
            group = group.sort_values("t")

            ax.plot(
                group["t"],
                group[feature],
                linewidth=1.4,
                label=f"Glycerol={gly:g}%, PS={ps:g}%"
            )

    else:

        ax.plot(
            df["t"],
            df[feature],
            linewidth=1.4
        )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} vs time")
    ax.grid(alpha=0.25)

    if "glycerol_pct" in df.columns:
        ax.legend(fontsize=8)

    plt.tight_layout()

    safe_name = feature.replace("/", "_").replace(" ", "_")

    fig.savefig(
        OUTPUT_DIR / f"feature_{safe_name}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 3. CONDITION-BY-CONDITION PLOTS
# ============================================================

if SAVE_INDIVIDUAL_CONDITIONS:

    if (
        "glycerol_pct" in df.columns
        and "concentration_wv" in df.columns
    ):

        for (gly, ps), group in df.groupby(
            ["glycerol_pct", "concentration_wv"]
        ):

            group = group.sort_values("t")

            fig, axes = plt.subplots(
                len(feature_cols),
                1,
                figsize=(11, 2.8 * len(feature_cols)),
                squeeze=False
            )

            axes = axes.ravel()

            for ax, feature in zip(axes, feature_cols):

                ax.plot(
                    group["t"],
                    group[feature],
                    linewidth=1.4
                )

                ax.set_ylabel(feature)
                ax.grid(alpha=0.25)

            axes[-1].set_xlabel("Normalized time")

            fig.suptitle(
                f"Previous Synthesiser\n"
                f"Glycerol = {gly:g}% | PS = {ps:g}%",
                fontsize=15
            )

            plt.tight_layout(rect=(0, 0, 1, 0.95))

            filename = (
                f"condition_gly{gly:g}_ps{ps:g}.png"
                .replace(".", "p")
            )

            fig.savefig(
                OUTPUT_DIR / filename,
                dpi=200,
                bbox_inches="tight"
            )

            plt.close(fig)


# ============================================================
# 4. FEATURE HEATMAP
# ============================================================

# Use the first condition for a clean time × feature visualization.

if (
    "glycerol_pct" in df.columns
    and "concentration_wv" in df.columns
):

    first_condition = list(
        df.groupby(["glycerol_pct", "concentration_wv"])
    )[0][0]

    gly, ps = first_condition

    plot_df = df[
        (df["glycerol_pct"] == gly)
        & (df["concentration_wv"] == ps)
    ].sort_values("t")

else:

    plot_df = df.sort_values("t")


X = plot_df[feature_cols].to_numpy(dtype=float)

# Normalize each feature independently for visualization.
# This does NOT alter the CSV.
X_norm = np.zeros_like(X)

for j in range(X.shape[1]):

    col = X[:, j]

    mean = np.nanmean(col)
    std = np.nanstd(col)

    if std > 1e-12:
        X_norm[:, j] = (col - mean) / std
    else:
        X_norm[:, j] = 0


fig, ax = plt.subplots(figsize=(14, 8))

im = ax.imshow(
    X_norm.T,
    aspect="auto",
    interpolation="nearest",
    origin="upper"
)

ax.set_yticks(np.arange(len(feature_cols)))
ax.set_yticklabels(feature_cols)

ax.set_xlabel("Time index")
ax.set_ylabel("Feature")

ax.set_title(
    "12-D Feature Trajectory Heatmap"
)

fig.colorbar(
    im,
    ax=ax,
    label="Within-feature standardized value"
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "02_feature_heatmap.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 5. PCA OF THE 12-D TRAJECTORY
# ============================================================

try:

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = plot_df[feature_cols].to_numpy(dtype=float)

    # Replace NaNs only for PCA
    # using column medians.
    for j in range(X.shape[1]):

        col = X[:, j]

        if np.any(~np.isfinite(col)):

            median = np.nanmedian(col)

            if not np.isfinite(median):
                median = 0.0

            X[:, j] = np.where(
                np.isfinite(col),
                col,
                median
            )

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)

    Z = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))

    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=plot_df["t"],
        s=18,
        alpha=0.8
    )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )

    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )

    ax.set_title("PCA — Previous 12-D Synthetic Trajectory")

    ax.grid(alpha=0.2)

    fig.colorbar(
        scatter,
        ax=ax,
        label="Normalized time"
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "03_PCA_trajectory.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "\nPCA explained variance:"
    )

    print(
        f"  PC1 = {pca.explained_variance_ratio_[0] * 100:.2f}%"
    )

    print(
        f"  PC2 = {pca.explained_variance_ratio_[1] * 100:.2f}%"
    )

except ImportError:

    print(
        "\nWARNING: scikit-learn is not installed."
    )

    print(
        "Install it with:"
    )

    print(
        "pip install scikit-learn"
    )


# ============================================================
# 6. OSCILLATION / FIRST-DIFFERENCE DIAGNOSTIC
# ============================================================

# This is useful for your original question:
# "Why are there oscillations at certain concentrations?"

for feature in feature_cols:

    if (
        "glycerol_pct" in df.columns
        and "concentration_wv" in df.columns
    ):

        fig, ax = plt.subplots(figsize=(11, 6))

        for (gly, ps), group in df.groupby(
            ["glycerol_pct", "concentration_wv"]
        ):

            group = group.sort_values("t")

            y = group[feature].to_numpy()

            if len(y) < 3:
                continue

            dy = np.gradient(y)

            ax.plot(
                group["t"],
                dy,
                linewidth=1.0,
                label=f"G={gly:g}%, PS={ps:g}%"
            )

        ax.axhline(0, linewidth=0.8)

        ax.set_xlabel("Normalized time")
        ax.set_ylabel(f"d({feature}) / dt")

        ax.set_title(
            f"Temporal derivative — {feature}"
        )

        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        plt.tight_layout()

        safe_name = feature.replace("/", "_")

        fig.savefig(
            OUTPUT_DIR / f"derivative_{safe_name}.png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PLOTTING COMPLETE")
print("=" * 60)

print(f"\nInput CSV:")
print(CSV_PATH)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\nRows plotted: {len(df)}")
print(f"Features plotted: {len(feature_cols)}")

print("\nGenerated:")
print("  01_all_feature_trajectories.png")
print("  02_feature_heatmap.png")
print("  03_PCA_trajectory.png")
print("  Individual feature plots")
print("  Condition-specific plots")
print("  Temporal derivative plots")
print("\nDone.")