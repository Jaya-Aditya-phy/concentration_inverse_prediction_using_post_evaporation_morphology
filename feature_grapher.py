"""
PLOT ALL FEATURES — morphodynamic fingerprinting visualizer
=============================================================
Pulls every tile-level feature (whatever is currently in
features_segmented — works for the 30-feature schema or the
extended 100-feature schema, 20 features x 5 tiles) and produces
a set of diagnostic figures:

  1. all_features_vs_glycerol.pdf
       Grid of scatter plots: one row per feature "family" (e.g.
       edge_density, fractal_dimension, glcm_contrast...), one
       column per tile (center/north/south/east/west). X-axis =
       glycerol%, color = concentration (w/v%). Multi-page so it
       stays readable even with 20 feature families.

  2. feature_condition_heatmaps.pdf
       Same grid layout (feature family x tile), but each panel is a
       heatmap: y=glycerol%, x=concentration (w/v%), color=mean
       feature value for that condition. Missing conditions show as
       gray cells, which doubles as a coverage map of your condition
       grid.

  3. feature_correlation_heatmap.png
       Full feature x feature Pearson correlation matrix — useful
       for spotting redundant features before LOEGO regression.

  3. missing_data_overview.png
       Bar chart of NaN fraction per feature — flags tiles/features
       that are systematically failing extraction (e.g. missing
       north tiles, bad flat-fielding).

  4. feature_variance_by_condition.png
       Ranks features by how much they vary ACROSS distinct
       (glycerol%, concentration) conditions vs WITHIN replicates
       of the same condition. High between/within ratio = a feature
       that's actually informative for the inverse regression;
       low ratio = mostly noise. Directly useful for diagnosing the
       negative-R2 LOEGO issue (are new conditions actually needed,
       or are the features themselves uninformative?).

  5. flatfield_comparison.png (only if --compare-flatfield is used
       with two databases / two feature tables — see bottom of file)

USAGE
-----
    python plot_all_features.py
    python plot_all_features.py --db "C:\\path\\to\\droplets.db" --out feature_plots

Adjust DB_PATH below or pass --db on the command line.
"""

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless/servers; remove if you want interactive windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# CONFIG — edit these if you don't want to pass CLI args
# -----------------------------
DB_PATH = r"C:\26_internship\droplets.db"
OUT_DIR = "feature_plots"
TILES = ["center", "north", "south", "east", "west"]


# -----------------------------
# DATA LOADING
# -----------------------------
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT e.droplet_id, e.glycerol_pct, e.concentration_wv,
                   e.particle, e.session_id,
                   f.*
            FROM experiments e
            JOIN features_segmented f ON e.droplet_id = f.droplet_id
            """,
            conn,
        )
    finally:
        conn.close()

    # de-dupe the join's repeated droplet_id column
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def detect_feature_columns(df):
    """Any column starting with '<tile>_' is a tile-level feature."""
    cols = []
    for c in df.columns:
        for t in TILES:
            if c.startswith(t + "_"):
                cols.append(c)
                break
    return sorted(cols)


def feature_family(feat_name):
    """Strip the tile prefix, e.g. 'north_glcm_contrast' -> 'glcm_contrast'."""
    for t in TILES:
        if feat_name.startswith(t + "_"):
            return feat_name[len(t) + 1:]
    return feat_name


def condition_key(df):
    """Group key for a distinct experimental condition."""
    return df["glycerol_pct"].round(3).astype(str) + "_" + df["concentration_wv"].round(4).astype(str)


# -----------------------------
# PLOT 1 — feature vs glycerol grid, colored by concentration
# -----------------------------
def plot_feature_grid(df, feat_cols, out_dir):
    base_names = sorted(set(feature_family(f) for f in feat_cols))
    pdf_path = os.path.join(out_dir, "all_features_vs_glycerol.pdf")

    per_page = 4  # base feature families per page -> per_page x len(TILES) subplots/page
    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(base_names), per_page):
            chunk = base_names[i:i + per_page]
            fig, axes = plt.subplots(
                len(chunk), len(TILES),
                figsize=(3.2 * len(TILES), 2.8 * len(chunk)),
                squeeze=False,
            )
            sc = None
            for r, base in enumerate(chunk):
                for c, tile in enumerate(TILES):
                    ax = axes[r][c]
                    col = f"{tile}_{base}"
                    if col not in df.columns:
                        ax.axis("off")
                        continue
                    sub = df[["glycerol_pct", "concentration_wv", col]].dropna()
                    if sub.empty:
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                fontsize=8, transform=ax.transAxes, color="gray")
                        ax.set_xticks([]); ax.set_yticks([])
                        continue
                    sc = ax.scatter(
                        sub["glycerol_pct"], sub[col],
                        c=sub["concentration_wv"], cmap="viridis",
                        s=16, alpha=0.75, edgecolor="none",
                    )
                    if r == 0:
                        ax.set_title(tile, fontsize=10, fontweight="bold")
                    if c == 0:
                        ax.set_ylabel(base, fontsize=8)
                    if r == len(chunk) - 1:
                        ax.set_xlabel("glycerol %", fontsize=7)
                    ax.tick_params(labelsize=6)

            fig.suptitle("Feature vs Glycerol % (color = concentration w/v%)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 0.93, 0.96])
            if sc is not None:
                cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
                fig.colorbar(sc, cax=cbar_ax, label="Concentration (w/v%)")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {pdf_path}")


# -----------------------------
# PLOT 1b — condition heatmaps: x=concentration, y=glycerol%, color=feature value
# -----------------------------
def plot_condition_heatmaps(df, feat_cols, out_dir):
    """
    One heatmap per feature: rows = distinct glycerol% values (sorted),
    columns = distinct concentration values (sorted), cell color = mean
    feature value across replicates of that condition. Cells with no
    data are left gray/blank.

    Grid is built from the actual distinct values present in the data,
    so it doesn't assume a regular/complete grid — missing conditions
    just show as blank cells (which is itself useful to see: it shows
    you where your condition coverage has holes).
    """
    base_names = sorted(set(feature_family(f) for f in feat_cols))
    gly_vals = sorted(df["glycerol_pct"].dropna().unique())
    conc_vals = sorted(df["concentration_wv"].dropna().unique())

    if len(gly_vals) < 2 or len(conc_vals) < 2:
        print("Need at least 2 distinct glycerol% and concentration values for heatmaps — skipping.")
        return

    pdf_path = os.path.join(out_dir, "feature_condition_heatmaps.pdf")
    per_page = 4  # base feature families per page -> per_page x len(TILES) subplots/page

    with PdfPages(pdf_path) as pdf:
        for i in range(0, len(base_names), per_page):
            chunk = base_names[i:i + per_page]
            fig, axes = plt.subplots(
                len(chunk), len(TILES),
                figsize=(3.4 * len(TILES), 3.0 * len(chunk)),
                squeeze=False,
            )
            for r, base in enumerate(chunk):
                for c, tile in enumerate(TILES):
                    ax = axes[r][c]
                    col = f"{tile}_{base}"
                    if col not in df.columns:
                        ax.axis("off")
                        continue

                    sub = df[["glycerol_pct", "concentration_wv", col]].dropna()
                    if sub.empty:
                        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                                fontsize=8, transform=ax.transAxes, color="gray")
                        ax.set_xticks([]); ax.set_yticks([])
                        continue

                    pivot = (
                        sub.groupby(["glycerol_pct", "concentration_wv"])[col]
                        .mean()
                        .unstack("concentration_wv")
                        .reindex(index=gly_vals, columns=conc_vals)
                    )

                    masked = np.ma.masked_invalid(pivot.values)
                    cmap = plt.get_cmap("magma").copy()
                    cmap.set_bad(color="#e5e5e5")  # gray for missing conditions

                    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower")
                    ax.set_xticks(range(len(conc_vals)))
                    ax.set_xticklabels([f"{v:g}" for v in conc_vals], rotation=90, fontsize=6)
                    ax.set_yticks(range(len(gly_vals)))
                    ax.set_yticklabels([f"{v:g}" for v in gly_vals], fontsize=6)

                    if r == 0:
                        ax.set_title(tile, fontsize=10, fontweight="bold")
                    if c == 0:
                        ax.set_ylabel(f"{base}\nglycerol %", fontsize=7)
                    if r == len(chunk) - 1:
                        ax.set_xlabel("concentration (w/v%)", fontsize=7)

                    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)

            fig.suptitle("Feature value by condition (x=concentration, y=glycerol%, color=feature)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {pdf_path}")


# -----------------------------
# PLOT 2 — correlation heatmap
# -----------------------------
def plot_correlation_heatmap(df, feat_cols, out_dir):
    corr_df = df[feat_cols].dropna(axis=1, how="all")
    corr_df = corr_df.loc[:, corr_df.nunique(dropna=True) > 1]  # drop constant cols
    corr = corr_df.corr()

    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(10, 0.16 * n), max(9, 0.16 * n)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title(f"Feature-feature correlation matrix ({n} features)", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "feature_correlation_heatmap.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# -----------------------------
# PLOT 3 — missing data overview
# -----------------------------
def plot_missing_data(df, feat_cols, out_dir):
    nan_frac = df[feat_cols].isna().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.18 * len(feat_cols))))
    colors = ["#DB2777" if v > 0 else "#F9A8D4" for v in nan_frac.values]
    ax.barh(nan_frac.index, nan_frac.values, color=colors)
    ax.set_xlabel("Fraction of droplets with NaN")
    ax.set_title("Missing data by feature (tile extraction failures)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "missing_data_overview.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    worst = nan_frac[nan_frac > 0]
    if not worst.empty:
        print("\nFeatures with missing data:")
        for name, frac in worst.items():
            print(f"  {name:<35} {frac * 100:5.1f}% missing")


# -----------------------------
# PLOT 4 — between-condition vs within-condition variance
# -----------------------------
def plot_variance_by_condition(df, feat_cols, out_dir):
    df = df.copy()
    df["_cond"] = condition_key(df)

    rows = []
    for col in feat_cols:
        sub = df[["_cond", col]].dropna()
        if sub["_cond"].nunique() < 2:
            continue
        grand_var = sub[col].var()
        if not np.isfinite(grand_var) or grand_var == 0:
            continue
        group_means = sub.groupby("_cond")[col].mean()
        between_var = group_means.var()
        within_var = sub.groupby("_cond")[col].apply(lambda x: x.var()).mean()
        within_var = 0.0 if np.isnan(within_var) else within_var
        ratio = between_var / (within_var + 1e-9)
        rows.append((col, between_var, within_var, ratio))

    if not rows:
        print("Not enough distinct conditions yet to compute between/within variance.")
        return

    result = pd.DataFrame(rows, columns=["feature", "between_var", "within_var", "ratio"])
    result = result.sort_values("ratio", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.18 * len(result))))
    ax.barh(result["feature"], result["ratio"], color="#C026D3")
    ax.set_xlabel("Between-condition variance / within-condition variance")
    ax.set_title("Feature informativeness ranking (higher = more discriminative across conditions)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "feature_variance_by_condition.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

    result.to_csv(os.path.join(out_dir, "feature_variance_by_condition.csv"), index=False)
    print("\nTop 10 most discriminative features:")
    print(result.head(10).to_string(index=False))
    print("\nBottom 10 least discriminative features (candidates to drop):")
    print(result.tail(10).to_string(index=False))


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize all droplet morphology features.")
    parser.add_argument("--db", default=DB_PATH, help="Path to droplets.db")
    parser.add_argument("--out", default=OUT_DIR, help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.db):
        print(f"ERROR: database not found at {args.db}")
        print("Pass the correct path with --db \"C:\\path\\to\\droplets.db\"")
        return

    df = load_data(args.db)
    if df.empty:
        print("No droplets found in the database yet.")
        return

    feat_cols = detect_feature_columns(df)
    n_conditions = condition_key(df).nunique()
    print(f"Loaded {len(df)} droplets across {n_conditions} distinct conditions.")
    print(f"Detected {len(feat_cols)} tile-level feature columns "
          f"({len(feat_cols) // len(TILES)} feature types x {len(TILES)} tiles).")

    plot_feature_grid(df, feat_cols, args.out)
    plot_condition_heatmaps(df, feat_cols, args.out)
    plot_correlation_heatmap(df, feat_cols, args.out)
    plot_missing_data(df, feat_cols, args.out)
    plot_variance_by_condition(df, feat_cols, args.out)

    print(f"\nAll plots written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()