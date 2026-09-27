"""
SEGMENTED FEATURE EXTRACTOR (v3)
Extracts local features from 5 segments of a dried droplet image.

For users: just call extract_all_features(image_paths_dict)

Segments: center, north, east, south, west
Each segment gives 20 features = 100 total per droplet
(6 original flat-fielded features + 14 new raw-tile features).

ORIGINAL 6 (flat-fielded, per tile):
    edge_density        edge fraction in the OUTWARD (ring-side) strip
    ring_peak_contrast  ring peak intensity / interior mean (from outward profile)
    ring_width_frac     ring peak width as fraction of tile length
    profile_variance    variance of the normalized outward intensity profile
    crack_density       edge fraction in the INTERIOR (non-ring) region
    inner_deposit_frac  fraction of deposit signal in the inner half
                        (Marangoni signature: inward flow moves deposit
                        toward the interior, raising this value)

NEW 14 (computed on the RAW, non-flat-fielded tile — see note below), per tile:
    Dendritic / branching complexity:
        fractal_dimension, skeleton_branch_density,
        skeleton_endpoint_density, skeleton_length_fraction
    Clustering / connectivity:
        cluster_count_density, cluster_size_cv,
        largest_cluster_fraction, cluster_nn_distance
    GLCM texture:
        glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation
    Flow / directionality:
        void_fraction, anisotropy

NOTE ON FLAT-FIELDING: the 14 new features are deliberately computed on the
raw (non-flat-fielded) tile. Flat-fielding (dividing by a blurred copy) can
inflate fractal dimension and skeleton branch counts in low-signal regions;
until that's validated against real images, these new features skip it.
The original 6 features still use flat-fielding as before.

AUTO-ARRANGEMENT (v3 addition): when the 9 raw overlapping tiles for a
droplet have no reliable position info (no usable filename keywords),
auto_arrange_tiles() recovers the 3x3 layout via edge-strip cross-
correlation (a mini stitcher), for the labeler UI to render + let the
user verify/correct before committing to feature extraction.

FIXES vs v1:
    - Outward strip follows tile orientation (north tile -> top, etc.)
    - radial_variance replaced by a physical outward radial profile
    - Flat-field correction removes vignetting / illumination gradients
    - Auto-Canny (median-based) instead of fixed thresholds
    - Missing or unreadable tiles -> NaN (never fake zeros), consistently
    - Crack density restricted to interior, decorrelated from edge_density
    - marangoni_number() physics helper for the labeler

NOTE: feature names/count changed vs v1 and v2 -> use a fresh droplets.db,
or drop and recreate the features_segmented table.
"""

import cv2
import numpy as np
import os
from itertools import permutations
from skimage.morphology import skeletonize
from skimage.feature import graycomatrix, graycoprops, structure_tensor, structure_tensor_eigenvalues
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

FEATURES = ["edge_density", "ring_peak_contrast", "ring_width_frac",
            "profile_variance", "crack_density", "inner_deposit_frac"]

# 14 new features, four families. Computed WITHOUT flat-fielding (raw tile),
# per Jaya's noise-amplification concern with fractal/skeleton features.
FEATURES_V2 = [
    # dendritic / branching complexity
    "fractal_dimension", "skeleton_branch_density", "skeleton_endpoint_density",
    "skeleton_length_fraction",
    # clustering / connectivity
    "cluster_count_density", "cluster_size_cv", "largest_cluster_fraction",
    "cluster_nn_distance",
    # GLCM texture
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    # flow / directionality
    "void_fraction", "anisotropy",
]
SEGMENTS = ["center", "north", "south", "east", "west"]


def get_feature_names():
    return [f"{s}_{f}" for s in SEGMENTS for f in FEATURES + FEATURES_V2]


def _nan_features():
    return {f: float("nan") for f in FEATURES + FEATURES_V2}


# -----------------------------
# PRE-PROCESSING
# -----------------------------
def flat_field(img):
    img = img.astype(np.float32)

    bg = cv2.blur(img, (101, 101))

    corrected = img / (bg + 1e-6)

    corrected = cv2.normalize(
        corrected,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return corrected.astype(np.float32)


def auto_canny(img_u8):
    """Median-based Canny thresholds — robust to exposure drift."""
    med = float(np.median(img_u8))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    if hi <= lo:
        lo, hi = 50, 150
    return cv2.Canny(img_u8, lo, hi)


# -----------------------------
# AUTO-ARRANGEMENT (9 overlapping raw tiles -> 3x3 grid positions)
# Position is recovered via edge-strip cross-correlation (mini stitcher).
# Orientation is assumed fixed/consistent across tiles (no rotation search).
# Tiles are flat-fielded before correlation so illumination/vignetting
# differences between captures don't masquerade as content mismatch — this
# is separate from, and doesn't affect, the flat-fielding choices used by
# the feature families above.
# -----------------------------
GRID_ORDER = ["nw", "north", "ne", "west", "center", "east", "sw", "south", "se"]
ADJ_PAIRS = []  # (slot_a_idx, slot_b_idx, 'right' or 'down') for 3x3 row-major grid
for _r in range(3):
    for _c in range(3):
        _idx = _r * 3 + _c
        if _c < 2:
            ADJ_PAIRS.append((_idx, _idx + 1, "right"))
        if _r < 2:
            ADJ_PAIRS.append((_idx, _idx + 3, "down"))


def grid_coords(size=180):
    """Shared position -> (x, y) top-left pixel map for a rendered 3x3 grid image."""
    return {
        "nw":     (0, 0),
        "north":  (size + 10, 0),
        "ne":     (2 * size + 20, 0),

        "west":   (0, size + 10),
        "center": (size + 10, size + 10),
        "east":   (2 * size + 20, size + 10),

        "sw":     (0, 2 * size + 20),
        "south":  (size + 10, 2 * size + 20),
        "se":     (2 * size + 20, 2 * size + 20),
    }


def position_at_pixel(px, py, size=180):
    """Inverse of grid_coords: which position slot contains this pixel, or None."""
    for pos, (x, y) in grid_coords(size).items():
        if x <= px < x + size and y <= py < y + size:
            return pos
    return None


def _arrange_strip(img, side, frac):
    h, w = img.shape[:2]
    if side == "right":
        return img[:, int((1 - frac) * w):]
    if side == "left":
        return img[:, :int(frac * w)]
    if side == "bottom":
        return img[int((1 - frac) * h):, :]
    if side == "top":
        return img[:int(frac * h), :]


def _ncc_match_score(strip_a, strip_b):
    """Normalized cross-correlation between two overlap strips (already flat-fielded)."""
    a = strip_a.astype(np.float32)
    b = strip_b.astype(np.float32)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h < 8 or w < 8:
        return 0.0
    a, b = a[:h, :w], b[:h, :w]
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.mean(a * b))


def compute_pairwise_scores(images, overlap_frac=0.20):
    """
    images: list of grayscale np arrays (flat-fielded), any order.
    Returns (right_score, down_score) n x n matrices.
    right_score[i, j] = how well tile i's right edge matches tile j's left edge.
    down_score[i, j]  = how well tile i's bottom edge matches tile j's top edge.
    """
    n = len(images)
    right_score = np.zeros((n, n))
    down_score = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            right_score[i, j] = _ncc_match_score(
                _arrange_strip(images[i], "right", overlap_frac),
                _arrange_strip(images[j], "left", overlap_frac))
            down_score[i, j] = _ncc_match_score(
                _arrange_strip(images[i], "bottom", overlap_frac),
                _arrange_strip(images[j], "top", overlap_frac))
    return right_score, down_score


def auto_arrange_tiles(tile_paths, overlap_frac=0.20):
    """
    tile_paths: list of 9 file paths for one droplet's raw overlapping tiles
                (unordered, no position info in filenames).
    Returns:
        assignment: dict {position: path} using GRID_ORDER labels
        confidence: dict {position: float} average matched-edge NCC score for that slot
                    (low score -> flag for manual review; featureless or corner tiles
                    tend to score lower even when correctly placed)

    Brute-forces all 9! permutations (~362k), which is cheap at this size.
    Orientation is assumed fixed across tiles -- only position is searched.
    """
    paths = list(tile_paths)
    if len(paths) != 9:
        raise ValueError(f"auto_arrange_tiles expects exactly 9 tiles, got {len(paths)}")

    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load tile: {p}")
        images.append(flat_field(img).astype(np.uint8))

    right_score, down_score = compute_pairwise_scores(images, overlap_frac)

    best_perm = None
    best_total = -np.inf
    for perm in permutations(range(9)):
        total = 0.0
        for a_idx, b_idx, direction in ADJ_PAIRS:
            i, j = perm[a_idx], perm[b_idx]
            total += right_score[i, j] if direction == "right" else down_score[i, j]
        if total > best_total:
            best_total = total
            best_perm = perm

    assignment = {GRID_ORDER[slot]: paths[tile_i] for slot, tile_i in enumerate(best_perm)}

    confidence = {}
    for slot_idx, pos in enumerate(GRID_ORDER):
        scores = []
        for a_idx, b_idx, direction in ADJ_PAIRS:
            if a_idx == slot_idx or b_idx == slot_idx:
                i, j = best_perm[a_idx], best_perm[b_idx]
                scores.append(right_score[i, j] if direction == "right" else down_score[i, j])
        confidence[pos] = float(np.mean(scores)) if scores else 0.0

    return assignment, confidence


# -----------------------------
# RAW-TILE BINARIZATION (no flat-fielding)
# Used only by the FEATURES_V2 family, per the concern that flat-fielding
# (dividing by a blurred copy) can inflate fractal dimension / skeleton
# branch counts in low-signal regions.
# -----------------------------
def _raw_deposit_mask(img_u8):
    """Otsu threshold on the raw (unflattened) tile -> binary deposit mask."""
    if img_u8 is None or img_u8.size == 0:
        return None
    _thresh, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Deposit is usually the darker (or brighter) phase depending on imaging mode;
    # assume deposit is the minority class so the mask isn't inverted by exposure drift.
    if np.mean(mask > 0) > 0.5:
        mask = 255 - mask
    return mask > 0


# -----------------------------
# FAMILY 1 — DENDRITIC / BRANCHING COMPLEXITY
# -----------------------------
def _fractal_dimension(mask):
    """Box-counting fractal dimension of a binary mask."""
    if mask is None or mask.sum() == 0:
        return float("nan")
    h, w = mask.shape
    n = min(h, w)
    sizes = []
    size = 2
    while size <= n // 2:
        sizes.append(size)
        size *= 2
    if len(sizes) < 2:
        return float("nan")

    counts = []
    for s in sizes:
        nh, nw = h // s, w // s
        if nh == 0 or nw == 0:
            continue
        cropped = mask[:nh * s, :nw * s]
        blocks = cropped.reshape(nh, s, nw, s)
        occupied = blocks.any(axis=(1, 3))
        counts.append(max(1, int(occupied.sum())))

    if len(counts) < 2:
        return float("nan")

    log_sizes = np.log(1.0 / np.array(sizes[:len(counts)]))
    log_counts = np.log(np.array(counts))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(coeffs[0])


def _skeleton_features(mask):
    """Branch/endpoint density (per skeleton pixel) and skeleton length fraction."""
    if mask is None or mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")

    skel = skeletonize(mask)
    n_skel = int(skel.sum())
    area = mask.size
    skeleton_length_fraction = n_skel / area

    if n_skel == 0:
        return 0.0, 0.0, skeleton_length_fraction

    # 8-neighbor count at each skeleton pixel via convolution
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count = neighbor_count - skel.astype(np.uint8)  # exclude self
    neighbor_count = neighbor_count[skel]

    branch_pts = int(np.sum(neighbor_count >= 3))
    end_pts = int(np.sum(neighbor_count == 1))

    skeleton_branch_density = branch_pts / n_skel
    skeleton_endpoint_density = end_pts / n_skel
    return skeleton_branch_density, skeleton_endpoint_density, skeleton_length_fraction


# -----------------------------
# FAMILY 2 — CLUSTERING / CONNECTIVITY
# -----------------------------
def _cluster_features(mask):
    if mask is None or mask.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    # label 0 is background
    sizes = stats[1:, cv2.CC_STAT_AREA]
    n_clusters = len(sizes)
    area = mask.size

    if n_clusters == 0:
        return 0.0, float("nan"), float("nan"), float("nan")

    cluster_count_density = n_clusters / area
    cluster_size_cv = float(np.std(sizes) / (np.mean(sizes) + 1e-9)) if n_clusters > 1 else 0.0
    largest_cluster_fraction = float(np.max(sizes) / (np.sum(sizes) + 1e-9))

    if n_clusters > 1:
        pts = centroids[1:]
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)  # k=1 is self (dist 0)
        cluster_nn_distance = float(np.mean(dists[:, 1]))
    else:
        cluster_nn_distance = float("nan")

    return cluster_count_density, cluster_size_cv, largest_cluster_fraction, cluster_nn_distance


# -----------------------------
# FAMILY 3 — GLCM TEXTURE
# -----------------------------
def _glcm_features(img_u8, levels=16):
    if img_u8 is None or img_u8.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Quantize to fewer gray levels for a well-conditioned co-occurrence matrix
    quantized = (img_u8.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    quantized = np.clip(quantized, 0, levels - 1)

    glcm = graycomatrix(quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=levels, symmetric=True, normed=True)

    contrast    = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy      = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    return contrast, homogeneity, energy, correlation


# -----------------------------
# FAMILY 4 — FLOW / DIRECTIONALITY
# -----------------------------
def _void_fraction(mask):
    if mask is None:
        return float("nan")
    return float(1.0 - np.mean(mask))


def _anisotropy(img_u8):
    """Structure-tensor anisotropy: (l1 - l2) / (l1 + l2), averaged over the tile."""
    if img_u8 is None or img_u8.size == 0:
        return float("nan")
    img_f = img_u8.astype(np.float64)
    Axx, Axy, Ayy = structure_tensor(img_f, sigma=1.5)
    l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
    denom = l1 + l2
    valid = denom > 1e-9
    if not np.any(valid):
        return float("nan")
    aniso = (l1[valid] - l2[valid]) / denom[valid]
    return float(np.mean(aniso))


def extract_v2_features(img_u8):
    """
    Compute the 14 new features on a RAW (non-flat-fielded) grayscale tile.
    Deliberately skips flat-fielding here, per the concern that it can
    inflate fractal dimension / skeleton branch counts in low-signal regions.
    """
    if img_u8 is None or img_u8.size == 0 or img_u8.ndim != 2:
        return {f: float("nan") for f in FEATURES_V2}

    mask = _raw_deposit_mask(img_u8)

    fractal_dimension = _fractal_dimension(mask)
    skel_branch, skel_end, skel_len_frac = _skeleton_features(mask)
    clus_count, clus_cv, clus_largest, clus_nn = _cluster_features(mask)
    glcm_contrast, glcm_homog, glcm_energy, glcm_corr = _glcm_features(img_u8)
    void_frac = _void_fraction(mask)
    aniso = _anisotropy(img_u8)

    return {
        "fractal_dimension":        fractal_dimension,
        "skeleton_branch_density":  skel_branch,
        "skeleton_endpoint_density": skel_end,
        "skeleton_length_fraction": skel_len_frac,
        "cluster_count_density":    clus_count,
        "cluster_size_cv":          clus_cv,
        "largest_cluster_fraction": clus_largest,
        "cluster_nn_distance":      clus_nn,
        "glcm_contrast":            glcm_contrast,
        "glcm_homogeneity":         glcm_homog,
        "glcm_energy":              glcm_energy,
        "glcm_correlation":         glcm_corr,
        "void_fraction":            void_frac,
        "anisotropy":               aniso,
    }


# -----------------------------
# ORIENTATION HELPERS
# -----------------------------
def outward_strip(arr, position, frac=0.2):
    """The 20% of the tile facing the ring edge (drop perimeter)."""
    h, w = arr.shape[:2]
    if position == "north":
        return arr[:int(frac * h), :]
    if position == "south":
        return arr[int((1 - frac) * h):, :]
    if position == "east":
        return arr[:, int((1 - frac) * w):]
    if position == "west":
        return arr[:, :int(frac * w)]
    # center tile has no ring side; use a perimeter frame
    m_h, m_w = int(frac * h), int(frac * w)
    mask = np.ones((h, w), bool)
    mask[m_h:h - m_h, m_w:w - m_w] = False
    return arr[mask]


def interior_region(arr, position, frac=0.6):
    """The inward 60% of the tile (away from the ring edge)."""
    h, w = arr.shape[:2]
    if position == "north":
        return arr[int((1 - frac) * h):, :]
    if position == "south":
        return arr[:int(frac * h), :]
    if position == "east":
        return arr[:, :int(frac * w)]
    if position == "west":
        return arr[:, int((1 - frac) * w):]
    # center tile: middle 60%
    m_h, m_w = int((1 - frac) / 2 * h), int((1 - frac) / 2 * w)
    return arr[m_h:h - m_h, m_w:w - m_w]


def outward_profile(img, position):
    """
    1D intensity profile along the outward radial direction.
    Index 0 = innermost (toward drop center), last = outermost (ring edge).
    For the center tile, an azimuthally averaged radial profile is used.
    """
    if position == "north":
        return img.mean(axis=1)[::-1]
    if position == "south":
        return img.mean(axis=1)
    if position == "east":
        return img.mean(axis=0)
    if position == "west":
        return img.mean(axis=0)[::-1]
    # center: radial profile about the tile center
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    return tbin / (nr + 1e-8)


# -----------------------------
# LOCAL FEATURE EXTRACTION
# -----------------------------
def extract_local_features(img_crop, position):
    """
    Extract 20 dimensionless features from one tile/crop.
    Returns NaNs if the crop is unusable.
    """
    if img_crop is None or img_crop.size == 0 or img_crop.ndim != 2:
        return _nan_features()

    img = flat_field(img_crop)
    img_u8 = img.astype(np.uint8)
    h, w = img.shape

    # --- edge density in the ring-side strip
    edges = auto_canny(img_u8)
    strip = outward_strip(edges, position)
    edge_density = float(np.mean(strip > 0)) if strip.size else float("nan")

    # --- crack density in the interior only (decorrelated from ring edge)
    interior = interior_region(img_u8, position)
    crack_density = (float(np.mean(auto_canny(interior) > 0))
                     if interior.size else float("nan"))

    # --- outward intensity profile features
    p = outward_profile(img, position)
    if p.size < 10:
        return _nan_features()

    n = p.size
    inner = p[:n // 2]
    outer = p[n // 2:]
    interior_mean = float(inner.mean()) + 1e-6

    # ring peak: strongest deviation from interior level in the outer half
    dev = np.abs(outer - interior_mean)
    peak_idx = int(np.argmax(dev))
    peak_val = float(outer[peak_idx])
    ring_peak_contrast = peak_val / interior_mean

    # ring width: extent where outer-half deviation exceeds half the peak's
    half = dev[peak_idx] / 2.0
    above = dev > half
    # contiguous run containing the peak
    left = peak_idx
    while left > 0 and above[left - 1]:
        left -= 1
    right = peak_idx
    while right < len(above) - 1 and above[right + 1]:
        right += 1
    ring_width_frac = float(right - left + 1) / n

    # profile variance (normalized -> exposure invariant)
    p_norm = p / (p.mean() + 1e-6)
    profile_variance = float(np.var(p_norm))

    # --- Marangoni signature: deposit fraction in the inner half.
    # Inward (Marangoni) recirculation moves particles toward the interior;
    # strong coffee-ring outward flow concentrates signal in the outer half.
    signal = np.abs(p - np.median(p))
    total = float(signal.sum()) + 1e-6
    inner_deposit_frac = float(signal[:n // 2].sum()) / total

    # --- 14 new features, computed on the RAW (non-flat-fielded) crop
    v2 = extract_v2_features(img_crop if img_crop.dtype == np.uint8 else img_crop.astype(np.uint8))

    return {
        "edge_density":       edge_density,
        "ring_peak_contrast": ring_peak_contrast,
        "ring_width_frac":    ring_width_frac,
        "profile_variance":   profile_variance,
        "crack_density":      crack_density,
        "inner_deposit_frac": inner_deposit_frac,
        **v2,
    }


# -----------------------------
# 5 IMAGES -> 100 FEATURES
# -----------------------------
def extract_from_5_images(image_paths):
    """
    image_paths: dict with keys center/north/south/east/west -> file path.
    Missing or unreadable tiles produce NaN features (never zeros).
    Returns flat dict of 100 features (20 per tile x 5 tiles).
    """
    all_features = {}
    for pos in SEGMENTS:
        path = image_paths.get(pos)
        img = None
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  WARNING: cannot load {pos} image: {path}")
        else:
            print(f"  WARNING: missing {pos} image -> NaN features")

        feats = extract_local_features(img, pos)
        for name, val in feats.items():
            all_features[f"{pos}_{name}"] = val
    return all_features


# -----------------------------
# SINGLE IMAGE -> 5 SEGMENTS
# -----------------------------
def get_segments(h, w):
    """Crop coordinates for 5 segments of a single whole-drop image."""
    cy, cx = h // 2, w // 2
    qh, qw = h // 4, w // 4
    return {
        "center": (cy - qh, cy + qh, cx - qw, cx + qw),
        "north":  (0,       h // 2,  cx - qw, cx + qw),
        "south":  (h // 2,  h,       cx - qw, cx + qw),
        "east":   (cy - qh, cy + qh, w // 2,  w),
        "west":   (cy - qh, cy + qh, 0,       w // 2),
    }


def extract_segments_from_image(image_path):
    """
    One whole-drop image (e.g. literature figure) -> 100 features.
    NOTE: segments overlap; feature statistics differ from 5-tile mode.
    Keep data_source distinct in the DB.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    h, w = img.shape
    all_features = {}
    for seg, (r1, r2, c1, c2) in get_segments(h, w).items():
        feats = extract_local_features(img[r1:r2, c1:c2], seg)
        for name, val in feats.items():
            all_features[f"{seg}_{name}"] = val
    return all_features


# -----------------------------
# MAIN USER-FACING FUNCTION
# -----------------------------
def extract_all_features(input_data):
    """
    Universal entry point:
        extract_all_features("path/to/ring.jpg")        # single image
        extract_all_features({"center": ..., ...})       # 5 tiles
    """
    if isinstance(input_data, str):
        return extract_segments_from_image(input_data)
    if isinstance(input_data, dict):
        return extract_from_5_images(input_data)
    raise ValueError("Input must be an image path (str) or dict of 5 paths")


# -----------------------------
# MARANGONI PHYSICS
# (experiment-level quantity — call from lab_labeler, store per droplet)
# -----------------------------
SIGMA_WATER = 71.99e-3     # N/m at 25 C
SIGMA_GLYCEROL = 63.4e-3   # N/m at 25 C
D_GLYCEROL_WATER = 0.95e-9 # m^2/s, glycerol diffusivity in water at 25 C


def marangoni_number(glycerol_mass_frac, h0_m, viscosity_pa_s):
    """
    Solutal Marangoni number for a drying water-glycerol droplet:

        Ma = delta_sigma * h0 / (eta * D_glycerol)

    delta_sigma: surface tension difference between pure water and the
    mixture (water evaporates preferentially at the contact line, locally
    enriching glycerol and lowering sigma there -> inward Marangoni flow
    that recirculates particles and suppresses the coffee ring).

    Ma >> 1 : Marangoni recirculation competes with radial flow
    Ma ~ 0  : pure coffee-ring behaviour (no glycerol)

    Args:
        glycerol_mass_frac: Cm in [0, 1] (use glycerol_mass_fraction())
        h0_m:               initial droplet apex height (spherical cap), m
        viscosity_pa_s:     mixture viscosity (Cheng 2008), Pa.s
    """
    sigma_mix = SIGMA_WATER - (SIGMA_WATER - SIGMA_GLYCEROL) * glycerol_mass_frac
    delta_sigma = SIGMA_WATER - sigma_mix
    return delta_sigma * h0_m / (viscosity_pa_s * D_GLYCEROL_WATER + 1e-30)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        feats = extract_all_features(sys.argv[1])
        print("\n100 LOCAL FEATURES:")
        for k, v in feats.items():
            print(f"  {k:<35} {v:.6f}")
    else:
        print("Usage: python feature_extractor.py path/to/image.jpg")
        print("\nFeature names (100 total):")
        for n in get_feature_names():
            print(f"  {n}")