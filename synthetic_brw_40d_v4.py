"""
SYNTHETIC MORPHODYNAMIC TIME-SERIES GENERATOR (v4 -- 3D biased random walk,
non-permanent inter-particle collisions)

Purpose
-------
Generate synthetic drying-droplet trajectories using the stochastic Monte-Carlo /
biased-random-walk (BRW) framework of:

    Crivoi & Duan, "Three-dimensional Monte Carlo model of the coffee-ring
    effect in evaporating colloidal droplets", Scientific Reports 4, 4310 (2014).

CHANGE FROM v3: BLOCKING RULE FIX (rush-hour depletion bug)
-------------------------------------------------------------------------------
Diagnosed via two instrumented diagnostic scripts (not included here):

  1. A mobility diagnostic showed ~97% of particles were already in the
     permanent `blocked` set by t~=0.6-0.7, well before the flow field's
     genuine late-time divergence (vmax rising into the tens of thousands
     as t->1, confirmed even WITHOUT any contact-angle-floor change).
  2. A transport-vs-feature diagnostic confirmed the edgeMean_edge_density
     feature plateau tracks the RAW particle count near the edge almost
     exactly -- i.e. the plateau is a genuine transport bottleneck, not a
     feature-extraction artifact.

Cross-checking against the primary literature (Marin, Gelderblom, Lohse,
Snoeijer, PRL 107, 085502 (2011) and the companion Phys. Fluids 23, 091111
(2011) paper) confirmed the "rush hour" effect specifically requires an
ongoing SUPPLY of still-in-transit particles that keep arriving late, at
high velocity, too fast to crystallize ("an avalanche of particles being
dragged in the last moments"). If ~97% of particles have already been
immobilized by t~0.6-0.7, there is no remaining population left for the
late-time flow divergence to act on -- the mechanism the velocity field
correctly encodes has nothing left to move.

ROOT CAUSE: v3's collision handling conflated two physically different
events into one PERMANENT blocking rule:

    (a) a transient inter-particle collision -- a proposed move lands on
        a cell already occupied by another particle, which can happen
        ANYWHERE in the bulk, not just at the ring. Physically this
        should just fail that one move (the particle stays mobile and
        tries again next step), the same way a real particle jostling
        past a neighbor in suspension doesn't get permanently stuck.
    (b) genuine substrate/contact-line pinning -- a particle actually
        reaches the pinned contact line at the substrate (z near 0, at
        the outer edge of the shrinking cap) and physically deposits.
        This IS supposed to be permanent -- it's literal coffee-ring
        deposition.

v3 treated (a) exactly like (b): ANY inter-particle collision, even deep
in the bulk, permanently added the whole cluster to `blocked`. That
artificially exhausted the mobile particle population several timesteps
after t=0, long before the physically real late-stage rush-hour window,
because ordinary in-transit jostling was being punished as if it were
contact-line arrest.

FIX: `_resolve_cluster_move()` now distinguishes the two cases explicitly.
Inter-particle collisions (a) simply cause the move to be skipped for this
step -- the cluster remains in `particles`, NOT in `blocked`, and gets a
fresh chance to move (possibly in a different direction) on the next MCS.
Only genuine contact-line pinning (b) -- leaving the liquid domain at a
position near the substrate (see NEAR_SUBSTRATE_Z) -- adds the cluster to
`blocked` permanently, matching the actual physical deposition mechanism.

Leaving the liquid domain elsewhere (i.e. through the shrinking free
surface, away from the substrate) is ALSO now treated as (a) -- a failed
move, not a pin -- since that is a geometric consequence of the cap
shrinking faster than a particle's step, not physical contact-line arrest.

Everything else (BRW architecture, velocity field, sticking/cluster logic,
image rendering, feature extraction hookup) is unchanged from v3.

This is a research simulator, not a claim of exact reproduction of every
closed-form coefficient in the paper's Hu-Larson velocity field. The BRW
architecture, shrinking spherical-cap domain, velocity bias, waiting move,
boundary blocking and sticking/cluster logic follow the paper.
"""

from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from feature_extractor import (
    extract_from_5_images,
    FEATURES,
    FEATURES_V2,
)
from lab_labeler import calculate_droplet_physics, DB_PATH


# =============================================================================
# CONFIG
# =============================================================================

OUT_ROOT = r"C:\26_internship\synthetic_droplets_brw"
CSV_OUT = r"C:\26_internship\synthetic_trajectories.csv"

# Same nominal physical assumptions used by the previous synthesizer.
CONTACT_ANGLE_DEG = 10.0
R_UM = 2000.0
EVAP_TIME_S = 1800.0
DROP_VOL_UL = 2.0

# Image / lattice resolution.
IMG_SIZE = 600
TILE_SIZE = 180

# Paper uses a 500-cell base diameter. We use a smaller scaled lattice so
# trajectories remain practical. Increase this for higher-fidelity runs.
BASE_RADIUS_CELLS = 80
LATTICE_H0 = 40.0

# Particle population. If None, derive from concentration and cap volume.
MAX_PARTICLES = 5000
MIN_PARTICLES = 250

# Monte-Carlo controls.
MAX_MCS = 6000
SNAPSHOT_TIMES = np.linspace(0.001, 1.00, 1000)

# Particle diffusion / random-walk strength. Larger -> less deterministic drift.
DIFFUSION_STRENGTH = 0.35

# DLA-like sticking parameter. The paper varies this and reports stronger
# clustering at higher concentration and sticking.
PSTICK_BASE = 0.10

# Extra image-level measurement noise is intentionally small because the
# stochasticity should primarily come from the BRW itself.
# Set to 0.0: this noise was cosmetic image realism, not physically
# informative, and adds pure variance to features (fractal dimension,
# GLCM, skeleton stats) with no compensating signal. Turn back on only if
# you specifically want to stress-test feature robustness to sensor noise.
IMAGE_NOISE = 0.0

# Ensemble averaging: number of independent BRW realizations run per
# (glycerol, concentration, particle_size) condition, averaged at each
# snapshot time before saving. A single BRW trajectory is a random walk --
# individual particle positions jitter between snapshots at these particle
# counts, so single-realization M(t) is dominated by Monte Carlo variance,
# not the underlying physical trend (same reason you'd never trust one
# real droplet over replicates). N_REPLICATES=1 reproduces the old
# (noisy) single-realization behavior.
N_REPLICATES = 20

# If true, render a separate image at every requested snapshot.
WRITE_IMAGES = True

# NEW: how close to the substrate (z=0) a particle must be for leaving the
# liquid domain to count as genuine contact-line pinning (permanent) rather
# than a transient bulk/free-surface collision (move just fails, stays
# mobile). 1 lattice cell above the substrate is the pinning zone -- real
# contact-line deposition happens essentially AT the substrate, not in the
# middle of the drop's shrinking free surface.
NEAR_SUBSTRATE_Z = 1


# =============================================================================
# 40D FEATURE DEFINITION
# =============================================================================

FEATURES_20 = list(FEATURES) + list(FEATURES_V2)

assert len(FEATURES) == 6, f"Expected 6 original features, got {len(FEATURES)}"
assert len(FEATURES_V2) == 14, f"Expected 14 V2 features, got {len(FEATURES_V2)}"
assert len(FEATURES_20) == 20


def feature_names_40() -> List[str]:
    """
    Stable 40D order:
        center_<20 features>
        edgeMean_<20 features>
    """
    return (
        [f"center_{f}" for f in FEATURES_20]
        + [f"edgeMean_{f}" for f in FEATURES_20]
    )


def reduce_to_40d(features_100d: Dict[str, float]) -> Dict[str, float]:
    """
    FIXED 40D reducer.

    Previous code used only FEATURES (6), producing 12 variables.
    This version includes all 20 features:
        6 original + 14 new raw-tile features.

    20 center + 20 mean(N,S,E,W) = 40.
    """
    out = {}

    for f in FEATURES_20:
        out[f"center_{f}"] = float(features_100d.get(f"center_{f}", np.nan))

    for f in FEATURES_20:
        vals = [
            features_100d.get(f"{pos}_{f}", np.nan)
            for pos in ("north", "south", "east", "west")
        ]
        vals = np.asarray(vals, dtype=float)
        out[f"edgeMean_{f}"] = (
            float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        )

    assert len(out) == 40
    return out


# =============================================================================
# SPHERICAL-CAP DOMAIN
# =============================================================================

def contact_angle_at_t(t: float) -> float:
    """Linear angle reduction used as a practical shrinking-cap approximation."""
    theta0 = math.radians(CONTACT_ANGLE_DEG)
    theta = theta0 * max(1.0 - t, 0.0)
    return max(theta, math.radians(0.25))


def cap_height_at_t(t: float) -> float:
    """Paper assumption: apex height decreases approximately linearly."""
    return LATTICE_H0 * max(1.0 - t, 0.0)


def cap_radius_at_z(z: float, t: float) -> float:
    """
    Radius of a spherical-cap cross-section at height z.

    Coordinates:
        substrate z=0
        apex z=h(t)

    For a shallow cap this gives the expected radius -> R at z=0 and
    radius -> 0 at the apex.
    """
    h = cap_height_at_t(t)
    if h <= 0 or z < 0 or z > h:
        return 0.0

    R = float(BASE_RADIUS_CELLS)
    # Spherical cap relation using sphere radius a = (R^2 + h^2)/(2h)
    sphere_R = (R * R + h * h) / (2.0 * h)
    zc = h - sphere_R
    rr2 = max(sphere_R * sphere_R - (z - zc) ** 2, 0.0)
    return math.sqrt(rr2)


def inside_liquid(x: int, y: int, z: int, t: float) -> bool:
    """Check whether a lattice cell lies inside the shrinking spherical cap."""
    h = cap_height_at_t(t)
    if h <= 0:
        return False
    if z < 0 or z > math.ceil(h):
        return False

    rho = math.sqrt(x * x + y * y)
    return rho <= cap_radius_at_z(float(z), t) + 0.5


def sample_initial_particles(concentration_wv: float, rng: np.random.Generator):
    """
    Uniform random particle placement inside the initial 3-D cap.

    concentration_wv is interpreted as a volume fraction, e.g. 0.01 = 1%.
    The lattice is deliberately scaled down from the paper's 500-cell base
    diameter, so a population cap is applied for tractability.
    """
    R = BASE_RADIUS_CELLS
    h = LATTICE_H0
    sphere_R = (R * R + h * h) / (2.0 * h)
    zc = h - sphere_R

    # Continuous cap volume in lattice-cell units.
    volume = math.pi * h * (3 * R * R + h * h) / 6.0
    target = int(round(volume * concentration_wv))

    target = max(MIN_PARTICLES, min(MAX_PARTICLES, target))

    particles = set()

    # Rejection sample uniformly over a bounding cylinder.
    attempts = 0
    max_attempts = target * 80 + 10000

    while len(particles) < target and attempts < max_attempts:
        attempts += 1
        x = int(rng.integers(-R, R + 1))
        y = int(rng.integers(-R, R + 1))
        z = int(rng.integers(0, max(1, int(h)) + 1))

        if inside_liquid(x, y, z, 0.0):
            particles.add((x, y, z))

    if len(particles) < max(10, target // 2):
        raise RuntimeError(
            f"Could only place {len(particles)} of {target} particles; "
            "increase lattice size or reduce concentration."
        )

    return particles


# =============================================================================
# FLOW FIELD
# =============================================================================

def deegan_lambda(theta: float) -> float:
    """
    Evaporative-flux singularity exponent near the pinned contact line,
    J(r) ~ (1 - (r/R)^2)^(-lambda), for contact angle theta < pi/2.

    lambda(theta) = (pi - 2*theta) / (2*(pi - theta))

    Checked against two known limits before use:
      theta -> 0     : lambda -> 0.5  (fully-wetting limit)
      theta -> pi/2  : lambda -> 0    (hemispherical cap has UNIFORM,
                                        non-singular flux -- confirmed
                                        directly in the sessile-droplet
                                        evaporation literature)

    VERIFY BEFORE TRUSTING FOR PUBLICATION: multiple papers state visually
    similar-looking exponent formulas with different normalizations/sign
    conventions (e.g. some reviews write lambda = pi/(2*pi - 2*theta),
    which does NOT satisfy the theta=pi/2 uniform-flux limit and appears
    to use a different definition). This implementation was chosen because
    it satisfies both known limits above, but confirm directly against
    Deegan, Bakajin, Dupont, Huber, Nagel, Witten, "Contact line deposits
    in an evaporating drop," Phys. Rev. E 62, 756 (2000) -- the exponent
    is derived there (see also Popov, Phys. Rev. E 71, 036313 (2005) for
    the full spherical-cap solution) -- before citing this exact form.

    Replaces the previous ad hoc approximation
        lam = max(0.05, 0.5 - theta/pi)
    which matched the theta->0 limit exactly but had roughly half the
    correct slope (i.e. underestimated how quickly the singularity weakens
    as contact angle grows).
    """
    theta = min(theta, math.pi / 2 - 1e-6)  # formula only valid for theta < pi/2
    lam = (math.pi - 2.0 * theta) / (2.0 * (math.pi - theta))
    return max(lam, 0.0)


def flow_velocity(
    x: float,
    y: float,
    z: float,
    t: float,
    Pe: float,
    Ma: float,
) -> Tuple[float, float, float]:
    """
    Dimensionless particle drift field.

    The paper computes velocity components from the Hu-Larson analytical
    flow field and converts them into move probabilities. Here the field is
    kept in a numerically stable form while preserving the physical structure:

      * outward capillary drift increases strongly toward r -> R
      * the drift depends on height in the cap
      * an optional inward Marangoni component opposes outward transport
      * vertical motion closes the recirculation

    The BRW layer below is the important stochastic part: velocities are
    converted to six directional probabilities plus a WAIT probability.
    """
    eps = 1e-6
    R = float(BASE_RADIUS_CELLS)
    h = max(cap_height_at_t(t), 1e-3)

    r = math.sqrt(x * x + y * y)
    rn = min(r / R, 0.9995)
    zn = min(max(z / h, 0.0), 1.0)

    theta = contact_angle_at_t(t)
    lam = max(deegan_lambda(theta), 0.05)  # floor kept to avoid lam->0 killing the singularity entirely

    # Dimensionless evaporative/capillary amplification near contact line.
    edge_factor = max(1.0 - rn * rn, eps) ** (-lam)
    time_factor = 1.0 / max(1.0 - t, 0.03)

    # Suppress the radial field at the exact center and apex.
    capillary = rn * edge_factor * (0.35 + 0.65 * zn) * time_factor

    # Compress the huge Pe range rather than letting one parameter dominate.
    pe_scale = math.log1p(max(Pe, 0.0)) / 8.0
    pe_scale = float(np.clip(pe_scale, 0.05, 3.0))

    vr_out = capillary * pe_scale

    # Optional inward Marangoni recirculation. It is strongest away from
    # center/edge and strongest in the upper/middle part of the drop.
    ma_scale = math.log1p(max(Ma, 0.0)) / 8.0
    ma_scale = float(np.clip(ma_scale, 0.0, 3.0))

    marangoni_shape = rn * (1.0 - rn) * (0.25 + 0.75 * zn)
    vr = vr_out - ma_scale * marangoni_shape

    # Convert radial velocity to x/y.
    if r > eps:
        vx = vr * x / r
        vy = vr * y / r
    else:
        vx = 0.0
        vy = 0.0

    # Downward component from shrinking cap + recirculating return flow.
    # Outward transport near the substrate is compensated by upward motion
    # in the inner region, giving a simple closed recirculation pattern.
    vz = (
        -0.35 * (1.0 + 2.0 * t) * (0.2 + 0.8 * rn)
        + 0.45 * ma_scale * (1.0 - rn) * (0.5 - zn)
    )

    return float(vx), float(vy), float(vz)


# =============================================================================
# PAPER-STYLE SEVEN-WAY BIASED RANDOM WALK
# =============================================================================

MOVE_NAMES = ("x+", "x-", "y+", "y-", "z+", "z-", "wait")


def move_probabilities(
    velocity: Tuple[float, float, float],
    vmax: float,
    diffusion_strength: float = DIFFUSION_STRENGTH,
):
    """
    Convert velocity into six directional probabilities + WAIT.

    This follows the logic described by Crivoi & Duan:
      - directional probabilities contain an unbiased 1/(2N) component
      - drift adds/subtracts according to the velocity projection
      - probabilities are scaled by the particle speed relative to the
        fastest mobile particle
      - the remaining probability is a WAIT move

    This is numerically equivalent to:
        q = |v| / vmax
        p(direction) = q * [1/(2N) +/- v_i/(2N |v|)]
        p(wait) = 1 - q

    A small diffusion-strength factor controls how strongly the directional
    distribution follows the drift.
    """
    vx, vy, vz = velocity
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    if vmax <= 1e-12 or speed <= 1e-12:
        return np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=float)

    q = float(np.clip(speed / vmax, 0.0, 1.0))

    # Unit drift direction.
    ux, uy, uz = vx / speed, vy / speed, vz / speed

    # 3 axes, two signs each. Unbiased component is 1/6.
    # diffusion_strength -> 1 means stronger Brownian component;
    # lower values make the walk more drift-biased.
    drift_weight = float(np.clip(1.0 - diffusion_strength, 0.0, 1.0))
    diff_weight = 1.0 - drift_weight

    directional = np.array(
        [
            1/6 + drift_weight * ux / 6,
            1/6 - drift_weight * ux / 6,
            1/6 + drift_weight * uy / 6,
            1/6 - drift_weight * uy / 6,
            1/6 + drift_weight * uz / 6,
            1/6 - drift_weight * uz / 6,
        ],
        dtype=float,
    )

    # Ensure the six components remain valid even under floating-point noise.
    directional = np.clip(directional, 0.0, None)

    # q controls the amount of actual motion during this MCS; the rest waits.
    p_move = q * directional

    # Diffusion_strength adds a small isotropic component without destroying
    # the paper's wait mechanism.
    if diff_weight > 0:
        isotropic = q * diff_weight / 6.0
        p_move = (1.0 - diff_weight) * p_move + isotropic

    p_wait = max(0.0, 1.0 - float(p_move.sum()))

    probs = np.concatenate([p_move, [p_wait]])
    probs /= probs.sum()

    return probs


# =============================================================================
# PARTICLE / CLUSTER UTILITIES
# =============================================================================

NEIGHBOURS_6 = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def _cluster_neighbors(pos, occupancy):
    x, y, z = pos
    for dx, dy, dz in NEIGHBOURS_6:
        q = (x + dx, y + dy, z + dz)
        if q in occupancy:
            yield q


def stick_particles(
    particles: set,
    blocked: set,
    p_stick: float,
    dt: float,
    rng: np.random.Generator,
):
    """
    DLA-style local sticking.

    We use the paper's continuous-time interpretation:
        P = 1 - exp(-p_stick * dt)

    Once two neighboring particles stick, they are represented by the same
    cluster id in cluster_of. Cluster motion is handled by moving all members
    together using the cluster centroid.
    """
    if not particles:
        return {}

    occupancy = set(particles)
    parent = {p: p for p in particles}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    p_event = 1.0 - math.exp(-max(p_stick, 0.0) * max(dt, 0.0))

    # Sparse local search. We only inspect each positive-direction pair once.
    for p in list(particles):
        x, y, z = p
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            q = (x + dx, y + dy, z + dz)
            if q in occupancy and rng.random() < p_event:
                union(p, q)

    cluster_of = {}
    for p in particles:
        cluster_of[p] = find(p)

    return cluster_of


def cluster_groups(cluster_of: Dict[Tuple[int, int, int], Tuple[int, int, int]]):
    groups = {}
    for p, root in cluster_of.items():
        groups.setdefault(root, []).append(p)
    return groups


# =============================================================================
# ONE BRW SIMULATION
# =============================================================================

def _is_near_substrate(members) -> bool:
    """
    True if any member of the cluster is within NEAR_SUBSTRATE_Z of the
    substrate (z=0). Used to distinguish genuine contact-line pinning from
    a transient collision/free-surface encounter elsewhere in the bulk.
    """
    return any(p[2] <= NEAR_SUBSTRATE_Z for p in members)


def simulate_brw(
    concentration_wv: float,
    Pe: float,
    Ma: float,
    p_stick: float,
    rng: np.random.Generator,
    snapshots: np.ndarray = SNAPSHOT_TIMES,
):
    """
    Run one full drying trajectory.

    Returns:
        {snapshot_t: set((x,y,z), ...)} for the particle positions.
    """
    particles = sample_initial_particles(concentration_wv, rng)
    blocked = set()

    snapshot_map = {}
    next_snap = 0
    t = 0.0

    # A practical starting time step. Actual dt is adjusted from vmax,
    # following the paper's fastest-particle clock logic.
    while t < 1.0 - 1e-12 and next_snap < len(snapshots):
        mobile = [p for p in particles if p not in blocked]

        # At the very end, immobilize everything.
        if not mobile:
            while next_snap < len(snapshots):
                snapshot_map[float(snapshots[next_snap])] = set(particles)
                next_snap += 1
            break

        # Exclude blocked/contact-line particles from vmax, as in the paper.
        velocities = [
            flow_velocity(*p, t, Pe, Ma)
            for p in mobile
        ]
        speeds = np.asarray(
            [math.sqrt(vx*vx + vy*vy + vz*vz) for vx, vy, vz in velocities],
            dtype=float,
        )
        vmax = float(np.max(speeds)) if len(speeds) else 1.0

        # Avoid dt -> 0 divergence near the contact line.
        vmax_eff = max(vmax, 1e-4)
        dt = min(0.004, 0.08 / vmax_eff)
        dt = max(dt, 1e-5)

        # Do not step past the next requested snapshot.
        if next_snap < len(snapshots):
            dt = min(dt, max(float(snapshots[next_snap]) - t, 1e-6))
        dt = min(dt, 1.0 - t)

        # Form clusters before movement.
        cluster_of = stick_particles(
            particles,
            blocked,
            p_stick=p_stick,
            dt=dt,
            rng=rng,
        )
        groups = cluster_groups(cluster_of)

        new_particles = set(particles)
        new_blocked = set(blocked)

        # Move one cluster at a time. Singleton particles are simply
        # one-member clusters.
        occupied_old = set(particles)

        for root, members in groups.items():
            if all(p in blocked for p in members):
                continue

            # Use mean position for cluster flow, as in the paper.
            cx = float(np.mean([p[0] for p in members]))
            cy = float(np.mean([p[1] for p in members]))
            cz = float(np.mean([p[2] for p in members]))

            vx, vy, vz = flow_velocity(cx, cy, cz, t, Pe, Ma)
            probs = move_probabilities(
                (vx, vy, vz),
                vmax=vmax,
                diffusion_strength=DIFFUSION_STRENGTH,
            )

            choice = int(rng.choice(7, p=probs))
            if choice == 6:
                continue

            dx, dy, dz = (
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            )[choice]

            proposed = [(p[0] + dx, p[1] + dy, p[2] + dz) for p in members]

            # If a particle is outside the shrinking liquid, only vertical
            # downward motion is allowed, matching the paper's post-evaporation
            # gravity rule.
            if any(not inside_liquid(*p, t) for p in members):
                if choice != 5:  # z-
                    continue

            # Boundary barrier condition. Always a hard stop (can't go
            # below the substrate) but NOT a permanent pin by itself --
            # just skip this move and stay mobile, unless the cluster is
            # already sitting at the substrate (see contact-line check
            # below), in which case it's genuine deposition.
            if any(p[2] < 0 for p in proposed):
                continue

            # --- FIX: separate transient collisions from genuine
            # contact-line pinning ---
            #
            # v3 treated ANY of the following as a PERMANENT block:
            #   (a) proposed cell occupied by another particle (bulk
            #       collision, can happen anywhere)
            #   (b) proposed cell outside the shrinking liquid domain
            #       (can happen anywhere the free surface recedes past
            #       a particle, not just at the substrate)
            #
            # That conflated ordinary in-transit jostling with real
            # coffee-ring deposition, and was diagnosed as the cause of
            # the mobile population being exhausted by t~0.6-0.7 --
            # long before the physically real late-stage rush-hour
            # window this project is trying to fit against.
            #
            # Now: only (b) occurring NEAR THE SUBSTRATE counts as
            # genuine contact-line pinning (permanent). Everything else
            # -- bulk particle-particle collisions, or leaving the
            # domain away from the substrate (the shrinking free
            # surface catching up to a particle mid-drop) -- just fails
            # this move. The cluster stays mobile and gets a fresh
            # chance on the next MCS, the same way a real particle in
            # suspension keeps jostling until it actually reaches the
            # pinned contact line.
            bulk_collision = False
            for p_new in proposed:
                if p_new in occupied_old and p_new not in members:
                    bulk_collision = True
                    break

            left_domain = any(
                not inside_liquid(*p_new, t) and p_new[2] > 0
                for p_new in proposed
            )

            if bulk_collision:
                # Transient -- just skip this move, stay mobile.
                continue

            if left_domain:
                if _is_near_substrate(members):
                    # Genuine contact-line pinning: physically real,
                    # permanent deposition.
                    for p in members:
                        new_blocked.add(p)
                    continue
                else:
                    # Shrinking free surface caught up to a bulk
                    # particle away from the substrate -- not real
                    # pinning, just fail this move.
                    continue

            # Remove old positions.
            for p in members:
                new_particles.discard(p)
                new_blocked.discard(p)

            # Add translated cluster.
            for p_new in proposed:
                new_particles.add(p_new)

        particles = new_particles
        blocked = new_blocked
        t += dt

        # Capture requested snapshots.
        while next_snap < len(snapshots) and t >= float(snapshots[next_snap]) - 1e-10:
            snapshot_map[float(snapshots[next_snap])] = set(particles)
            next_snap += 1

    # Fill any missing final snapshots.
    while next_snap < len(snapshots):
        snapshot_map[float(snapshots[next_snap])] = set(particles)
        next_snap += 1

    return snapshot_map


# =============================================================================
# 3-D PARTICLES -> 2-D DEPOSIT IMAGE
# =============================================================================

def render_particle_deposit(
    particles: set,
    t: float,
    size: int = IMG_SIZE,
    rng: np.random.Generator | None = None,
):
    """
    Top-down projection of the 3-D particle deposit.

    Pixel intensity represents local particle occupancy / stack height.
    This gives the feature extractor a genuine stochastic image rather than
    adding arbitrary noise to a deterministic radial curve.
    """
    rng = rng or np.random.default_rng()

    canvas = np.zeros((size, size), dtype=np.float32)
    if not particles:
        return canvas.astype(np.uint8)

    R = float(BASE_RADIUS_CELLS)
    scale = (size * 0.46) / R
    cx = cy = size / 2.0

    for x, y, z in particles:
        px = int(round(cx + x * scale))
        py = int(round(cy + y * scale))
        if 0 <= px < size and 0 <= py < size:
            canvas[py, px] += 1.0

    # Slightly spread individual lattice particles to resemble an optical
    # deposit texture while retaining discrete stochastic morphology.
    sigma = max(0.7, scale * 0.35)
    k = int(max(3, 2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1

    canvas = cv2.GaussianBlur(canvas, (k, k), sigmaX=sigma)

    # Mild multiplicative illumination + very small sensor noise.
    if canvas.max() > 0:
        canvas /= canvas.max()

    if IMAGE_NOISE > 0:
        canvas += rng.normal(0.0, IMAGE_NOISE, canvas.shape).astype(np.float32)

    # Keep substrate close to black and deposit bright.
    canvas = np.clip(canvas, 0.0, 1.0)
    return np.uint8(canvas * 255.0)


# =============================================================================
# SAME 5-TILE ACQUISITION GEOMETRY
# =============================================================================

def crop_5_tiles(full_img: np.ndarray, tile_size: int = TILE_SIZE):
    h, w = full_img.shape
    cy, cx = h // 2, w // 2
    half = tile_size // 2
    edge_offset = int(0.42 * h)

    def crop(cy_, cx_):
        y0, y1 = max(0, cy_ - half), min(h, cy_ + half)
        x0, x1 = max(0, cx_ - half), min(w, cx_ + half)
        return full_img[y0:y1, x0:x1]

    return {
        "center": crop(cy, cx),
        "north": crop(cy - edge_offset, cx),
        "south": crop(cy + edge_offset, cx),
        "east": crop(cy, cx + edge_offset),
        "west": crop(cy, cx - edge_offset),
    }


# =============================================================================
# ONE EXPERIMENTAL CONDITION
# =============================================================================

def simulate_one_condition(
    glycerol_pct: float,
    concentration_wv: float,
    particle_size_nm: float = 1000.0,
    batch_idx: int = 0,
    rng: np.random.Generator | None = None,
):
    rng = rng or np.random.default_rng(batch_idx)

    phys = calculate_droplet_physics(
        R_um=R_UM,
        d_nm=particle_size_nm,
        t_s=EVAP_TIME_S,
        V_uL=DROP_VOL_UL,
        glycerol_pct=glycerol_pct,
        contact_angle_deg=CONTACT_ANGLE_DEG,
    )

    if phys.get("error"):
        print(
            f"SKIP gly={glycerol_pct} conc={concentration_wv}: "
            f"{phys['error']}"
        )
        return []

    Pe = float(phys["Pe"])
    Ma = float(phys["Ma"])

    # Concentration-dependent sticking:
    # keep a baseline p_stick but make high-concentration systems more prone
    # to collision/aggregation, consistent with the paper's concentration
    # dependence.
    p_stick = PSTICK_BASE * (1.0 + 3.0 * concentration_wv)

    print(
        f"  BRW: gly={glycerol_pct:g}% "
        f"PS={concentration_wv:g}% "
        f"Pe={Pe:.3g} Ma={Ma:.3g} "
        f"p_stick={p_stick:.4g}"
    )

    snapshots = simulate_brw(
        concentration_wv=concentration_wv,
        Pe=Pe,
        Ma=Ma,
        p_stick=p_stick,
        rng=rng,
        snapshots=SNAPSHOT_TIMES,
    )

    batch_name = (
        f"SYN__BRW__gly{glycerol_pct:g}_"
        f"ps{concentration_wv:g}_b{batch_idx}"
    )
    batch_dir = Path(OUT_ROOT) / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for t_snap in SNAPSHOT_TIMES:
        t_snap = float(t_snap)
        particles = snapshots[t_snap]

        img = render_particle_deposit(
            particles,
            t=t_snap,
            rng=rng,
        )

        tiles = crop_5_tiles(img)

        tile_dir = batch_dir / f"t_{t_snap:.3f}"
        if WRITE_IMAGES:
            tile_dir.mkdir(parents=True, exist_ok=True)

        tile_paths = {}
        for pos, arr in tiles.items():
            p = tile_dir / f"{pos}.png"
            if WRITE_IMAGES:
                cv2.imwrite(str(p), arr)
                tile_paths[pos] = str(p)
            else:
                # Feature extractor accepts paths, so when images are disabled
                # we still need temporary files. Keep this branch simple.
                tile_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), arr)
                tile_paths[pos] = str(p)

        # SAME production feature extractor used for experimental images.
        feats_100 = extract_from_5_images(tile_paths)

        # 100D -> 40D:
        # 20 center + 20 edge mean.
        feats_40 = reduce_to_40d(feats_100)

        row = {
            "droplet_id": f"{batch_name}__t{t_snap:.3f}",
            "t": t_snap,
            "Pe": Pe,
            "Ma": Ma,
            "glycerol_pct": glycerol_pct,
            "concentration_wv": concentration_wv,
            "particle_size_nm": particle_size_nm,
            "model": "BRW_Crivoi_Duan_style_v4_nonpermanent_collisions",
            **feats_40,
        }

        rows.append(row)

        print(
            f"    t={t_snap:.3f} "
            f"N={len(particles):5d} "
            f"features={len(feats_40):2d}"
        )

    return rows


def simulate_one_condition_averaged(
    glycerol_pct: float,
    concentration_wv: float,
    particle_size_nm: float = 1000.0,
    batch_idx: int = 0,
    n_replicates: int = N_REPLICATES,
    write_replicate_images: bool = False,
):
    """
    Run n_replicates independent BRW realizations of the SAME physical
    condition (different RNG seeds), and average the 40D feature vector at
    each snapshot time. This is the recommended entry point for building
    SINDy-ready trajectories -- a single realization is dominated by
    particle-position Monte Carlo noise (see module docstring / IMAGE_NOISE
    comment); averaging replicates the same way you'd average real
    droplet replicates rather than trust one run.

    Images are only written for the FIRST replicate by default
    (write_replicate_images=False) since with n_replicates=8 you don't need
    8x the PNGs on disk to get the averaged trajectory -- set True if you
    want to inspect the spread across replicates visually.
    """
    per_replicate_dfs = []
    for rep in range(n_replicates):
        old_write = globals()["WRITE_IMAGES"]
        if not write_replicate_images and rep > 0:
            globals()["WRITE_IMAGES"] = False
        try:
            rows = simulate_one_condition(
                glycerol_pct=glycerol_pct,
                concentration_wv=concentration_wv,
                particle_size_nm=particle_size_nm,
                batch_idx=batch_idx * n_replicates + rep,
                rng=np.random.default_rng(batch_idx * 1000 + rep),
            )
        finally:
            globals()["WRITE_IMAGES"] = old_write

        if not rows:
            continue
        df_rep = pd.DataFrame(rows).set_index("t")
        per_replicate_dfs.append(df_rep)

    if not per_replicate_dfs:
        return []

    feature_cols = feature_names_40()
    stacked = pd.concat(per_replicate_dfs)
    averaged = stacked.groupby(level=0)[feature_cols].mean()
    std_cols = stacked.groupby(level=0)[feature_cols].std()
    std_cols.columns = [f"{c}_std" for c in std_cols.columns]

    meta_ref = per_replicate_dfs[0][["Pe", "Ma", "glycerol_pct", "concentration_wv",
                                      "particle_size_nm", "model"]]

    batch_name = (
        f"SYN__BRW__gly{glycerol_pct:g}_ps{concentration_wv:g}_b{batch_idx}"
        f"_avg{n_replicates}"
    )
    out_rows = []
    for t_snap in averaged.index:
        row = {
            "droplet_id": f"{batch_name}__t{t_snap:.3f}",
            "t": float(t_snap),
            "n_replicates": n_replicates,
            **meta_ref.loc[t_snap].to_dict(),
            **averaged.loc[t_snap].to_dict(),
            **std_cols.loc[t_snap].to_dict(),
        }
        out_rows.append(row)

    print(f"  -> averaged {n_replicates} replicates into {len(out_rows)} rows")
    return out_rows


# =============================================================================
# SAVE
# =============================================================================

def _ensure_table_schema(conn, table, df):
    """
    If `table` already exists (e.g. from an earlier run of an older version
    of this script) but is missing columns present in df, ALTER TABLE to add
    them as REAL/TEXT before inserting. Prevents
    'table X has no column named Y' crashes when the schema has grown
    (e.g. particle_size_nm / model added in v3) without forcing a manual
    DROP TABLE on your end.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    existing_cols = {row[1] for row in cur.fetchall()}
    if not existing_cols:
        return  # table doesn't exist yet -- to_sql will create it fresh

    missing = [c for c in df.columns if c not in existing_cols]
    for col in missing:
        sqlite_type = "TEXT" if df[col].dtype == object else "REAL"
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sqlite_type}")
    if missing:
        conn.commit()
        print(f"  Added {len(missing)} missing column(s) to {table}: {missing}")


def save_trajectories(
    rows,
    csv_path: str = CSV_OUT,
    db_path: str = DB_PATH,
):
    df = pd.DataFrame(rows)

    if df.empty:
        print("No rows generated.")
        return df

    expected = feature_names_40()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing 40D features: {missing}")

    # Explicit ordering makes SINDy / downstream ML reproducible.
    metadata = [
        "droplet_id",
        "t",
        "Pe",
        "Ma",
        "glycerol_pct",
        "concentration_wv",
        "particle_size_nm",
        "model",
    ]
    # n_replicates / <feature>_std only exist when rows came from
    # simulate_one_condition_averaged() -- keep them if present rather than
    # silently dropping the uncertainty info averaging gives you.
    extra_cols = [c for c in df.columns if c == "n_replicates" or c.endswith("_std")]
    df = df[metadata + expected + extra_cols]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows -> {csv_path}")
    print(f"Feature dimension = {len(expected)}")

    conn = sqlite3.connect(db_path)
    try:
        _ensure_table_schema(conn, "synthetic_trajectories", df)
        df.to_sql(
            "synthetic_trajectories",
            conn,
            if_exists="append",
            index=False,
        )
        print(
            f"Appended {len(df)} rows -> {db_path} "
            f"(table: synthetic_trajectories)"
        )
    finally:
        conn.close()

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    glycerol_values = [0.0]
    concentration_values = [0.05,0.07, 0.10,0.15, 0.20, 0.30]
    particle_sizes_nm = [1000.0]

    all_rows = []
    idx = 0

    for gly in glycerol_values:
        for conc in concentration_values:
            for ps_nm in particle_sizes_nm:
                print(
                    f"\n=== BRW condition "
                    f"gly={gly:g}% "
                    f"PS={conc:g}% "
                    f"d={ps_nm:g} nm ==="
                )

                try:
                    rows = simulate_one_condition_averaged(
                        glycerol_pct=gly,
                        concentration_wv=conc,
                        particle_size_nm=ps_nm,
                        batch_idx=idx,
                        n_replicates=N_REPLICATES,
                    )
                    all_rows.extend(rows)
                except Exception as exc:
                    print(f"  ERROR: {type(exc).__name__}: {exc}")

                idx += 1

    save_trajectories(all_rows)


if __name__ == "__main__":
    main()