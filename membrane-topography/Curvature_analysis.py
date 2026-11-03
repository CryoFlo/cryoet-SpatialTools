#!/usr/bin/env python3
"""
Estimate a membrane mid-surface and curvature from a subtomogram average (STA).

Required input
--------------
  --map STA.mrc
  --level <density threshold>   # analogous to a ChimeraX volume level

Optional
--------
  --protein-mask mask.mrc       # mask on the same grid as the STA
  --pdb model.pdb               # fitted model; coordinates must match the MRC world coordinates

Main outputs
------------
  height_map.png
  mean_curvature.png
  gaussian_curvature.png
  membrane_surface.csv
  membrane_surface.npz
  membrane_3vox_mask.mrc
  membrane_3vox_density.mrc
  membrane_average_2d.mrc
  aligned_sta_local.mrc
  membrane_detection_qc.png
  support_qc.png
  summary.tsv
  membrane_profile_template.csv
  membrane_surface_in_map.mrc   # slab on the input grid, overlay on the input map
  transform.txt

Dependencies
------------
  numpy, scipy, mrcfile, matplotlib

The algorithm:
  1. Threshold the STA at the user-supplied level.
  2. Exclude the protein region if a mask/PDB is supplied.
  3. Find the membrane without a threshold: sheet-like ridges at bilayer
     scale (Hessian filter) -> several candidate planes (RANSAC) -> each grown
     into a quadric that follows the curvature -> keep the one covering the
     largest area. Bulky protein, however bright, cannot win that.
  4. Sample density profiles along the frame normal, around that guide.
  5. First estimate of the membrane center per lateral position from --level:
     two leaflet peaks where possible, otherwise the thresholded band
     (slope-corrected; two passes).
  6. Refine without a threshold: the map's own average bilayer profile is
     cross-correlated along every column. A column is kept only if both
     halves of the bilayer agree on the height, which drops positions where
     protein is fused to the membrane or the map mask cuts into it.
  7. Interpolate holes (under protein) and smooth the mid-surface. By default
     (--support disk) the surface is reported over a circle: the largest one
     that stays on the membrane at least --mask-edge-ang inside the cut edge
     of the map, or --disk-radius-ang. Detection never uses density closer
     than --mask-edge-ang to the cut; the rim of the circle beyond the last
     detections is extrapolated from them. Curvature is not reported where
     the surface is only inter-/extrapolated.
  8. Compute height, mean curvature, Gaussian curvature and principal curvatures.
  9. Generate an aligned membrane representation with <=3 voxel thickness.

To compare maps that share box size and centering, run all of them with
--frame box (plus --box-axis if the membrane normal is not along z),
--detrend none, --fix-height-scale and one common --disk-radius-ang (the
smallest "largest that fits" printed over your maps): height maps then share
pixel grid, circle and colors, and summary.tsv holds one comparable line per
map.

--level only seeds step 5; with the default settings the final surface comes
from step 6 and barely depends on it.

Python 3.8+ compatible.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import mrcfile
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull, Delaunay, cKDTree


# ----------------------------- I/O helpers -----------------------------

def read_mrc(path):
    with mrcfile.open(path, permissive=True) as m:
        data = np.asarray(m.data, dtype=np.float32).copy()

        vx = float(m.voxel_size.x)
        vy = float(m.voxel_size.y)
        vz = float(m.voxel_size.z)
        voxel = np.array([vx, vy, vz], dtype=float)  # xyz

        mapc = int(m.header.mapc)
        mapr = int(m.header.mapr)
        maps = int(m.header.maps)
        if (mapc, mapr, maps) != (1, 2, 3):
            raise ValueError(
                "This script currently expects standard MRC axis mapping "
                "mapc/mapr/maps = 1/2/3. Found %d/%d/%d." % (mapc, mapr, maps)
            )

        origin = np.array(
            [float(m.header.origin.x), float(m.header.origin.y), float(m.header.origin.z)],
            dtype=float,
        )
        starts = np.array(
            [int(m.header.nxstart), int(m.header.nystart), int(m.header.nzstart)],
            dtype=float,
        )

        # Cryo-EM software differs in whether physical placement is encoded in
        # origin or nstart. Prefer origin when it is non-zero; otherwise use nstart.
        if np.any(np.abs(origin) > 1e-6):
            world_origin = origin
            origin_source = "header.origin"
        else:
            world_origin = starts * voxel
            origin_source = "nstart * voxel_size"

    return data, voxel, world_origin, origin_source


def write_mrc(path, data, voxel_A, origin_xyz_A=None):
    arr = np.asarray(data, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(arr)
        m.voxel_size = tuple(float(x) for x in voxel_A)
        if origin_xyz_A is not None:
            m.header.origin.x = float(origin_xyz_A[0])
            m.header.origin.y = float(origin_xyz_A[1])
            m.header.origin.z = float(origin_xyz_A[2])
        m.update_header_stats()


def parse_cif_atoms(path):
    """Atom coordinates from the _atom_site loop of an mmCIF file."""
    xyz = []
    cols = []
    in_loop = False
    with open(path, "r") as f:
        for line in f:
            t = line.strip()
            if t.startswith("_atom_site."):
                cols.append(t.split()[0])
                in_loop = True
                continue
            if not in_loop:
                continue
            if not t or t.startswith("#") or t.startswith("loop_") or t.startswith("_") or t.startswith("data_"):
                if xyz:
                    break
                in_loop = bool(t.startswith("_atom_site."))
                cols = cols if in_loop else []
                continue
            try:
                ix, iy, iz = (cols.index("_atom_site.Cartn_" + c) for c in "xyz")
                tok = t.split()
                if "_atom_site.type_symbol" in cols and tok[cols.index("_atom_site.type_symbol")].upper() in ("H", "D"):
                    continue
                xyz.append((float(tok[ix]), float(tok[iy]), float(tok[iz])))
            except (ValueError, IndexError):
                continue
    if not xyz:
        raise ValueError("No _atom_site coordinates could be parsed from %s" % path)
    return np.asarray(xyz, dtype=float)


def parse_pdb_atoms(path):
    if str(path).lower().endswith((".cif", ".mmcif")):
        return parse_cif_atoms(path)
    xyz = []
    with open(path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            if line[76:78].strip().upper() in ("H", "D"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                xyz.append((x, y, z))
            except ValueError:
                continue
    if not xyz:
        raise ValueError("No ATOM/HETATM coordinates could be parsed from %s" % path)
    return np.asarray(xyz, dtype=float)


def pdb_to_mask(pdb_path, shape_zyx, voxel_xyz_A, origin_xyz_A, radius_A):
    atoms = parse_pdb_atoms(pdb_path)
    seeds = np.zeros(shape_zyx, dtype=bool)

    # world xyz -> voxel xyz -> array zyx
    ijk = np.rint((atoms - origin_xyz_A[None, :]) / voxel_xyz_A[None, :]).astype(int)
    x = ijk[:, 0]
    y = ijk[:, 1]
    z = ijk[:, 2]
    valid = (
        (x >= 0) & (x < shape_zyx[2]) &
        (y >= 0) & (y < shape_zyx[1]) &
        (z >= 0) & (z < shape_zyx[0])
    )
    seeds[z[valid], y[valid], x[valid]] = True
    if not np.any(seeds):
        raise ValueError(
            "The PDB atoms do not overlap the MRC grid. The fitted PDB and map "
            "must share the same coordinate system."
        )

    # Exact Euclidean distance in physical units for isotropic voxels.
    vox = float(voxel_xyz_A[0])
    dist = ndimage.distance_transform_edt(~seeds, sampling=vox)
    return dist <= float(radius_A)


# ----------------------------- Geometry -----------------------------

def xyz_from_zyx_indices(idx_zyx, voxel_xyz_A, origin_xyz_A):
    idx_zyx = np.asarray(idx_zyx, dtype=float)
    xyz_idx = idx_zyx[:, ::-1]  # zyx -> xyz
    return origin_xyz_A[None, :] + xyz_idx * voxel_xyz_A[None, :]


def fit_plane_pca(points_xyz):
    center = points_xyz.mean(axis=0)
    q = points_xyz - center
    _, _, vh = np.linalg.svd(q, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    return center, normal


def robust_plane_ransac(points_xyz, n_iter=500, inlier_A=20.0, max_points=100000, seed=1):
    rng = np.random.default_rng(seed)
    pts = points_xyz
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]

    best_count = -1
    best_mask = None

    for _ in range(int(n_iter)):
        ii = rng.choice(len(pts), size=3, replace=False)
        a, b, c = pts[ii]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n /= nn
        d = np.abs((pts - a) @ n)
        mask = d <= inlier_A
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_mask is None or best_count < 100:
        raise RuntimeError("Could not robustly fit a membrane plane.")

    center, normal = fit_plane_pca(pts[best_mask])

    # Refine once using distances to the PCA plane.
    d = np.abs((pts - center) @ normal)
    refine = d <= inlier_A
    if refine.sum() >= 100:
        center, normal = fit_plane_pca(pts[refine])

    return center, normal, int(refine.sum())


def make_plane_basis(normal_xyz):
    n = np.asarray(normal_xyz, dtype=float)
    n /= np.linalg.norm(n)

    # Choose a stable reference vector not parallel to n.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, n


def map_corners_xyz(shape_zyx, voxel_xyz_A, origin_xyz_A):
    nz, ny, nx = shape_zyx
    corners_idx = np.array(
        [
            [0, 0, 0],
            [nx - 1, 0, 0],
            [0, ny - 1, 0],
            [0, 0, nz - 1],
            [nx - 1, ny - 1, 0],
            [nx - 1, 0, nz - 1],
            [0, ny - 1, nz - 1],
            [nx - 1, ny - 1, nz - 1],
        ],
        dtype=float,
    )
    return origin_xyz_A[None, :] + corners_idx * voxel_xyz_A[None, :]


# ----------------------------- Sampling -----------------------------

def world_xyz_to_array_zyx(xyz, voxel_xyz_A, origin_xyz_A):
    ijk_xyz = (xyz - origin_xyz_A[None, :]) / voxel_xyz_A[None, :]
    return ijk_xyz[:, ::-1].T  # 3 x N in z,y,x order


def sample_uv_w_volume(data, u, v, w, center_xyz, e1, e2, n, voxel_xyz_A, origin_xyz_A,
                       order=1, cval=0.0):
    """Sample a local aligned volume with output order [w, v, u]."""
    out = np.empty((len(w), len(v), len(u)), dtype=np.float32)

    # Process one v-row at a time to limit memory.
    U, W = np.meshgrid(u, w, indexing="xy")  # [w, u]
    for j, vv in enumerate(v):
        pts = (
            center_xyz[None, None, :]
            + U[:, :, None] * e1[None, None, :]
            + float(vv) * e2[None, None, :]
            + W[:, :, None] * n[None, None, :]
        )
        flat = pts.reshape(-1, 3)
        coords = world_xyz_to_array_zyx(flat, voxel_xyz_A, origin_xyz_A)
        vals = map_coordinates(data, coords, order=order, mode="constant", cval=cval)
        out[:, j, :] = vals.reshape(len(w), len(u))
    return out


def quadric_design(u, v):
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    return np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])


def eval_quadric(coef, U, V):
    a, b, c, d, e, f = coef
    return a * U * U + b * U * V + c * V * V + d * U + e * V + f


def fit_quadric_irls(u, v, w, coef0, inlier_A, n_iter=20, max_points=200000, seed=1):
    """
    Robust quadric w = a u^2 + b uv + c v^2 + d u + e v + f, grown from an
    initial guess (the RANSAC plane): refit on the points within inlier_A of
    the current surface until the inlier set is stable. A curved membrane is
    followed down to its rim, where a plane loses it.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if len(u) > max_points:
        sel = np.random.default_rng(seed).choice(len(u), size=max_points, replace=False)
        u, v, w = u[sel], v[sel], w[sel]
    A = quadric_design(u, v)
    coef = np.asarray(coef0, dtype=float).copy()
    n_in = 0
    for _ in range(int(n_iter)):
        m = np.abs(w - A @ coef) <= inlier_A
        n_in = int(m.sum())
        if n_in < 50:
            break
        new, _, _, _ = np.linalg.lstsq(A[m], w[m], rcond=None)
        done = np.allclose(new, coef, rtol=1e-4, atol=1e-9)
        coef = new
        if done:
            break
    return coef, n_in


def quadric_curvature_at_center(coef):
    """H, K, k1, k2 [1/A] and the in-plane angle of k1 [deg] at u=v=0."""
    a, b, c, d, e, _ = coef
    hx, hy, hxx, hxy, hyy = d, e, 2.0 * a, b, 2.0 * c
    g = 1.0 + hx * hx + hy * hy
    H = ((1.0 + hy * hy) * hxx - 2.0 * hx * hy * hxy + (1.0 + hx * hx) * hyy) / (2.0 * g ** 1.5)
    K = (hxx * hyy - hxy * hxy) / (g * g)
    root = math.sqrt(max(H * H - K, 0.0))
    evals, evecs = np.linalg.eigh(np.array([[hxx, hxy], [hxy, hyy]]))
    vec = evecs[:, int(np.argmax(evals))]
    ang = math.degrees(math.atan2(vec[1], vec[0])) % 180.0
    return H, K, H + root, H - root, ang


def estimate_mask_region(data):
    """
    Region of the map that survived masking: everything that is not the
    constant value found at the box corners (any mask shape or position).
    Returns a boolean volume, or None if the map does not look masked.
    """
    corners = data[[0, 0, 0, 0, -1, -1, -1, -1], [0, 0, -1, -1, 0, 0, -1, -1], [0, -1, 0, -1, 0, -1, 0, -1]]
    if float(np.ptp(corners)) != 0.0:
        return None
    outside = data == corners[0]
    if outside.mean() < 0.02:
        return None
    # Only the constant region connected to the box border is "outside".
    lab, _ = ndimage.label(outside)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                       lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    border = border[border > 0]
    outside = np.isin(lab, border)
    if outside.mean() < 0.02:
        return None
    return ~outside


def refine_surface_by_template(local_wvu, w_A, surf_vu, cos_vu, clean_vu,
                               halfwidth_A, tol_A, step_A, usable_wvu=None,
                               min_cc=0.4, min_amp=0.3, min_cover=0.7, agree_A=10.0, grow_px=0, patch_min_px=1,
                               protein_wvu=None, to_cut_edge=False, grad_uv=None, dl_A=None, rim_agree_A=0.0):
    """
    Threshold-free refinement. The average density profile across the membrane
    (taken from the map itself, along the local normal) is slid along every
    column; the best normalised cross-correlation within +/- tol_A of the
    current surface gives the membrane center. Samples that are masked out
    (usable_wvu False: outside the map mask, inside the protein mask) are
    ignored rather than treated as zero density.

    Returns refined surface (NaN where rejected), cc, amplitude relative to
    the average membrane, and the template (offsets, values).
    """
    nw, nv, nu = local_wvu.shape
    dw = float(w_A[1] - w_A[0])
    no = max(2, int(round(halfwidth_A / step_A)))
    ns = max(1, int(math.ceil(tol_A / step_A)))
    m = np.arange(-(no + ns), no + ns + 1)
    jj, ii = np.indices(surf_vu.shape)
    if grad_uv is None:
        # Profile sampled along the column (w), stretched by 1/cos(slope).
        pos = surf_vu[None, :, :] + m[:, None, None] * step_A / cos_vu[None, :, :]
        coords = np.array([
            (pos - w_A[0]) / dw,
            np.broadcast_to(jj[None], pos.shape),
            np.broadcast_to(ii[None], pos.shape),
        ], dtype=float)
    else:
        # Profile sampled along the local membrane normal: on steep flanks the
        # template window then stays inside the map instead of running along
        # the column far past the membrane.
        gu, gv = grad_uv
        g = np.sqrt(1.0 + gu * gu + gv * gv)
        o = m[:, None, None] * step_A
        pos = surf_vu[None, :, :] + o / g[None]
        coords = np.array([
            (pos - w_A[0]) / dw,
            jj[None] - o * (gv / g)[None] / dl_A,
            ii[None] - o * (gu / g)[None] / dl_A,
        ], dtype=float)
    Q = map_coordinates(local_wvu, coords, order=1, mode="constant", cval=np.nan)
    use = np.isfinite(Q)
    if usable_wvu is not None:
        use &= map_coordinates(usable_wvu.astype(np.float32), coords, order=0, mode="constant", cval=0.0) > 0.5
    Wmap = use.astype(float)      # limited by the map only
    if protein_wvu is not None:
        use &= map_coordinates(np.asarray(protein_wvu, dtype=np.float32), coords, order=0, mode="constant", cval=0.0) < 0.5
    Q = np.where(use, Q, 0.0)
    W = use.astype(float)

    core = slice(ns, ns + 2 * no + 1)
    cols = clean_vu & np.all(use[core], axis=0)
    if np.count_nonzero(cols) < 30:
        return None
    T = np.median(Q[core][:, cols], axis=1)
    if float(np.ptp(T)) <= 0:
        return None

    corr = lambda arr, k: ndimage.correlate1d(arr, k, axis=0, mode="constant", cval=0.0)[no:no + 2 * ns + 1]
    take = lambda arr, kk: np.take_along_axis(arr, kk[None], axis=0)[0]
    QQ = Q * Q

    def match(part):
        """Best shift (in steps), cc and amplitude using the template samples in `part`."""
        k1 = part.astype(float)
        t = np.where(part, T - T[part].mean(), 0.0)
        Sw, Sq, Sqq = corr(W, k1), corr(Q, k1), corr(QQ, k1)
        St, Stt, Sqt = corr(W, t), corr(W, t * t), corr(Q, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = Sqt - Sq * St / Sw
            vq = Sqq - Sq * Sq / Sw
            vt = Stt - St * St / Sw
            cc = cov / np.sqrt(vq * vt)
            amp = cov / vt
        ok = (Sw >= min_cover * k1.sum()) & (vt > 0.3 * float(np.sum(t * t))) & (vq > 0) & np.isfinite(cc)
        cc = np.where(ok, cc, -np.inf)
        k = np.argmax(cc, axis=0)
        c0 = take(cc, k)
        cm = take(cc, np.clip(k - 1, 0, 2 * ns))
        cp = take(cc, np.clip(k + 1, 0, 2 * ns))
        with np.errstate(divide="ignore", invalid="ignore"):
            sub = 0.5 * (cm - cp) / (cm - 2.0 * c0 + cp)
        sub = np.where(np.isfinite(sub) & (np.abs(sub) <= 1.0), sub, 0.0)
        found = (k > 0) & (k < 2 * ns) & np.isfinite(c0)
        # Hidden by the protein mask only (the map itself covers this part).
        map_cover = corr(Wmap, k1)[ns] >= min_cover * k1.sum()
        hidden = ~np.any(ok, axis=0) & map_cover
        cut = ~np.any(ok, axis=0) & ~map_cover
        return k - ns + sub, np.where(np.isfinite(c0), c0, 0.0), take(np.nan_to_num(amp), k), found, hidden, cut

    offs = m[core]
    sh, c0, a0, ok, _, _ = match(np.ones(len(T), dtype=bool))
    # Each half of the bilayer (one leaflet + its flank) must place the
    # membrane at the same height. Where protein is fused to one side, or the
    # map mask cuts the profile, they do not, and the column is left out.
    sh_lo, c_lo, a_lo, ok_lo, hid_lo, cut_lo = match(offs <= 0)
    sh_up, c_up, a_up, ok_up, hid_up, cut_up = match(offs >= 0)
    agree = np.abs(sh_lo - sh_up) * step_A <= agree_A
    good = ok & ok_lo & ok_up & agree & (c0 >= min_cc) & (a0 >= min_amp) & (np.minimum(c_lo, c_up) >= min_cc)
    if rim_agree_A > agree_A and np.count_nonzero(good) >= 10:
        # Outside the area covered by strictly accepted columns (the rim, away
        # from the complex) a broadened or fading membrane is accepted with a
        # looser agreement between the two halves of the profile.
        _, hull_strict = support_from_detections(good, mode="hull")
        loose = (ok & ok_lo & ok_up & (np.abs(sh_lo - sh_up) * step_A <= rim_agree_A)
                 & (c0 >= min_cc) & (a0 >= min_amp) & (np.minimum(c_lo, c_up) >= min_cc))
        good |= loose & ~hull_strict
    # One side hidden: by the user's protein mask, or (at the rim) by the cut
    # edge of the map. The other half alone decides, so that the membrane is
    # followed right up to the cut.
    gone_lo = cut_lo if to_cut_edge else np.zeros_like(cut_lo)
    gone_up = cut_up if to_cut_edge else np.zeros_like(cut_up)
    if protein_wvu is not None:
        gone_lo = gone_lo | hid_lo
        gone_up = gone_up | hid_up
    only_lo = ~good & ok_lo & gone_up & (c_lo >= max(min_cc, 0.6)) & (a_lo >= min_amp)
    only_up = ~good & ok_up & gone_lo & (c_up >= max(min_cc, 0.6)) & (a_up >= min_amp)
    sh = np.where(only_lo, sh_lo, np.where(only_up, sh_up, sh))
    c0 = np.where(only_lo, c_lo, np.where(only_up, c_up, c0))
    good |= only_lo | only_up
    # Columns bordering a sizeable rejected patch (a protein footprint) are
    # partly contaminated too: step back from it.
    if grow_px > 0:
        lab, nlab = ndimage.label(ndimage.binary_opening(~good & np.any(use[core], axis=0), iterations=1))
        if nlab > 0:
            counts = np.bincount(lab.ravel())
            big = counts >= patch_min_px
            big[0] = False
            good &= ~ndimage.binary_dilation(big[lab], iterations=int(grow_px))
    refined = np.where(good, surf_vu + sh * step_A / cos_vu, np.nan).astype(np.float32)
    # Columns where the cut edge of the map takes away too much of either half
    # of the profile window (at the current surface): the template cannot judge them.
    Wc = Wmap[core]
    lo_half, up_half = offs <= 0, offs >= 0
    cut_cols = ((Wc[lo_half].mean(axis=0) < min_cover) | (Wc[up_half].mean(axis=0) < min_cover))
    return refined, c0.astype(np.float32), a0.astype(np.float32), (offs * step_A, T), cut_cols


def membrane_sheet_candidates(data, inside_vol, vox_A, sigma_A=15.0, edge_A=20.0,
                              min_ratio=0.7, noise_k=3.0, min_component_vox=50):
    """
    Threshold-free membrane candidates: voxels on a bright sheet-like ridge at
    the scale of a bilayer (one strongly negative Hessian eigenvalue, the two
    others small). Protein is mostly blob- or rod-like at that scale and is
    depleted, however bright it is.
    """
    sv = max(1.0, sigma_A / vox_A)
    vol = np.asarray(data, dtype=np.float32)
    if inside_vol is not None:
        vol = np.where(inside_vol, vol, np.float32(vol[inside_vol].mean()))
        sel = ndimage.distance_transform_edt(inside_vol, sampling=vox_A) > edge_A + sv * vox_A
    else:
        sel = np.ones(vol.shape, dtype=bool)
    nsel = int(sel.sum())
    if nsel < 1000:
        return np.zeros(vol.shape, dtype=bool)
    M = np.empty((nsel, 3, 3), dtype=np.float32)
    for a, b in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        order = [0, 0, 0]
        order[a] += 1
        order[b] += 1
        h = ndimage.gaussian_filter(vol, sv, order=order, mode="nearest")[sel]
        M[:, a, b] = h
        M[:, b, a] = h
    ev = np.linalg.eigvalsh(M)
    l1 = -ev[:, 0]
    l2 = np.maximum(-ev[:, 1], 0.0)
    ratio = np.where(l1 > 0, (l1 - l2) / np.maximum(l1, 1e-30), 0.0)
    cand = np.zeros(vol.shape, dtype=bool)
    # Above the noise floor, but never so high that a faint membrane next to
    # bright protein is lost.
    floor = min(noise_k * float(np.median(np.abs(ev[:, 0]))), 0.3 * float(np.percentile(l1, 99)))
    cand[sel] = (ratio > min_ratio) & (l1 > floor)
    lab, nlab = ndimage.label(cand, structure=np.ones((3, 3, 3)))
    if nlab > 0:
        counts = np.bincount(lab.ravel())
        keep = counts >= min_component_vox
        keep[0] = False
        cand = keep[lab]
    return cand


def _coverage(a, b, cell_A, min_count=1):
    """Number of well-occupied lateral cells: how far a set of points spreads as a sheet."""
    key = np.floor(a / cell_A).astype(np.int64) * 100003 + np.floor(b / cell_A).astype(np.int64)
    _, counts = np.unique(key, return_counts=True)
    return int(np.count_nonzero(counts >= min_count))


def ransac_plane_seeds(points_xyz, n_iter=600, inlier_A=25.0, max_points=30000, seed=1,
                       prior_normal=None, max_tilt_deg=90.0, n_keep=8, min_count=1):
    """
    Several distinct candidate planes, ranked by the lateral area their
    inliers cover (not by inlier count: a slab through bulky protein holds
    many voxels but covers little area). Each is refined by PCA.
    Returns a list of (center_xyz, normal_xyz).
    """
    rng = np.random.default_rng(seed)
    pts = points_xyz
    if len(pts) > max_points:
        min_count = max(1, int(round(min_count * max_points / float(len(pts)))))
        pts = pts[rng.choice(len(pts), size=max_points, replace=False)]
    min_cos = math.cos(math.radians(max_tilt_deg))
    found = []
    for _ in range(int(n_iter)):
        a, b, c = pts[rng.choice(len(pts), size=3, replace=False)]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-8:
            continue
        nrm /= nn
        if prior_normal is not None and abs(float(nrm @ prior_normal)) < min_cos:
            continue
        rel = pts - a
        mask = np.abs(rel @ nrm) <= inlier_A
        if int(mask.sum()) < 100:
            continue
        b1, b2, _ = make_plane_basis(nrm)
        found.append((_coverage(rel[mask] @ b1, rel[mask] @ b2, inlier_A, min_count), a, nrm))
    if not found:
        raise RuntimeError("Could not robustly fit a membrane plane.")
    found.sort(key=lambda t: -t[0])

    seeds = []
    for _, a, nrm in found[:200]:
        mask = np.abs((pts - a) @ nrm) <= inlier_A
        center, normal = fit_plane_pca(pts[mask])
        if prior_normal is not None and abs(float(normal @ prior_normal)) < min_cos:
            center, normal = pts[mask].mean(axis=0), nrm
        if any(abs(float(normal @ n2)) > 0.985 and abs(float((center - c2) @ n2)) < inlier_A for c2, n2 in seeds):
            continue
        seeds.append((center, normal))
        if len(seeds) >= n_keep:
            break
    return seeds


def orient_normal(normal_xyz):
    """Reproducible sign: the dominant component of the normal is positive."""
    nrm = np.asarray(normal_xyz, dtype=float).copy()
    if nrm[int(np.argmax(np.abs(nrm)))] < 0:
        nrm *= -1.0
    return nrm


def grow_guide(pts_xyz, center_xyz, e1, e2, n, plane_center_xyz, plane_normal_xyz, inlier_A, seed=1, min_count=1):
    """Plane -> quadric in the frame (e1, e2, n); returns coef, inlier count, lateral coverage."""
    q = pts_xyz - center_xyz[None, :]
    u, v, w = q @ e1, q @ e2, q @ n
    nn = float(n @ plane_normal_xyz)
    if abs(nn) < 0.2:
        return None, 0, 0
    off = float((plane_center_xyz - center_xyz) @ plane_normal_xyz)
    coef0 = np.array([0.0, 0.0, 0.0, -float(e1 @ plane_normal_xyz) / nn, -float(e2 @ plane_normal_xyz) / nn, off / nn])
    coef, n_in = fit_quadric_irls(u, v, w, coef0, inlier_A=inlier_A, seed=seed)
    m = np.abs(w - eval_quadric(coef, u, v)) <= inlier_A
    if int(m.sum()) < 50:
        return None, 0, 0
    return coef, int(m.sum()), _coverage(u[m], v[m], inlier_A, min_count)


def _crossing(pp, w_A, i_in, i_out, level):
    """Sub-voxel position where the profile crosses `level` between two samples."""
    if i_out < 0 or i_out >= len(pp) or pp[i_out] < -1e29:
        return float(w_A[i_in])
    a, b = pp[i_in], pp[i_out]
    if a <= b:
        return float(w_A[i_in])
    t = min(max((a - level) / (a - b), 0.0), 1.0)
    return float(w_A[i_in] + t * (w_A[i_out] - w_A[i_in]))


def detect_surface_from_profiles(local_vol_wvu, w_A, level,
                                 leaflet_min_A=20.0, leaflet_max_A=60.0,
                                 protein_local_wvu=None,
                                 guide_vu=None, cos_vu=None,
                                 halfwidth_A=80.0, tol_A=40.0,
                                 max_thick_A=80.0, thick_ref_A=None,
                                 baseline=0.0, min_dip=0.1):
    """
    Detect membrane center for each (v,u) profile, searching around a guide
    surface (not around w=0) and correcting distances along the profile for
    the local membrane slope (cos_vu).

    Preference:
      1. Midpoint between two above-threshold peaks separated by a plausible
         bilayer distance, with a real density dip between them.
      2. Density-weighted center of the thresholded band at the guide, if the
         band is about as thick as a membrane (not a lone leaflet, not fused
         with protein).
      3. If that band is thicker (membrane fused with protein density): the
         clean edge of the band +/- half the reference membrane thickness.
    Anything further than tol_A from the guide is left missing.
    """
    nw, nv, nu = local_vol_wvu.shape
    surface = np.full((nv, nu), np.nan, dtype=np.float32)
    confidence = np.zeros((nv, nu), dtype=np.float32)
    method = np.zeros((nv, nu), dtype=np.uint8)  # 0 missing, 1 pair, 2 band, 3 one edge (4 template, set later)
    thickness = np.full((nv, nu), np.nan, dtype=np.float32)  # along the local normal

    if guide_vu is None:
        guide_vu = np.zeros((nv, nu), dtype=float)
    if cos_vu is None:
        cos_vu = np.ones((nv, nu), dtype=float)

    dw = float(abs(w_A[1] - w_A[0])) if len(w_A) > 1 else 1.0
    sig = 0.5 * float(tol_A)

    for j in range(nv):
        for i in range(nu):
            g = float(guide_vu[j, i])
            c = float(cos_vu[j, i])
            if not np.isfinite(g):
                continue
            p = np.asarray(local_vol_wvu[:, j, i], dtype=float).copy()

            if protein_local_wvu is not None:
                pm = protein_local_wvu[:, j, i] > 0.5
                # If the protein occupies the membrane search zone substantially,
                # do not force a surface estimate; interpolate later.
                central = np.abs(w_A - g) <= max(leaflet_max_A, 40.0)
                if np.count_nonzero(pm & central) >= 2:
                    continue
                p[pm] = -np.inf

            pp = p.copy()
            pp[~np.isfinite(pp)] = -1e30
            above = pp >= level
            win = np.abs(w_A - g) <= halfwidth_A
            if not np.any(above & win):
                continue

            lab, nlab = ndimage.label(above.astype(np.uint8))

            # Find leaflet-like peaks.
            min_sep_px = max(1, int(round(leaflet_min_A / (c * dw))))
            peaks, _ = find_peaks(pp, height=level, distance=min_sep_px)
            peaks = peaks[win[peaks]]

            best = None
            best_score = -np.inf
            for a_idx in range(len(peaks)):
                for b_idx in range(a_idx + 1, len(peaks)):
                    pa = peaks[a_idx]
                    pb = peaks[b_idx]
                    sep = abs(w_A[pb] - w_A[pa]) * c
                    if not (leaflet_min_A <= sep <= leaflet_max_A):
                        continue
                    mid = 0.5 * (w_A[pa] + w_A[pb])
                    if abs(mid - g) > tol_A:
                        continue
                    low = min(pp[pa], pp[pb])
                    if low - pp[pa:pb + 1].min() < min_dip * (low - baseline):
                        continue
                    # Strong peaks, close to the guide surface (scale-free).
                    score = (pp[pa] + pp[pb] - 2.0 * level) * math.exp(-0.5 * ((mid - g) / sig) ** 2)
                    if score > best_score:
                        best_score = score
                        best = (pa, pb, mid)

            if best is not None:
                pa, pb, mid = best
                surface[j, i] = float(mid)
                confidence[j, i] = float(max(0.0, pp[pa] - level) + max(0.0, pp[pb] - level))
                method[j, i] = 1
                ia = np.where(lab == lab[pa])[0][0]
                ib = np.where(lab == lab[pb])[0][-1]
                t = (_crossing(pp, w_A, ib, ib + 1, level) - _crossing(pp, w_A, ia, ia - 1, level)) * c
                if t <= max_thick_A:
                    thickness[j, i] = t
                continue

            # Fall back to the thresholded band at the guide surface.
            best_idx = None
            best_integ = -np.inf
            for lab_id in range(1, nlab + 1):
                idx = np.where(lab == lab_id)[0]
                gap = max(w_A[idx[0]] - g, g - w_A[idx[-1]], 0.0)
                if gap > tol_A:
                    continue
                integ = float(np.sum(np.maximum(pp[idx] - level, 0.0)))
                if integ > best_integ:
                    best_integ = integ
                    best_idx = idx

            if best_idx is None:
                continue

            idx = best_idx
            lo = _crossing(pp, w_A, idx[0], idx[0] - 1, level)
            hi = _crossing(pp, w_A, idx[-1], idx[-1] + 1, level)
            thick = (hi - lo) * c

            if thick_ref_A is not None and thick < 0.6 * float(thick_ref_A):
                # A single leaflet above the level: its center is not the mid-surface.
                continue
            if thick <= max_thick_A:
                weights = np.maximum(pp[idx] - level, 0.0)
                if np.sum(weights) > 0:
                    mid = float(np.sum(w_A[idx] * weights) / np.sum(weights))
                else:
                    mid = float(np.mean(w_A[idx]))
                if abs(mid - g) > tol_A:
                    continue
                surface[j, i] = mid
                confidence[j, i] = best_integ
                method[j, i] = 2
                thickness[j, i] = thick
            elif thick_ref_A is not None:
                half = 0.5 * float(thick_ref_A) / c
                cands = (lo + half, hi - half)
                mid = cands[0] if abs(cands[0] - g) <= abs(cands[1] - g) else cands[1]
                if abs(mid - g) > tol_A:
                    continue
                surface[j, i] = mid
                confidence[j, i] = 0.5 * best_integ * float(thick_ref_A) / thick
                method[j, i] = 3

    return surface, confidence, method, thickness


def smooth_surface(surface, valid_mask, base, sigma_px, n_relax=2000):
    """
    Smooth the detected surface as (global quadric) + (smoothed residual).
    Holes are filled by harmonic interpolation of the residual; the residual
    is then smoothed by Gaussian-weighted local plane fits, which (unlike a
    plain Gaussian) do not flatten a slope at the rim of the data.
    """
    if np.count_nonzero(valid_mask) < 10:
        raise RuntimeError("Too few valid membrane positions were detected.")

    invalid = ~valid_mask
    resid = np.where(valid_mask, surface - base, 0.0).astype(float)
    dist, indices = ndimage.distance_transform_edt(invalid, return_indices=True)
    resid[invalid] = resid[tuple(indices[:, invalid])]

    if np.any(invalid):
        for _ in range(int(n_relax)):
            pad = np.pad(resid, 1, mode="edge")
            avg = 0.25 * (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
            resid[invalid] = avg[invalid]

    if sigma_px <= 0:
        return base + resid

    # Filled-in values count less the further they are from real detections.
    # Holes inside the detected area keep a small weight; beyond its outline
    # the fill counts for (almost) nothing, so the rim continues the local
    # slope of the real detections instead of levelling off.
    _, hull = support_from_detections(valid_mask, mode="hull")
    wgt = np.where(valid_mask, 1.0,
                   np.where(hull, 0.2 * np.exp(-0.5 * (dist / (2.0 * sigma_px)) ** 2) + 1e-4, 1e-4))
    Y, X = np.indices(resid.shape, dtype=float)

    def local_plane(sig):
        G = lambda arr: ndimage.gaussian_filter(arr, sigma=float(sig), mode="constant", cval=0.0)
        m0, mx, my = G(wgt), G(wgt * X), G(wgt * Y)
        mxx, mxy, myy = G(wgt * X * X), G(wgt * X * Y), G(wgt * Y * Y)
        z0, zx, zy = G(wgt * resid), G(wgt * X * resid), G(wgt * Y * resid)
        # Moments about each pixel.
        cxx = mxx - 2.0 * X * mx + X * X * m0
        cxy = mxy - X * my - Y * mx + X * Y * m0
        cyy = myy - 2.0 * Y * my + Y * Y * m0
        cx, cy = mx - X * m0, my - Y * m0
        czx, czy = zx - X * z0, zy - Y * z0
        ridge = 1e-3 * m0 * sig * sig + 1e-12
        A = np.empty(resid.shape + (3, 3))
        A[..., 0, 0] = m0 + 1e-12
        A[..., 0, 1] = A[..., 1, 0] = cx
        A[..., 0, 2] = A[..., 2, 0] = cy
        A[..., 1, 1] = cxx + ridge
        A[..., 1, 2] = A[..., 2, 1] = cxy
        A[..., 2, 2] = cyy + ridge
        rhs = np.stack([z0, czx, czy], axis=-1)[..., None]
        return np.linalg.solve(A, rhs)[..., 0, 0]

    sol = local_plane(sigma_px)
    if float(dist.max()) > sigma_px:
        # Away from real detections the fit widens with the distance to them,
        # so that what is inter-/extrapolated follows the broad trend of the
        # data around it, not the noise of the nearest few points.
        stack = np.stack([sol, local_plane(2.0 * sigma_px), local_plane(4.0 * sigma_px)])
        lev = np.clip(np.log2(np.maximum(dist / sigma_px, 1.0)), 0.0, 2.0)
        i0 = np.clip(np.floor(lev).astype(int), 0, 1)
        frac_ = lev - i0
        lo_ = np.take_along_axis(stack, i0[None], axis=0)[0]
        hi_ = np.take_along_axis(stack, (i0 + 1)[None], axis=0)[0]
        sol = (1.0 - frac_) * lo_ + frac_ * hi_
    return base + sol


def support_from_detections(valid_mask, mode="hull", min_px=25):
    """
    Lateral region in which a surface may be reported. Nothing is ever
    extrapolated beyond it.
      detected: only where a membrane was detected
      hull:     convex hull of the detections (holes under protein are
                interpolated; the outside, e.g. beyond a spherical mask, is not)
    """
    lab, nlab = ndimage.label(valid_mask)
    clean = valid_mask.copy()
    if nlab > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep = counts >= min(min_px, counts.max())
        clean = keep[lab]
    if mode == "detected":
        return clean, clean.copy()

    pts = np.argwhere(clean).astype(float)
    try:
        hull = ConvexHull(pts)
        tri = Delaunay(pts[hull.vertices])
        grid = np.indices(clean.shape).reshape(2, -1).T.astype(float)
        support = (tri.find_simplex(grid) >= 0).reshape(clean.shape) | clean
    except Exception:
        support = clean.copy()
    return clean, support


def detrend_plane(surface_vu, u_A, v_A, fit_mask):
    U, V = np.meshgrid(u_A, v_A, indexing="xy")
    m = fit_mask & np.isfinite(surface_vu)
    A = np.column_stack([U[m], V[m], np.ones(np.count_nonzero(m))])
    b = surface_vu[m]
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    trend = coef[0] * U + coef[1] * V + coef[2]
    return surface_vu - trend, coef, trend


def compute_curvature(h_A, du_A, dv_A):
    # h array order [v, u], so axis0 spacing=dv, axis1 spacing=du
    hy, hx = np.gradient(h_A, dv_A, du_A)
    hyy, hyx = np.gradient(hy, dv_A, du_A)
    hxy, hxx = np.gradient(hx, dv_A, du_A)
    hxy = 0.5 * (hxy + hyx)

    denom_H = 2.0 * np.power(1.0 + hx * hx + hy * hy, 1.5)
    H_Ainv = ((1.0 + hy * hy) * hxx - 2.0 * hx * hy * hxy + (1.0 + hx * hx) * hyy) / denom_H

    denom_K = np.power(1.0 + hx * hx + hy * hy, 2.0)
    K_Ainv2 = (hxx * hyy - hxy * hxy) / denom_K

    disc = np.maximum(H_Ainv * H_Ainv - K_Ainv2, 0.0)
    root = np.sqrt(disc)
    k1_Ainv = H_Ainv + root
    k2_Ainv = H_Ainv - root

    return H_Ainv, K_Ainv2, k1_Ainv, k2_Ainv


# ----------------------------- Plotting -----------------------------

def finite_range(a):
    x = np.asarray(a)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    return float(np.nanmin(x)), float(np.nanmax(x))


def _apply_disk_view(ax, disk, artists=(), lw=1.0, margin=1.06):
    """
    Clean outline and the same field of view for every map.
    disk = (center_x, center_y, view_radius, U, V, edge_field); the outline is
    the zero contour of edge_field (positive inside the reported region).
    """
    if disk is None:
        return
    cu, cv, view_radius, U, V, field = disk
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    cs = ax.contour(U, V, field, levels=[0.0], colors="k", linewidths=lw)
    try:
        paths = list(cs.get_paths())
    except Exception:
        paths = list(cs.collections[0].get_paths())
    paths = [p_ for p_ in paths if len(p_.vertices) > 2]
    if paths:
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import PathPatch
        clip = PathPatch(MplPath.make_compound_path(*paths), transform=ax.transData,
                         facecolor="none", edgecolor="none")
        ax.add_patch(clip)
        for art in artists:
            if art is None:
                continue
            try:
                art.set_clip_path(clip)
            except Exception:
                for coll in getattr(art, "collections", []):
                    coll.set_clip_path(clip)
    ax.set_xlim(cu - margin * view_radius, cu + margin * view_radius)
    ax.set_ylim(cv - margin * view_radius, cv + margin * view_radius)


def model_footprint(au, av, aw, resolution_A=10.0):
    """
    Outline of a model seen along the membrane normal, at a given resolution:
    the model is rendered as a density map at that resolution (Gaussian per
    atom, sigma = 0.225 x resolution, as ChimeraX molmap), contoured at the
    level that encloses the molecular volume, and its silhouette is taken.
    Returns (x, y, field) on a fine grid; field < 0 inside, zero contour = outline.
    """
    sigma = 0.225 * float(resolution_A)
    step = max(1.0, float(resolution_A) / 5.0)
    pad = 4.0 * sigma + 2.0 * step
    pts = np.column_stack([au, av, aw])
    edges = [np.arange(pts[:, k].min() - pad, pts[:, k].max() + pad + step, step) for k in range(3)]
    rho, _ = np.histogramdd(pts, bins=edges)
    rho = ndimage.gaussian_filter(rho, sigma / step, mode="constant")
    # Level enclosing the molecular volume (~17.4 A^3 per non-hydrogen atom).
    n_vox = int(np.clip(round(len(pts) * 17.4 / step ** 3), 1, rho.size - 1))
    level = float(np.partition(rho.ravel(), rho.size - n_vox)[rho.size - n_vox])
    proj = rho.max(axis=2).T          # [y, x]
    gx = 0.5 * (edges[0][:-1] + edges[0][1:])
    gy = 0.5 * (edges[1][:-1] + edges[1][1:])
    return gx, gy, level - proj


def plot_height(path, h_A, u_A, v_A, contour_step_A, protein_footprint=None, vmin=None, vmax=None,
                interpolated=None, cmap="YlGnBu", disk=None, outlines=(), outline_color="w",
                outline_style="solid", outline_width=1.2, gamma=1.0, curv_axes=None, curv_color="#404040"):
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    # Pixel centers are at u_A, v_A: half a pixel of extent on each side.
    du = 0.5 * (u_A[1] - u_A[0])
    dv = 0.5 * (v_A[1] - v_A[0])
    extent = [u_A[0] - du, u_A[-1] + du, v_A[0] - dv, v_A[-1] + dv]

    kwargs = dict(origin="lower", extent=extent, cmap=cmap, aspect="equal")
    lo, hi = (vmin, vmax) if (vmin is not None and vmax is not None) else finite_range(h_A)
    if gamma != 1.0:
        # Power-law color scale (identical for every map with the same limits).
        from matplotlib.colors import PowerNorm
        kwargs.update(norm=PowerNorm(gamma=gamma, vmin=lo, vmax=hi))
    elif vmin is not None and vmax is not None:
        kwargs.update(vmin=vmin, vmax=vmax)

    im = ax.imshow(h_A, **kwargs)

    # Determine contour levels based on fixed limits or finite data range
    if vmin is not None and vmax is not None:
        cmin, cmax = vmin, vmax
    else:
        cmin, cmax = finite_range(h_A)

    if contour_step_A > 0 and cmax > cmin:
        lo = math.floor(cmin / contour_step_A) * contour_step_A
        hi = math.ceil(cmax / contour_step_A) * contour_step_A
        levels = np.arange(lo, hi + 0.5 * contour_step_A, contour_step_A)
        if len(levels) >= 2:
            U, V = np.meshgrid(u_A, v_A, indexing="xy")
            cs = ax.contour(U, V, h_A, levels=levels, colors="k", linewidths=0.8)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%g")

    if protein_footprint is not None and np.any(protein_footprint):
        U, V = np.meshgrid(u_A, v_A, indexing="xy")
        ax.contour(U, V, protein_footprint.astype(float), levels=[0.5], colors="k", linewidths=1.5)

    ci = None
    if interpolated is not None and np.any(interpolated):
        # White dotted outline: no membrane detected inside (under protein, or
        # at the cut edge of the map); the height there is inter-/extrapolated.
        U, V = np.meshgrid(u_A, v_A, indexing="xy")
        ci = ax.contour(U, V, interpolated.astype(float), levels=[0.5], colors="w", linewidths=1.3, linestyles=":")

    # Dotted outline(s) of the model(s) given with --outline-pdb.
    for gx, gy, field in outlines:
        ax.contour(gx, gy, field, levels=[0.0], colors=outline_color, linewidths=outline_width,
                   linestyles={"dotted": ":", "dashed": "--", "solid": "-"}[outline_style])

    _apply_disk_view(ax, disk, [im, ci] + list(ax.collections))
    if curv_axes and disk is not None:
        _draw_curv_axes(ax, disk, curv_axes, curv_color, 1.2, 12)
        ax.set_title("   ".join("\u03ba%d = %+.4f \u00b1 %.4f nm\u207b\u00b9 (axis %d, %.0f\u00b0)"
                               % (a["axis"], 10 * a["kappa"], 10 * a["se"], a["axis"], a["angle"])
                               for a in curv_axes), fontsize=9)
    ax.set_xlabel("x' (Å)")
    ax.set_ylabel("y' (Å)")
    cb = fig.colorbar(im, ax=ax)
    if gamma != 1.0:
        # Ticks every 10/20/50 A, dropping those squeezed together by the power scale.
        step = 10.0 if hi - lo <= 80 else (20.0 if hi - lo <= 200 else 50.0)
        kept = []
        for t in np.arange(math.ceil(lo / step) * step, hi + 1e-6, step):
            if not kept or kwargs["norm"](t) - kwargs["norm"](kept[-1]) >= 0.07:
                kept.append(t)
        cb.set_ticks(kept)
    cb.set_label("Membrane height (Å)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _height_norm(h_A, vmin, vmax, gamma):
    lo, hi = (vmin, vmax) if (vmin is not None and vmax is not None) else finite_range(h_A)
    if gamma != 1.0:
        from matplotlib.colors import PowerNorm
        return PowerNorm(gamma=gamma, vmin=lo, vmax=hi), lo, hi
    from matplotlib.colors import Normalize
    return Normalize(vmin=lo, vmax=hi), lo, hi


def _colorbar_ticks(norm, lo, hi, gamma, max_ticks=None, min_frac=0.07):
    step = 10.0 if hi - lo <= 80 else (20.0 if hi - lo <= 200 else 50.0)
    ticks = list(np.arange(math.ceil(lo / step) * step, hi + 1e-6, step))
    if gamma == 1.0:
        if max_ticks and len(ticks) > max_ticks:
            k = int(math.ceil(len(ticks) / float(max_ticks)))
            ticks = ticks[::-1][::k][::-1]
        return ticks
    kept = []
    for t in ticks:
        if not kept or norm(t) - norm(kept[-1]) >= min_frac:
            kept.append(t)
    # Make sure the top of the scale is labelled.
    if kept and kept[-1] < ticks[-1]:
        if norm(ticks[-1]) - norm(kept[-1]) < min_frac:
            kept[-1] = ticks[-1]
        else:
            kept.append(ticks[-1])
    return kept


def curvature_along_axes(U, V, surface, measured, center, angle_deg, band_A=20.0, max_A=None):
    """
    Curvature of the membrane along two perpendicular lines through `center`
    (axis 1 at angle_deg counter-clockwise from x', axis 2 at +90 deg): the
    height profile of the measured membrane within band_A of each line is
    fitted with h(s) = a s^2 + b s + c, and kappa = 2a / (1 + b^2)^1.5 at the
    center. Returns a list of dicts (angle, kappa [1/A], standard error,
    number of pixels, extent of the data along the line).
    """
    out = []
    for k, ang in enumerate((angle_deg, angle_deg + 90.0)):
        th = math.radians(ang)
        du, dv = U - center[0], V - center[1]
        s_ = du * math.cos(th) + dv * math.sin(th)
        d_ = -du * math.sin(th) + dv * math.cos(th)
        m = measured & np.isfinite(surface) & (np.abs(d_) <= band_A)
        if max_A:
            m &= np.abs(s_) <= max_A
        res = dict(axis=k + 1, angle=ang % 180.0, kappa=float("nan"), se=float("nan"), n=int(m.sum()),
                   s_min=float("nan"), s_max=float("nan"), reach=max_A)
        if m.sum() >= 10:
            A = np.column_stack([s_[m] ** 2, s_[m], np.ones(int(m.sum()))])
            coef, _, _, _ = np.linalg.lstsq(A, surface[m], rcond=None)
            r = surface[m] - A @ coef
            cov = np.linalg.pinv(A.T @ A) * float(r @ r) / max(1, int(m.sum()) - 3)
            g = (1.0 + coef[1] ** 2) ** 1.5
            res.update(kappa=float(2.0 * coef[0] / g), se=float(2.0 * math.sqrt(max(cov[0, 0], 0.0)) / g),
                       s_min=float(s_[m].min()), s_max=float(s_[m].max()))
        out.append(res)
    return out


def _axis_number_xy(disk, angle_deg):
    """Where an axis number goes: near the end of the line that points right (or up)."""
    th = math.radians(angle_deg)
    c, s_ = math.cos(th), math.sin(th)
    sign = 1.0 if (c > 1e-6 or (abs(c) <= 1e-6 and s_ > 0)) else -1.0
    return disk[0] + sign * 0.9 * disk[2] * c, disk[1] + sign * 0.9 * disk[2] * s_


def _label_sign(angle_deg):
    """Which end of an axis carries its number: the end that points right (or up)."""
    th = math.radians(angle_deg)
    c, s_ = math.cos(th), math.sin(th)
    return 1.0 if (c > 1e-6 or (abs(c) <= 1e-6 and s_ > 0)) else -1.0


def _rotate_panel_inputs(h_A, u_A, v_A, disk, outlines, curv_axes, rot_cw_deg):
    """
    Rotate everything drawn on a panel clockwise by rot_cw_deg about the circle
    center: the height map is resampled on a new grid centered on the circle,
    outlines get rotated coordinates, axis angles (and the end that carries
    their number) turn with the picture.
    """
    cu, cv, rad = disk[0], disk[1], disk[2]
    t = math.radians(rot_cw_deg)
    ct, st = math.cos(t), math.sin(t)
    step = float(u_A[1] - u_A[0])
    half = int(math.ceil((max(abs(u_A[0] - cu), abs(u_A[-1] - cu), abs(v_A[0] - cv), abs(v_A[-1] - cv))) / step))
    g = np.arange(-half, half + 1) * step
    X, Y = np.meshgrid(g, g, indexing="xy")
    # Source point of each output pixel (inverse of a clockwise rotation).
    xs = cu + X * ct - Y * st
    ys = cv + X * st + Y * ct
    ii = (xs - u_A[0]) / step
    jj = (ys - v_A[0]) / step
    fin = np.isfinite(h_A)
    hv = map_coordinates(np.where(fin, h_A, 0.0), [jj, ii], order=1, mode="constant", cval=0.0)
    wv = map_coordinates(fin.astype(float), [jj, ii], order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        h_new = np.where(wv > 0.5, hv / np.maximum(wv, 1e-6), np.nan)
    rot = lambda x, y: ((x - cu) * ct + (y - cv) * st, -(x - cu) * st + (y - cv) * ct)
    new_outlines = []
    for gx, gy, field in outlines:
        GX, GY = np.meshgrid(gx, gy, indexing="xy")
        RX, RY = rot(GX, GY)
        new_outlines.append((RX, RY, field))
    new_disk = (0.0, 0.0, rad, X, Y, rad - np.hypot(X, Y))
    new_axes = None
    if curv_axes:
        new_axes = [dict(a, angle=a["angle"] - rot_cw_deg, label_sign=_label_sign(a["angle"])) for a in curv_axes]
    return h_new, g, g, new_disk, new_outlines, new_axes


def _draw_curv_axes_outside(ax, disk, axes_info, color, lw, font_pt, ext=0.1, gap=0.07):
    """
    Panel style: dashed axis across the circle, short solid ticks continuing it
    outside the circle at both ends, the axis number beyond the labelled end.
    """
    cu, cv, rad = disk[0], disk[1], disk[2]
    for a in axes_info:
        th = math.radians(a["angle"])
        c, s_ = math.cos(th), math.sin(th)
        L = min(rad, a.get("reach") or rad)
        ax.plot([cu - L * c, cu + L * c], [cv - L * s_, cv + L * s_], color=color, linewidth=lw,
                linestyle=(0, (4, 2.5)), zorder=4, solid_capstyle="butt")
        for sgn in (1.0, -1.0):
            ax.plot([cu + sgn * rad * c, cu + sgn * (1.0 + ext) * rad * c],
                    [cv + sgn * rad * s_, cv + sgn * (1.0 + ext) * rad * s_],
                    color="k", linewidth=1.5 * lw, solid_capstyle="butt", zorder=5, clip_on=False)
        sgn = a.get("label_sign", _label_sign(a["angle"]))
        r_lab = (1.0 + ext + gap) * rad
        ax.text(cu + sgn * r_lab * c, cv + sgn * r_lab * s_, str(a["axis"]), ha="center", va="center",
                fontsize=font_pt, fontweight="bold", color="k", zorder=5, clip_on=False)


def _draw_curv_axes(ax, disk, axes_info, color, lw, font_pt, label=True):
    """Dashed lines for the two curvature axes across the circle, numbered at their ends."""
    import matplotlib.patheffects as pe
    cu, cv, rad = disk[0], disk[1], disk[2]
    for a in axes_info:
        th = math.radians(a["angle"])
        c, s_ = math.cos(th), math.sin(th)
        # The line spans the part of the axis used for the curvature.
        L = min(rad, a.get("reach") or rad)
        ax.plot([cu - L * c, cu + L * c], [cv - L * s_, cv + L * s_], color=color, linewidth=lw,
                linestyle=(0, (4, 2.5)), zorder=4, solid_capstyle="butt")
        if label:
            # Number at the end of the line that points right (or up).
            x_, y_ = _axis_number_xy(disk, a["angle"])
            ax.text(x_, y_, str(a["axis"]),
                    ha="center", va="center", fontsize=font_pt, fontweight="bold", color=color, zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="w")])


def plot_height_panel(path_base, h_A, u_A, v_A, contour_step_A, size_mm=30.0, font_pt=6.0, label_pt=6.0,
                      vmin=None, vmax=None, gamma=1.0, cmap="YlGnBu", disk=None, outlines=(),
                      outline_color="w", outline_style="solid", outline_pt=0.8, axes=False, scalebar_A=100.0,
                      curv_axes=None, curv_color="#404040", ref_radius_A=None, units="A",
                      rotate_cw_deg=0.0):
    """
    Height map at its final printed size (the circle is size_mm across), fonts in
    true points, no color bar (see plot_height_colorbar). Written as .png
    (600 dpi) and .pdf. With curvature axes, the axes continue outside the
    circle with their numbers, and the curvatures are printed on one line below.
    """
    if rotate_cw_deg and disk is not None:
        h_A, u_A, v_A, disk, outlines, curv_axes = _rotate_panel_inputs(
            h_A, u_A, v_A, disk, outlines, curv_axes, rotate_cw_deg)
    unit_scale, unit_txt, k_fmt = (1.0, "\u00c5", "%+.4f") if units == "A" else (10.0, "nm", "%+.3f")
    import matplotlib as mpl
    with mpl.rc_context({"font.size": font_pt, "axes.linewidth": 0.4, "xtick.major.width": 0.4,
                         "ytick.major.width": 0.4, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        if ref_radius_A and disk is not None:
            # Same Å per mm as panels drawn for a circle of ref_radius_A.
            size_mm = size_mm * disk[2] / float(ref_radius_A)
        # Outside margin for the axis ticks and numbers (fraction of the radius).
        m_out = 0.24 if (curv_axes and not axes) else 0.0
        k_txt = None
        if curv_axes:
            fmt_k = lambda k: ("0." + "0" * (int(k_fmt[-2]))) if abs(unit_scale * k) < 0.5 * 10 ** -int(k_fmt[-2]) \
                else (k_fmt % (unit_scale * k)).replace("-", "\u2212").replace("+", "")
            k_txt = "    ".join("\u03ba%s %s %s\u207b\u00b9" % ("\u2081" if a["axis"] == 1 else "\u2082",
                                                              fmt_k(a["kappa"]), unit_txt) for a in curv_axes)
        map_in = size_mm / 25.4 * (1.0 + m_out)
        text_in = 0.0
        if k_txt:
            from matplotlib.textpath import TextPath
            from matplotlib.font_manager import FontProperties
            text_in = TextPath((0, 0), k_txt, size=font_pt, prop=FontProperties()).get_extents().width / 72.0 + 0.06
        width_in = max(map_in, text_in)
        extra = (1.35 * font_pt / 72.0 + 0.06) if k_txt else 0.0
        fig = plt.figure(figsize=(width_in, map_in + extra))
        fw, fh = map_in / width_in, map_in / (map_in + extra)
        if axes:
            ax = fig.add_axes([0.5 - 0.5 * fw + 0.27 * fw, 1.0 - fh + 0.22 * fh, 0.71 * fw, 0.71 * fh])
        else:
            ax = fig.add_axes([0.5 - 0.5 * fw, 1.0 - fh, fw, fh])
        f_map = fh
        du = 0.5 * (u_A[1] - u_A[0])
        dv = 0.5 * (v_A[1] - v_A[0])
        extent = [u_A[0] - du, u_A[-1] + du, v_A[0] - dv, v_A[-1] + dv]
        norm, lo, hi = _height_norm(h_A, vmin, vmax, gamma)
        im = ax.imshow(h_A, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal",
                       interpolation="bilinear")
        U, V = np.meshgrid(u_A, v_A, indexing="xy")
        avoid = np.zeros((0, 2))
        for gx, gy, field in outlines:
            co = ax.contour(gx, gy, field, levels=[0.0], colors=outline_color, linewidths=outline_pt,
                            linestyles={"dotted": ":", "dashed": "--", "solid": "-"}[outline_style], zorder=3)
            try:
                segs = co.allsegs[0]
            except Exception:
                segs = [p_.vertices for p_ in co.collections[0].get_paths()]
            if segs:
                avoid = np.concatenate([avoid] + [sg for sg in segs if len(sg)], axis=0)
        if curv_axes and disk is not None:
            # Contour labels also keep clear of the curvature axes and their numbers.
            tt = np.linspace(-1.0, 1.0, 81)[:, None] * disk[2]
            for a in curv_axes:
                th = math.radians(a["angle"])
                avoid = np.concatenate([avoid, np.column_stack([disk[0] + tt[:, 0] * math.cos(th),
                                                                disk[1] + tt[:, 0] * math.sin(th)])], axis=0)
        if contour_step_A > 0:
            levels = np.arange(math.floor(lo / contour_step_A) * contour_step_A,
                               math.ceil(hi / contour_step_A) * contour_step_A + 0.5 * contour_step_A, contour_step_A)
            cs = ax.contour(U, V, h_A, levels=levels, colors="k", linewidths=0.3,
                            linestyles="solid")
            if label_pt > 0:
                if disk is not None:
                    # One label per contour, placed well inside the circle and
                    # spread out from the other labels (never cut by the edge).
                    cu, cv, rad = disk[0], disk[1], disk[2]
                    chosen = []
                    try:
                        seg_sets = cs.allsegs
                    except Exception:
                        seg_sets = [[p.vertices for p in c.get_paths()] for c in cs.collections]
                    for lev_segs in seg_sets:
                        pts = np.concatenate([sg for sg in lev_segs if len(sg)], axis=0) if any(len(sg) for sg in lev_segs) else np.zeros((0, 2))
                        if len(pts) == 0:
                            continue
                        if np.hypot(*(pts.max(axis=0) - pts.min(axis=0))) < 0.4 * rad:
                            continue   # small closed contour (a summit): a label would hide it
                        r = np.hypot(pts[:, 0] - cu, pts[:, 1] - cv)
                        pts = pts[r <= 0.78 * rad]
                        if len(pts) == 0:
                            continue
                        r = np.hypot(pts[:, 0] - cu, pts[:, 1] - cv)
                        if chosen:
                            sep = np.min(np.hypot(pts[:, None, 0] - np.array(chosen)[None, :, 0],
                                                  pts[:, None, 1] - np.array(chosen)[None, :, 1]), axis=1)
                        else:
                            sep = np.full(len(pts), 2.0 * rad)
                        score = np.minimum(sep, 0.5 * rad) - 0.3 * np.abs(r - 0.6 * rad)
                        if len(avoid):
                            # Stay clear of the model outline (and of the curvature axes).
                            d_out = cKDTree(avoid).query(pts)[0]
                            score = score - 2.0 * np.clip(0.12 * rad - d_out, 0.0, None)

                        chosen.append(tuple(pts[int(np.argmax(score))]))
                    if chosen:
                        ax.clabel(cs, inline=True, fontsize=label_pt, fmt="%g", inline_spacing=1.5,
                                  manual=chosen)
                else:
                    ax.clabel(cs, inline=True, fontsize=label_pt, fmt="%g", inline_spacing=1.5)
        _apply_disk_view(ax, disk, [im] + list(ax.collections), lw=0.5,
                         margin=(1.0 + m_out) if not axes else 1.04)
        if curv_axes and disk is not None:
            if axes:
                _draw_curv_axes(ax, disk, curv_axes, curv_color, 0.6, font_pt)
            else:
                _draw_curv_axes_outside(ax, disk, curv_axes, curv_color, 0.6, font_pt)
            fig.text(0.5, 0.5 * (1.0 - f_map), k_txt, ha="center", va="center", fontsize=font_pt)
        if axes:
            ax.set_xlabel("x' (Å)", labelpad=1.0)
            ax.set_ylabel("y' (Å)", labelpad=1.0)
            ax.tick_params(pad=1.0)
            if disk is not None:
                tk = [t for t in (-200, 0, 200) if abs(t) <= disk[2]]
                ax.set_xticks([disk[0] + t for t in tk]); ax.set_xticklabels(["%d" % t for t in tk])
                ax.set_yticks([disk[1] + t for t in tk]); ax.set_yticklabels(["%d" % t for t in tk])
        else:
            ax.set_axis_off()
            if scalebar_A and disk is not None:
                # Scale bar tucked into the empty lower-left corner, outside the circle.
                cu, cv, rad = disk[0], disk[1], disk[2]
                x0, y0 = cu - (0.995 + m_out) * rad, cv - (0.985 + m_out) * rad
                ax.plot([x0, x0 + scalebar_A], [y0, y0], color="k", linewidth=1.0, solid_capstyle="butt",
                        clip_on=False)
                bar_txt = ("%g \u00c5" % scalebar_A) if units == "A" else ("%g nm" % (scalebar_A / 10.0))
                ax.text(x0, y0 + 0.035 * rad, bar_txt, ha="left", va="bottom", fontsize=font_pt)
        fig.savefig(str(path_base) + ".png", dpi=600, transparent=True)
        fig.savefig(str(path_base) + ".pdf", transparent=True)
        plt.close(fig)


def plot_section_panel(path_base, data, inside_vol, voxel_xyz_A, origin_xyz_A, center_xyz, e1, e2, n,
                       circle_uv, angle_deg, surface_vu, support_vu, u_A, v_A, size_mm=30.0, font_pt=6.0,
                       slab_A=17.3, scalebar_A=100.0, units="nm", label="1", label_sign=1.0,
                       slab_color="#00c8e6", mask_color="#d62728", smooth_vox=0.8, half_A=None,
                       circle_A=None, up_A=0.0):
    """
    Density section through the map along a curvature axis (the plane holding
    the axis and the membrane normal), with the cut edge of the map (dashed)
    and the reported membrane slab (outline). Written as .png (600 dpi) and .pdf.
    """
    import matplotlib as mpl
    vox = float(voxel_xyz_A[0])
    th = math.radians(angle_deg)
    c, s_ = math.cos(th), math.sin(th)
    cu, cv = circle_uv
    # In-plane window: the section of the mask sphere (or the circle if unmasked).
    if inside_vol is not None:
        mz, my, mx = ndimage.center_of_mass(inside_vol)
        q = origin_xyz_A + np.array([mx, my, mz]) * voxel_xyz_A - center_xyz
        du, dv = float(q @ e1) - cu, float(q @ e2) - cv
        s_mid, w_mid = du * c + dv * s_, float(q @ n)
        d_off = -du * s_ + dv * c
        r_mask = (3.0 * inside_vol.sum() / (4.0 * math.pi)) ** (1.0 / 3.0) * vox
        half = 1.03 * math.sqrt(max(r_mask ** 2 - d_off ** 2, (0.3 * r_mask) ** 2))
    else:
        s_mid, w_mid, half = 0.0, 0.0, 1.2 * float(np.hypot(u_A[-1] - u_A[0], v_A[-1] - v_A[0])) / 2.0
    if half_A:
        # Fixed window centered on the circle center (same scale as the height panel).
        s_mid, half = 0.0, float(half_A)
    step = 0.5 * vox
    sg = np.arange(s_mid - half, s_mid + half + 1e-6, step)
    # Reported surface along the axis, inside the circle.
    su = (cu + sg * c - u_A[0]) / (u_A[1] - u_A[0])
    sv = (cv + sg * s_ - v_A[0]) / (v_A[1] - v_A[0])
    fin = np.isfinite(surface_vu)
    hs = map_coordinates(np.where(fin, surface_vu, 0.0), [sv, su], order=1, mode="constant", cval=0.0)
    ok = map_coordinates((support_vu & fin).astype(float), [sv, su], order=0, mode="constant", cval=0.0) > 0.5
    if circle_A and np.any(ok):
        # Round view centered on the membrane at the circle center.
        i0 = int(np.argmin(np.abs(sg)))
        w_mid = (float(hs[i0]) if ok[i0] else float(np.median(hs[ok]))) + float(up_A)
    wg = np.arange(w_mid - half, w_mid + half + 1e-6, step)
    S, W = np.meshgrid(sg, wg, indexing="xy")
    P = (center_xyz[None, None, :] + (cu + S * c)[..., None] * e1 + (cv + S * s_)[..., None] * e2
         + W[..., None] * n)
    idx = (P - origin_xyz_A) / voxel_xyz_A
    co = [idx[..., 2], idx[..., 1], idx[..., 0]]
    sm = ndimage.gaussian_filter(np.asarray(data, dtype=np.float32), smooth_vox) if smooth_vox > 0 else data
    img = map_coordinates(sm, co, order=1, mode="constant", cval=0.0)
    vals = data[inside_vol] if inside_vol is not None else data.ravel()
    lo, hi = np.percentile(vals, [5, 99.5])
    with mpl.rc_context({"font.size": font_pt, "pdf.fonttype": 42}):
        inch = size_mm / 25.4
        fig = plt.figure(figsize=(inch, inch))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        im_ = ax.imshow(img, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="bilinear",
                        extent=[sg[0] - step / 2, sg[-1] + step / 2, wg[0] - step / 2, wg[-1] + step / 2])
        if inside_vol is not None:
            ins = map_coordinates(inside_vol.astype(np.float32), co, order=1, mode="constant", cval=0.0)
            cm_ = ax.contour(S, W, ins, levels=[0.5], colors=mask_color, linewidths=0.6)
            for coll in (getattr(cm_, "collections", None) or [cm_]):
                coll.set_linestyle((0, (3, 2)))
        if np.any(ok):
            # Slab outline, segment by segment where the surface is reported.
            lab, nl = ndimage.label(ok)
            for k in range(1, nl + 1):
                i = np.where(lab == k)[0]
                xs = np.concatenate([sg[i], sg[i][::-1], sg[i][:1]])
                ys = np.concatenate([hs[i] + 0.5 * slab_A, (hs[i] - 0.5 * slab_A)[::-1], hs[i][:1] + 0.5 * slab_A])
                ax.plot(xs, ys, color=slab_color, linewidth=0.6)
        if circle_A:
            # Clip everything to a circle the size of the height map, outlined like it.
            circle = plt.Circle((0.0, w_mid), circle_A, transform=ax.transData, facecolor="none",
                                edgecolor="k", linewidth=0.5, zorder=6)
            ax.add_patch(circle)
            for art in [im_] + list(ax.collections) + list(ax.lines):
                art.set_clip_path(circle)
            if label:
                # Number of the axis end on the right, outside the circle at the membrane.
                j = np.where(ok & (sg * label_sign > 0))[0]
                y_lab = float(hs[j[np.argmax(np.abs(sg[j]))]]) if len(j) else w_mid
                ax.text(label_sign * 1.12 * circle_A, y_lab, label, ha="center", va="center",
                        fontsize=font_pt, fontweight="bold", color="k")
            if scalebar_A:
                m_out = half / circle_A - 1.0
                x0, y0 = -(0.995 + m_out) * circle_A, w_mid - (0.985 + m_out) * circle_A
                ax.plot([x0, x0 + scalebar_A], [y0, y0], color="k", linewidth=1.0, solid_capstyle="butt")
                bar_txt = ("%g \u00c5" % scalebar_A) if units == "A" else ("%g nm" % (scalebar_A / 10.0))
                ax.text(x0, y0 + 0.035 * circle_A, bar_txt, ha="left", va="bottom", fontsize=font_pt)
            scalebar_A = 0
            label = ""
        if label:
            # Which end of the axis this is (as numbered on the height map).
            x_lab = s_mid + label_sign * 0.95 * half
            ax.text(x_lab, w_mid + 0.88 * half, label, ha="center", va="center", fontsize=font_pt,
                    fontweight="bold", color="w")
        if scalebar_A:
            x0, y0 = s_mid - 0.95 * half, w_mid - 0.93 * half
            ax.plot([x0, x0 + scalebar_A], [y0, y0], color="w", linewidth=1.0, solid_capstyle="butt")
            bar_txt = ("%g \u00c5" % scalebar_A) if units == "A" else ("%g nm" % (scalebar_A / 10.0))
            ax.text(x0, y0 + 0.03 * half, bar_txt, ha="left", va="bottom", fontsize=font_pt, color="w")
        ax.set_xlim(sg[0], sg[-1])
        ax.set_ylim(wg[0], wg[-1])
        ax.set_axis_off()
        fig.savefig(str(path_base) + ".png", dpi=600, transparent=bool(circle_A))
        fig.savefig(str(path_base) + ".pdf", transparent=bool(circle_A))
        plt.close(fig)


def plot_height_colorbar(path_base, vmin, vmax, gamma=1.0, cmap="YlGnBu", width_mm=30.0, font_pt=6.0,
                         label="Membrane height (Å)"):
    """Horizontal color bar matching plot_height_panel, as .png (600 dpi) and .pdf."""
    import matplotlib as mpl
    with mpl.rc_context({"font.size": font_pt, "axes.linewidth": 0.4, "xtick.major.width": 0.4,
                         "xtick.major.size": 2.0, "pdf.fonttype": 42}):
        w, h = width_mm / 25.4, 11.0 / 25.4
        fig = plt.figure(figsize=(w, h))
        cax = fig.add_axes([0.1, 0.62, 0.8, 0.26])
        norm, lo, hi = _height_norm(np.array([vmin, vmax], float), vmin, vmax, gamma)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        # Tick labels need ~3.5 characters of room at this font size and bar width.
        min_frac = (3.5 * 0.62 * font_pt + 3.0) / (0.8 * width_mm / 25.4 * 72.0)
        cb.set_ticks(_colorbar_ticks(norm, lo, hi, gamma, max_ticks=max(2, int(1.0 / min_frac)),
                                     min_frac=min_frac))
        cb.ax.tick_params(pad=1.0)
        cb.outline.set_linewidth(0.4)
        cb.set_label(label, labelpad=1.5)
        fig.savefig(str(path_base) + ".png", dpi=600, transparent=True)
        fig.savefig(str(path_base) + ".pdf", transparent=True)
        plt.close(fig)


def plot_curvature(path, arr, u_A, v_A, label, symmetric=True, disk=None):
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    extent = [u_A[0], u_A[-1], v_A[0], v_A[-1]]

    kwargs = dict(origin="lower", extent=extent, cmap="BuPu", aspect="equal")
    if symmetric:
        finite = np.abs(arr[np.isfinite(arr)])
        if finite.size:
            lim = float(np.percentile(finite, 99))
            if lim > 0:
                kwargs.update(vmin=-lim, vmax=lim)

    im = ax.imshow(arr, **kwargs)
    _apply_disk_view(ax, disk, [im])
    ax.set_xlabel("x' (Å)")
    ax.set_ylabel("y' (Å)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_qc(path, local_vol_wvu, w_A, u_A, v_A, surface_A, level):
    # Three orthogonal-style QC panels: central local slice + two profiles.
    j = len(v_A) // 2
    i = len(u_A) // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Local aligned density, central v slice: w vs u
    sl = local_vol_wvu[:, j, :]
    axes[0].imshow(
        sl,
        origin="lower",
        aspect="auto",
        extent=[u_A[0], u_A[-1], w_A[0], w_A[-1]],
        cmap="gray",
    )
    axes[0].plot(u_A, surface_A[j, :], linewidth=1.2)
    axes[0].set_xlabel("x' (Å)")
    axes[0].set_ylabel("normal coordinate w (Å)")
    axes[0].set_title("Detected membrane surface")

    # Central profile
    p = local_vol_wvu[:, j, i]
    axes[1].plot(w_A, p)
    axes[1].axhline(level, linestyle="--", linewidth=1)
    axes[1].axvline(surface_A[j, i], linestyle="--", linewidth=1)
    axes[1].set_xlabel("w (Å)")
    axes[1].set_ylabel("density")
    axes[1].set_title("Central normal profile")

    # Valid surface height distribution
    vals = surface_A[np.isfinite(surface_A)]
    axes[2].hist(vals, bins=50)
    axes[2].set_xlabel("surface position w (Å)")
    axes[2].set_ylabel("count")
    axes[2].set_title("Detected surface distribution")

    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_support_qc(path, method, support, analysis_mask, u_A, v_A):
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    extent = [u_A[0], u_A[-1], v_A[0], v_A[-1]]
    im = ax.imshow(method, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=4, aspect="equal")
    U, V = np.meshgrid(u_A, v_A, indexing="xy")
    ax.contour(U, V, support.astype(float), levels=[0.5], colors="w", linewidths=1.8)
    if np.any(analysis_mask):
        ax.contour(U, V, analysis_mask.astype(float), levels=[0.5], colors="w", linewidths=1.0, linestyles="--")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x' (Å)")
    ax.set_ylabel("y' (Å)")
    ax.set_title("solid: reported surface, dashed: curvature region")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(["none", "leaflet pair", "band", "one edge", "template"])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------- Main -----------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Estimate membrane height and curvature from an STA MRC map."
    )
    p.add_argument("--map", required=True, help="Input STA map (.mrc/.map)")
    p.add_argument("--level", required=True, type=float,
                   help="Density threshold, analogous to a ChimeraX volume level")
    p.add_argument("--outdir", default="membrane_curvature_out")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--protein-mask", help="Optional protein mask on the same grid as the STA")
    g.add_argument("--pdb", help="Optional fitted PDB in the same world coordinate system as the MRC")

    p.add_argument("--pdb-radius-ang", type=float, default=5.0,
                   help="Atomic exclusion radius used to voxelize a PDB [Å] (default: 5)")
    p.add_argument("--protein-dilate-ang", type=float, default=5.0,
                   help="Additional dilation of protein exclusion mask [Å] (default: 5)")

    p.add_argument("--analysis-sigma-vox", type=float, default=0.0,
                   help="Optional Gaussian sigma for profile analysis [vox]. Default 0 keeps the ChimeraX-like level on raw density.")
    p.add_argument("--min-component-vox", type=int, default=50,
                   help="Discard thresholded components smaller than this (default: 50)")

    p.add_argument("--ransac-iter", type=int, default=600)
    p.add_argument("--plane-inlier-ang", type=float, default=25.0,
                   help="RANSAC membrane-plane inlier distance [Å] (default: 25)")
    p.add_argument("--flip-normal", action="store_true",
                   help="Flip the fitted membrane normal; changes curvature sign")

    p.add_argument("--profile-halfwidth-ang", type=float, default=80.0,
                   help="Search +/- this distance around the guide surface, along the frame normal [Å] (default: 80)")
    p.add_argument("--guide-tol-ang", type=float, default=20.0,
                   help="Final pass: reject detections further than this from the smooth guide surface [Å] (default: 20)")
    p.add_argument("--max-membrane-thickness-ang", type=float, default=80.0,
                   help="Thresholded bands thicker than this (along the local normal) are membrane fused "
                        "with protein and are not used as a plain band [Å] (default: 80)")
    p.add_argument("--leaflet-min-dip", type=float, default=0.1,
                   help="Minimum relative density dip between two leaflet peaks (default: 0.1)")

    p.add_argument("--template-iter", type=int, default=2,
                   help="Rounds of threshold-free refinement with the map's own average membrane "
                        "profile. 0: level-based detection only (default: 2)")
    p.add_argument("--template-halfwidth-ang", type=float, default=50.0,
                   help="Half-width of the membrane profile template, along the local normal [Å] (default: 50)")
    p.add_argument("--template-min-cc", type=float, default=0.4,
                   help="Minimum cross-correlation with the template to accept a column (default: 0.4)")
    p.add_argument("--rim-level-fallback", action="store_true",
                   help="Where the cut edge of the map leaves the template too little membrane profile, "
                        "keep the level-based detection (if it agrees with the surface) instead of "
                        "extrapolating. Follows a membrane further towards the mask")
    p.add_argument("--template-along-normal", action="store_true",
                   help="Steep-membrane mode: sample the membrane profile and check the mask margin along "
                        "the local membrane normal instead of along the box axis. Use for strongly curved "
                        "membranes with steep flanks")
    p.add_argument("--template-rim-agree-ang", type=float, default=0.0,
                   help="Outside the area covered by strictly accepted columns, accept columns whose two "
                        "profile halves agree within this distance instead [Å]; for membranes whose density "
                        "fades or broadens towards the edge. 0: off (default: 0)")
    p.add_argument("--template-agree-ang", type=float, default=10.0,
                   help="The two halves of the bilayer profile must locate the membrane within this "
                        "distance of each other, else the column is left out (default: 10)")
    p.add_argument("--template-margin-ang", type=float, default=5.0,
                   help="Also leave out columns within this distance of a rejected patch such as a "
                        "protein footprint. Larger: cleaner but less coverage; 0: off [Å] (default: 5)")
    p.add_argument("--template-min-amp", type=float, default=0.3,
                   help="Minimum membrane contrast relative to the average membrane (default: 0.3)")

    p.add_argument("--frame", choices=["fit", "box"], default="fit",
                   help="fit: axes from the fitted membrane plane (differs per map). "
                        "box: axes and grid of the input box, identical for all maps of the same "
                        "box size; use this to compare maps (default: fit)")
    p.add_argument("--box-axis", choices=["x", "y", "z"], default="z",
                   help="With --frame box: box axis taken as membrane normal (default: z)")
    p.add_argument("--box-max-tilt-deg", type=float, default=35.0,
                   help="With --frame box: the membrane plane is searched within this angle of --box-axis (default: 35)")
    p.add_argument("--guide-from", choices=["sheet", "level"], default="sheet",
                   help="Voxels used to find the membrane: a threshold-free sheet filter, or the "
                        "density above --level as in the original script (default: sheet)")
    p.add_argument("--sheet-sigma-ang", type=float, default=15.0,
                   help="Scale of the sheet filter, about half a bilayer thickness [Å] (default: 15)")
    p.add_argument("--detrend", choices=["plane", "offset", "none"], default="plane",
                   help="What to remove from the height map: best-fit plane, mean height, or nothing (default: plane)")
    p.add_argument("--support", choices=["disk", "hull", "detected"], default="disk",
                   help="Where a surface is reported. disk: a circle, by default the largest one that stays "
                        "on the membrane at least --mask-edge-ang inside the cut edge of the map (holes "
                        "interpolated, the rim extrapolated from the nearest detections). hull: convex "
                        "hull of the detections, no extrapolation. detected: detections only (default: disk)")
    p.add_argument("--disk-radius-ang", type=float, default=0.0,
                   help="Radius of the circle [Å]. 0: the largest that fits this map (printed). Give all "
                        "maps you compare the same value; a part that reaches past the cut edge of a map "
                        "is extrapolated, with a warning (default: 0)")
    p.add_argument("--disk-center-best", action="store_true",
                   help="Place the circle where the largest one fits on the membrane inside the map "
                        "(ignored if --disk-center-ang is given)")
    p.add_argument("--disk-center-ang", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="Center of the circle in the box-frame coordinates x', y' [Å] (the plots are then "
                        "drawn relative to it). Default: below the center of the map mask (box center if "
                        "the map is not masked)")
    p.add_argument("--height-cmap", default="YlGnBu",
                   help="Matplotlib colormap of the height map; append _r to invert it (default: YlGnBu)")
    p.add_argument("--height-gamma", type=float, default=1.0,
                   help="Power-law color scale for the height map: >1 spreads the colors over the upper "
                        "(higher) part of the range, <1 over the lower part. 1: linear (default: 1)")
    p.add_argument("--show-interp-outline", action="store_true",
                   help="Outline the inter-/extrapolated parts on the height map (dotted)")
    p.add_argument("--outline-pdb", nargs="+", default=None, metavar="MODEL",
                   help="Model(s) (.pdb or .cif, fitted in the coordinates of the input map) whose footprint "
                        "is drawn as an outline on the height map. Display only: it does not "
                        "affect the membrane detection")
    p.add_argument("--outline-resolution-ang", type=float, default=9.0,
                   help="Level of detail of the --outline-pdb outline: silhouette of the model rendered "
                        "as a map at this resolution [Å] (default: 9)")
    p.add_argument("--outline-color", default="w",
                   help="Color of the --outline-pdb outline (default: w)")
    p.add_argument("--outline-style", choices=["dotted", "dashed", "solid"], default="solid",
                   help="Line style of the --outline-pdb outline (default: solid)")
    p.add_argument("--outline-width", type=float, default=1.2,
                   help="Line width of the --outline-pdb outline [pt] (default: 1.2)")
    p.add_argument("--panel-mm", type=float, default=0.0,
                   help="Also write the height map as a figure-ready panel of this printed width [mm] "
                        "(height_map_panel.png/.pdf, fonts at true size, no color bar) plus a matching "
                        "height_colorbar.png/.pdf. 0: off (default: 0)")
    p.add_argument("--panel-font-pt", type=float, default=6.0,
                   help="Panel text size (scale bar, axes, color bar) [pt] (default: 6)")
    p.add_argument("--panel-label-pt", type=float, default=6.0,
                   help="Panel contour-label size [pt]; 0 hides the labels (default: 6)")
    p.add_argument("--panel-contour-step-ang", type=float, default=20.0,
                   help="Contour interval on the panel [Å] (default: 20)")
    p.add_argument("--panel-outline-pt", type=float, default=0.8,
                   help="Line width of the --outline-pdb outline on the panel [pt] (default: 0.8)")
    p.add_argument("--panel-axes", action="store_true",
                   help="Draw axes with ticks on the panel instead of a scale bar")
    p.add_argument("--panel-units", choices=["A", "nm"], default="nm",
                   help="Units of the panel scale bar and curvatures: nm (nm, nm^-1) or Å (Å, Å^-1) (default: nm)")
    p.add_argument("--panel-section", type=int, nargs="+", default=None, choices=[1, 2], metavar="AXIS",
                   help="Also write a density section through the map along curvature axis 1 and/or 2 "
                        "(section_axis<N>_panel.png/.pdf): map in grey, cut edge dashed, reported "
                        "membrane slab outlined. Needs --axes-angle-deg and --panel-mm")
    p.add_argument("--panel-section-square", action="store_true",
                   help="Square section view instead of a round one the size of the height map circle")
    p.add_argument("--panel-section-up-ang", type=float, default=0.0,
                   help="Round section view: move the view up by this much from the membrane at the center "
                        "(shows more of the protein, less below the membrane) [Å] (default: 0)")
    p.add_argument("--panel-section-mm", type=float, default=0.0,
                   help="Size of the section panel [mm]; 0: same scale and size as the height map area "
                        "(default: 0)")
    p.add_argument("--panel-rotate-deg", type=float, default=0.0,
                   help="Rotate the panel (map, outline, contours and curvature axes) clockwise by this "
                        "angle [deg] (default: 0)")
    p.add_argument("--panel-ref-radius-ang", type=float, default=0.0,
                   help="Draw the panel at the scale of a --panel-mm panel for a circle of this radius, so "
                        "that maps with smaller circles keep the same Å per mm (default: 0 = fill the panel)")
    p.add_argument("--height-offset-ang", type=float, default=0.0,
                   help="Add this to all reported heights [Å] (default: 0)")
    p.add_argument("--panel-scalebar-ang", type=float, default=100.0,
                   help="Length of the panel scale bar [Å]; 0: none (default: 100 = 10 nm)")
    p.add_argument("--axes-angle-deg", type=float, default=None,
                   help="Measure the curvature along two perpendicular axes through the circle center: "
                        "axis 1 at this angle counter-clockwise from x', axis 2 at +90 deg. Drawn on the "
                        "height maps, values in summary.tsv. Use one angle for all maps you compare")
    p.add_argument("--axes-band-ang", type=float, default=20.0,
                   help="Membrane within this distance of an axis is used for its curvature [Å] (default: 20)")
    p.add_argument("--axes-max-ang", type=float, default=0.0,
                   help="Use the membrane only up to this distance from the center along the axes [Å]; "
                        "0: the whole circle (default: 0)")
    p.add_argument("--axes-color", default="#404040",
                   help="Color of the curvature axes on the plots (default: #404040)")
    p.add_argument("--summary-radius-ang", type=float, default=0.0,
                   help="summary.tsv uses only the membrane within this lateral distance of the box "
                        "center, so that maps are compared over the same region. 0: all of it [Å] (default: 0)")
    p.add_argument("--mask-radius-ang", type=float, default=-1.0,
                   help="Mask that was applied to the map. -1: auto-detect the constant region around the "
                        "data (any shape/position), 0: none, >0: centered sphere of this radius [Å] (default: -1)")
    p.add_argument("--mask-edge-ang", type=float, default=12.0,
                   help="Margin kept from the cut edge of the map: no density closer than this to the mask "
                        "boundary is used for detection. Increase for a soft-edged mask [Å] (default: 12)")
    p.add_argument("--rim-one-leaflet", action="store_true",
                   help="At the rim, where the cut hides one leaflet, locate the membrane from the other "
                        "one alone instead of extrapolating. Reaches further out, but is noisier")
    p.add_argument("--leaflet-min-ang", type=float, default=20.0,
                   help="Minimum plausible leaflet-peak separation [Å] (default: 20)")
    p.add_argument("--leaflet-max-ang", type=float, default=60.0,
                   help="Maximum plausible leaflet-peak separation [Å] (default: 60)")
    p.add_argument("--surface-smooth-ang", type=float, default=20.0,
                   help="Gaussian smoothing length of extracted mid-surface [Å] (default: 20)")
    p.add_argument("--margin-ang", type=float, default=10.0,
                   help="Lateral margin around membrane candidate [Å] (default: 10)")

    p.add_argument("--contour-step-ang", type=float, default=10,
                   help="Contour interval for height map [Å] (default: 10)")
    p.add_argument("--max-thickness-vox", type=int, default=3,
                   help="Maximum aligned membrane slab thickness [vox] (default: 3)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fix-height-scale", action="store_true",
                   help="Fix the height map color scale between --height-vmin and --height-vmax")
    p.add_argument("--height-vmin", type=float, default=-60.0,
                   help="Minimum limit for height map colorbar [Å] (default: -60)")
    p.add_argument("--height-vmax", type=float, default=60.0,
                   help="Maximum limit for height map colorbar [Å] (default: 60)")
    return p


def main():
    args = build_argparser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data, voxel_xyz_A, origin_xyz_A, origin_source = read_mrc(args.map)

    if not np.allclose(voxel_xyz_A, voxel_xyz_A[0], rtol=1e-4, atol=1e-4):
        raise ValueError(
            "Anisotropic voxels are not supported in this version. "
            "Voxel size is x/y/z = %s Å." % voxel_xyz_A
        )
    vox_A = float(voxel_xyz_A[0])

    print("Input map:", args.map)
    print("Shape (z,y,x):", data.shape)
    print("Voxel size (Å):", voxel_xyz_A)
    print("World origin (Å):", origin_xyz_A, "from", origin_source)
    print("Density level:", args.level)

    # Optional protein exclusion mask.
    protein_mask = None
    if args.protein_mask:
        protein_mask, pm_vox, pm_origin, _ = read_mrc(args.protein_mask)
        if protein_mask.shape != data.shape:
            raise ValueError("Protein mask and STA map must have identical array shape.")
        if not np.allclose(pm_vox, voxel_xyz_A, rtol=1e-4, atol=1e-4):
            raise ValueError("Protein mask and STA map must have identical voxel size.")
        if not np.allclose(pm_origin, origin_xyz_A, atol=0.25 * vox_A):
            print("WARNING: protein mask origin differs from STA origin; assuming grids are index-aligned.")
        protein_mask = protein_mask > 0.5
    elif args.pdb:
        protein_mask = pdb_to_mask(
            args.pdb, data.shape, voxel_xyz_A, origin_xyz_A, args.pdb_radius_ang
        )

    if protein_mask is not None and args.protein_dilate_ang > 0:
        dil_px = int(math.ceil(args.protein_dilate_ang / vox_A))
        if dil_px > 0:
            protein_mask = ndimage.binary_dilation(protein_mask, iterations=dil_px)
        write_mrc(outdir / "protein_exclusion_mask.mrc", protein_mask.astype(np.float32), voxel_xyz_A, origin_xyz_A)

    # Threshold raw density exactly at the requested level.
    candidate = data >= args.level
    if protein_mask is not None:
        candidate &= ~protein_mask

    # Remove very small disconnected islands.
    labels, nlab = ndimage.label(candidate)
    if nlab > 0:
        counts = np.bincount(labels.ravel())
        keep_ids = np.where(counts >= args.min_component_vox)[0]
        keep = np.zeros_like(counts, dtype=bool)
        keep[keep_ids] = True
        keep[0] = False
        candidate = keep[labels]

    n_candidate = int(candidate.sum())
    if n_candidate < 500:
        raise RuntimeError(
            "Only %d candidate voxels remain after threshold/masking. "
            "Check --level and protein mask." % n_candidate
        )
    write_mrc(outdir / "membrane_candidate_thresholded.mrc", candidate.astype(np.float32), voxel_xyz_A, origin_xyz_A)

    # The frame origin is always the geometric center of the MRC map volume.
    grid_center_zyx = (np.array(data.shape, dtype=float) - 1.0) / 2.0
    center_xyz = origin_xyz_A + grid_center_zyx[::-1] * voxel_xyz_A
    shape_xyz = np.array(data.shape[::-1], dtype=int)

    # Mask applied to the input map: nothing in or beyond its edge is membrane.
    if args.mask_radius_ang < 0:
        inside_vol = estimate_mask_region(data)
    elif args.mask_radius_ang > 0:
        zz, yy, xx = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
        rr = np.sqrt((zz - grid_center_zyx[0]) ** 2 + (yy - grid_center_zyx[1]) ** 2 + (xx - grid_center_zyx[2]) ** 2)
        inside_vol = rr * vox_A <= args.mask_radius_ang
    else:
        inside_vol = None
    usable_vol = None
    if inside_vol is None:
        print("Map mask: none detected")
    else:
        cz, cy, cx = ndimage.center_of_mass(inside_vol)
        print("Map mask: %.1f%% of the box, equivalent radius %.1f Å, center offset from box center (x,y,z) %.1f %.1f %.1f Å"
              % (100.0 * inside_vol.mean(), (3.0 * inside_vol.sum() / (4.0 * math.pi)) ** (1.0 / 3.0) * vox_A,
                 (cx - grid_center_zyx[2]) * vox_A, (cy - grid_center_zyx[1]) * vox_A, (cz - grid_center_zyx[0]) * vox_A))
        if args.mask_edge_ang > 0:
            usable_vol = ndimage.distance_transform_edt(inside_vol, sampling=vox_A) > args.mask_edge_ang
        else:
            usable_vol = inside_vol

    # Membrane candidates for the guide surface.
    fit_candidate = candidate
    if args.guide_from == "sheet":
        sheet = membrane_sheet_candidates(data, inside_vol, vox_A, sigma_A=args.sheet_sigma_ang,
                                          edge_A=args.mask_edge_ang)
        if protein_mask is not None:
            sheet &= ~protein_mask
        if int(sheet.sum()) >= 500:
            fit_candidate = sheet
            print("Sheet-like (membrane) candidate voxels:", int(sheet.sum()))
        else:
            print("WARNING: sheet filter found too little; using thresholded density for the guide.")
        write_mrc(outdir / "membrane_candidate_for_fit.mrc", fit_candidate.astype(np.float32), voxel_xyz_A, origin_xyz_A)

    idx_zyx = np.argwhere(fit_candidate)
    pts_xyz = xyz_from_zyx_indices(idx_zyx, voxel_xyz_A, origin_xyz_A)

    prior = None
    if args.frame == "box":
        ax_n = "xyz".index(args.box_axis)
        ax_1 = (ax_n + 1) % 3
        prior = np.zeros(3)
        prior[ax_n] = 1.0
    # A cell of the lateral coverage grid counts only if a sheet's worth of
    # candidate voxels falls in it (stray noise voxels do not make a membrane).
    min_count = max(1, int(round(0.15 * 2.0 * (args.plane_inlier_ang / vox_A) ** 3)))
    seeds = ransac_plane_seeds(
        pts_xyz,
        n_iter=args.ransac_iter,
        inlier_A=args.plane_inlier_ang,
        seed=args.seed,
        min_count=min_count,
        prior_normal=prior,
        max_tilt_deg=args.box_max_tilt_deg,
    )

    def frame_for(normal_xyz):
        if args.frame == "box":
            fn = np.zeros(3)
            fn[ax_n] = -1.0 if args.flip_normal else 1.0
            f1 = np.zeros(3)
            f1[ax_1] = 1.0
            return f1, np.cross(fn, f1), fn
        nrm = orient_normal(normal_xyz)
        if args.flip_normal:
            nrm = -nrm
        return make_plane_basis(nrm)

    # Grow every seed plane into a quadric; keep the one that covers the
    # largest area. A slab of protein cannot grow into a wide sheet.
    best = None
    for s_center, s_normal in seeds:
        f1, f2, fn = frame_for(s_normal)
        coef, n_in, cover = grow_guide(pts_xyz, center_xyz, f1, f2, fn, s_center, s_normal,
                                       args.plane_inlier_ang, seed=args.seed, min_count=min_count)
        if coef is not None and (best is None or cover > best[0]):
            best = (cover, n_in, coef, f1, f2, fn, s_center, s_normal)
    if best is None:
        raise RuntimeError("Could not fit a membrane guide surface. Check --level / --guide-from.")
    _, n_guide, guide_coef, e1, e2, n, ransac_center_xyz, normal_xyz = best

    if args.frame != "box":
        # Frame normal = best-fit plane normal of the whole membrane patch.
        q = pts_xyz - center_xyz[None, :]
        m = np.abs(q @ n - eval_quadric(guide_coef, q @ e1, q @ e2)) <= args.plane_inlier_ang
        p_center, p_normal = fit_plane_pca(pts_xyz[m])
        f1, f2, fn = frame_for(p_normal)
        qm = pts_xyz[m] - center_xyz[None, :]
        coef0, _, _, _ = np.linalg.lstsq(quadric_design(qm @ f1, qm @ f2), qm @ fn, rcond=None)
        coef, n_in = fit_quadric_irls(q @ f1, q @ f2, q @ fn, coef0, inlier_A=args.plane_inlier_ang, seed=args.seed)
        if n_in >= 0.8 * n_guide:
            guide_coef, n_guide, e1, e2, n, normal_xyz = coef, n_in, f1, f2, fn, p_normal
    else:
        ax_2 = int(np.argmax(np.abs(e2)))
        tilt_deg = math.degrees(math.acos(min(1.0, abs(float(normal_xyz @ n)))))
        print("Box frame: seed membrane plane is %.1f deg from box %s" % (tilt_deg, args.box_axis))
    print("Frame center xyz (Å):", center_xyz)
    print("Frame normal xyz:", n)
    print("Guide surface: %d candidate voxels within %.0f Å (of %d), from %d seed planes"
          % (n_guide, args.plane_inlier_ang, len(pts_xyz), len(seeds)))

    q = pts_xyz - center_xyz[None, :]
    u_pts = q @ e1
    v_pts = q @ e2

    if args.frame == "box":
        # Input grid: every map of the same box size gives the same (u, v) pixels.
        u_A = (np.arange(shape_xyz[ax_1]) - (shape_xyz[ax_1] - 1) / 2.0) * vox_A
        v_A = (np.arange(shape_xyz[ax_2]) - (shape_xyz[ax_2] - 1) / 2.0) * vox_A
    else:
        # Lateral extent from membrane-candidate points, not full map corners.
        umin = float(np.percentile(u_pts, 0.5) - args.margin_ang)
        umax = float(np.percentile(u_pts, 99.5) + args.margin_ang)
        vmin = float(np.percentile(v_pts, 0.5) - args.margin_ang)
        vmax = float(np.percentile(v_pts, 99.5) + args.margin_ang)
        u_A = np.arange(umin, umax + 0.5 * vox_A, vox_A, dtype=float)
        v_A = np.arange(vmin, vmax + 0.5 * vox_A, vox_A, dtype=float)
    U, V = np.meshgrid(u_A, v_A, indexing="xy")

    guide = eval_quadric(guide_coef, U, V)
    gu = 2.0 * guide_coef[0] * U + guide_coef[1] * V + guide_coef[3]
    gv = guide_coef[1] * U + 2.0 * guide_coef[2] * V + guide_coef[4]
    cos_guide = 1.0 / np.sqrt(1.0 + gu * gu + gv * gv)

    # Normal range: wherever the guide goes, +/- the search half-width.
    hw = float(args.profile_halfwidth_ang)
    reach = max(hw, 1.5 * (args.template_halfwidth_ang + args.guide_tol_ang))
    ext = 0.5 * float(np.sum(np.abs(n) * (shape_xyz - 1) * voxel_xyz_A))
    lat_ext = 0.5 * float(np.sum((np.abs(e1) + np.abs(e2)) * (shape_xyz - 1) * voxel_xyz_A))
    near = (np.abs(U) <= lat_ext) & (np.abs(V) <= lat_ext)
    wlo = max(float(guide[near].min()) - reach, -ext)
    whi = min(float(guide[near].max()) + reach, ext)
    if args.frame == "box":
        w_A = (np.arange(shape_xyz[ax_n]) - (shape_xyz[ax_n] - 1) / 2.0) * vox_A
        w_A = w_A[(w_A >= wlo - 0.5 * vox_A) & (w_A <= whi + 0.5 * vox_A)]
    else:
        w_A = np.arange(math.floor(wlo / vox_A), math.ceil(whi / vox_A) + 1, dtype=float) * vox_A

    # Mild smoothing for peak detection only; thresholding above used raw density.
    if args.analysis_sigma_vox > 0:
        analysis_data = ndimage.gaussian_filter(data, sigma=args.analysis_sigma_vox)
    else:
        analysis_data = data

    print("Sampling aligned local volume: (w,v,u) =", (len(w_A), len(v_A), len(u_A)))
    local = sample_uv_w_volume(
        analysis_data, u_A, v_A, w_A, center_xyz, e1, e2, n,
        voxel_xyz_A, origin_xyz_A, order=1, cval=float(data.min())
    )

    # Raw aligned volume for output/QC.
    local_raw = sample_uv_w_volume(
        data, u_A, v_A, w_A, center_xyz, e1, e2, n,
        voxel_xyz_A, origin_xyz_A, order=1, cval=0.0
    )

    local_protein = None
    protein_footprint = None
    if protein_mask is not None:
        local_protein = sample_uv_w_volume(
            protein_mask.astype(np.float32), u_A, v_A, w_A, center_xyz, e1, e2, n,
            voxel_xyz_A, origin_xyz_A, order=0, cval=0.0
        )
        protein_footprint = np.max(local_protein, axis=0) > 0.5

    # Voxels that may be used: inside the map mask (minus its edge).
    local_inside = None
    if usable_vol is not None:
        local_inside = sample_uv_w_volume(
            usable_vol.astype(np.float32), u_A, v_A, w_A, center_xyz, e1, e2, n,
            voxel_xyz_A, origin_xyz_A, order=0, cval=0.0
        ) > 0.5

    sigma_px = args.surface_smooth_ang / vox_A
    baseline = float(np.median(data[inside_vol])) if inside_vol is not None else float(np.median(data))
    detect_kw = dict(
        leaflet_min_A=args.leaflet_min_ang,
        leaflet_max_A=args.leaflet_max_ang,
        protein_local_wvu=local_protein,
        halfwidth_A=hw,
        max_thick_A=args.max_membrane_thickness_ang,
        baseline=baseline,
        min_dip=args.leaflet_min_dip,
    )

    # Pass 1: loose search around the quadric guide.
    tol1 = 0.5 * hw
    surface1, _, method1, thick1 = detect_surface_from_profiles(
        local, w_A, args.level, guide_vu=guide, cos_vu=cos_guide, tol_A=tol1, **detect_kw
    )
    thick_ref = float(np.nanmedian(thick1)) if np.any(np.isfinite(thick1)) else None

    def clipped_by_mask(surf, cosang, slope_surf=None):
        """Columns whose membrane (full thickness) is not entirely inside the map mask."""
        if local_inside is None:
            return np.zeros(surf.shape, dtype=bool)
        if args.template_along_normal and slope_surf is not None:
            # Membrane thickness checked along the local normal (taken from the
            # smooth slope_surf), not along the column: on steep flanks the
            # column cuts the membrane over a long stretch.
            half_n = 0.5 * (thick_ref if thick_ref is not None else args.leaflet_max_ang)
            gv_s, gu_s = np.gradient(slope_surf, vox_A, vox_A)
            g_s = np.sqrt(1.0 + gu_s * gu_s + gv_s * gv_s)
            jj_, ii_ = np.indices(surf.shape, dtype=float)
            sf = np.nan_to_num(surf, nan=0.0)
            dw_ = float(w_A[1] - w_A[0])
            out_ = np.zeros(surf.shape, dtype=bool)
            for o in np.linspace(-half_n, half_n, 9):
                wq = sf + o / g_s
                inside_q = map_coordinates(local_inside.astype(np.float32),
                                           [(wq - w_A[0]) / dw_, jj_ - o * gv_s / g_s / vox_A,
                                            ii_ - o * gu_s / g_s / vox_A], order=0, mode="constant", cval=0.0) > 0.5
                out_ |= ~inside_q
            return out_
        half = 0.5 * (thick_ref if thick_ref is not None else args.leaflet_max_ang) / cosang
        near_w = np.abs(w_A[:, None, None] - np.nan_to_num(surf, nan=0.0)[None]) <= half[None]
        beyond = (surf - half < w_A[0]) | (surf + half > w_A[-1])
        return np.any(near_w & ~local_inside, axis=0) | beyond

    steep = args.template_along_normal
    surface1[clipped_by_mask(surface1, cos_guide, guide)
             | (False if steep else clipped_by_mask(guide, cos_guide))] = np.nan
    valid1 = np.isfinite(surface1)
    if np.count_nonzero(valid1) < 50:
        raise RuntimeError(
            "Almost no membrane was detected around the fitted surface. "
            "Try a different --level or provide a protein mask."
        )
    if thick_ref is not None:
        print("Membrane thickness at this level (Å): %.1f" % thick_ref)

    # Pass 2: tight search around the smoothed pass-1 surface.
    coef1, _ = fit_quadric_irls(U[valid1], V[valid1], surface1[valid1], guide_coef, inlier_A=tol1, seed=args.seed)
    base1 = eval_quadric(coef1, U, V)
    keep1 = valid1 & (np.abs(surface1 - base1) <= tol1)
    guide2 = smooth_surface(surface1, keep1, base1, sigma_px=max(2.0 * sigma_px, 2.0))
    g2v, g2u = np.gradient(guide2, vox_A, vox_A)
    cos2 = 1.0 / np.sqrt(1.0 + g2u * g2u + g2v * g2v)
    surface_raw, confidence, method, _ = detect_surface_from_profiles(
        local, w_A, args.level, guide_vu=guide2, cos_vu=cos2,
        tol_A=args.guide_tol_ang, thick_ref_A=thick_ref, **detect_kw
    )

    # Drop detections where the spherical mask cuts into the membrane: the
    # truncated band is biased towards the box center (flattens the rim).
    out = np.isfinite(surface_raw) & (clipped_by_mask(surface_raw, cos2, guide2)
                                      | (False if steep else clipped_by_mask(guide2, cos2)))
    surface_raw[out] = np.nan
    confidence[out] = 0.0
    method[out] = 0

    support_mode = "hull" if args.support == "disk" else args.support
    valid, support = support_from_detections(np.isfinite(surface_raw), mode=support_mode)
    surface_raw[~valid] = np.nan
    confidence[~valid] = 0.0
    method[~valid] = 0

    frac = float(np.count_nonzero(valid)) / max(1, np.count_nonzero(support))
    print("Valid surface detections: %.1f%% of the reported area" % (100.0 * frac))
    if np.count_nonzero(valid) < 0.02 * valid.size or np.count_nonzero(valid) < 50:
        raise RuntimeError(
            "Too few lateral positions yielded membrane detections. "
            "Try a different --level, larger --profile-halfwidth-ang, or provide a protein mask."
        )

    # Global quadric of the detections (also the summary shape descriptor),
    # then the local surface = quadric + smoothed residual.
    fit_coef, _ = fit_quadric_irls(
        U[valid], V[valid], surface_raw[valid], coef1,
        inlier_A=max(args.guide_tol_ang, 3.0 * vox_A), seed=args.seed
    )
    surface_full = smooth_surface(surface_raw, valid, eval_quadric(fit_coef, U, V), sigma_px=sigma_px)

    support &= ~(clipped_by_mask(surface_full, cos2, surface_full)
                 | (False if steep else clipped_by_mask(guide2, cos2)))

    # Threshold-free refinement: slide the map's own average membrane profile
    # along each column, starting from the level-based surface.
    template = None
    if args.template_iter > 0:
        usable = local_inside
        clean_cols = valid.copy()
        level_raw = surface_raw.copy()
        level_method = method.copy()
        for it in range(int(args.template_iter)):
            gv_, gu_ = np.gradient(surface_full, vox_A, vox_A)
            cos_r = 1.0 / np.sqrt(1.0 + gu_ * gu_ + gv_ * gv_)
            res = refine_surface_by_template(
                local, w_A, surface_full, cos_r, clean_cols,
                halfwidth_A=args.template_halfwidth_ang, tol_A=args.guide_tol_ang, step_A=0.5 * vox_A,
                usable_wvu=usable, min_cc=args.template_min_cc, min_amp=args.template_min_amp,
                to_cut_edge=args.rim_one_leaflet,
                agree_A=args.template_agree_ang, grow_px=int(math.ceil(args.template_margin_ang / vox_A)),
                protein_wvu=local_protein,
                grad_uv=(gu_, gv_) if args.template_along_normal else None, dl_A=vox_A,
                rim_agree_A=args.template_rim_agree_ang,
            )
            if res is None or np.count_nonzero(np.isfinite(res[0])) < 50:
                print("WARNING: template refinement found too little membrane; keeping the level-based surface.")
                break
            refined, cc_map, amp_map, template, cut_cols = res
            from_level = np.zeros(refined.shape, dtype=bool)
            if args.rim_level_fallback:
                # Near the cut edge the template window is incomplete; keep the
                # level-based detection there if it agrees with the surface.
                from_level = (~np.isfinite(refined) & cut_cols & np.isfinite(level_raw)
                              & (np.abs(level_raw - surface_full) * cos_r <= args.guide_tol_ang))
                refined = np.where(from_level, level_raw, refined)
            valid, support = support_from_detections(np.isfinite(refined), mode=support_mode)
            surface_raw = np.where(valid, refined, np.nan).astype(np.float32)
            confidence = np.where(valid, cc_map, 0.0).astype(np.float32)
            method = np.where(valid, np.where(from_level, level_method, 4), 0).astype(np.uint8)
            fit_coef, _ = fit_quadric_irls(
                U[valid], V[valid], surface_raw[valid], fit_coef,
                inlier_A=max(args.guide_tol_ang, 3.0 * vox_A), seed=args.seed
            )
            surface_full = smooth_surface(surface_raw, valid, eval_quadric(fit_coef, U, V), sigma_px=sigma_px)
            clean_cols = valid
            print("Template refinement %d: %d columns, median cc %.2f%s"
                  % (it + 1, np.count_nonzero(valid), float(np.median(cc_map[valid & ~from_level])),
                     (" (+%d level-based at the cut edge)" % np.count_nonzero(valid & from_level))
                     if args.rim_level_fallback else ""))
        if local_inside is not None:
            # Reported only where the surface itself lies inside the map mask.
            wi = np.clip(np.rint((surface_full - w_A[0]) / vox_A).astype(int), 0, len(w_A) - 1)
            support &= np.take_along_axis(local_inside, wi[None], axis=0)[0]
        frac = float(np.count_nonzero(valid)) / max(1, np.count_nonzero(support))
        print("Template detections: %.1f%% of the detected outline" % (100.0 * frac))

    # Regular reporting region: a circle, the same shape for every map.
    disk = None
    if args.support == "disk":
        if inside_vol is not None:
            # Depth of every surface point inside the map mask [A]; the circle
            # is centered below the center of the mask.
            depth_vol = ndimage.distance_transform_edt(inside_vol, sampling=vox_A).astype(np.float32)
            p_xyz = (center_xyz[None, None, :] + U[..., None] * e1 + V[..., None] * e2
                     + surface_full[..., None] * n)
            idx = (p_xyz - origin_xyz_A) / voxel_xyz_A
            depth = map_coordinates(depth_vol, [idx[..., 2], idx[..., 1], idx[..., 0]],
                                    order=1, mode="constant", cval=0.0)
            within = depth >= args.mask_edge_ang
            mz, my, mx = ndimage.center_of_mass(inside_vol)
            m_xyz = origin_xyz_A + np.array([mx, my, mz]) * voxel_xyz_A - center_xyz
            cu, cv = float(m_xyz @ e1), float(m_xyz @ e2)
        else:
            # Unmasked map: inside the outline of the detections.
            within = support.copy()
            cu, cv = 0.0, 0.0
        if args.disk_center_ang is not None:
            cu, cv = float(args.disk_center_ang[0]), float(args.disk_center_ang[1])
        elif args.disk_center_best:
            # Center of the largest circle that fits on the membrane inside the map.
            lab_b, _ = ndimage.label(within)
            if lab_b.max() > 0:
                counts = np.bincount(lab_b.ravel())
                counts[0] = 0
                big = lab_b == counts.argmax()
                dist_in = ndimage.distance_transform_edt(np.pad(big, 1))[1:-1, 1:-1]
                jb, ib = np.unravel_index(np.argmax(dist_in), dist_in.shape)
                cu, cv = float(u_A[ib]), float(v_A[jb])
        rr = np.hypot(U - cu, V - cv)
        lab_a, _ = ndimage.label(within)
        jc = int(np.argmin(np.abs(v_A - cv)))
        ic = int(np.argmin(np.abs(u_A - cu)))
        if lab_a[jc, ic] == 0:
            raise RuntimeError("The circle center is not on the membrane inside the map; set --disk-center-ang.")
        within = lab_a == lab_a[jc, ic]
        # Largest circle that stays on the membrane inside the margin.
        r_fit = float(rr[~within].min() - 0.5 * vox_A) if np.any(~within) else float(rr.max())
        r_fit = min(r_fit, u_A[-1] - cu, cu - u_A[0], v_A[-1] - cv, cv - v_A[0])
        radius = float(args.disk_radius_ang) if args.disk_radius_ang > 0 else r_fit
        support = rr <= radius
        disk = (cu, cv, radius, U, V, radius - rr)
        eq_radius = radius
        near = ndimage.distance_transform_edt(~valid) * vox_A
        frac = float(np.count_nonzero(valid & support)) / max(1, np.count_nonzero(support))
        print("Circle: center x', y' = %.1f, %.1f Å, radius %.1f Å (largest that fits this map: %.1f Å)"
              % (cu, cv, radius, r_fit))
        if radius > r_fit + 1e-6:
            print("WARNING: the circle goes up to %.0f Å beyond the %.0f Å margin from the cut edge of this "
                  "map (%.1f%% of its area); the surface there is extrapolated."
                  % (radius - r_fit, args.mask_edge_ang, 100.0 * np.count_nonzero(support & ~within) / max(1, np.count_nonzero(support))))
        print("Circle: %.1f%% of it detected directly; furthest inter-/extrapolation %.0f Å from a detection"
              % (100.0 * frac, float(near[support].max())))

    # Height map. Curvature is unaffected by the subtraction of a plane.
    if args.detrend == "plane":
        height_full, trend_coef, trend = detrend_plane(surface_full, u_A, v_A, valid)
    elif args.detrend == "offset":
        trend_coef = np.array([0.0, 0.0, float(np.mean(surface_full[valid]))])
        height_full = surface_full - trend_coef[2]
    else:
        trend_coef = np.zeros(3)
        height_full = surface_full.copy()
    if args.height_offset_ang:
        # Shift all reported heights, e.g. to put a complex that sits lower in this
        # box at the same height as in the other maps.
        height_full = height_full + args.height_offset_ang

    H_Ainv, K_Ainv2, k1_Ainv, k2_Ainv = compute_curvature(height_full, vox_A, vox_A)
    H_nminv = H_Ainv * 10.0
    K_nminv2 = K_Ainv2 * 100.0
    k1_nminv = k1_Ainv * 10.0
    k2_nminv = k2_Ainv * 10.0

    # Nothing is reported outside the support.
    surface_smooth = np.where(support, surface_full, np.nan)
    height_A = np.where(support, height_full, np.nan)
    for arr in (H_nminv, K_nminv2, k1_nminv, k2_nminv):
        arr[~support] = np.nan

    # Curvature region: support minus a rim where derivatives are dominated by
    # the boundary.
    edge = max(2, int(math.ceil(2.0 * sigma_px)))
    outline = ndimage.binary_fill_holes(ndimage.binary_closing(
        np.pad(support, 8), iterations=max(1, int(math.ceil(sigma_px)))))[8:-8, 8:-8] | support
    analysis_mask = ndimage.binary_erosion(outline, iterations=max(1, int(math.ceil(1.5 * sigma_px))), border_value=1) & support
    if edge * 2 < min(analysis_mask.shape) and args.frame != "box":
        analysis_mask[:edge, :] = False
        analysis_mask[-edge:, :] = False
        analysis_mask[:, :edge] = False
        analysis_mask[:, -edge:] = False

    # Curvature is only meaningful near real detections: an interpolated hole
    # (under protein) has a height estimate but no measured shape.
    interpolated = support & (ndimage.distance_transform_edt(~valid) > max(1.5, sigma_px))
    analysis_mask &= ~interpolated

    H_plot = H_nminv.copy()
    K_plot = K_nminv2.copy()
    H_plot[~analysis_mask] = np.nan
    K_plot[~analysis_mask] = np.nan

    # Save arrays.
    # Footprint of the model(s) to draw on the height map.
    outlines = []
    pdb_footprint = np.zeros(U.shape, dtype=bool)
    for model in (args.outline_pdb or []):
        atoms = parse_pdb_atoms(model) - center_xyz[None, :]
        au, av = atoms @ e1, atoms @ e2
        in_box = (au >= u_A[0]) & (au <= u_A[-1]) & (av >= v_A[0]) & (av <= v_A[-1])
        print("Outline model %s: %d atoms, %.0f%% inside the map laterally, heights %.0f to %.0f Å"
              % (os.path.basename(model), len(atoms), 100.0 * in_box.mean(),
                 float((atoms @ n).min()), float((atoms @ n).max())))
        if in_box.mean() < 0.5:
            print("WARNING: most of %s lies outside the map; is it in the coordinate system of the input map?"
                  % os.path.basename(model))
        gx, gy, field = model_footprint(au, av, atoms @ n, resolution_A=args.outline_resolution_ang)
        outlines.append((gx, gy, field))
        iy = np.clip(np.rint((V - gy[0]) / (gy[1] - gy[0])).astype(int), 0, len(gy) - 1)
        ix = np.clip(np.rint((U - gx[0]) / (gx[1] - gx[0])).astype(int), 0, len(gx) - 1)
        inside_grid = (U >= gx[0]) & (U <= gx[-1]) & (V >= gy[0]) & (V <= gy[-1])
        pdb_footprint |= inside_grid & (field[iy, ix] < 0)

    if template is not None:
        np.savetxt(outdir / "membrane_profile_template.csv", np.column_stack(template), delimiter=",",
                   header="offset_along_normal_A,density", comments="")
    np.savez_compressed(
        outdir / "membrane_surface.npz",
        u_A=u_A,
        v_A=v_A,
        surface_raw_A=surface_raw,
        surface_smooth_A=surface_smooth,
        height_A=height_A,
        confidence=confidence,
        detection_method=method,
        mean_curvature_nm_inv=H_nminv,
        gaussian_curvature_nm_inv2=K_nminv2,
        principal_curvature_1_nm_inv=k1_nminv,
        principal_curvature_2_nm_inv=k2_nminv,
        valid_detection=valid,
        support=support,
        interpolated=interpolated,
        pdb_footprint=pdb_footprint,
        analysis_mask=analysis_mask,
        quadric_coef=fit_coef,
        disk_center_radius_A=np.array([disk[0], disk[1], eq_radius] if disk is not None else [np.nan] * 3, dtype=float),
    )

    # Long-form CSV.
    csv = np.column_stack([
        U.ravel(), V.ravel(),
        surface_raw.ravel(), surface_smooth.ravel(), height_A.ravel(),
        H_nminv.ravel(), K_nminv2.ravel(), k1_nminv.ravel(), k2_nminv.ravel(),
        confidence.ravel(), method.ravel(), valid.astype(np.uint8).ravel(),
        support.astype(np.uint8).ravel(),
    ])
    header = (
        "xprime_A,yprime_A,surface_raw_A,surface_smooth_A,height_A,"
        "mean_curvature_nm^-1,gaussian_curvature_nm^-2,"
        "principal_k1_nm^-1,principal_k2_nm^-1,confidence,detection_method,valid,support"
    )
    np.savetxt(outdir / "membrane_surface.csv", csv, delimiter=",", header=header, comments="")

    curv_axes = None
    if args.axes_angle_deg is not None:
        ax_center = (disk[0], disk[1]) if disk is not None else (0.0, 0.0)
        ax_meas = valid & support if disk is not None else valid
        curv_axes = curvature_along_axes(U, V, surface_full, ax_meas, ax_center, args.axes_angle_deg,
                                         band_A=args.axes_band_ang, max_A=args.axes_max_ang or None)
        for a in curv_axes:
            print("Curvature along axis %d (%.0f deg): %+.4f +- %.4f nm^-1 (R = %.0f nm), %d px, data from %.0f to %.0f Å"
                  % (a["axis"], a["angle"], 10 * a["kappa"], 10 * a["se"],
                     abs(0.1 / a["kappa"]) if a["kappa"] else float("inf"), a["n"], a["s_min"], a["s_max"]))

    # One-line summary, comparable between maps (use --frame box for that).
    sum_coef = fit_coef
    am = analysis_mask & np.isfinite(H_nminv)
    if args.summary_radius_ang > 0 or disk is not None:
        # Same lateral region for every map: the disk, and/or a disc around the box center.
        disc = np.ones(U.shape, dtype=bool)
        if disk is not None:
            disc &= support
        if args.summary_radius_ang > 0:
            disc &= np.hypot(U, V) <= args.summary_radius_ang
        am &= disc
        if np.count_nonzero(valid & disc) >= 50:
            sum_coef, _ = fit_quadric_irls(
                U[valid & disc], V[valid & disc], surface_raw[valid & disc], fit_coef,
                inlier_A=max(args.guide_tol_ang, 3.0 * vox_A), seed=args.seed
            )
        else:
            print("WARNING: too few detections inside the summary region; summary uses all detections.")
    qH, qK, qk1, qk2, qang = quadric_curvature_at_center(sum_coef)
    summary = [
        ("map", os.path.basename(args.map)),
        ("frame", args.frame),
        ("level", "%.6g" % args.level),
        ("area_nm2", "%.1f" % (np.count_nonzero(support) * vox_A * vox_A / 100.0)),
        ("detected_fraction", "%.3f" % frac),
        ("curvature_area_nm2", "%.1f" % (np.count_nonzero(analysis_mask) * vox_A * vox_A / 100.0)),
        ("summary_radius_A", "%.6g" % args.summary_radius_ang),
        ("disk_radius_A", "%.2f" % (eq_radius if disk is not None else float("nan"))),
        ("disk_center_x_A", "%.2f" % (disk[0] if disk is not None else float("nan"))),
        ("disk_center_y_A", "%.2f" % (disk[1] if disk is not None else float("nan"))),
        ("max_extrapolation_A", "%.1f" % float((ndimage.distance_transform_edt(~valid) * vox_A)[support].max())),
        ("center_height_A", "%.2f" % sum_coef[5]),
        ("tilt_deg", "%.2f" % math.degrees(math.atan(math.hypot(sum_coef[3], sum_coef[4])))),
        ("quadric_H_nm^-1", "%.5f" % (qH * 10.0)),
        ("quadric_K_nm^-2", "%.6f" % (qK * 100.0)),
        ("quadric_k1_nm^-1", "%.5f" % (qk1 * 10.0)),
        ("quadric_k2_nm^-1", "%.5f" % (qk2 * 10.0)),
        ("quadric_k1_angle_deg", "%.1f" % qang),
        ("local_H_mean_nm^-1", "%.5f" % (float(np.mean(H_nminv[am])) if np.any(am) else float("nan"))),
        ("local_H_sd_nm^-1", "%.5f" % (float(np.std(H_nminv[am])) if np.any(am) else float("nan"))),
        ("height_range_A", "%.1f" % (float(np.nanmax(height_A) - np.nanmin(height_A)))),
    ]
    if curv_axes:
        for a in curv_axes:
            th = math.radians(a["angle"])
            q_ = sum_coef
            k_q = (2 * q_[0] * math.cos(th) ** 2 + 2 * q_[1] * math.cos(th) * math.sin(th)
                   + 2 * q_[2] * math.sin(th) ** 2)
            summary += [
                ("axis%d_angle_deg" % a["axis"], "%.1f" % a["angle"]),
                ("axis%d_kappa_nm^-1" % a["axis"], "%.5f" % (10 * a["kappa"])),
                ("axis%d_kappa_se_nm^-1" % a["axis"], "%.5f" % (10 * a["se"])),
                ("axis%d_radius_nm" % a["axis"], "%.1f" % (0.1 / a["kappa"]) if a["kappa"] else "inf"),
                ("axis%d_quadric_kappa_nm^-1" % a["axis"], "%.5f" % (10 * k_q)),
            ]
    with open(outdir / "summary.tsv", "w") as f:
        f.write("\t".join(k for k, _ in summary) + "\n")
        f.write("\t".join(v for _, v in summary) + "\n")
    print("Quadric fit: H = %.4f nm^-1, k1 = %.4f, k2 = %.4f nm^-1" % (qH * 10.0, qk1 * 10.0, qk2 * 10.0))

    # Plots.
    height_plot = height_A
    interp_plot = interpolated
    if disk is not None:
        # Fill the pixel staircase under the outline (the image is clipped by it).
        grown = ndimage.binary_dilation(support, iterations=2)
        height_plot = np.where(grown, height_full, np.nan)
        interp_plot = interpolated | (grown & ~support & ndimage.binary_dilation(interpolated, iterations=3))
    # Plots are drawn in coordinates relative to the circle center, so that
    # maps analysed with different --disk-center-ang share the same axes.
    pu, pv, pdisk, poutlines = u_A, v_A, disk, outlines
    if disk is not None:
        cu0, cv0 = disk[0], disk[1]
        pu, pv = u_A - cu0, v_A - cv0
        pdisk = (0.0, 0.0, disk[2], disk[3] - cu0, disk[4] - cv0, disk[5])
        poutlines = [(gx - cu0, gy - cv0, fld) for gx, gy, fld in outlines]
    plot_height(
        outdir / "height_map.png",
        height_plot,
        pu,
        pv,
        args.contour_step_ang,
        protein_footprint=protein_footprint,
        interpolated=interp_plot if args.show_interp_outline else None,
        outlines=poutlines,
        outline_color=args.outline_color,
        outline_style=args.outline_style,
        outline_width=args.outline_width,
        gamma=args.height_gamma,
        curv_axes=curv_axes,
        curv_color=args.axes_color,
        cmap=args.height_cmap,
        disk=pdisk,
        vmin=args.height_vmin if args.fix_height_scale else None,
        vmax=args.height_vmax if args.fix_height_scale else None,
    )
    if args.panel_mm > 0:
        plot_height_panel(
            outdir / "height_map_panel", height_plot, pu, pv, args.panel_contour_step_ang,
            size_mm=args.panel_mm, font_pt=args.panel_font_pt, label_pt=args.panel_label_pt,
            vmin=args.height_vmin if args.fix_height_scale else None,
            vmax=args.height_vmax if args.fix_height_scale else None,
            gamma=args.height_gamma, cmap=args.height_cmap, disk=pdisk, outlines=poutlines,
            outline_color=args.outline_color, outline_style=args.outline_style,
            outline_pt=args.panel_outline_pt, axes=args.panel_axes,
            scalebar_A=args.panel_scalebar_ang,
            curv_axes=curv_axes, curv_color=args.axes_color, ref_radius_A=args.panel_ref_radius_ang or None,
            units=args.panel_units, rotate_cw_deg=args.panel_rotate_deg,
        )
        lo_c, hi_c = ((args.height_vmin, args.height_vmax) if args.fix_height_scale
                      else finite_range(height_plot))
        plot_height_colorbar(outdir / "height_colorbar", lo_c, hi_c, gamma=args.height_gamma,
                             cmap=args.height_cmap, width_mm=args.panel_mm, font_pt=args.panel_font_pt)
        if args.panel_section and curv_axes and disk is not None:
            for a in curv_axes:
                if a["axis"] not in args.panel_section:
                    continue
                plot_section_panel(
                    outdir / ("section_axis%d_panel" % a["axis"]), data, inside_vol, voxel_xyz_A, origin_xyz_A,
                    center_xyz, e1, e2, n, (disk[0], disk[1]), a["angle"], surface_full, support, u_A, v_A,
                    size_mm=args.panel_section_mm or args.panel_mm * 1.24, half_A=1.24 * disk[2],
                    circle_A=None if args.panel_section_square else disk[2], up_A=args.panel_section_up_ang,
                    font_pt=args.panel_font_pt,
                    slab_A=max(1, int(args.max_thickness_vox)) * vox_A, scalebar_A=args.panel_scalebar_ang,
                    units=args.panel_units, label=str(a["axis"]), label_sign=_label_sign(a["angle"]),
                )
    plot_curvature(
        outdir / "mean_curvature.png", H_plot, pu, pv,
        "Mean curvature H (nm$^{-1}$)", symmetric=True, disk=pdisk
    )
    plot_curvature(
        outdir / "gaussian_curvature.png", K_plot, pu, pv,
        "Gaussian curvature K (nm$^{-2}$)", symmetric=True, disk=pdisk
    )
    plot_qc(
        outdir / "membrane_detection_qc.png",
        local_raw, w_A, u_A, v_A, surface_smooth, args.level
    )
    plot_support_qc(
        outdir / "support_qc.png", method, support, analysis_mask, u_A, v_A
    )

    # Generate a <=N voxel membrane slab in the aligned local coordinate frame.
    thickness_vox = max(1, int(args.max_thickness_vox))
    thickness_vox = min(thickness_vox, len(w_A))

    # Select exactly the N closest sampled planes to the fitted surface at each
    # lateral pixel. This guarantees that the membrane representation is never
    # thicker than --max-thickness-vox (unlike a simple distance cutoff, which
    # can occasionally select N+1 planes when the surface lies between voxels).
    dist_to_surface = np.abs(w_A[:, None, None] - surface_full[None, :, :])
    nearest = np.argpartition(dist_to_surface, kth=thickness_vox - 1, axis=0)[:thickness_vox]
    slab_mask = np.zeros_like(local_raw, dtype=bool)
    np.put_along_axis(slab_mask, nearest, True, axis=0)
    slab_mask &= support[None, :, :]

    # The same slab on the grid of the input map, to overlay on it directly.
    in_map = np.zeros(data.shape, dtype=np.float32)
    yy, xx = np.indices(data.shape[1:], dtype=float)
    for iz in range(data.shape[0]):
        qx = xx * voxel_xyz_A[0] + origin_xyz_A[0] - center_xyz[0]
        qy = yy * voxel_xyz_A[1] + origin_xyz_A[1] - center_xyz[1]
        qz = iz * voxel_xyz_A[2] + origin_xyz_A[2] - center_xyz[2]
        cu = (qx * e1[0] + qy * e1[1] + qz * e1[2] - u_A[0]) / vox_A
        cv = (qx * e2[0] + qy * e2[1] + qz * e2[2] - v_A[0]) / vox_A
        cw = qx * n[0] + qy * n[1] + qz * n[2]
        s_here = map_coordinates(surface_full, [cv, cu], order=1, mode="constant", cval=np.nan)
        ok_here = map_coordinates(support.astype(np.float32), [cv, cu], order=0, mode="constant", cval=0.0) > 0.5
        with np.errstate(invalid="ignore"):
            in_map[iz] = ok_here & (np.abs(cw - s_here) <= 0.5 * thickness_vox * vox_A)
        if inside_vol is not None:
            in_map[iz] *= inside_vol[iz]
    write_mrc(outdir / "membrane_surface_in_map.mrc", in_map, voxel_xyz_A, origin_xyz_A)

    slab_density = np.where(slab_mask, local_raw, 0.0).astype(np.float32)
    slab_mask_f = slab_mask.astype(np.float32)

    # Per-(v,u) average over the selected slab voxels.
    denom = slab_mask_f.sum(axis=0)
    denom[denom == 0] = 1.0
    avg2d = (slab_density.sum(axis=0) / denom).astype(np.float32)

    # Local aligned volumes all share the same arbitrary local coordinate system:
    # x'=u, y'=v, z'=w. Use [u_min, v_min, w_min] as MRC origin.
    local_origin = np.array([u_A[0], v_A[0], w_A[0]], dtype=float)
    write_mrc(outdir / "aligned_sta_local.mrc", local_raw, np.array([vox_A, vox_A, vox_A]), local_origin)
    write_mrc(outdir / "membrane_3vox_mask.mrc", slab_mask_f, np.array([vox_A, vox_A, vox_A]), local_origin)
    write_mrc(outdir / "membrane_3vox_density.mrc", slab_density, np.array([vox_A, vox_A, vox_A]), local_origin)
    write_mrc(
        outdir / "membrane_average_2d.mrc",
        avg2d[None, :, :],
        np.array([vox_A, vox_A, vox_A]),
        np.array([u_A[0], v_A[0], 0.0]),
    )

    # Human-readable transform information.
    with open(outdir / "transform.txt", "w") as f:
        f.write("Input map: %s\n" % args.map)
        f.write("Voxel size A: %.8g\n" % vox_A)
        f.write("Input world origin xyz A: %.8g %.8g %.8g\n" % tuple(origin_xyz_A))
        f.write("Frame: %s\n" % args.frame)
        f.write("Frame center xyz A: %.8g %.8g %.8g\n" % tuple(center_xyz))
        f.write("Local x' basis e1 xyz: %.10g %.10g %.10g\n" % tuple(e1))
        f.write("Local y' basis e2 xyz: %.10g %.10g %.10g\n" % tuple(e2))
        f.write("Local normal basis n xyz: %.10g %.10g %.10g\n" % tuple(n))
        f.write("World coordinate relation:\n")
        f.write("  xyz = center + u*e1 + v*e2 + w*n\n")
        f.write("Removed from height map (--detrend %s), [a,b,c] for a*u+b*v+c:\n" % args.detrend)
        f.write("  %.10g %.10g %.10g\n" % tuple(trend_coef))
        f.write("Quadric w = a u^2 + b uv + c v^2 + d u + e v + f, [a,b,c,d,e,f]:\n")
        f.write("  %.10g %.10g %.10g %.10g %.10g %.10g\n" % tuple(fit_coef))
        if disk is not None:
            f.write("Circle center x', y' and radius A: %.4f %.4f %.4f\n" % (disk[0], disk[1], eq_radius))
            f.write("Plots (height/curvature maps) use x', y' relative to the circle center.\n")
        f.write("Curvature sign depends on the chosen normal direction.\n")

    print("\nDone. Output written to:", outdir)
    print("Primary figures:")
    print("  ", outdir / "height_map.png")
    print("  ", outdir / "mean_curvature.png")
    print("  ", outdir / "membrane_detection_qc.png")
    print("  ", outdir / "support_qc.png")
    print("Summary:", outdir / "summary.tsv")


if __name__ == "__main__":
    main()
