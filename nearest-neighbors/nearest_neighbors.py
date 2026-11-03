#!/usr/bin/env python3
"""
RELION5 tomography particle nearest-neighbor analysis (per-tomogram kNN, aggregated across tomograms)

Outputs (in --outdir):
- knn_k{K}_top_xy.png      (XY in particle frame; "top view")
- knn_k{K}_side_xz.png     (XZ in particle frame; "side view")
- knn_k{K}_front_yz.png    (YZ in particle frame; "front view")
- k2_first_nn_stats.txt and k2_first_nn_distance_hist.png (only when first-NN stats are collected)

Optional:
- Overlay a target structure map (--target_mrc) as a grayscale projection in each view.
- Highlight the k-th neighbor points (outline) (--highlight_kth).
- Interactive 3D view in Plotly (--view3d plotly) -> knn_k{K}_3d.html

Alpha-by-density (2D + 3D):
- Enable with: --alpha_by_density
- Control levels with: --alpha_lo --alpha_mid --alpha_hi
- Thresholds are density percentiles (default 30/70) adjustable via --alpha_lo_pct --alpha_hi_pct

Notes:
- kNN is computed strictly within each tomogram group (per _rlnTomoName).
- Robust MRC voxel size handling: avoids m.voxel_size (can crash on some permissive reads).
  If MRC header voxel size is missing and --target_pix_nm is not set, script tries to infer from STAR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import starfile


# -------------------- kNN backend --------------------
def build_knn_backend(X_ang: np.ndarray):
    """
    Returns a function query(k) -> (dist_ang, idx) where:
      dist_ang: (N,k) distances in Angstrom
      idx:      (N,k) indices
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(X_ang)

        def query(k: int):
            d_ang, j = tree.query(X_ang, k=k, workers=-1)
            return d_ang, j

        return query
    except Exception:
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore

            def query(k: int):
                nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_ang)
                d_ang, j = nn.kneighbors(X_ang, return_distance=True)
                return d_ang, j

            return query
        except Exception as e:
            raise RuntimeError("Need either scipy (preferred) or scikit-learn for nearest-neighbor search.") from e


# -------------------- RELION angles -> rotation matrices (vectorized) --------------------
def relion_rot_tilt_psi_to_R_batch(rot_deg: np.ndarray, tilt_deg: np.ndarray, psi_deg: np.ndarray) -> np.ndarray:
    """
    RELION rotation matrix A (src/euler.cpp, Euler_angles2matrix), vectorized:
      A = (Rz(rot) @ Ry(tilt) @ Rz(psi))^T    (Rz, Ry: active, right-handed)
    A maps reference -> particle orientation, so a tomogram-frame vector v is
    A^T @ v in the particle (reference) frame.
    Returns A with shape (N,3,3).
    """
    rot = np.deg2rad(rot_deg)
    tilt = np.deg2rad(tilt_deg)
    psi = np.deg2rad(psi_deg)

    cr, sr = np.cos(rot), np.sin(rot)
    ct, st = np.cos(tilt), np.sin(tilt)
    cp, sp = np.cos(psi), np.sin(psi)

    # Rz(rot)
    Rr = np.zeros((rot_deg.size, 3, 3), dtype=np.float64)
    Rr[:, 0, 0] = cr
    Rr[:, 0, 1] = -sr
    Rr[:, 1, 0] = sr
    Rr[:, 1, 1] = cr
    Rr[:, 2, 2] = 1.0

    # Ry(tilt)
    Rt = np.zeros((rot_deg.size, 3, 3), dtype=np.float64)
    Rt[:, 0, 0] = ct
    Rt[:, 0, 2] = st
    Rt[:, 1, 1] = 1.0
    Rt[:, 2, 0] = -st
    Rt[:, 2, 2] = ct

    # Rz(psi)
    Rp = np.zeros((rot_deg.size, 3, 3), dtype=np.float64)
    Rp[:, 0, 0] = cp
    Rp[:, 0, 1] = -sp
    Rp[:, 1, 0] = sp
    Rp[:, 1, 1] = cp
    Rp[:, 2, 2] = 1.0

    # A = (Rr @ Rt @ Rp)^T
    return np.transpose(Rr @ Rt @ Rp, (0, 2, 1))


def apply_axis_flips(arr: np.ndarray, flip_x: bool, flip_y: bool, flip_z: bool) -> np.ndarray:
    """
    In-place sign flips on last dimension (x,y,z). Returns same array.
    Works for shapes (...,3).
    """
    if flip_x:
        arr[..., 0] *= -1.0
    if flip_y:
        arr[..., 1] *= -1.0
    if flip_z:
        arr[..., 2] *= -1.0
    return arr


# -------------------- density coloring --------------------
def point_density_2d(x: np.ndarray, y: np.ndarray, bins: int = 250) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float64)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    xi = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, H.shape[0] - 1)
    yi = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, H.shape[1] - 1)
    return H[xi, yi].astype(np.float64)


def point_density_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, bins: int = 80) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float64)

    mins = np.array([np.min(x), np.min(y), np.min(z)], dtype=np.float64)
    maxs = np.array([np.max(x), np.max(y), np.max(z)], dtype=np.float64)
    for i in range(3):
        if mins[i] == maxs[i]:
            mins[i] -= 1.0
            maxs[i] += 1.0

    H, edges = np.histogramdd(
        np.stack([x, y, z], axis=1),
        bins=bins,
        range=[[mins[0], maxs[0]], [mins[1], maxs[1]], [mins[2], maxs[2]]],
    )
    xi = np.clip(np.searchsorted(edges[0], x, side="right") - 1, 0, H.shape[0] - 1)
    yi = np.clip(np.searchsorted(edges[1], y, side="right") - 1, 0, H.shape[1] - 1)
    zi = np.clip(np.searchsorted(edges[2], z, side="right") - 1, 0, H.shape[2] - 1)
    return H[xi, yi, zi].astype(np.float64)


# -------------------- alpha mapping (by density percentiles) --------------------
def alpha_from_density_percentile(
    dens: np.ndarray,
    lo_pct: float,
    hi_pct: float,
    lo_alpha: float,
    mid_alpha: float,
    hi_alpha: float,
) -> np.ndarray:
    if dens.size == 0:
        return np.array([], dtype=np.float32)
    p_lo, p_hi = np.percentile(dens, [lo_pct, hi_pct])
    a = np.full(dens.shape, float(mid_alpha), dtype=np.float32)
    a[dens <= p_lo] = float(lo_alpha)
    a[dens >= p_hi] = float(hi_alpha)
    return np.clip(a, 0.0, 1.0)


# -------------------- MRC helpers (robust voxel size + polarity) --------------------
def _safe_mrc_voxel_size_angstrom(m) -> tuple[float, float, float]:
    """
    Return (vx,vy,vz) in Angstrom, or (0,0,0) if unavailable.
    Avoids m.voxel_size property (can crash if header is None).
    """
    hdr = getattr(m, "header", None)
    if hdr is not None:
        try:
            mx, my, mz = int(hdr.mx), int(hdr.my), int(hdr.mz)
            cella = hdr.cella
            vx = float(cella.x) / mx if mx > 0 else 0.0
            vy = float(cella.y) / my if my > 0 else 0.0
            vz = float(cella.z) / mz if mz > 0 else 0.0
            if vx > 0 and vy > 0 and vz > 0:
                return vx, vy, vz
        except Exception:
            pass
    return 0.0, 0.0, 0.0


def _apply_polarity(vol: np.ndarray, polarity: str) -> np.ndarray:
    if polarity not in ("auto", "pos", "neg"):
        raise ValueError("--target_polarity must be one of: auto,pos,neg")
    if polarity == "pos":
        return vol
    if polarity == "neg":
        return -vol
    p1, p99 = np.percentile(vol, [1.0, 99.0])
    return -vol if abs(p1) > abs(p99) else vol


# -------------------- optional target (.mrc) overlay --------------------
def load_target_projections(
    mrc_path: str,
    pix_nm_override: float = 0.0,
    proj_mode: str = "max",
    clip_pct: tuple[float, float] = (1.0, 99.7),
    polarity: str = "auto",
    invert: bool = False,
    bg_white: bool = False,
    bg_floor: float = 0.02,
    bg_alpha0: bool = False,
):
    """
    Returns dict: view -> (img_2d float(0..1), extent (xmin,xmax,ymin,ymax) nm, alpha01 float(0..1))
      - 'xy' : x horiz, y vert
      - 'xz' : x horiz, z vert
      - 'yz' : y horiz, z vert
    """
    try:
        import mrcfile  # type: ignore
    except Exception as e:
        raise RuntimeError("Target overlay requires 'mrcfile' (pip/conda install mrcfile).") from e

    if proj_mode not in ("max", "sum"):
        raise ValueError("--target_proj must be 'max' or 'sum'.")

    with mrcfile.open(mrc_path, permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32)  # (z,y,x)
        nz, ny, nx = vol.shape
        vxA, vyA, vzA = _safe_mrc_voxel_size_angstrom(m)

    vol = _apply_polarity(vol, polarity)

    if pix_nm_override > 0:
        px_nm = py_nm = pz_nm = float(pix_nm_override)
    else:
        if vxA <= 0 or vyA <= 0 or vzA <= 0:
            raise ValueError("Target MRC voxel size not found/invalid in header. Provide --target_pix_nm (nm/voxel).")
        px_nm, py_nm, pz_nm = vxA / 10.0, vyA / 10.0, vzA / 10.0

    def norm_clip(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(img, [clip_pct[0], clip_pct[1]])
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
        img = np.clip(img, lo, hi)
        return (img - lo) / (hi - lo)

    def postprocess(img01: np.ndarray):
        bg_mask = (img01 <= float(bg_floor)) if bg_white else None

        if invert:
            img01 = 1.0 - img01
            if bg_white and bg_mask is not None:
                img01[bg_mask] = 1.0

        alpha01 = np.ones_like(img01, dtype=np.float32)
        if bg_alpha0 and bg_white and bg_mask is not None:
            alpha01[bg_mask] = 0.0

        return img01, alpha01

    if proj_mode == "max":
        xy = vol.max(axis=0)  # (y,x)
        xz = vol.max(axis=1)  # (z,x)
        yz = vol.max(axis=2)  # (z,y)
    else:
        xy = vol.sum(axis=0)
        xz = vol.sum(axis=1)
        yz = vol.sum(axis=2)

    xy01, xy_a01 = postprocess(norm_clip(xy))
    xz01, xz_a01 = postprocess(norm_clip(xz))
    yz01, yz_a01 = postprocess(norm_clip(yz))

    x_extent = (-0.5 * nx * px_nm, 0.5 * nx * px_nm)
    y_extent = (-0.5 * ny * py_nm, 0.5 * ny * py_nm)
    z_extent = (-0.5 * nz * pz_nm, 0.5 * nz * pz_nm)

    return {
        "xy": (xy01, (x_extent[0], x_extent[1], y_extent[0], y_extent[1]), xy_a01),
        "xz": (xz01, (x_extent[0], x_extent[1], z_extent[0], z_extent[1]), xz_a01),
        "yz": (yz01, (y_extent[0], y_extent[1], z_extent[0], z_extent[1]), yz_a01),
    }


def load_target_isosurface_mesh_nm(
    mrc_path: str,
    pix_nm_override: float = 0.0,
    clip_pct: tuple[float, float] = (1.0, 99.7),
    iso: float = 0.25,
    polarity: str = "auto",
):
    """
    Returns (x_nm, y_nm, z_nm, faces) for Plotly Mesh3d.
    Mesh is built from 0..1 normalized (clipped) volume; iso is in 0..1.
    """
    try:
        import mrcfile  # type: ignore
    except Exception as e:
        raise RuntimeError("3D target mesh requires 'mrcfile'.") from e
    try:
        from skimage.measure import marching_cubes  # type: ignore
    except Exception as e:
        raise RuntimeError("3D target mesh requires 'scikit-image' (skimage).") from e

    with mrcfile.open(mrc_path, permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32)  # (z,y,x)
        nz, ny, nx = vol.shape
        vxA, vyA, vzA = _safe_mrc_voxel_size_angstrom(m)

    vol = _apply_polarity(vol, polarity)

    if pix_nm_override > 0:
        px_nm = py_nm = pz_nm = float(pix_nm_override)
    else:
        if vxA <= 0 or vyA <= 0 or vzA <= 0:
            raise ValueError("Target MRC voxel size not found/invalid. Provide --target_pix_nm (nm/voxel).")
        px_nm, py_nm, pz_nm = vxA / 10.0, vyA / 10.0, vzA / 10.0

    lo, hi = np.percentile(vol, [clip_pct[0], clip_pct[1]])
    if hi <= lo:
        raise ValueError("Target volume contrast is degenerate after clipping; adjust --target_clip_lo/hi.")
    vol01 = np.clip(vol, lo, hi)
    vol01 = (vol01 - lo) / (hi - lo)

    verts_zyx, faces, _normals, _values = marching_cubes(vol01, level=float(iso))

    z_nm = (verts_zyx[:, 0] - (nz / 2.0)) * pz_nm
    y_nm = (verts_zyx[:, 1] - (ny / 2.0)) * py_nm
    x_nm = (verts_zyx[:, 2] - (nx / 2.0)) * px_nm

    return x_nm.astype(np.float32), y_nm.astype(np.float32), z_nm.astype(np.float32), faces.astype(np.int32)


# -------------------- Plotly 3D --------------------
def write_plotly_3d_html(
    out_html: Path,
    rel_nm: np.ndarray,
    highlight_nm: np.ndarray | None,
    target_mesh: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    max_points: int = 200000,
    bins3d: int = 80,
    points_alpha: float = 0.5,
    points_size: float = 1.0,
    lim3d_nm: float | None = None,
    alpha_by_density: bool = False,
    alpha_lo_pct: float = 30.0,
    alpha_hi_pct: float = 70.0,
    alpha_lo: float = 0.2,
    alpha_mid: float = 0.6,
    alpha_hi: float = 1.0,
    slab3d: bool = False,
    slab3d_step_nm: float = 1.0,
):
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore

    X_nm = rel_nm[:, 0].astype(np.float64, copy=False)
    Y_nm = rel_nm[:, 1].astype(np.float64, copy=False)
    Z_nm = rel_nm[:, 2].astype(np.float64, copy=False)

    if max_points and rel_nm.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(rel_nm.shape[0], size=max_points, replace=False)
        X_nm, Y_nm, Z_nm = X_nm[idx], Y_nm[idx], Z_nm[idx]

    dens = point_density_3d(X_nm, Y_nm, Z_nm, bins=bins3d)

    fig = go.Figure()

    # --- Target mesh (unchanged) ---
    if target_mesh is not None:
        mx, my, mz, faces = target_mesh

        fig.add_trace(
            go.Mesh3d(
                x=mx, y=my, z=mz,
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="lightgray",
                opacity=0.35,
                flatshading=True,
                showscale=False,
                name="target",
                lighting=dict(
                    ambient=0.65,
                    diffuse=1.00,
                    specular=0.25,
                    roughness=0.35,
                    fresnel=0.10,
                ),
                lightposition=dict(x=400, y=400, z=400),
            )
        )

        # --- Outline wireframe (vectorized unique edges, None-separated segments) ---
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        e = np.unique(np.sort(e, axis=1), axis=0)
        def _segs(v):
            s = np.empty((e.shape[0], 3), dtype=object)
            s[:, 0], s[:, 1], s[:, 2] = v[e[:, 0]].tolist(), v[e[:, 1]].tolist(), None
            return s.ravel().tolist()

        fig.add_trace(
            go.Scatter3d(
                x=_segs(mx), y=_segs(my), z=_segs(mz),
                mode="lines",
                line=dict(width=1, color="rgba(0,0,0,0.25)"),
                showlegend=False,
                name="target_outline",
                hoverinfo="skip",
            )
        )

    # --- Neighbor points trace (single trace; slab3d updates this) ---
    fig.add_trace(
        go.Scatter3d(
            x=X_nm, y=Y_nm, z=Z_nm,
            mode="markers",
            marker=dict(
                size=float(points_size),
                color=dens,
                colorscale="Viridis",
                cmin=float(np.min(dens)) if dens.size else 0.0,
                cmax=float(np.max(dens)) if dens.size else 1.0,
                opacity=float(points_alpha),
                showscale=True,
                colorbar=dict(title="Local density (a.u.)"),
            ),
            name="neighbors",
            showlegend=False,
        )
    )

    # Origin
    fig.add_trace(
        go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="origin",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Highlight k-th neighbors (optional)
    if highlight_nm is not None and highlight_nm.size:
        hk = highlight_nm.astype(np.float64, copy=False)
        fig.add_trace(
            go.Scatter3d(
                x=hk[:, 0], y=hk[:, 1], z=hk[:, 2],
                mode="markers",
                marker=dict(size=4, color="gold", opacity=0.95),
                name="k-th NN",
                showlegend=False,
            )
        )

    # --- Bounds centered at origin ---
    L = 0.0
    if X_nm.size:
        L = max(L, float(np.max(np.abs(X_nm))), float(np.max(np.abs(Y_nm))), float(np.max(np.abs(Z_nm))))
    if highlight_nm is not None and highlight_nm.size:
        L = max(L, float(np.max(np.abs(highlight_nm))))
    if target_mesh is not None:
        mx, my, mz, _faces = target_mesh
        if mx.size:
            L = max(L, float(np.max(np.abs(mx))), float(np.max(np.abs(my))), float(np.max(np.abs(mz))))
    if lim3d_nm is not None and lim3d_nm > 0:
        L = float(lim3d_nm)
    if L <= 0:
        L = 1.0
    L *= 1.05

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-L, L]),
            yaxis=dict(range=[-L, L]),
            zaxis=dict(range=[-L, L]),
            aspectmode="cube",
        ),
        scene_dragmode="orbit",
        margin=dict(l=0, r=0, t=30, b=0),
        title=str(out_html.name),
        showlegend=False,
    )
    
    # Always write the custom HTML with sliders
    slab3d = True

    #if not slab3d:
    #    fig.write_html(str(out_html), include_plotlyjs="cdn")
    #    return

    # -------------------- Custom HTML with 3 sliders + density threshold --------------------
    fig.update_layout(autosize=True, height=None)
    fig_div = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"displaylogo": False, "responsive": True},
        default_width="100%",
        default_height="100%",
    )

    Xl = np.asarray(X_nm, dtype=np.float32).tolist()
    Yl = np.asarray(Y_nm, dtype=np.float32).tolist()
    Zl = np.asarray(Z_nm, dtype=np.float32).tolist()
    Dl = np.asarray(dens, dtype=np.float32).tolist()

    step = float(slab3d_step_nm) if slab3d_step_nm and slab3d_step_nm > 0 else 1.0
    L0 = float(L)

    map_controls = (
        """
  <div>Map</div>
  <div class="toggles">
    <label><input id="tmap" type="checkbox" checked> surface</label>
    <label><input id="twire" type="checkbox" checked> wireframe</label>
  </div>
  <div></div>
"""
        if target_mesh is not None
        else ""
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{out_html.name}</title>
  <style>
    html, body {{ height: 100%; }}
    body {{ margin:0; font-family: sans-serif; overflow: hidden; display: flex; flex-direction: column; }}
    .toggles {{ display: flex; gap: 16px; }}

    .controls {{
      padding: 10px 12px;
      border-bottom: 1px solid #ddd;
      display: grid;
      grid-template-columns: 120px 1fr 90px;
      gap: 8px 12px;
      align-items: center;
    }}
    .controls input[type=range] {{ width: 100%; }}
    .val {{ text-align: right; font-variant-numeric: tabular-nums; }}

    #plotwrap {{
      flex: 1 1 auto;
      min-height: 0;
      width: 100vw;
    }}
    #plotwrap > div {{
      height: 100% !important;
      width: 100% !important;
    }}
    #plotwrap .js-plotly-plot {{
      height: 100% !important;
      width: 100% !important;
    }}
  </style>
</head>
<body>

<div class="controls">
  <div>X slab half-width (nm)</div>
  <input id="sx" type="range" min="0" max="{L0:.3f}" step="{step}" value="{L0:.3f}">
  <div class="val" id="sxv">{L0:.1f}</div>

  <div>Y slab half-width (nm)</div>
  <input id="sy" type="range" min="0" max="{L0:.3f}" step="{step}" value="{L0:.3f}">
  <div class="val" id="syv">{L0:.1f}</div>

  <div>Z slab half-width (nm)</div>
  <input id="sz" type="range" min="0" max="{L0:.3f}" step="{step}" value="{L0:.3f}">
  <div class="val" id="szv">{L0:.1f}</div>

  <div>Min density (a.u.)</div>
  <input id="sd" type="range">
  <div class="val" id="sdv"></div>
{map_controls}
  <div>View</div>
  <div class="toggles">
    <label><input id="tclean" type="checkbox"> clean white background (no axes)</label>
  </div>
  <div></div>
</div>

<div id="plotwrap">
{fig_div}
</div>

<script>
  const X = {Xl};
  const Y = {Yl};
  const Z = {Zl};
  const D = {Dl};

  function getPlotDiv() {{
    return document.querySelector('.js-plotly-plot');
  }}

  function findNeighborsTraceIndex(gd) {{
    const data = gd.data;
    for (let i = 0; i < data.length; i++) {{
      if (data[i].name === 'neighbors') return i;
    }}
    return -1;
  }}

  function initDensitySlider() {{
    let dmin = Infinity, dmax = -Infinity;
    for (let i = 0; i < D.length; i++) {{
      const v = D[i];
      if (v < dmin) dmin = v;
      if (v > dmax) dmax = v;
    }}
    if (!isFinite(dmin) || !isFinite(dmax) || dmax <= dmin) {{
      dmin = 0; dmax = 1;
    }}

    const gd = getPlotDiv();
    if (gd) {{
      const ti = findNeighborsTraceIndex(gd);
      if (ti >= 0) {{
        const mk = (gd.data[ti] && gd.data[ti].marker) ? gd.data[ti].marker : null;
        if (mk) {{
          if (typeof mk.cmin === 'number') dmin = mk.cmin;
          if (typeof mk.cmax === 'number') dmax = mk.cmax;
        }}
      }}
    }}

    const sd = document.getElementById('sd');
    const sdv = document.getElementById('sdv');

    const span = dmax - dmin;
    let step = span / 300.0;
    if (step <= 0) step = 1;
    const pow = Math.pow(10, Math.floor(Math.log10(step)));
    step = Math.round(step / pow * 100) / 100 * pow;

    sd.min = dmin;
    sd.max = dmax;
    sd.step = step;

    sd.value = dmin;
    sdv.textContent = Number(sd.value).toFixed(2);
  }}

  function applySlab() {{
    const sx = parseFloat(document.getElementById('sx').value);
    const sy = parseFloat(document.getElementById('sy').value);
    const sz = parseFloat(document.getElementById('sz').value);
    const dthr = parseFloat(document.getElementById('sd').value);

    document.getElementById('sxv').textContent = sx.toFixed(1);
    document.getElementById('syv').textContent = sy.toFixed(1);
    document.getElementById('szv').textContent = sz.toFixed(1);
    document.getElementById('sdv').textContent = dthr.toFixed(2);

    const xs = [];
    const ys = [];
    const zs = [];
    const ds = [];

    for (let i = 0; i < X.length; i++) {{
      if (D[i] < dthr) continue;
      if (Math.abs(X[i]) <= sx && Math.abs(Y[i]) <= sy && Math.abs(Z[i]) <= sz) {{
        xs.push(X[i]);
        ys.push(Y[i]);
        zs.push(Z[i]);
        ds.push(D[i]);
      }}
    }}

    const gd = getPlotDiv();
    if (!gd) return;
    const ti = findNeighborsTraceIndex(gd);
    if (ti < 0) return;

    Plotly.restyle(gd, {{
      x: [xs],
      y: [ys],
      z: [zs],
      'marker.color': [ds],
    }}, [ti]);
  }}

  function forceFullSize() {{
    const gd = getPlotDiv();
    const wrap = document.getElementById('plotwrap');
    if (!gd || !wrap) return;
    const h = wrap.clientHeight;
    const w = wrap.clientWidth;
    Plotly.relayout(gd, {{width: w, height: h}});
    Plotly.Plots.resize(gd);
  }}

  document.getElementById('sx').addEventListener('input', () => {{ applySlab(); }});
  document.getElementById('sy').addEventListener('input', () => {{ applySlab(); }});
  document.getElementById('sz').addEventListener('input', () => {{ applySlab(); }});
  document.getElementById('sd').addEventListener('input', () => {{ applySlab(); }});

  function setTraceVisible(name, on) {{
    const gd = getPlotDiv();
    if (!gd) return;
    const idx = [];
    gd.data.forEach((t, i) => {{ if (t.name === name) idx.push(i); }});
    if (idx.length) Plotly.restyle(gd, {{visible: on}}, idx);
  }}
  [['tmap', 'target'], ['twire', 'target_outline']].forEach(([id, name]) => {{
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', () => setTraceVisible(name, el.checked));
  }});

  document.getElementById('tclean').addEventListener('change', (ev) => {{
    const gd = getPlotDiv();
    if (!gd) return;
    const show = !ev.target.checked;
    Plotly.relayout(gd, {{
      'scene.xaxis.visible': show,
      'scene.yaxis.visible': show,
      'scene.zaxis.visible': show,
      'scene.bgcolor': 'white',
      'paper_bgcolor': 'white',
      'plot_bgcolor': 'white',
      'title.text': show ? {json.dumps(out_html.name)} : '',
    }});
  }});

  window.addEventListener('resize', forceFullSize);

  initDensitySlider();
  applySlab();
  setTimeout(forceFullSize, 50);
</script>

</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")


# -------------------- 2D plotting --------------------
def plot_neighbor_cloud(
    rel_xyz_nm: np.ndarray,
    out_png: Path,
    title: str,
    view: str,
    lim_nm: float | None,
    bins: int,
    target_overlay=None,  # (img, extent) or (img, extent, alpha01) or None
    target_alpha: float = 0.7,
    highlight_kth_nm: np.ndarray | None = None,
    cmap: str | None = None,
    show_colorbar: bool = True,
    points_alpha: float = 1.0,
    points_size: float = 2.0,
    target_on_top: bool = False,
    alpha_by_density: bool = False,
    alpha_lo_pct: float = 30.0,
    alpha_hi_pct: float = 70.0,
    alpha_lo: float = 0.2,
    alpha_mid: float = 0.6,
    alpha_hi: float = 1.0,
):
    assert view in ("xy", "xz", "yz")

    if view == "xy":
        a = rel_xyz_nm[:, 0]
        b = rel_xyz_nm[:, 1]
        xlabel, ylabel = "X (nm)", "Y (nm)"
        hk_a, hk_b = (
            (highlight_kth_nm[:, 0], highlight_kth_nm[:, 1])
            if (highlight_kth_nm is not None and highlight_kth_nm.size)
            else (None, None)
        )
    elif view == "xz":
        a = rel_xyz_nm[:, 0]
        b = rel_xyz_nm[:, 2]
        xlabel, ylabel = "X (nm)", "Z (nm)"
        hk_a, hk_b = (
            (highlight_kth_nm[:, 0], highlight_kth_nm[:, 2])
            if (highlight_kth_nm is not None and highlight_kth_nm.size)
            else (None, None)
        )
    else:  # yz
        a = rel_xyz_nm[:, 1]
        b = rel_xyz_nm[:, 2]
        xlabel, ylabel = "Y (nm)", "Z (nm)"
        hk_a, hk_b = (
            (highlight_kth_nm[:, 1], highlight_kth_nm[:, 2])
            if (highlight_kth_nm is not None and highlight_kth_nm.size)
            else (None, None)
        )

    dens = point_density_2d(a, b, bins=bins)
    order = np.argsort(dens)
    a, b, dens = a[order], b[order], dens[order]

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=200)

    def _draw_target(z):
        if target_overlay is None:
            return
        if len(target_overlay) == 2:
            img, extent = target_overlay
            alpha = float(target_alpha)
        else:
            img, extent, a01 = target_overlay
            alpha = (a01 * float(target_alpha))
        ax.imshow(img, extent=extent, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, alpha=alpha, zorder=z)

    if target_overlay is not None and (not target_on_top):
        _draw_target(z=0)

    if not alpha_by_density:
        sc = ax.scatter(a, b, c=dens, s=points_size, linewidths=0, zorder=2, cmap=cmap, alpha=float(points_alpha))
        mappable_for_cbar = sc
    else:
        import matplotlib as mpl

        cmap_obj = mpl.colormaps[cmap if cmap is not None else "viridis"]
        vmin = float(dens.min()) if dens.size else 0.0
        vmax = float(dens.max()) if dens.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        rgba = cmap_obj(norm(dens))
        rgba[:, 3] = alpha_from_density_percentile(
            dens,
            lo_pct=alpha_lo_pct,
            hi_pct=alpha_hi_pct,
            lo_alpha=alpha_lo,
            mid_alpha=alpha_mid,
            hi_alpha=alpha_hi,
        )

        ax.scatter(a, b, c=rgba, s=points_size, linewidths=0, zorder=2)
        mappable_for_cbar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable_for_cbar.set_array([])

    if target_overlay is not None and target_on_top:
        _draw_target(z=6)

    ax.scatter([0.0], [0.0], c="red", s=40, linewidths=0, zorder=10)

    if hk_a is not None and hk_a.size:
        ax.scatter(hk_a, hk_b, s=18, facecolors="none", edgecolors="gold", linewidths=0.8, zorder=9, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")

    if lim_nm is None:
        m = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        lim_nm = float(np.ceil(m / 5.0) * 5.0) if m > 0 else 10.0

    ax.set_xlim(-lim_nm, lim_nm)
    ax.set_ylim(-lim_nm, lim_nm)

    if show_colorbar:
        cbar = fig.colorbar(mappable_for_cbar, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Local density (a.u.)")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# -------------------- STAR parsing helpers --------------------
def pick_particles_table(star_obj):
    # Normal single loop block -> DataFrame
    if hasattr(star_obj, "columns") and hasattr(star_obj, "shape"):
        return star_obj, "<single_table>"

    # Multiple named STAR blocks
    if isinstance(star_obj, dict):
        if "particles" in star_obj:
            return star_obj["particles"], "particles"

        tables = [
            (k, v)
            for k, v in star_obj.items()
            if hasattr(v, "columns") and hasattr(v, "shape")
        ]

        if not tables:
            raise ValueError(
                f"No tabular blocks found in STAR. Blocks: {list(star_obj.keys())}"
            )

        key, val = max(tables, key=lambda kv: int(kv[1].shape[0]))
        return val, key

    # Older starfile versions can return a list
    if isinstance(star_obj, (list, tuple)):
        tables = [
            (i, v)
            for i, v in enumerate(star_obj)
            if hasattr(v, "columns") and hasattr(v, "shape")
        ]

        if not tables:
            raise ValueError(
                f"STAR was returned as a list, but no DataFrame was found. "
                f"Element types: {[type(x).__name__ for x in star_obj[:10]]}"
            )

        # Usually the particle table is the largest table
        idx, val = max(tables, key=lambda kv: int(kv[1].shape[0]))
        return val, f"list[{idx}]"

    raise ValueError(
        f"Unsupported STAR object type: {type(star_obj)}"
    )


def colname(df, logical: str) -> str:
    for c in (logical, logical.lstrip("_")):
        if c in df.columns:
            return c
    raise KeyError(logical)


def maybe_colname(df, logical: str) -> str | None:
    for c in (logical, logical.lstrip("_")):
        if c in df.columns:
            return c
    return None


def infer_target_pix_nm_from_star(df) -> float | None:
    candidates = [
        "_rlnTomoTiltSeriesPixelSize",
        "_rlnTomoTomogramPixelSize",
        "_rlnImagePixelSize",
        "_rlnPixelSize",
    ]
    for logical in candidates:
        c = maybe_colname(df, logical)
        if c is None:
            continue
        vals = df[c].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vA = float(np.nanmedian(vals))
        if vA > 0:
            return vA / 10.0
    return None


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--outdir", default="knn_out")
    ap.add_argument("--k", default="2,7")
    ap.add_argument(
        "--k_mode",
        choices=["includes_self", "excludes_self"],
        default="includes_self",
        help="includes_self: k passed to kNN includes self. excludes_self: treat k as neighbor rank excluding self.",
    )
    ap.add_argument(
        "--neighbor_set",
        choices=["up_to_k", "kth_only"],
        default="up_to_k",
        help="up_to_k: plot all neighbors up to k (excluding self). kth_only: plot only the k-th neighbor vectors.",
    )
    ap.add_argument("--tomos", default="ALL")
    ap.add_argument("--max_per_tomo", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=250)
    ap.add_argument("--lim_nm", type=float, default=60.0)

    # target overlay options
    ap.add_argument("--target_mrc", default="", help="Optional target density map (.mrc) to overlay.")
    ap.add_argument(
        "--target_pix_nm",
        type=float,
        default=0.0,
        help="Override target voxel size (nm/voxel). If 0, read from MRC header; if missing, infer from STAR.",
    )
    ap.add_argument("--target_proj", choices=["max", "sum"], default="max")
    ap.add_argument("--target_clip_lo", type=float, default=1.0)
    ap.add_argument("--target_clip_hi", type=float, default=99.7)
    ap.add_argument("--target_alpha", type=float, default=0.7)
    ap.add_argument("--target_polarity", choices=["auto", "pos", "neg"], default="auto")
    ap.add_argument("--target_invert", action="store_true", default=None)
    ap.add_argument("--target_bg_white", action="store_true", default=None)
    ap.add_argument("--target_bg_floor", type=float, default=None)
    ap.add_argument("--target_bg_alpha0", action="store_true", default=None)
    ap.add_argument("--target_on_top", action="store_true")
    ap.add_argument("--highlight_kth", action="store_true")

    ap.add_argument(
        "--points_alpha",
        type=float,
        default=-1.0,
        help="Uniform point alpha (used only when --alpha_by_density is NOT set). -1 => auto (1.0 if no target, else 0.5).",
    )
    ap.add_argument(
        "--points_size",
        type=float,
        default=-1.0,
        help="Point marker size. -1 => auto (2.0 if no target, else 1.0).",
    )

    # alpha-by-density options (2D + 3D)
    ap.add_argument("--alpha_by_density", action="store_true", help="Use density-percentile based per-point alpha.")
    ap.add_argument("--alpha_lo", type=float, default=0.2)
    ap.add_argument("--alpha_mid", type=float, default=0.6)
    ap.add_argument("--alpha_hi", type=float, default=1.0)
    ap.add_argument("--alpha_lo_pct", type=float, default=30.0, help="Percentile defining low-density cutoff.")
    ap.add_argument("--alpha_hi_pct", type=float, default=70.0, help="Percentile defining high-density cutoff.")

    ap.add_argument(
        "--flip_local_axes",
        default="",
        help="Flip sign of local coordinates after rotation (mirrors the cloud). Example: 'x', 'yz'. Default: none.",
    )
    ap.add_argument("--screenshot_style", action="store_true", help="Blue cloud + no colorbar (for screenshot-like look).")

    # 3D options
    ap.add_argument("--view3d", choices=["none", "plotly"], default="none")
    ap.add_argument("--no_2d", action="store_true", help="Skip 2D PNG outputs.")
    ap.add_argument("--view3d_max_points", type=int, default=200000, help="Downsample points for 3D viewer (0 disables).")
    ap.add_argument("--view3d_bins3d", type=int, default=80)
    ap.add_argument("--target_iso", type=float, default=0.25, help="Isosurface level on 0..1 normalized target volume for 3D mesh.")
    ap.add_argument("--slab3d", action="store_true", help="3 sliders (X/Y/Z + density) in the Plotly HTML to clip points.")
    ap.add_argument("--slab3d_step_nm", type=float, default=1.0, help="Slider step in nm for the 3D slab controls.")

    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    k_list = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    if any(k < 1 for k in k_list):
        raise ValueError("k must be >= 1.")
    max_k = int(max(k_list))

    star_obj = starfile.read(args.star)
    df, picked_key = pick_particles_table(star_obj)

    if not args.quiet:
        if isinstance(star_obj, dict):
            print("STAR blocks:", list(star_obj.keys()))
        print("Picked particle block:", picked_key)
        print("Picked columns (first 30):", list(df.columns)[:30])

    # Resolve columns
    xcol = colname(df, "_rlnCenteredCoordinateXAngst")
    ycol = colname(df, "_rlnCenteredCoordinateYAngst")
    zcol = colname(df, "_rlnCenteredCoordinateZAngst")

    oxcol = maybe_colname(df, "_rlnOriginXAngst")
    oycol = maybe_colname(df, "_rlnOriginYAngst")
    ozcol = maybe_colname(df, "_rlnOriginZAngst")

    rotcol = colname(df, "_rlnAngleRot")
    tiltcol = colname(df, "_rlnAngleTilt")
    psiccol = colname(df, "_rlnAnglePsi")
    tomocol = colname(df, "_rlnTomoName")

    for logical in ("_rlnTomoSubtomogramRot", "_rlnTomoSubtomogramTilt", "_rlnTomoSubtomogramPsi"):
        c = maybe_colname(df, logical)
        if c is not None and np.any(np.abs(df[c].to_numpy(dtype=np.float64)) > 1e-6):
            print(f"[warning] {logical} is non-zero: _rlnAngle* are relative to the subtomogram frame, "
                  "and this script uses _rlnAngle* only. Local frames will be wrong.")
            break

    flip = set(args.flip_local_axes.lower().replace(",", "").replace(" ", ""))
    flip_x, flip_y, flip_z = ("x" in flip), ("y" in flip), ("z" in flip)

    # Conditional defaults when a target is provided
    if args.target_mrc:
        points_alpha = 0.5 if args.points_alpha < 0 else float(args.points_alpha)
        points_size = 1.0 if args.points_size < 0 else float(args.points_size)

        if args.target_invert is None:
            args.target_invert = True
        if args.target_bg_white is None:
            args.target_bg_white = True
        if args.target_bg_alpha0 is None:
            args.target_bg_alpha0 = True
        if args.target_bg_floor is None:
            args.target_bg_floor = 0.05
    else:
        points_alpha = 1.0 if args.points_alpha < 0 else float(args.points_alpha)
        points_size = 2.0 if args.points_size < 0 else float(args.points_size)

        if args.target_invert is None:
            args.target_invert = False
        if args.target_bg_white is None:
            args.target_bg_white = False
        if args.target_bg_alpha0 is None:
            args.target_bg_alpha0 = False
        if args.target_bg_floor is None:
            args.target_bg_floor = 0.02

    rng = np.random.default_rng(args.seed)

    if args.tomos.strip().upper() == "ALL":
        tomo_names = sorted(df[tomocol].astype(str).unique().tolist())
    else:
        tomo_names = [t.strip() for t in args.tomos.split(",") if t.strip()]

    # If target voxel size is not provided, infer from STAR (nm) to avoid broken MRC headers.
    if args.target_mrc and args.target_pix_nm <= 0:
        inferred = infer_target_pix_nm_from_star(df)
        if inferred is not None:
            args.target_pix_nm = inferred
            if not args.quiet:
                print(f"[info] Inferred --target_pix_nm {args.target_pix_nm:.6f} nm from STAR.")

    # Aggregate
    relvec_by_k_nm = {k: [] for k in k_list}   # all neighbors up to query_k (excluding self)
    kthvec_by_k_nm = {k: [] for k in k_list}   # only kth neighbor (excluding self)
    first_nn_dists_nm: list[np.ndarray] = []

    target_proj = None
    target_mesh = None

    if args.target_mrc:
        target_proj = load_target_projections(
            args.target_mrc,
            pix_nm_override=float(args.target_pix_nm),
            proj_mode=args.target_proj,
            clip_pct=(float(args.target_clip_lo), float(args.target_clip_hi)),
            polarity=str(args.target_polarity),
            invert=bool(args.target_invert),
            bg_white=bool(args.target_bg_white),
            bg_floor=float(args.target_bg_floor),
            bg_alpha0=bool(args.target_bg_alpha0),
        )

    if args.view3d == "plotly" and args.target_mrc:
        target_mesh = load_target_isosurface_mesh_nm(
            args.target_mrc,
            pix_nm_override=float(args.target_pix_nm),
            clip_pct=(float(args.target_clip_lo), float(args.target_clip_hi)),
            iso=float(args.target_iso),
            polarity=str(args.target_polarity),
        )

    # Determine the single kNN query size per tomogram
    if args.k_mode == "includes_self":
        max_query_k = max_k
    else:
        max_query_k = max_k + 1  # to include self + max_k neighbors

    for tomo in tomo_names:
        g = df[df[tomocol].astype(str) == str(tomo)]
        if len(g) < 2:
            continue

        if args.max_per_tomo and len(g) > args.max_per_tomo:
            g = g.sample(n=args.max_per_tomo, random_state=int(rng.integers(0, 2**31 - 1)))

        n = len(g)
        if max_query_k > n or max_query_k < 2:
            continue

        ox_ang = g[oxcol].to_numpy(dtype=np.float64) if oxcol else np.zeros(n, dtype=np.float64)
        oy_ang = g[oycol].to_numpy(dtype=np.float64) if oycol else np.zeros(n, dtype=np.float64)
        oz_ang = g[ozcol].to_numpy(dtype=np.float64) if ozcol else np.zeros(n, dtype=np.float64)

        xyz_ang = np.stack(
            [
                g[xcol].to_numpy(dtype=np.float64) - ox_ang,  # RELION: true centre = coord - origin
                g[ycol].to_numpy(dtype=np.float64) - oy_ang,
                g[zcol].to_numpy(dtype=np.float64) - oz_ang,
            ],
            axis=1,
        )  # (N,3) Angstrom

        rot_deg = g[rotcol].to_numpy(dtype=np.float64)
        tilt_deg = g[tiltcol].to_numpy(dtype=np.float64)
        psi_deg = g[psiccol].to_numpy(dtype=np.float64)

        # kNN once at max_query_k, then slice for each k
        knn_query = build_knn_backend(xyz_ang)
        d_all_ang, idx_all = knn_query(max_query_k)  # (N,max_query_k)

        # Precompute R^T per particle for local transform
        R = relion_rot_tilt_psi_to_R_batch(rot_deg, tilt_deg, psi_deg)  # (N,3,3)
        Rt = np.transpose(R, (0, 2, 1))  # (N,3,3)

        for k in k_list:
            if args.k_mode == "includes_self":
                query_k = k
                kth_col = k - 1
            else:
                query_k = k + 1
                kth_col = k

            if query_k > n or query_k < 2:
                continue

            # first-NN stats
            if (args.k_mode == "includes_self" and k == 2) or (args.k_mode == "excludes_self" and k == 1):
                first_nn_dists_nm.append((d_all_ang[:, 1] / 10.0).copy())

            # neighbors excluding self: idx columns [1:query_k]
            nbr_idx = idx_all[:, 1:query_k]  # (N, M)
            nbr_xyz_ang = xyz_ang[nbr_idx]   # (N, M, 3)
            rel_global_ang = nbr_xyz_ang - xyz_ang[:, None, :]  # (N, M, 3)

            # Batched local transform: rel_local = Rt @ rel_global
            # out[n,m,i] = sum_j Rt[n,i,j] * rel_global[n,m,j]
            rel_local_ang = np.einsum("nij,nmj->nmi", Rt, rel_global_ang, optimize=True)
            apply_axis_flips(rel_local_ang, flip_x, flip_y, flip_z)

            relvec_by_k_nm[k].append(rel_local_ang.reshape(-1, 3) / 10.0)

            # kth neighbor (excluding self) vector
            kth_idx = idx_all[:, kth_col]  # (N,)
            kth_xyz_ang = xyz_ang[kth_idx]  # (N,3)
            kth_rel_global_ang = kth_xyz_ang - xyz_ang  # (N,3)
            kth_rel_local_ang = np.einsum("nij,nj->ni", Rt, kth_rel_global_ang, optimize=True)  # (N,3)
            apply_axis_flips(kth_rel_local_ang, flip_x, flip_y, flip_z)

            kthvec_by_k_nm[k].append(kth_rel_local_ang / 10.0)

    for k in k_list:
        relvec_by_k_nm[k] = np.concatenate(relvec_by_k_nm[k], axis=0) if relvec_by_k_nm[k] else np.zeros((0, 3))
        kthvec_by_k_nm[k] = np.concatenate(kthvec_by_k_nm[k], axis=0) if kthvec_by_k_nm[k] else np.zeros((0, 3))

    if first_nn_dists_nm:
        first_nn = np.concatenate(first_nn_dists_nm)
        stats_txt = (
            f"First NN distance: mean={np.mean(first_nn):.2f} nm, "
            f"median={np.median(first_nn):.2f} nm, std={np.std(first_nn):.2f} nm, n={first_nn.size}"
        )
        (outdir / "k2_first_nn_stats.txt").write_text(stats_txt + "\n")

        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        ax.hist(first_nn, bins=60)
        ax.set_xlabel("First nearest-neighbor distance (nm)")
        ax.set_ylabel("Count")
        ax.set_title("First NN distance distribution")
        fig.tight_layout()
        fig.savefig(outdir / "k2_first_nn_distance_hist.png")
        plt.close(fig)

        if not args.quiet:
            print(stats_txt)

    lim_nm = None if args.lim_nm <= 0 else float(args.lim_nm)

    for k in k_list:
        rel_nm = kthvec_by_k_nm[k] if args.neighbor_set == "kth_only" else relvec_by_k_nm[k]
        if rel_nm.size == 0:
            continue

        highlight = (kthvec_by_k_nm[k] if (args.highlight_kth and args.neighbor_set != "kth_only") else None)

        cmap = "Blues" if args.screenshot_style else None
        show_cbar = (not args.screenshot_style)

        if not args.no_2d:
            plot_neighbor_cloud(
                rel_nm, outdir / f"knn_k{k}_top_xy.png",
                f"k={k} neighbor distribution (top view, XY)",
                "xy", lim_nm, args.bins,
                target_overlay=(target_proj["xy"] if target_proj else None),
                target_alpha=args.target_alpha,
                highlight_kth_nm=highlight,
                cmap=cmap, show_colorbar=show_cbar,
                points_alpha=points_alpha, points_size=points_size,
                target_on_top=bool(args.target_on_top),
                alpha_by_density=bool(args.alpha_by_density),
                alpha_lo_pct=float(args.alpha_lo_pct),
                alpha_hi_pct=float(args.alpha_hi_pct),
                alpha_lo=float(args.alpha_lo),
                alpha_mid=float(args.alpha_mid),
                alpha_hi=float(args.alpha_hi),
            )
            plot_neighbor_cloud(
                rel_nm, outdir / f"knn_k{k}_side_xz.png",
                f"k={k} neighbor distribution (side view, XZ)",
                "xz", lim_nm, args.bins,
                target_overlay=(target_proj["xz"] if target_proj else None),
                target_alpha=args.target_alpha,
                highlight_kth_nm=highlight,
                cmap=cmap, show_colorbar=show_cbar,
                points_alpha=points_alpha, points_size=points_size,
                target_on_top=bool(args.target_on_top),
                alpha_by_density=bool(args.alpha_by_density),
                alpha_lo_pct=float(args.alpha_lo_pct),
                alpha_hi_pct=float(args.alpha_hi_pct),
                alpha_lo=float(args.alpha_lo),
                alpha_mid=float(args.alpha_mid),
                alpha_hi=float(args.alpha_hi),
            )
            plot_neighbor_cloud(
                rel_nm, outdir / f"knn_k{k}_front_yz.png",
                f"k={k} neighbor distribution (front view, YZ)",
                "yz", lim_nm, args.bins,
                target_overlay=(target_proj["yz"] if target_proj else None),
                target_alpha=args.target_alpha,
                highlight_kth_nm=highlight,
                cmap=cmap, show_colorbar=show_cbar,
                points_alpha=points_alpha, points_size=points_size,
                target_on_top=bool(args.target_on_top),
                alpha_by_density=bool(args.alpha_by_density),
                alpha_lo_pct=float(args.alpha_lo_pct),
                alpha_hi_pct=float(args.alpha_hi_pct),
                alpha_lo=float(args.alpha_lo),
                alpha_mid=float(args.alpha_mid),
                alpha_hi=float(args.alpha_hi),
            )

        if args.view3d == "plotly":
            out_html = outdir / f"knn_k{k}_3d.html"
            write_plotly_3d_html(
                out_html=out_html,
                rel_nm=rel_nm,
                highlight_nm=highlight,
                target_mesh=target_mesh,
                max_points=int(args.view3d_max_points),
                bins3d=int(args.view3d_bins3d),
                points_alpha=float(points_alpha),
                points_size=float(points_size),
                lim3d_nm=(float(args.lim_nm) if args.lim_nm and args.lim_nm > 0 else None),
                alpha_by_density=bool(args.alpha_by_density),
                alpha_lo_pct=float(args.alpha_lo_pct),
                alpha_hi_pct=float(args.alpha_hi_pct),
                alpha_lo=float(args.alpha_lo),
                alpha_mid=float(args.alpha_mid),
                alpha_hi=float(args.alpha_hi),
                slab3d=True,
                slab3d_step_nm=float(args.slab3d_step_nm),
            )

    if not args.quiet:
        print(f"Wrote outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()