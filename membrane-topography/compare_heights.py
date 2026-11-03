#!/usr/bin/env python3
"""
Compare membrane height maps produced by Curvature_analysis.py (--frame box).

Each run is taken relative to its own circle center (--disk-center-ang), so
maps whose complex sits at a different place in the box are compared complex
to complex. Heights are the reported heights (height_A, i.e. including any
--height-offset-ang). For every map this writes:

  diff_<label>.png/.pdf     height of <label> minus the reference, as a
                            figure-ready circular panel, plus diff_colorbar.png/.pdf
  profiles.png/.pdf         membrane height along the two axes for all maps
                            (solid: measured, thin dotted: inter-/extrapolated)
  compare_summary.tsv       per map: height difference to the reference
                            (median, 5th/95th percentile), slope at the center
                            and curvature along each axis, all computed over the
                            same radius for every map (the smallest circle)

Usage:
  compare_heights.py --runs Crista=crista_out Oxphos=oxphos_out Dimer=dimer_out \
                     --reference Crista --axes-angle-deg 20 --outdir compare_out
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import map_coordinates


def load(run_dir):
    d = np.load(Path(run_dir) / "membrane_surface.npz")
    out = {k: d[k] for k in d.files}
    cu, cv, rad = (float(x) for x in out["disk_center_radius_A"])
    out["center"] = (cu, cv)
    out["radius"] = rad
    return out


def to_relative(run, grid):
    """Resample a run onto a grid in coordinates relative to its circle center."""
    u, v = run["u_A"], run["v_A"]
    step = float(u[1] - u[0])
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    iu = (X + run["center"][0] - u[0]) / step
    iv = (Y + run["center"][1] - v[0]) / step
    h = run["height_A"].astype(float)
    finite = np.isfinite(h)
    hv = map_coordinates(np.where(finite, h, 0.0), [iv, iu], order=1, mode="constant", cval=0.0)
    wv = map_coordinates(finite.astype(float), [iv, iu], order=1, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        hrel = np.where(wv > 0.5, hv / np.maximum(wv, 1e-6), np.nan)
    near = lambda a: map_coordinates(a.astype(float), [iv, iu], order=0, mode="constant", cval=0.0) > 0.5
    rel = dict(h=hrel, support=near(run["support"]) & np.isfinite(hrel) & (np.hypot(X, Y) <= run["radius"]),
               valid=near(run["valid_detection"]), radius=run["radius"], X=X, Y=Y)
    rel["valid"] &= rel["support"]
    rel["fp"] = near(run["pdb_footprint"]) if "pdb_footprint" in run else np.zeros(X.shape, bool)
    return rel


def axis_coords(rel, angle_deg):
    th = math.radians(angle_deg)
    s_ = rel["X"] * math.cos(th) + rel["Y"] * math.sin(th)
    d_ = -rel["X"] * math.sin(th) + rel["Y"] * math.cos(th)
    return s_, d_


def axis_fit(rel, angle_deg, band_A, max_A):
    """Parabola through the measured profile within max_A of the center: slope (deg) and curvature (1/nm)."""
    s_, d_ = axis_coords(rel, angle_deg)
    m = rel["valid"] & (np.abs(d_) <= band_A) & (np.abs(s_) <= max_A)
    if np.count_nonzero(m) < 10:
        return float("nan"), float("nan"), int(m.sum())
    a, b, _ = np.polyfit(s_[m], rel["h"][m], 2)
    return math.degrees(math.atan(b)), 10.0 * 2.0 * a / (1.0 + b * b) ** 1.5, int(m.sum())


def profile(rel, angle_deg, band_A, s_grid):
    s_, d_ = axis_coords(rel, angle_deg)
    inside = rel["support"] & (np.abs(d_) <= band_A)
    meas = inside & rel["valid"]
    step = s_grid[1] - s_grid[0]
    prof, prof_meas = np.full(len(s_grid), np.nan), np.full(len(s_grid), np.nan)
    for i, s0 in enumerate(s_grid):
        b = np.abs(s_ - s0) <= 0.5 * step
        if np.any(inside & b):
            prof[i] = float(np.median(rel["h"][inside & b]))
        if np.count_nonzero(meas & b) >= 2:
            prof_meas[i] = float(np.median(rel["h"][meas & b]))
    return prof, prof_meas


def _rot_resample(arr, g, rot_cw_deg, order=1):
    """Resample an array on the centered grid g after a clockwise rotation of the picture."""
    if not rot_cw_deg:
        return arr
    t = math.radians(rot_cw_deg)
    X, Y = np.meshgrid(g, g, indexing="xy")
    xs, ys = X * math.cos(t) - Y * math.sin(t), X * math.sin(t) + Y * math.cos(t)
    step = g[1] - g[0]
    co = [(ys - g[0]) / step, (xs - g[0]) / step]
    if arr.dtype == bool:
        return map_coordinates(arr.astype(float), co, order=0, mode="constant", cval=0.0) > 0.5
    fin = np.isfinite(arr)
    hv = map_coordinates(np.where(fin, arr, 0.0), co, order=order, mode="constant", cval=0.0)
    wv = map_coordinates(fin.astype(float), co, order=order, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wv > 0.5, hv / np.maximum(wv, 1e-6), np.nan)


def _label_sign(angle_deg):
    th = math.radians(angle_deg)
    c, s_ = math.cos(th), math.sin(th)
    return 1.0 if (c > 1e-6 or (abs(c) <= 1e-6 and s_ > 0)) else -1.0


def diff_panel(path_base, rel, ref, label, ref_label, vlim, size_mm, font_pt, label_pt, step_A, cmap,
               outline_color, angles, ref_radius_A, rotate_cw_deg=0.0, numbers=True):
    rad = min(rel["radius"], ref["radius"])
    m_out = 0.24 if numbers else 0.0
    with mpl.rc_context({"font.size": font_pt, "pdf.fonttype": 42}):
        size_mm = size_mm * rad / ref_radius_A
        inch = size_mm / 25.4 * (1.0 + m_out)
        extra = 1.35 * font_pt / 72.0 + 0.03
        fig = plt.figure(figsize=(inch, inch + extra))
        f_map = inch / (inch + extra)
        ax = fig.add_axes([0.0, 1.0 - f_map, 1.0, f_map])
        X, Y = rel["X"], rel["Y"]
        g = X[0]
        sup = _rot_resample(rel["support"] & ref["support"], g, rotate_cw_deg)
        grown = ndimage.binary_dilation(sup, iterations=2)
        dh_raw = _rot_resample(rel["h"] - ref["h"], g, rotate_cw_deg)
        dh = np.where(grown, np.nan_to_num(dh_raw, nan=0.0), np.nan)
        fp = _rot_resample(rel["fp"], g, rotate_cw_deg)
        du = 0.5 * (g[1] - g[0])
        im = ax.imshow(dh, origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim,
                       extent=[g[0] - du, g[-1] + du, g[0] - du, g[-1] + du], interpolation="bilinear")
        lev = [x for x in np.arange(-vlim, vlim + 1e-6, step_A) if abs(x) > 1e-6]
        cs = ax.contour(X, Y, dh, levels=lev, colors="k", linewidths=0.3)
        if label_pt > 0:
            pos = []
            for segs in cs.allsegs:
                if not any(len(sg) for sg in segs):
                    continue
                pts = np.concatenate([sg for sg in segs if len(sg)], axis=0)
                r = np.hypot(pts[:, 0], pts[:, 1])
                pts = pts[r <= 0.78 * rad]
                if len(pts):
                    pos.append(tuple(pts[np.argmin(np.abs(np.hypot(pts[:, 0], pts[:, 1]) - 0.55 * rad))]))
            if pos:
                ax.clabel(cs, inline=True, fontsize=label_pt, fmt="%+g", manual=pos, inline_spacing=1.5)
        if np.any(fp):
            ax.contour(X, Y, ndimage.gaussian_filter(fp.astype(float), 1.0), levels=[0.5],
                       colors=outline_color, linewidths=0.8)
        for k, ang0 in enumerate(angles):
            ang = ang0 - rotate_cw_deg
            th = math.radians(ang)
            c, s_ = math.cos(th), math.sin(th)
            ax.plot([-rad * c, rad * c], [-rad * s_, rad * s_],
                    color="#404040", linewidth=0.6, linestyle=(0, (4, 2.5)))
            if numbers:
                for sgn in (1.0, -1.0):
                    ax.plot([sgn * rad * c, sgn * 1.1 * rad * c], [sgn * rad * s_, sgn * 1.1 * rad * s_],
                            color="k", linewidth=0.9, solid_capstyle="butt", clip_on=False)
                sgn = _label_sign(ang0 % 180.0)
                ax.text(sgn * 1.17 * rad * c, sgn * 1.17 * rad * s_, str(k + 1), ha="center", va="center",
                        fontsize=font_pt, fontweight="bold", clip_on=False)
        circle = plt.Circle((0, 0), rad, facecolor="none", edgecolor="k", linewidth=0.5)
        ax.add_patch(circle)
        for art in [im] + list(ax.collections):
            art.set_clip_path(circle)
        ax.set_xlim(-(1.0 + m_out) * rad, (1.0 + m_out) * rad)
        ax.set_ylim(-(1.0 + m_out) * rad, (1.0 + m_out) * rad)
        ax.set_axis_off()
        fig.text(0.5, 0.5 * (1.0 - f_map), "%s − %s" % (label, ref_label), ha="center", va="center",
                 fontsize=font_pt)
        fig.savefig(str(path_base) + ".png", dpi=600, transparent=True)
        fig.savefig(str(path_base) + ".pdf", transparent=True)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="LABEL=output_dir of Curvature_analysis.py runs")
    p.add_argument("--reference", required=True, help="Label of the reference map for the difference panels")
    p.add_argument("--axes-angle-deg", type=float, default=20.0, help="Axis 1 angle; axis 2 at +90 (default: 20)")
    p.add_argument("--band-ang", type=float, default=20.0, help="Half-width of the axis strips [Å] (default: 20)")
    p.add_argument("--common-radius-ang", type=float, default=0.0,
                   help="Radius used for the per-axis numbers of every map; 0: the smallest circle (default)")
    p.add_argument("--diff-range-ang", type=float, default=30.0, help="Color range of the difference panels, +/- [Å]")
    p.add_argument("--diff-contour-ang", type=float, default=10.0, help="Contour interval of the difference panels [Å]")
    p.add_argument("--diff-cmap", default="RdBu_r", help="Diverging colormap for the difference (default: RdBu_r)")
    p.add_argument("--outline-color", default="#b2182b")
    p.add_argument("--panel-mm", type=float, default=30.0, help="Circle diameter for the largest circle [mm]")
    p.add_argument("--units", choices=["A", "nm"], default="nm", help="Units along the profile axes (default: nm)")
    p.add_argument("--rotate-deg", type=float, default=0.0,
                   help="Rotate the difference panels clockwise by this angle [deg], as --panel-rotate-deg")
    p.add_argument("--no-axis-numbers", action="store_true",
                   help="Draw the axes as dashed lines only, without outside ticks and numbers")
    p.add_argument("--panel-font-pt", type=float, default=6.0)
    p.add_argument("--panel-label-pt", type=float, default=6.0)
    p.add_argument("--profile-width-mm", type=float, default=60.0)
    p.add_argument("--outdir", default="compare_out")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = []
    for item in args.runs:
        label, path = item.split("=", 1)
        raw.append((label, load(path)))
    r_max = max(r["radius"] for _, r in raw)
    r_common = args.common_radius_ang or min(r["radius"] for _, r in raw)
    step = float(raw[0][1]["u_A"][1] - raw[0][1]["u_A"][0])
    grid = np.arange(-math.ceil(r_max / step), math.ceil(r_max / step) + 1) * step
    runs = [(label, to_relative(r, grid)) for label, r in raw]
    ref = dict(runs)[args.reference]

    angles = (args.axes_angle_deg, args.axes_angle_deg + 90.0)
    s_grid = np.arange(-r_max, r_max + 1e-6, 2.0 * step)

    rows, profiles = [], {}
    for label, rel in runs:
        both = rel["support"] & ref["support"]
        dh = (rel["h"] - ref["h"])[both]
        fits = [axis_fit(rel, ang, args.band_ang, r_common) for ang in angles]
        for ang in angles:
            profiles[(label, ang)] = profile(rel, ang, args.band_ang, s_grid)
        rows.append((label, np.nanmedian(dh), np.nanpercentile(dh, 5), np.nanpercentile(dh, 95), fits))
        if label != args.reference:
            diff_panel(outdir / ("diff_" + label), rel, ref, label, args.reference, args.diff_range_ang,
                       args.panel_mm, args.panel_font_pt, args.panel_label_pt, args.diff_contour_ang,
                       args.diff_cmap, args.outline_color, angles, r_max,
                       rotate_cw_deg=args.rotate_deg, numbers=not args.no_axis_numbers)

    with mpl.rc_context({"font.size": args.panel_font_pt, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(args.panel_mm / 25.4, 11.0 / 25.4))
        cax = fig.add_axes([0.1, 0.62, 0.8, 0.26])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-args.diff_range_ang, args.diff_range_ang),
                                                cmap=args.diff_cmap), cax=cax, orientation="horizontal")
        cb.set_ticks([-args.diff_range_ang, 0, args.diff_range_ang])
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(pad=1.0, width=0.4, length=2.0)
        cb.set_label("Height difference (Å)", labelpad=1.5)
        fig.savefig(outdir / "diff_colorbar.png", dpi=600, transparent=True)
        fig.savefig(outdir / "diff_colorbar.pdf", transparent=True)
        plt.close(fig)

    with mpl.rc_context({"font.size": args.panel_font_pt, "pdf.fonttype": 42, "axes.linewidth": 0.4,
                         "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                         "xtick.major.size": 2.0, "ytick.major.size": 2.0}):
        w = args.profile_width_mm / 25.4
        two_rows = len(runs) > 4
        fig, axs = plt.subplots(1, 2, figsize=(w, (0.6 if two_rows else 0.5) * w), sharey=True)
        colors = plt.cm.tab10(np.arange(len(runs)) % 10)
        for k, ang in enumerate(angles):
            ax = axs[k]
            for (label, _), col in zip(runs, colors):
                prof, prof_meas = profiles[(label, ang)]
                xs_ = s_grid / (10.0 if args.units == "nm" else 1.0)
                ax.plot(xs_, prof, color=col, linewidth=0.5, linestyle=":")
                ax.plot(xs_, prof_meas, color=col, linewidth=1.0, label=label)
            ax.set_title("axis %d" % (k + 1), pad=2)
            ax.set_xlabel("position along axis (%s)" % ("nm" if args.units == "nm" else "\u00c5"), labelpad=1)
            ax.tick_params(pad=1)
        axs[0].set_ylabel("membrane height (Å)", labelpad=1)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=args.panel_font_pt, handlelength=1.2,
                   loc="lower center", ncol=(len(runs) if len(runs) <= 4 else int(math.ceil(len(runs) / 2.0))),
                   columnspacing=1.0)
        fig.tight_layout(pad=0.3, rect=(0, 0.25 if two_rows else 0.13, 1, 1))
        fig.savefig(outdir / "profiles.png", dpi=600, transparent=True)
        fig.savefig(outdir / "profiles.pdf", transparent=True)
        plt.close(fig)

    with open(outdir / "compare_summary.tsv", "w") as f:
        f.write("map\treference\tradius_A\tcommon_radius_A\tdh_median_A\tdh_p5_A\tdh_p95_A\t"
                "slope_axis1_deg\tkappa_axis1_nm^-1\tslope_axis2_deg\tkappa_axis2_nm^-1\n")
        print("Per-axis numbers over the common radius of %.0f Å:" % r_common)
        for (label, med, p5, p95, fits), (_, rel) in zip(rows, runs):
            (s1, k1, _), (s2, k2, _) = fits
            f.write("%s\t%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.4f\t%.1f\t%.4f\n"
                    % (label, args.reference, rel["radius"], r_common, med, p5, p95, s1, k1, s2, k2))
            print("%-10s minus %-8s median %+5.1f A (5-95%%: %+5.1f .. %+5.1f) | axis 1: slope %+5.1f deg, kappa %+.4f | axis 2: slope %+5.1f deg, kappa %+.4f nm^-1"
                  % (label, args.reference, med, p5, p95, s1, k1, s2, k2))
    print("Written to", outdir)


if __name__ == "__main__":
    main()
