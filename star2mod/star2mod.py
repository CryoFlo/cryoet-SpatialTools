#!/usr/bin/env python3
"""
RELION particle STAR -> IMOD model (.mod) of scattered points, for viewing picks in 3dmod.

Coordinates
- RELION 5 (_rlnCenteredCoordinate{X,Y,Z}Angst): Angstrom from the tomogram centre,
  converted to voxels of the tomogram you will open. Size from --tomo or --dims;
  voxel size from --angpix, else the --tomo header, else
  _rlnTomoTiltSeriesPixelSize x --binning.
- RELION 3/4 (_rlnCoordinate{X,Y,Z}): pixels from the corner, scaled by --coord_scale
  (e.g. 0.25 for unbinned coordinates on a bin-4 tomogram).
- _rlnOrigin{X,Y,Z}Angst shifts are applied (true centre = coordinate - origin).

One .mod (+ .txt point list) per STAR file, or per tomogram if a STAR file holds several.
Requires IMOD's point2model on PATH.
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import starfile

# Tomogram size (voxels X Y Z) used when neither --dims nor --tomo is given
DEFAULT_DIMS = (1024, 1024, 512)


def tomo_geometry(path: str) -> tuple[tuple[int, int, int], float]:
    """(nx, ny, nz) and voxel size (Angstrom) from an MRC header."""
    import mrcfile

    with mrcfile.open(path, header_only=True, permissive=True) as m:
        h = m.header
        dims = (int(h.nx), int(h.ny), int(h.nz))
        mx = int(h.mx)
        angpix = float(h.cella.x) / mx if mx > 0 else 0.0
    return dims, angpix


def read_star(path: str):
    """(particles table, optics table or None)."""
    data = starfile.read(path)
    if isinstance(data, dict):
        return data.get("particles", next(iter(data.values()))), data.get("optics")
    return data, None


def star_pixel_size(df, optics) -> float:
    """_rlnTomoTiltSeriesPixelSize from the particles table, else the optics table; 0 if absent."""
    for t in (df, optics):
        if t is not None and "rlnTomoTiltSeriesPixelSize" in t.columns:
            return float(np.median(t["rlnTomoTiltSeriesPixelSize"].to_numpy(float)))
    return 0.0


def particle_voxels(df, dims, angpix, coord_scale) -> np.ndarray:
    cols = df.columns
    if "rlnCenteredCoordinateXAngst" in cols:
        if dims is None or not angpix:
            sys.exit("RELION 5 centred coordinates need --tomo or --dims, and a voxel size "
                     "(--tomo, --angpix, or _rlnTomoTiltSeriesPixelSize in the STAR file).")
        xyz = np.stack(
            [df[f"rlnCenteredCoordinate{a}Angst"].to_numpy(float) / angpix + n / 2.0
             for a, n in zip("XYZ", dims)], axis=1)
    elif "rlnCoordinateX" in cols:
        xyz = np.stack([df[f"rlnCoordinate{a}"].to_numpy(float) * coord_scale for a in "XYZ"], axis=1)
    else:
        raise ValueError("no _rlnCoordinate* or _rlnCenteredCoordinate*Angst columns")

    ocols = [f"rlnOrigin{a}Angst" for a in "XYZ"]
    if all(c in cols for c in ocols):
        orig = df[ocols].to_numpy(float)
        if np.any(orig != 0):
            if not angpix:
                sys.exit("Non-zero _rlnOrigin*Angst: give --tomo or --angpix to apply them.")
            xyz -= orig / angpix
    return xyz


def write_model(xyz, txt: Path, mod: Path, sphere: int, image: str | None):
    np.savetxt(txt, xyz, fmt="%.3f")
    cmd = ["point2model", "-scat"]
    if sphere > 0:
        cmd += ["-sphere", str(sphere)]
    if image:
        cmd += ["-image", image]
    subprocess.run(cmd + [str(txt), str(mod)], check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stars", nargs="*", help="STAR files (default: *_particles.star in the current folder)")
    ap.add_argument("--tomo", help="Tomogram (.mrc) the model is for: sets --dims and --angpix, and the model's coordinate system")
    ap.add_argument("--dims", nargs=3, type=int, metavar=("X", "Y", "Z"),
                    help="Tomogram size in voxels (default: --tomo header, else %d %d %d)" % DEFAULT_DIMS)
    ap.add_argument("--angpix", type=float, default=0.0,
                    help="Tomogram voxel size (Angstrom). Default: --tomo header, else "
                         "_rlnTomoTiltSeriesPixelSize x --binning")
    ap.add_argument("--binning", type=float, default=1.0,
                    help="Tomogram binning relative to _rlnTomoTiltSeriesPixelSize (used without --angpix/--tomo)")
    ap.add_argument("--coord_scale", type=float, default=1.0, help="Scale for _rlnCoordinate* (RELION 3/4), e.g. 1/binning")
    ap.add_argument("--sphere", type=int, default=5, help="Sphere radius in voxels for display (0: points only)")
    ap.add_argument("--outdir", default="mod")
    args = ap.parse_args()

    if shutil.which("point2model") is None:
        sys.exit("point2model not found: load IMOD first (e.g. 'module load IMOD').")

    dims = tuple(args.dims) if args.dims else None
    tomo_angpix = 0.0
    if args.tomo:
        t_dims, tomo_angpix = tomo_geometry(args.tomo)
        dims = dims or t_dims
    dims = dims or DEFAULT_DIMS

    stars = args.stars or sorted(glob.glob("*_particles.star"))
    if not stars:
        sys.exit("No STAR files given and no *_particles.star here.")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for star in stars:
        df, optics = read_star(star)
        base = Path(star).stem

        star_px = star_pixel_size(df, optics)
        if args.angpix:
            angpix, src = args.angpix, "--angpix"
        elif tomo_angpix:
            angpix, src = tomo_angpix, "--tomo header"
        else:
            angpix, src = star_px * args.binning, f"_rlnTomoTiltSeriesPixelSize x {args.binning:g}"
        if angpix:
            print(f"{star}: voxel size {angpix:.4g} A ({src})")
        if star_px and angpix and abs(star_px * args.binning - angpix) > 1e-3 * angpix:
            print(f"[note] {star}: _rlnTomoTiltSeriesPixelSize x binning = {star_px * args.binning:.4g} A "
                  f"differs from {angpix:.4g} A; tomogram binning is {angpix / star_px:.3g} if {src} is right")
        if "rlnTomoName" in df.columns and df["rlnTomoName"].nunique() > 1:
            groups = [(f"{base}_{re.sub(r'[^A-Za-z0-9._-]', '_', str(t))}", g) for t, g in df.groupby("rlnTomoName")]
        else:
            groups = [(base, df)]

        for name, g in groups:
            xyz = particle_voxels(g, dims, angpix, args.coord_scale)
            if dims is not None:
                out = np.any((xyz < 0) | (xyz >= np.array(dims)), axis=1).sum()
                if out:
                    print(f"[warning] {name}: {out}/{len(xyz)} points outside {dims} - check --dims / --angpix / binning")
            mod = outdir / f"{name}.mod"
            write_model(xyz, outdir / f"{name}.txt", mod, args.sphere, args.tomo)
            print(f"{star} -> {mod} ({len(xyz)} points)")


if __name__ == "__main__":
    main()
