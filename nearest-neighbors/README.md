# nearest-neighbors

Where a particle's neighbours sit, seen from the particle itself.

For every particle in a RELION 5 tomography STAR file, `nearest_neighbors.py` finds
its k nearest neighbours in the same tomogram, rotates each neighbour vector into
that particle's own frame (from its Euler angles), and pools the vectors over all
particles and tomograms. Preferred contacts (dimers, lattice neighbours, packing)
show up as dense spots in the cloud, which can be drawn over the average of the
particle.

```
pip install -r requirements.txt
python nearest_neighbors.py -h
```

## Input

- `--star`: RELION 5 particle STAR file with
  - `_rlnTomoName`
  - `_rlnCenteredCoordinate{X,Y,Z}Angst`
  - `_rlnAngleRot`, `_rlnAngleTilt`, `_rlnAnglePsi`
  - optional `_rlnOrigin{X,Y,Z}Angst` (subtracted from the coordinates)
- `--target_mrc` (optional): the average of the particle, drawn as a projection
  under the 2D clouds and as an isosurface in the 3D viewer. Voxel size comes from
  `--target_pix_nm`, the MRC header, or the pixel-size column of the STAR file.

Neighbours are searched only within each tomogram.

## Examples

First neighbour and six nearest neighbours, 2D views:

```
python nearest_neighbors.py --star particles.star --k 2,7 --outdir knn_out
```

With the average as overlay, k-th neighbour highlighted, interactive 3D viewer:

```
python nearest_neighbors.py --star particles.star --k 2,7 \
    --target_mrc average.mrc --highlight_kth --view3d plotly --outdir knn_out
```

## Counting neighbours

- `--k_mode includes_self` (default): k counts the particle itself, as in a raw kNN
  query. `--k 2` is the first nearest neighbour, `--k 7` the six nearest.
- `--k_mode excludes_self`: k is the neighbour rank (`--k 1` is the first neighbour).
- `--neighbor_set up_to_k` (default) plots all neighbours up to k; `kth_only` plots
  only the k-th.

## Frame convention

The rotation matrix is RELION's (`Euler_angles2matrix`, ZYZ), which maps reference to
particle orientation; a neighbour vector `v` in the tomogram becomes `Aᵀ v` in the
frame of the reference, so the cloud lines up with the average. `--flip_local_axes`
mirrors the cloud and is off by default.

If `_rlnTomoSubtomogram{Rot,Tilt,Psi}` are non-zero in the STAR file, the particle
angles are relative to the subtomogram frame; the script warns, since it uses
`_rlnAngle*` only.

## Outputs (in `--outdir`)

- `knn_k{K}_top_xy.png`, `knn_k{K}_side_xz.png`, `knn_k{K}_front_yz.png`: neighbour
  clouds in the particle frame, coloured by local density
- `k2_first_nn_stats.txt`, `k2_first_nn_distance_hist.png`: first nearest-neighbour
  distance statistics
- `knn_k{K}_3d.html` (`--view3d plotly`): interactive 3D viewer with slab and
  density sliders; open in a browser

Display options: `--lim_nm`, `--bins`, `--points_size`, `--points_alpha`,
`--alpha_by_density` (2D), `--screenshot_style`, `--target_*`. See `-h`.
