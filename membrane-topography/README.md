# membrane-topography

Membrane mid-surface, height maps and curvature from subtomogram averages of
membrane-embedded complexes, and height differences between maps.

- `Curvature_analysis.py` — detects the membrane in one map (threshold-free sheet
  filter + bilayer-profile cross-correlation), reports its height and curvature
  inside a circle, and writes figure-ready panels.
- `compare_heights.py` — height differences between runs of `Curvature_analysis.py`
  (difference panels, height profiles along the curvature axes, summary table).

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11, mrcfile 1.5.

```
pip install -r requirements.txt
python Curvature_analysis.py -h
python compare_heights.py -h
```

## Input

- Maps in MRC format, all with the same box, voxel size and centring when they are
  compared. The mask applied to a map (any shape) is detected automatically.
- Optional atomic models (`.pdb` / `.cif`) in the coordinate system of the map,
  for the model outline (`--outline-pdb`). If a map was padded without updating
  its header origin, fix the origin first so that the model lines up.

## Example (settings used for the figures)

Height map, curvature along two perpendicular axes, and density section along axis 1:

```
python Curvature_analysis.py --map map.mrc --level 0.0025 --outline-pdb model.pdb \
    --frame box --box-axis y --detrend none \
    --template-iter 6 --template-rim-agree-ang 25 \
    --disk-center-ang 0 0 --disk-radius-ang 230 --axes-angle-deg 20 \
    --fix-height-scale --height-vmin -140 --height-vmax 20 \
    --height-cmap YlGnBu_r --height-gamma 2 --outline-color "#b2182b" \
    --panel-mm 30 --panel-rotate-deg 90 --panel-section 1 \
    --outdir map_out
```

Height differences to a reference map:

```
python compare_heights.py --runs Reference=ref_out MapA=mapA_out MapB=mapB_out \
    --reference Reference --axes-angle-deg 20 --rotate-deg 90 --outdir compare_out
```

## Main outputs

`Curvature_analysis.py` (per map):
- `height_map_panel.pdf/.png`, `height_colorbar.pdf/.png`, `section_axis1_panel.pdf/.png`:
  figure panels at final printed size
- `height_map.png`, `mean_curvature.png`, `gaussian_curvature.png`, `support_qc.png`:
  overview and QC plots
- `summary.tsv`: curvature along the axes (κ1, κ2), quadric principal curvatures,
  detected fraction, extrapolation distance
- `membrane_surface.npz` / `.csv`: surface, heights, curvature maps, masks
- `membrane_surface_in_map.mrc`: reported membrane slab on the input grid (overlay in ChimeraX)

`compare_heights.py`:
- `diff_<label>.pdf/.png`, `diff_colorbar.pdf/.png`, `profiles.pdf/.png`, `compare_summary.tsv`

## Citation

If you use these scripts, please cite: [reference]
