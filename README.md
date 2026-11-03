# cryoet-SpatialTools

Spatial analysis of cryo-electron tomography data.

| Tool | What it does | Input |
|---|---|---|
| [`nearest-neighbors`](nearest-neighbors/) | k-nearest-neighbour clouds in each particle's own frame, pooled over tomograms; 2D views and an interactive 3D viewer | RELION 5 particle STAR, optional average (MRC) |
| [`membrane-topography`](membrane-topography/) | Membrane mid-surface, height map and curvature from a subtomogram average; height differences between maps | Subtomogram average (MRC), optional model (PDB/mmCIF) |

Each folder is self-contained, with its own README and `requirements.txt`.

```
git clone https://github.com/CryoFlo/cryoet-SpatialTools.git
cd cryoet-SpatialTools
pip install -r nearest-neighbors/requirements.txt -r membrane-topography/requirements.txt
```

## Citation

If you use these scripts, please cite: [reference]

## License

MIT, see [LICENSE](LICENSE).
