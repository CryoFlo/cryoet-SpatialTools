# cryoet-SpatialTools

Spatial analysis of cryo-electron tomography data: how particles are arranged
relative to each other, and the shape of the membranes they sit in.

| Tool | What it does | Input |
|---|---|---|
| [`nearest-neighbors`](nearest-neighbors/) | k-nearest-neighbour clouds in each particle's own frame, pooled over tomograms; 2D views and an interactive 3D viewer | RELION 5 particle STAR, optional average (MRC) |
| [`membrane-topography`](membrane-topography/) | Membrane mid-surface, height map and curvature from a subtomogram average; height differences between maps | Subtomogram average (MRC), optional model (PDB/mmCIF) |
| [`star2mod`](star2mod/) | Particle list as an IMOD model, to check particle positions on the tomogram (e.g. after template matching) | RELION 3/4/5 particle STAR, tomogram (MRC) or its size |

Each folder is self-contained, with its own README and `requirements.txt`.

```
git clone https://github.com/CryoFlo/cryoet-SpatialTools.git
cd cryoet-SpatialTools
pip install -r nearest-neighbors/requirements.txt -r membrane-topography/requirements.txt -r star2mod/requirements.txt
```

## Citation

If you use these scripts, please cite: [reference]

## License

MIT, see [LICENSE](LICENSE).
