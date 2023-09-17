

# Experiment: 3DGS20 - Force Prediction in 3D Graphic Statics

## Dataset

The dataset comprises 30,000 $1$-complexes. Within each complex, there are 20 $0$-cells, $\sigma\_i$, characterised by a positional vector $\mathbf{p}\_i \in \mathbb{R}^3$. Each complex contains 34 $1$-cells, $\tau\_{ij} = \{\sigma\_i,\sigma\_j\}$.

For each 1-cell, denoted as $\tau\_i$, a magnitude $f\_{\tau\_i} [\text{N}]$ is assigned. This scalar value quantifies the magnitude of the compression force exerted on the 1-cell. To construct a force vector $\mathbf{f}\_{\tau\_{ij}}$ in $\mathbb{R}^3$, this magnitude is combined with a direction. The direction is determined by the positional vectors $\mathbf{p}\_j$ and $\mathbf{p}\_i$, which are the spatial coordinates of the neighboring 0-cell $\sigma\_j$ and the 0-cell $\sigma\_i$, respectively.

The force vector $\mathbf{f}\_{\tau\_{ij}}$ thus encapsulates both the magnitude and direction of the force exerted on $\sigma\_i$ due to its interaction with $\sigma\_j$. For a given 0-cell $\sigma\_i$, there exist $n$ such force vectors, where $n$ corresponds to the number of neighboring 0-cells. The aggregation of these force vectors yields a resultant force vector, which serves as the ground truth vector, $\mathbf{y}\_{\sigma\_i} \in \mathbb{R}^3$, for the $i$-th 0-cell:

$$\mathbf{y}\_{\sigma\_i} = \sum\_{j=1}^{n} \mathbf{f}\_{\tau\_{ij}}$$

Each $0$-cell $\sigma\_i$ is associated with a feature vector $\mathbf{x}\_{\sigma\_i} \in \mathbb{R}^3$, defined as:
$$\mathbf{x}\_{\sigma\_i} = \sum\_{j=1}^{n} \mathbf{d}\_{ji}$$
where $\mathbf{d}\_{ji} = \mathbf{p}\_i - \mathbf{p}\_j$.

Lastly, each $1$-cell $\tau\_{i}$ is associated with an attribute $\mathbf{a}\_{\tau\_i}$, which is calculated as the Euclidean distance between the position vectors of its constituent $0$-cells:
$$\mathbf{a}\_{\tau\_i} = \lVert \mathbf{p}\_j - \mathbf{p}\_i \rVert$$



<div align="center">
  <img src="img/3d_graphic_statics_data_example_prepro-annot-v3.png" width="600">
</div>

*Processed data sample. Left: compression force vectors $\mathbf{f}\_{\tau\_{ij}} \subset \mathbf{y}\_{\sigma\_i}$. Right: glyph plot of the spherical harmonic embeddings $\tilde{\mathbf{y}}\_{\sigma\_{i}}$.*








## Computational Implementation

The e3nn library is used for the effective implementation of composite steerable vectors and irreducible representations. The model consists of three message-passing layers and has a total parameter count of 692,000.

## Training

The model is trained using a batch size of 64 with an initial learning rate of $3e^{-4}$. The optimization criterion is the Mean Absolute Error (MAE) between the predicted and ground-truth force vectors.

## Results

| Method | MAE | Time [s] |
|--------|-----|----------|
| G-map & Nonscal. | .xxxx±.xxxxx | .xxxx |
| + & - | .0204±.00019 | .xxxx |
| + & + | **.0186±.00015** | .0165 |

![Comparison of MAE](img/mp-steer-non-steer-comparison.png)
![MAE on Validation Dataset](img/mae-sh-sample-v3-upscaled-bw.png)

<div align="center">
  <img src="img/3dgs20-complex-level-mae-density.png" width="300">
</div>



## Limitations and Future Work

Validation of the method has been constrained to synthetic datasets. Future work should aim to validate the model using datasets derived from actual physical configurations.

## Summary

The G-equivariant and nonscalar method constitutes a robust and computationally efficient framework for the accurate prediction of physical properties in structural design.

