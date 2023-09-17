# Experiment: 3DGS20 - Force Prediction in 3D Graphic Statics

## Dataset

The 3DGS20 dataset is synthesized using 3D Graphic Statics, a tool for structural design and engineering. This dataset is tailored for the generation of compression-only funicular structures along with their corresponding force diagrams.

The dataset comprises 30,000 1-complexes, each containing 20 0-cells and 34 1-cells. For each 1-cell, a magnitude is assigned that quantifies the compression force exerted on it. The dataset also includes feature vectors and attributes for each cell.

![Processed data sample](img/3d_graphic_statics_data_example_prepro-annot-v3.png)

## Computational Implementation

The e3nn library is used for the effective implementation of composite steerable vectors and irreducible representations. The model consists of three message-passing layers and has a total parameter count of 692,000.

## Training

The model is trained using a batch size of 64 with an initial learning rate of \(3e^{-4}\). The optimization criterion is the Mean Absolute Error (MAE) between the predicted and ground-truth force vectors.

## Results

| Method | MAE | Time [s] |
|--------|-----|----------|
| G-map & Nonscal. | .xxxx±.xxxxx | .xxxx |
| + & - | .0204±.00019 | .xxxx |
| + & + | **.0186±.00015** | .0165 |

![Comparison of MAE](img/mp-steer-non-steer-comparison.png)
![MAE on Validation Dataset](img/mae-sh-sample-v3-upscaled-bw.png)
![MAE Density](img/mae-sh-sample-v3-upscaled-bw-mae.png)

## Limitations and Future Work

Validation of the method has been constrained to synthetic datasets. Future work should aim to validate the model using datasets derived from actual physical configurations.

## Summary

The G-equivariant and nonscalar method constitutes a robust and computationally efficient framework for the accurate prediction of physical properties in structural design.

