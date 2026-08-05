# Data Docs

A physical brain phantom was imaged using a gradient echo spiral on a [3T GE UHP MRI scanner](https://cni.su.domains/wiki/index.php?title=MR_Scanner) using a 32 receive channel NOVA head coil array. We use 60 shots in order to be robust to any field imperfections. The channel dimension was compressed down to 13 using coil copmression, and coil sensitivities were estimated by ESPIRiT from the fully
sampled central region. 

## Explanations of Saved Tensors
- **mps** (13, 220, 220), measured (via ESPIRIT) coil sensitivity maps, which were SVD compressed from 32 down to 13 channels.
- **evals** (220, 220) the eigen-value maps (via ESPIRIT) used for masking.
- **ksp** (13, 1536, 60), the measured 60-shot spiral k-space data, the same SVD compression matrix was applied.
- **trj** (1536, 60, 2), the 60-shot spiral trajectory (each shot is 3X undersampled). Designed to hit 0.8mm resolution at a 22cm field of view. Values scaled between [-220/2, +220/2].
- **dcf** (1536, 60), the corresponding density compensation function for the above trajectory.


