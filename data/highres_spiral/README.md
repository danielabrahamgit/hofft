

## Dataset Details

Multi-shot spiral readout dataset with b0 and gradient correction with FESTIVE. 

Data was collected on TerraX Impule 7T Scanner at Stanford on Apr 08 2026.

| Parameter             | Value                                   |
|-----------------------|-----------------------------------------|
| Sequence Type         | Axial Gradient Echo Spiral              |
| Number of Slices      | 10 (only lowest slice included here)    |
| Flip Angle (FA)       | 60°                                     |
| Echo Time (TE)        | 1.7 ms                                  |
| Repetition Time (TR)  | 0.6 s                                   |
| Number of Averages    | 4                                       |
| Number of Shots       | 16                                      |
| Scan Time             | ~41 ms                                  |
| FOV                   | 220 mm                                  |
| In-plane Resolution   | 0.3 mm                                  |
| Slice Thickness       | 0.5 mm (located +30.5 mm from isocenter)|
| Sampling Rate         | 1 μs                                    |
| Spiral Gmax           | 100 mT/m                                |
| Spiral Smax           | 500 T/m/s                               |

### Calibrations: 
- b0 estimated with 8-echo GRE
- kspha estimated with FESTiVE 

## Dataset contents
All data needed for recon is in `data.pt`. 

### Dimension names
| Abbreviation | Meaning          | Value     |
|--------------|------------------|-----------|
| C            | Coil             | 20        |
| T            | Readout axis     | 23210     |
| P            | Shot index       | 16        |
| N = X = Y    | Image dimension  | 734       |

### Dataset contents
| Key           | Shape            | Note                                                   | Units                                     |
|---------------|------------------|--------------------------------------------------------|-------------------------------------------|
| `ksp`         | (C, T, P)        | K-space data                                           |                                           |
| `mps`         | (C, X, Y)        | Sensitivity maps                                       |                                           |
| `trj`         | (T, P, 2)        | Trajectory, cycles/image (-N/2 to N/2 convention)      | [cycles/image]                            |
| `dcf`         | (T, P)           | Density compensation factors                           |                                           |
| `coords`      | (X, Y, d)        | Voxel coordinates                                      | [meters]                                  |
| `times`       | (T, P)           | Sampling times (uniform dt)                            | [seconds]                                 |
| `b0`          | (X, Y)           | B0 field map                                           | [Hz]                                      |
| `kspha`       | (T, P, 16)       | FESTIVE sh coefficients (up to 3rd order)              | [rad / m^[n]]                             |
| `kspha_bases` | (X, Y, 16)       | Basis functions for field coefficients                 | [m^[n]]                                   |

### Dataset preprocessing
Data is preprocessed already with the following steps (don't worry about any of this, just listing for reference)
1. FOV shifted 8mm relative to acquired data
2. Coil compression to 20 coils
3. Demodulation of axial concomitant phase (not much here bc 7T but still up to ~5Hz)
4. RF Spike filtering (don't worry about this, but you might see that like 1-2% of ksp data is zeroed out, this is why)


## Reference recons
My initial recons are in `recon.pt`. Keys:
- `hofft`: HOFFT (params and initialization below)
    - Note: max eigenvalue = 2.30
- `matrix`: Expanded encoding
    - Note: max eigenvalue = 3.85
- `uncorr`: Same as HOFFT recon, just no fields (only trj). also L=4 instead of L=14

### HOFFT Strategy
HOFFT strategy roughly used something like this:
```python
hparams = hofft_params(
    kern_size=(3,3),
    os=1.25,
    L=14,
    use_type3=False,
    check_convergence=False,
)
apods = K_alphas_apod_init(
    phis, alphas, hparams,
    method = 'minmax',
    apod_init_method = 'seg',
    check_convergence=False,
    mask = mask,
    num_als_iter = 8,
    K = 80,
)
apods_init = apods # don't do additional ALS for apods
phase_model = matvec_naive(phis, alphas, mask=mask)
weights = lstsq_temporal(phase_model, kern_bases, apods_init, mask=mask)
# --> return weights, apods
```

### Rough Recon Computational Benchmark

| Method  | Total Time        | Decomp Time   | Recon Time (CG+ME)  | Max Eigen Time | CG Time     | CG Iter/sec | Peak Memory |
|---------|-------------------|---------------|---------------------|----------------|-------------|-------------|-------------|
| uncorr  | 6 sec             | 5 sec         | 0.8 sec             | ~0.4 sec       | 0.5 sec     | 1500        | 38.7 GB     |
| HOFFT   | 13 sec            | 10 sec        | 3 sec               | ~1.5 sec       | 1.5 sec     | 868         | 38.7 GB     |
| Matrix  | 40 min (~2400 sec)| -             | ~40 min             | ~20 min        | ~20 min     | 0.011       | 20.5 GB     |

Recons are all CG recon 12 iterations, l2 lamda = 1e-3, and include time to compute max eigenvalue and any other decmps beforehand. I haven't over-optimized any of these so take the times with a grain of salt; just found some set of params that worked and gave good quality images. 