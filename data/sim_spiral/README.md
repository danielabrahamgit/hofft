# Data Docs

## Purpose
This dataset is great for simulating non-Cartesian MRI reconstruction. We use this dataset for the `test_nufft_sim.py` file.

## Explanations of Saved Tensors
- **img** (220, 220), a shepp-logan image phantom slice.
- **mps** (16, 220, 220), the 16 coil sensivity maps which were simulated using biot savart law on 16 surface loop coils + espirit.
- **trj** (26296, 2), single shot spiral trajectory with 3X radial undersampling, designed to hit 1mm resolution at a 22cm field of view. Values scaled between [-220/2, +220/2].
- **dcf** (26296,), the corresponding density compensation function for the above trajectory.

