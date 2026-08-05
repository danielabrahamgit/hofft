# Paper Experiments

We want to answer the following questions:
- Is HOFFT mathematically sound?   
Answered in 1,2
- Does HOFFT improve the forward model accuracy vs computation tradeoff?   
Answered in 3, 4
- Does the compressed HOFFT idea work?   
Answered in 4
- Does it matter on real MRI data?   
Answered in 5, 6

## Experiments
### Experiment 1: Learning NUFFT with HOFFT
Using the shepp logan phantom, we look at the learned NUFFT kernel for L=1: W=2, W=3 and the accuracy of each using HOFFT and NUFFT. The kernels and spatial factors should basically look identical. We can show this in 1D and 2D
### Experiment 2: Simulated High Order Phase
Next we'll use the same shepp logan phantom to simulate a quadratic high order phase evolution. We can show the efficacy of HOFFT vs spatio temporal splitting for a grid of images corresponding to different L,W, with expanded encoding as ground truth
### Experiment 3: Exploring the L,W HOFFT Tradeoff Space
 Here we will look at some quantiative plots of L, W vs things like NRMSE and forward model evaluation time. We want to make sure HOFFT gives a better tradeoff. We'll ignore decomposition time for now. 
### Experiment 4: Validating and Understanding Sparse Compressed HOFFT
Now we want to show that HOFFT decomposition time is differnt using the sparse model (compressed ALS + sparsity least squares) or that it takes less memory. We also want to understand the Q, S tradeoff space. We can evaluate this on the simulated dataset still.
### Experiment 5: Real MRI Data Case 1 -- 2D High Order Phase
Coco phantom, show L,W plot and images right where HOFFT flattens out. Should see ~3X improvement for same error
### Experiment 6: Real MRI Data Case 2 -- 3D High Order Phase
Same as above, 3D case

## Figures
### Main HOFFT Decomp Figure -- Contrast with NUFFT

### Compressed HOFFT Figure

### Forward Model Figure?

### Learned NUFFT Kernels
- Show the reconstructed HOFFT kernel/apod against KB-NUFFT kernel/apod for W=2, W=3, os=1.25
- Show reconstruced images/error maps for each
- Uses data from Experiment 1

### Simulated Quadratic Phase
- Show uncorrected, EE, and a few (L,W) pairs for HOFFT vs Splitting 
- Uses data from Experiment 2

### Accuracy vs Computation Tradeoff
- Perform quanitative L,W nrmse sweep analysis on exp 1 and exp 2 data
- Forward model recon time L,W sweep
- Error vs time L,W sweep 

### Compressed HOFFT
- For quad phase data only, show how compressed HOFFT gives similar results for some Q,S value
- Qunatify NRMSE/Decomp-Time/Forward-time/Memory vs Q,S
- Also add CUR decomp sketching step?

### 2D MRI Reconstruction
- (L,W) grid of reconstructions for splitting and HOFFT
- Maybe some lineplots
- Repeat for structural spiral and rossette 

### High Res 3D MRI Reconstruction
- Single recon, big gains showed here 
- HOFFT --> minutes
- TS-NUFFT --> tens of minutes
- Expanded encoding --> hours


