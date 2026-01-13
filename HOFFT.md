# HOFFT Model Overview
The goal of HOFFT is to evaluate the following MRI forward model (with field imperfections) efficiently:
$$
y(t) = \int_\mathbf{r} x(\mathbf{r}) e^{-j2\pi \mathbf{r} \cdot \mathbf{k}(t)} \underbrace{e^{-j2\pi \boldsymbol{\phi}(\mathbf{r})\cdot \boldsymbol{\alpha}(t)}}_\text{Field-Imperfection} d\mathbf{r}.
$$

Our approach is to combine the [NUFFT algorithm](NUFFT.md), which originally uses compact kernels in the spatial frequency domain to model non-Cartesian spatially linear phase functions, and [prior factorization approaches](PRIOR_METHODS.md). We train the NUFFT kernels to model spatially non-linear phase functions, and synergize them with $L$ spatial factors according to 
$$
e^{-j2\pi \boldsymbol{\phi}(\mathbf{r})\cdot \boldsymbol{\alpha}(t)} \approx \sum_{l=1}^L \sum_{i=1}^K w_{l,i}(t) b_l(\mathbf{r}) e^{-j 2\pi \mathbf{r} \cdot \frac{\mathbf{z}_i}{\sigma}},
$$
where $L$ is the number of spatial factors, $K = W \times W, \cdots, \times W$ is the kernel size, $e^{-j 2\pi \mathbf{r} \cdot \frac{\mathbf{z}_i}{\sigma}}$ are the spatial domain kernel phase functions ($\sigma$ is the spatial oversampling factor), $w_{l,i}(t)$ are the temporally-varying HOFFT kernel weights, and $b_l(r)$ are the HOFFT spatial factors. 

# 1.) Combine Non-Cartesian Deviations into $\boldsymbol{\phi}(\mathbf{r}), \boldsymbol{\alpha}(t)$

# 2.) Initialize Spatial Factors

# 3.) Peform HOFFT Decomposition with Alternating Least Squares

# 4.) Assemble Forward Model
