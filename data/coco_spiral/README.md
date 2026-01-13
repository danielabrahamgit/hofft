# Data Docs

## Purpose
We performed a corronal (X-Z) plane R3 0.8mm resolution 22cm FOV spiral acquisition on a water/plastic phantom, which is sensitive to high order conocmitant fields.Tthis dataset was collected on our [3T GE MRI scanner](https://cni.su.domains/wiki/index.php?title=MR_Scanner). 

## Explanations of Saved Tensors
- **b0** (276, 276), the spatially varying off-resonance field map in units of Hz.
- **mps** (17, 220, 220), measured (via ESPIRIT) coil sensitivity maps, which were SVD compressed from 32 down to 17 channels.
- **ksp** (17, 26475, 3), the measured k-space data, the same SVD compression matrix was applied.
- **trj** (26475, 3, 2), the 3-shot spiral trajectory (each shot is 3X undersampled). Designed to hit 0.8mm resolution at a 22cm field of view. Values scaled between [-276/2, +276/2].
- **dcf** (26475, 3), the corresponding density compensation function for the above trajectory.

## High Order Field Model
The recieved signal model is 
$$\mathbf{y}(t) = \int_\mathbf{r} x(\mathbf{r}) \mathbf{c}(\mathbf{r}) e^{-j2\pi \mathbf{k}(t) \cdot \mathbf{r}} e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha}(t)} d\mathbf{r}$$
where $\mathbf{y}(t)$ is the multi-channel k-space data, $\mathbf{c}(\mathbf{r})$ are the coil sensivity maps, $\mathbf{k}(t)$ is the k-space trajectory, $\boldsymbol{\phi}(\mathbf{r})$ are the spatial phase basis functions, and $\boldsymbol{\alpha}(t)$ are the temporal phase coefficients. 

In this case, the high order phase terms are due to both X-Z plane concomitant fields and $B_0$ fields. As such, the $\boldsymbol{\phi}(\mathbf{r}), \boldsymbol{\alpha}(t)$ functions are
$$
\boldsymbol{\phi}(\mathbf{r}) = \begin{bmatrix}
    \Delta B_0(\mathbf{r}) \\
    r_x^2 \\
    r_z^2
\end{bmatrix}, \boldsymbol{\alpha}(t) = \begin{bmatrix}
    t \\
    \frac{\bar{\gamma}}{8 B_0} \int_0^t G_z^2(\tau) d\tau \\
    \frac{\bar{\gamma}}{2 B_0} \int_0^t G_x^2(\tau) d\tau 
\end{bmatrix}.
$$
