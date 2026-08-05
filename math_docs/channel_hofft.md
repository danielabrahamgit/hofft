# Applying HOFFT Per Channel

## Motivation
We are given spatial phase functions $\boldsymbol{\phi}(\mathbf{r}) \in \mathbb{R}^B$ and temporal phase features $\boldsymbol{\alpha}(t) \in \mathbb{R}^B$, and would like to make the following HOFFT decomposition:
$$
e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha} (t)} \approx \sum_{l=1}^L \sum_{w=1}^W g_l(\mathbf{r}) e^{-j 2\pi \mathbf{z}_w \cdot \mathbf{r} / \sigma} h_{l,w}(t),
$$
where $g_l(\mathbf{r})$ are the L HOFFT spatial factors, $\mathbf{z}_w$ form a compact square fourier kernel in $d=2$ or $d=3$ dimensions, and $h_{l,w}(t)$ are HOFFT kernels.

Normally, we choose $L$ HOFFT spatial factors by picking $L$ vectors $\boldsymbol{\beta}_1, \dots, \boldsymbol{\beta}_L$ to represent our massive set of $M$ phase coefficients $\boldsymbol{\alpha}_1, \dots, \boldsymbol{\alpha}_M$, and the $l^\text{th}$ spatial factor is simply $g_l(\mathbf{r}) = e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\beta}_l}$. One interpretation of this is that each HOFFT kernel only needs to interpolate within some neighborhood around each $\boldsymbol{\beta}$ cluster. 

An alternative approach is to let each HOFFT kernels be responsible for a compact spatial region. This can be acheived by setting the spatial factors $g_l(\mathbf{r})$ to masks, each of which are responsible for some region in the field of view. The idea is that over a small enough region, the spatial phase is very low spatial frequency, and hence can be well approximated by the HOFFT kernels. 

## Strategy 1: Better HOFFT Spatial Factor Init
The first is to simply initialize with some spatially selective spatial factors. Perhaps a good initial strategy is to spatially mask according to vector quantization on the spatial phase functions $\boldsymbol{\phi}(\mathbf{r})$:
$$
g_l(\mathbf{r}) = \begin{cases} 1 & \text{if }l = \text{argmin}_{l'} ||\boldsymbol{\phi}(\mathbf{r}) - \boldsymbol{\psi}_{l'}||_2\} \\ 
0 & \text{else} \end{cases}
$$
where $\boldsymbol{\psi}_l$ is the $l^\text{th}$ vector quantization bin of $\boldsymbol{\phi}(\mathbf{r})$. 

Another potential initialization strategy is to use spatially contiguous localized regions that sum up to one.

## Strategy 2: HOFFT Per Channel
Instead of using the HOFFT spatial factors $g_l(\mathbf{r})$ as spatially selective functions, we can use the inherent coil sensitivity information as a spatial factor. This gives a new HOFFT decomposition that can be applied for each of the $C$ spatial channel with coil sensivity profiles $s_c(\mathbf{r})$
$$
s_c(\mathbf{r})e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha} (t)} \approx s_c(\mathbf{r}) \sum_{l=1}^L \sum_{w=1}^W g_l(\mathbf{r}) e^{-j 2\pi \mathbf{z}_w \cdot \mathbf{r} / \sigma} h_{l,w,c}(t),
$$

## Strategy 3: HOFFT With Channel Mixing
The final form of the HOFFT channel exploitation is to allow for channel mixing, similar to a GRAPPA operator:
$$
s_c(\mathbf{r})e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha} (t)} \approx  \sum_{c'=1}^{C} \sum_{l=1}^L \sum_{w=1}^W g_l(\mathbf{r}) s_{c'}(\mathbf{r}) e^{-j 2\pi \mathbf{z}_w \cdot \mathbf{r} / \sigma} h_{l,w,c, c'}(t),
$$
It is possible that the above decomposition doesn't even need an $L$ dimension if the coil sensitivies are good enough at spatially localizing smooth phase accruals.
