# Getting Started 
To use our library, you will need to run the following commands:
```
a
```
We reccomend running sample code from the `hofft/experiments` directiory, which gives examples of how to use our library for several MRI reconstruction cases. If you have a unique case that you believe is not covered, please reach out (abrahamd@stanford.edu) and we will be glad to help!

# What Does the HOFFT Library Do?
The conventional MRI model involves a Fourier transform:
$$
y(t) = \int_\mathbf{r} x(\mathbf{r}) e^{-j2\pi \mathbf{r} \cdot \mathbf{k}(t)} d\mathbf{r}.
$$
Here we use $\mathbf{r} \in \mathbb{R}^3, \mathbf{k}(t) \in \mathbb{R}^3$ to denote the spatial coordinates and imaging trajectory respectively. After proper discretization, the NUFFT algorithm can be used to effiicently impliment the above forward model. NUFFTs are designed for such problems where the phase is uniform in one domain (spatial) but non-uniform in the other domain (temporal).

While the above model is interpretable and well understood, it is not always an accurate depiction of MRI data acquisiton. MRI data is almost always collected in the presence of field imperfections, which are usually ignored. Such field imperfections in MRI are commonly modeled by a spatio-temporal phase evolution term into the above expression
$$
y(t) = \int_\mathbf{r} x(\mathbf{r}) e^{-j2\pi \mathbf{r} \cdot \mathbf{k}(t)} \underbrace{e^{-j2\pi \boldsymbol{\phi}(\mathbf{r})\cdot \boldsymbol{\alpha}(t)}}_\text{Field-Imperfection} d\mathbf{r}.
$$
Here $\boldsymbol{\phi}(\mathbf{r}) \in \mathbb{R}^B, \boldsymbol{\alpha}(t) \in \mathbb{R}^B$ represent spatial phase bases and temporal coefficients. We show [common examples of such functions below](#common). Unfortunately, for most field imperfections, $\boldsymbol{\phi}(\mathbf{r})$ are spatially non-linear, which means that a NUFFT/FFT type algorithm cannot be directly applied. Such spatio-temporal phase evolutions will be refered to as high order phase.

**The goal of this library is to evaluate the above integral as fast as possible!**

The current 'gold standard' approach to modeling such integrals with high order phases is to decompose the problematic higher order phase terms into spatial and temporal factors, as discussed in the [prior methods](PRIOR_METHODS.md) section. We introduce the HOFFT algorithm to overcome the limitations imposed by the current 'gold standard' solutions, with algorithmic details described in the [HOFFT](HOFFT.md) section. 

# <a id="common"></a>Common $\boldsymbol{\phi}(\mathbf{r}), \boldsymbol{\alpha}(t)$
| Field-Imperfection  | $\boldsymbol{\phi}(\mathbf{r})$ | $\boldsymbol{\alpha}(t)$ | $B$ |
|---|---|---|---|
| Main Field Inhomogeneity |  $B_0(\mathbf{r})$ | t | 1 |
| Concomitant Fields |  $\bigg( r_x^2 + r_y^2, r_x r_z, r_y r_z, r_z^2\bigg)$ | $\bar{\gamma} \int_0^t \bigg( G_z^2, G_x G_z, G_yG_z, G_x^2 + G_y^2\bigg)(\tau)d\tau$ | 4 |
| Eddy-Current Fields |  $\bigg( 1, r_x, r_y, r_z, r_x r_y, \cdots, r_x^3 - 3r_x r_y^2\bigg)$ | $ \bigg( \alpha_0(t), \alpha_1(t), \alpha_2(t), \alpha_3(t), \alpha_4(t), \cdots, \alpha_{15}(t) \bigg)$ | 16 |

In the above table, we show common field imperfections in MRI. In all of these models, the spatial phase bases usually have non-linear terms. In the case of Eddy-Current fields, the spatially linear terms can be simply combined with the imaging trajectory.


