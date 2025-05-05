# High Order Phase Modeling with Fast Fourier Transforms (HOFFT) 

## Motivation
In magnetic resonance imaging, field imperfections are well modeled as some spatio-temporal phase evolution:
$$
e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha}(t)}
$$

where $\boldsymbol{\phi}(\mathbf{r})$ are a set of $B$ spatial phase maps, and $\boldsymbol{\alpha}(t)$ are a set of $B$ temporal coeffients. Be sure to checkout [FIELD.md](FIELD.md) to see how different field imperfections in MRI fit into the above model.

**The goal of this library** is to evaluate the following **high order transform**
$$
y(t) = \int_\mathbf{r} x(\mathbf{r}) e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha}(t)} d\mathbf{r}
$$
and its **adjoint operation**
$$
x(\mathbf{r}) = \int_t y(t) e^{+j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha}(t)} dt
$$
**as fast as we possibly can!**

## Methods
Evaluating the above operators quickly is done by first decomposing $e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha}(t)}$ into some easy to apply low-dimensional structure. This enables fast forward and adjoint transforms.

#### Low-Rank Approach
One such low-dimensional structure is a spatio-temporal low-rank factorization:
$$
e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha}(t)} \approx \sum_{l=1}^L b_l(r) h_l(t).
$$
Be sure to checkout [LOWRANK.md](LOWRANK.md) for more details on this.

#### HOFFT Kernel Approach
We also propose an extention of this low-rank model to synergize with compact Fourier kernels:
$$
e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha}(t)} \approx \sum_{k=1}^K \sum_{l=1}^L b_l(r) e^{-j 2\pi \mathbf{d}_k \cdot \mathbf{r}} h_{l, k}(t)
$$
where $\mathbf{d}_1, \cdots, \mathbf{d}_K$ describe some compact fourier kernel -- often used in [NUFFT.md](NUFFT.md) models. Be sure to check out [HOFFT.md](HOFFT.md) for more details.

