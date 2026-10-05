# Warm Starting HOFFT

## Motivation
The HOFFT decomposition step can be computationally expensive, and hence we'd like to offload as much of that complexity offline as possible. 

## Strategy 1: Compute Trajectory Imperfections Offline
The high order phase functions $\boldsymbol{\phi}(\mathbf{r}) \in \mathbb{R}^B, \boldsymbol{\alpha}(t) \in \mathbb{R}^B$ contain both subject induced field imperfections, which we will use the first $b=0$ dimension for
$$
\phi_0(\mathbf{r}) = B_0(\mathbf{r}), \alpha_0(t) = t,
$$
as well as trajectory induced imperfections that are known ahead of time 
$$
\boldsymbol{\phi}_\text{trj}(\mathbf{r}) = \begin{bmatrix} \phi_1(\mathbf{r}) \\ \vdots \\ \phi_{B-1}(\mathbf{r})\end{bmatrix}, \boldsymbol{\alpha}_\text{trj}(t) = \begin{bmatrix} \alpha_1(t) \\ \vdots \\ \alpha_{B-1}(t)\end{bmatrix}.
$$
The trajectory terms include things like eddy currents, concomitant fields, and the non-Cartesian grid deviations.

The main idea will be to perform an offline HOFFT decomposition on the trajectory terms only
$$
e^{-j2\pi \boldsymbol{\phi}_\text{trj}(\mathbf{r}) \cdot \boldsymbol{\alpha}_\text{trj} (t)} \approx \sum_{\tilde{l}=1}^{\tilde{L}} \sum_{w=1}^W \tilde{h}_{\tilde{l},w}(t) e^{-j 2\pi \mathbf{z}_w \cdot \mathbf{r} / \sigma } \tilde{g}_{\tilde{l}}(\mathbf{r}),
$$
and then perform a quick low-rank or time segmented decomposition of the $B_0$ term
$$
e^{-j2\pi B_0(\mathbf{r}) t} \approx \sum_{q=1}^Q c_{q}(t) b_q(\mathbf{r}).
$$

Then we can apply an SVD to compress the product of the HOFFT and $B_0$ spatial factors  
$$
\tilde{g}_{\tilde{l}}(\mathbf{r})b_q(\mathbf{r}) \approx \sum_{l=1}^L g_l(\mathbf{r}) v_{l,q,\tilde{l}}.
$$

After this compression, the new HOFFT spatial factors become $g_l(\mathbf{r})$, and the new HOFFT kernels become
$$
h_{l,w}(t) = \sum_{\tilde{l}=1}^{\tilde{L}} \sum_{q=1}^Q \tilde{h}_{\tilde{l},w}(t) c_q(t) v_{l, q, \tilde{l}}.
$$

