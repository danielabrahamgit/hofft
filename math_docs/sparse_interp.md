# Sparse Interpolation Strategies

## Problem Setup
We are given spatial phase functions $\boldsymbol{\phi}(\mathbf{r}) \in \mathbb{R}^B$ and temporal phase features $\boldsymbol{\alpha}(t) \in \mathbb{R}^B$, and would like to make the following approximation using sparse interpolators:
$$
e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\alpha} (t)} \approx a(\mathbf{r}) \sum_{q=1}^Q e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r}) \cdot \boldsymbol{\beta}_q} c_q(t),
$$
where $a(\mathbf{r})$ is an optional spatial weighting function to help reduce approximation error, $\boldsymbol{\beta}_q$ are chosen features bases, and $c_q(t)$ are optimized sparse interpolation coefficients. We want interpolation strategies that force K-sparse coefficients $\text{nnz}(c_1(t), \cdots, c_Q(t)) = K$. 

Assuming the approximation is sufficiently accurate, we can get away with applying HOFFT to the "compressed" set of high order phase bases:
$$
\tilde{\mathbf{H}}, \mathbf{G} = \text{HOFFT-Decomp}\{ \tilde{\boldsymbol{\Phi}}\}
$$
where $\tilde{H}$ are the compressed HOFFT kernels with shape $(L \cdot W, Q)$, $\mathbf{G}$ are the HOFFT $L$ spatial factors with shape $(N, L)$, and $\tilde{\Phi}_{n, q} = a(\mathbf{r}_n) e^{-j 2\pi \boldsymbol{\phi}(\mathbf{r_n}) \cdot \boldsymbol{\beta}_q}$ is the $(N, Q)$ matrix that we factorize with HOFFT. The HOFFT kernel weights can then be recovered by using the sparse interpolation coefficients
$$
\mathbf{H} \approx \tilde{\mathbf{H}} \mathbf{C}
$$
where $C_{q, m} = c_q(t_m)$ with shape $(Q, M)$ are the interpolation weights with K-sparse column vectors.

The sparsity factor $K$ will control how efficiently we can query HOFFT kernels in the forward model, while the total number of bases $Q$ determines how expensive the HOFFT decomposition step will be. Ideally we'd like $K$, $Q$, and the approximation error to be as low as possible. 

## Strategy 1: Unstructured Barycentric Interpolation

### Basis Selection 
The basis features $\boldsymbol{\beta}_1, \cdots, \boldsymbol{\beta}_Q$ are chosen according to a furthest sampling strategy or K-means clustering approach on all of the $\boldsymbol{\alpha}(t)$ features. 

### Sparse Indices
For each sample $m$, we find the $K$ nearest basis features, yielding a matrix $\mathbf{Z}$ with shape $(M, K)$ containing the $K$ nearest indices:
$$
(Z_{m,1}, \cdots, Z_{m, K}) = \text{argmin-topK}_{q \in \{1, ..., Q\}} ||\boldsymbol{\alpha}(t_m) - \boldsymbol{\beta}_q||_2
$$

### Interpolation Coefficients
*Barycentric Weights*   
The non-zero coefficients $\mathbf{c}_m \in \mathbb{C}^K$ of for the $m^\text{th}$ sample are estimated according to 
$$
\begin{align}
\begin{split}
\min_{\mathbf{c}_m} ||\boldsymbol{\alpha}(t_m)& - \begin{bmatrix} \beta_{Z_{m,1}} & \cdots & \beta_{Z_{m,K}} \end{bmatrix} \mathbf{c}_m||_2 \\
\text{subject to  } ||\mathbf{c}_m||_1 & = 1, \mathbf{c}_m \ge 0
\end{split}
\end{align}
$$

## Strategy 2: Unstructured RBF Interpolation

### Basis Selection
The basis features $\boldsymbol{\beta}_1, \cdots, \boldsymbol{\beta}_Q$ are chosen according to a furthest sampling strategy or K-means clustering approach on all of the $\boldsymbol{\alpha}(t)$ features. 

### Sparse Indices
For each sample $m$, we find the $K$ nearest basis features, yielding a matrix $\mathbf{Z}$ with shape $(M, K)$ containing the $K$ nearest indices:
$$
(Z_{m,1}, \cdots, Z_{m, K}) = \text{argmin-topK}_{q \in \{1, ..., Q\}} ||\boldsymbol{\alpha}(t_m) - \boldsymbol{\beta}_q||_2
$$

### Interpolation Coefficients
The RBF model calls for estimating complex weights $w_1, \cdots, w_Q$ such that
$$
\mathbf{c}_m = \begin{bmatrix} w_{Z_m,1} h_{Z_m,1}(\boldsymbol{\alpha}(t_m)) \\ \vdots \\ w_{Z_m,K} h_{Z_m,K}(\boldsymbol{\alpha}(t_m)) \end{bmatrix},
$$
where 
$$
h_q(\boldsymbol{\alpha}) = \exp\{ -\frac{1}{2} (\boldsymbol{\alpha} - \boldsymbol{\beta}_q)^T \boldsymbol{\Sigma_q}^{-1}(\boldsymbol{\alpha} - \boldsymbol{\beta}_q)\}.
$$
The weights $w_1, \cdots, w_Q$ can be estimated according to 
$$
\min_\mathbf{w} \sum_m ||\mathbf{A}_m \mathbf{w}  - \mathbf{b}_m||_2^2.
$$
where $\mathbf{A}_m = \mathbf{B} \mathbf{D}_m \mathbf{I}_m$, $\mathbf{\tilde{A}} = \begin{bmatrix} e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}_1) \cdot \boldsymbol{\beta}_1} & \cdots & e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}_1) \cdot \boldsymbol{\beta}_Q} \\ & \vdots & \\ e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}_N) \cdot \boldsymbol{\beta}_1} & \cdots & e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}_N) \cdot \boldsymbol{\beta}_Q} \end{bmatrix}$, $\mathbf{D}_m = \text{diag}\big( h_1(\boldsymbol{\alpha}(t_m)) , \cdots h_Q(\boldsymbol{\alpha}(t_m))\big)$, and $\mathbf{I}_m$ is a K-row subset of the $Q\times Q$ identity matrix selected according to $\mathbf{Z}_{m}$.

Initializing $\boldsymbol{\Sigma}_q$ is quite important. This can be done by estimating a covariance matrix via all the $\boldsymbol{\alpha}$ that are closest to that RBF function. 

An optional step that helps with stability is to renormalize the interpolation functions $h_q(\boldsymbol{\alpha}(t_m))$ for each sample $m$ such that the sum of the $K$ non-zero RBF functions must sum to one.

## Strategy 3: Structured Grid Interpolation

### Basis Selection 
We start by determing the grid spacing $\Delta \beta$ and interpolation width $W_\beta \in \mathbb{Z^+}$. The sparse interpolation stencil is described by the vectors $\boldsymbol{\beta}_1, \cdots, \boldsymbol{\beta}_K$ form a $W_\beta \times \cdots B\text{-times} \cdots \times W_\beta$ compact grid ($K = W_\beta^B$) with spacing $\Delta \beta$.

$Q$ is deterimed by the number of unique grid points needed to interpolate the entire trajectory. The unique grid points can be computed according to 
$$
\text{\# grid points } (Q) = \text{unique}\bigg( \Delta \beta \cdot \text{round}(\frac{\boldsymbol{\alpha}(t_m)+ \boldsymbol{\beta}_k}{\Delta \beta}) \bigg).
$$

### Sparse Indices
For each sample $m$, we find the $K$ nearest basis features, yielding a matrix $\mathbf{Z}$ with shape $(M, K)$ containing the $K$ nearest indices:
$$
(Z_{m,1}, \cdots, Z_{m, K}) = \text{argmin-topK}_{q \in \{1, ..., Q\}} ||\boldsymbol{\alpha}(t_m) - \boldsymbol{\beta}_q||_2
$$

### Interpolation Coefficients
We'll first renomalize things so that each $\phi_b(\mathbf{r})$ lies in the range $[-\frac{1}{2}, +\frac{1}{2}]$ which helps normalize the effect of $\alpha_b$. We further will asume that the feature trajectory can be decomposed according to 
$$
\boldsymbol{\alpha}(t_m) = \boldsymbol{\beta}_{Z_{m,1}} + \Delta \boldsymbol{\alpha}(t_m),
$$
for some value of $q \in \{1, \cdots, Q\}$. The trajectory deviations $\Delta \alpha_b(t_m)$ must lie in the range $[-\frac{\Delta \beta}{2}, +\frac{\Delta \beta}{2}]$ for each value of $b \in \{1, \cdots, B\}$.

We decouple the interpolation into a sequence of 1D problems. For each $b \in \{1, \cdots, B\}$, we solve for
$$
\min_{a_b(\mathbf{r}), \mathbf{w}_b(\Delta \alpha)} \sum_n ||e^{-j 2\pi \phi_b(\mathbf{r_n}) \Delta \alpha} - \begin{bmatrix} a(\mathbf{r}_n) e^{-j2\pi \phi_b(\mathbf{r}_n) (W_\beta//2) \Delta \beta} & \cdots & a(\mathbf{r}_n) e^{+j2\pi \phi_b(\mathbf{r}_n) (W_\beta // 2) \Delta \beta} \end{bmatrix} \mathbf{w}_b(\Delta \alpha)||_2^2.
$$
The final spatial weighting function is then $a(\mathbf{r}) = \Pi_{b=1}^B a_b(\mathbf{r})$ and the sparse coefficients are 
$$
\mathbf{c}_m = \text{flatten}\bigg( \mathbf{w}_1(\Delta \alpha_1(t_m)) \otimes \cdots \otimes \mathbf{w}_B(\Delta \alpha_B(t_m)) \bigg).
$$

