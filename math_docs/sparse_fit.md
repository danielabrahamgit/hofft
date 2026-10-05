# Sparse Fitting Strategies

## Problem Setup
Unlike the sparse interpolation approach, here we seek to directly fit the HOFFT kernels to a sparse model. Suppose that we've already computed the $L$ HOFFT spatial factors $g_1(\mathbf{r}) \cdots g_L(\mathbf{r})$ and combined them with the fourier phase bases $e^{-j2\pi \frac{\mathbf{z}_1}{\sigma} \cdot \mathbf{r}} \cdots e^{-j2\pi \frac{\mathbf{z}_W}{\sigma} \cdot \mathbf{r}}$ to get the combined matrix $\mathbf{E} \in \mathbb{C}^{N \times L \cdot W }, E_{n, l + Lw} = g_l(\mathbf{r}_n) e^{-j2\pi \frac{\mathbf{z}_w}{\sigma} \cdot \mathbf{r}_n}$. To solve for the $M$ HOFFT kernels $h_{l,w}(t_1) \cdots h_{l,w}(t_m)$ represented by the matrix $\mathbf{H} \in \mathbb{C}^{L \cdot W \times M}, H_{l + Lw, M} = h_{l, w}(t_m)$, we solve the following least squares problem
$$
\min_\mathbf{H} || \boldsymbol{\Phi} - \mathbf{E} \mathbf{H}||_F^2 
$$ 
where the high order phase matrix is $\boldsymbol{\Phi} \in \mathbb{C}^{N \times M}, \Phi_{n,m} = e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}_n) \cdot \boldsymbol{\alpha}(t_m)}$.

For 3D problems, we usually don't have the memory to store the massive $\mathbf{H}$ matrix with $L \cdot W \cdot M$ entries. As such, we'd like to find an **S-sparse compressed represntation of $\mathbf{H}_\text{comp}$** such that
$$
\mathbf{H} = \mathbf{H}_\text{comp} \mathbf{C}
$$
where $\mathbf{H}_\text{comp} \in \mathbb{C}^{L \cdot W \times Q}$ $(Q << M)$ are the compressed basis HOFFT kernels, and $\mathbf{C} \in \mathbb{C}^{Q \times M}$ are coefficients with $S$-sparse columns $(||\mathbf{c}_m||_0 \le S)$. The sparsity quantity $S$ should be as small as possible (under accuracy constraints) to ensure that the HOFFT forward model is computationally efficient while the total number of compressed HOFFT kernels $Q$ should be also be as small as possible (under accuracy constraints) to ensure that we can store $\mathbf{H}_\text{comp}$ with $L \cdot W \cdot Q$ terms in memory.


## Strategy 1: Use a Known $\mathbf{C}$ Matrix
If we know the coefficients $\mathbf{C}$ ahead of time, we can solve for the compressed basis kernels by solving the least squares problem 
$$
\min_{\mathbf{H}_\text{comp}} || \boldsymbol{\Phi} - \mathbf{E} \mathbf{H}_\text{comp}\mathbf{C}||_F^2 
$$
The solution to this requires forming a gram matrix 
$$
\mathbf{M} \leftarrow \mathbf{E}^H \mathbf{E} \otimes \mathbf{C} \mathbf{C}^H \in \mathbb{C}^{L \cdot W \cdot Q \times L \cdot W \cdot Q}
$$
and an 'rhs' term 
$$
\mathbf{f} \leftarrow \text{vec}_{n,m} \{{\mathbf{E} \otimes \mathbf{C}} \}^H  \text{vec}_{n,m}\{{\Phi}\} \in \mathbb{C}^{L \cdot W \cdot Q}
$$
and we solve the normal equations 
$$
\mathbf{M} \mathbf{h}_\text{comp} = \mathbf{f}.
$$
Note that constructing $\mathbf{M}, \mathbf{f}$ require a massive summation over $N, M$. However, we can likely sketch the $N, M$ dimensions since the underlying number of unknowns $(L \cdot W \cdot Q)$ is much smaller than the number of observations $(N \cdot M)$.

In practice, we can get try to get a 'good enough' $\mathbf{C}$ matrix by using techniques discussed in [sparse_interp](./sparse_interp.md), such as barycentric interpolation or RBF interpolation.

## Strategy 2: Use a Known $\mathbf{H}_\text{comp}$ Matrix
If we know the compressed basis kernels $\mathbf{H}_\text{comp}$, we can solve for the $m^\text{th}$ column of $\mathbf{C}$ according to 
$$
\begin{align*}
\min_{\mathbf{c}_m} & ||\boldsymbol{\phi}_m - \mathbf{E} \mathbf{H}_\text{comp}\mathbf{c}_m ||_2^2 \\
& \text{s.t. } ||\mathbf{c}_m||_0 \le S
\end{align*}
$$
which is a non-convex problem needing a convex reformulation, or a potentially sub-optimal algorithm like Orthogonal Matching Pursuit.

In practice, we can get a 'known' $\mathbf{H}_\text{comp}$ Matrix by applying HOFFT on $Q$ representative vectors $\boldsymbol{\beta}_1 \cdots \boldsymbol{\beta}_Q$ via k-means or a furthest sampling strategy, and applying a HOFFT decomposition on the reduced matrix $e^{-j2\pi \boldsymbol{\phi}(\mathbf{r}_n) \boldsymbol{\beta}_q }$.