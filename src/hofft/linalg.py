import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Optional, Tuple
from einops import einsum

def svd_product(B: torch.Tensor,
                C: torch.Tensor,
                rank: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Thin SVD of A = B @ C without forming A, using QR of the factors.

        B = Qb Rb
        C^H = Qc Rc
        Rb Rc^H = Um Σ Vm^H
        A = (Qb Um) Σ (Qc Vm)^H

    The inner product is K×K, so this is cheap when K << min(M, N).
    Real and complex dtypes are supported (Hermitian transposes throughout).

    Args
    ----
    B : torch.Tensor
        left factor with shape (..., M, K)
    C : torch.Tensor
        right factor with shape (..., K, N)
    rank : int, optional
        keep the leading ``rank`` singular components. Default keeps all.

    Returns
    -------
    U : torch.Tensor
        left singular vectors with shape (..., M, R)
    S : torch.Tensor
        singular values with shape (..., R)
    Vh : torch.Tensor
        conjugate-transpose right singular vectors with shape (..., R, N)
    """
    if B.shape[-1] != C.shape[-2]:
        raise ValueError(f'Inner dims of B @ C must match, got {tuple(B.shape)} @ {tuple(C.shape)}')
    Qb, Rb = torch.linalg.qr(B, mode='reduced')
    Qc, Rc = torch.linalg.qr(C.mH, mode='reduced')
    Um, S, Vmh = torch.linalg.svd(Rb @ Rc.mH, full_matrices=False)
    U = Qb @ Um
    Vh = Vmh @ Qc.mH
    if rank is not None:
        U = U[..., :rank]
        S = S[..., :rank]
        Vh = Vh[..., :rank, :]
    return U, S, Vh

def svd_operator(A: callable,
                 AHA: callable,
                 inp_example: torch.Tensor,
                 rank: int,
                 num_iter: Optional[int] = 15,
                 tol: Optional[float] = 1e-2,
                 lobpcg: Optional[bool] = True,
                 verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses power method or lobpcg to to eigen-step in SVD on a matrix operator.

    Args
    ----
    A : callable
        linear operator mapping from (N, *inp_shape) to (N, *out_shape)
        where N is a batch dimension
    AHA : callable
        linear operator square shape mapping from (N, *inp_shape) to (N, *inp_shape)
    inp_example : torch.Tensor
        example input tensor with shape (*inp_shape) (also contains device and dtype info)
    rank : int
        number of ordered svd terms
    num_iter : int
        number of iterations to run power method
        OR
        number of iterations to run lobpcg if lobpcg is True
    tol : float
        LOBPCG relative residual tolerance (see :func:`lobpcg_operator`)
    lobpcg : bool
        toggles use of lobpcg instead of power method
    verbose : bool
        toggles progress bar
    
    Returns
    -------
    U : torch.Tensor
        left vectors with shape (*out_shape, rank)
    S : torch.Tensor
        singular values with shape (rank,)
    Vh : torch.Tensor
        right vectors with shape (rank, *inp_shape)
    """
    
    # Compute right singular vectors via eigen decomposition of AHA
    V, S = eigen_decomp_operator(AHA, inp_example, num_eigen=rank, num_iter=num_iter, tol=tol, lobpcg=lobpcg, verbose=verbose)
    
    # Sort by singular values
    idx = torch.argsort(S.abs(), descending=True)
    V = V[idx]
    S = S[idx] ** 0.5
    U = None
    
    # Clear GPU mem
    import gc
    gc.collect()
    with torch.cuda.device(inp_example.device):
        torch.cuda.empty_cache()
    
    # Compute left singular vectors
    bs = rank
    for l1 in tqdm(range(0, rank, bs), 'Calculating Left Singular Vectors', disable=not verbose):
        l2 = min(l1 + bs, rank)
        u = A(V[l1:l2])
        u = u.moveaxis(0, -1) / S[l1:l2]
        if U is None:
            U = u
        else:
            U = torch.cat((U, u), dim=-1)
    
    return U, S, V.conj()

def power_method_operator(A: callable,
                          x0: torch.Tensor,
                          num_iter: Optional[int] = 15,
                          verbose: Optional[bool] = True) -> Tuple[torch.Tensor, float]:
    """
    Uses power method to find largest eigenvalue and corresponding eigenvector

    Parameters:
    -----------
    A : callable
        linear operator
    vec_init : torch.Tensor
        initial guess of eigenvector with shape (*vec_shape)
    num_iter : int
        number of iterations to run power method
    verbose : bool
        toggles progress bar
    
    Returns:
    --------
    eigen_vec : torch.Tensor
        eigenvector with shape (*vec_shape)
    eigen_val : float
        eigenvalue
    """
    
    for _ in tqdm(range(num_iter), 'Max Eigenvalue', disable=not verbose):
        
        z = A(x0)
        ll = torch.norm(z)
        x0 = z / ll
    
    if verbose:
        print(f'Max Eigenvalue = {ll}')
    
    return x0, ll.item()

def eigen_decomp_operator(A: callable,
                          x0: torch.Tensor,
                          num_eigen: int,
                          num_iter: Optional[int] = 15,
                          tol: Optional[float] = 1e-2,
                          lobpcg: Optional[bool] = True,
                          largest: Optional[bool] = True,
                          verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses power method to find largest num_eigen eigenvalues and corresponding eigenvectors

    Args
    ----
    A : callable
        linear operator mapping from (N, *vec_shape) to (N, *vec_shape)
        where N is a batch dimension
    x0 : torch.Tensor
        initial guess of eigenvector with shape (*vec_shape)
    num_eigen : int
        number of eigenvalues to find
    num_iter : int
        number of iterations to run power method
        OR
        number of iterations to run lobpcg if lobpcg is True
    tol : float
        LOBPCG relative residual tolerance (see :func:`lobpcg_operator`)
    lobpcg : bool
        toggles use of lobpcg instead of power method
    largest : bool
        If True, computes the largest eigenpairs; otherwise, the smallest.
    verbose : bool
        toggles progress bar
    
    Returns
    -------
    eigen_vecs : torch.Tensor
        eigenvectors with shape (num_eigen, *vec_shape)
    eigen_vals : torch.Tensor
        eigenvalues with shape (num_eigen,)
    """
    if lobpcg:
        
        # linops
        def matvec(x_flt):
            # x_flt (n, k) 
            x_vec = x_flt.T.reshape((x_flt.shape[1], *x0.shape))
            out_vec = A(x_vec) # k, *vec_shape
            out_flt = out_vec.reshape((x_flt.shape[1], -1)).T # (n, k)
            return out_flt
        
        # Run lobpcg
        n = x0.numel()
        k = num_eigen
        X = torch.randn((n, k), device=x0.device, dtype=x0.dtype)
        eigen_vals, eigen_vecs = lobpcg_operator(matvec, X, maxiter=num_iter, tol=tol, largest=largest, verbose=verbose)
        eigen_vecs = eigen_vecs.T.reshape((k, *x0.shape))
    else:
        eigen_vecs = torch.zeros(num_eigen, *x0.shape, device=x0.device, dtype=x0.dtype)
        eigen_vals = torch.zeros(num_eigen, device=x0.device, dtype=x0.dtype)
        A_clone = A
        
        def A_resid_operator(x):
            y = A_clone(x[None,])[0]
            VH_x = einsum(eigen_vecs.conj(), x, 'n ..., ... -> n') * eigen_vals
            V_diag_VH_x = einsum(eigen_vecs, VH_x, 'n ..., n -> ...')
            y = y - V_diag_VH_x
            return y
        
        for r in tqdm(range(num_eigen), 'Eigen Iterations', disable=not verbose):
            init = torch.randn_like(x0)
            init /= torch.linalg.norm(init)
            
            vec, val = power_method_operator(A_resid_operator, init, 
                                            num_iter=num_iter, 
                                            verbose=False)
            eigen_vecs[r] = vec
            eigen_vals[r] = val

    return eigen_vecs, eigen_vals

def _hermitize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.mH)


def _orth_drop(X: torch.Tensor, rank_tol: float = 1e-6) -> Optional[torch.Tensor]:
    """Orthonormalize columns, dropping near-zero / dependent ones.

    QR of a numerically rank-deficient block (tiny residuals or a converged
    search direction) injects random orthonormal columns. Those junk
    directions make the next Rayleigh–Ritz basis jump by ~90°, which is
    what made :func:`_subspace_change` stick at 1 after the eigenpairs had
    already settled.
    """
    if X is None or X.numel() == 0 or X.shape[1] == 0:
        return None
    col_nrm = torch.linalg.norm(X, dim=0)
    eps = torch.tensor(rank_tol, device=X.device, dtype=col_nrm.dtype)
    keep = col_nrm > rank_tol * col_nrm.max().clamp(min=eps)
    if not torch.any(keep):
        return None
    Q, R = torch.linalg.qr(X[:, keep], mode='reduced')
    diag = torch.abs(torch.diagonal(R))
    keep_r = diag > rank_tol * diag.max().clamp(min=eps)
    if not torch.any(keep_r):
        return None
    return Q[:, keep_r]


def _relative_residual(R: torch.Tensor, evals: torch.Tensor) -> torch.Tensor:
    """max_i ||A x_i - λ_i x_i|| / |λ_i|, invariant to column phase/sign."""
    col_res = torch.linalg.norm(R, dim=0)
    lam = evals.real.abs() if evals.is_complex() else evals.abs()
    scale = lam.clamp(min=torch.finfo(col_res.dtype).eps)
    return (col_res / scale).max()


def _subspace_change(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Max sine of principal angles between two column-spaces.

    Invariant to column phase/sign and within-subspace rotations, so it does
    *not* detect the eigh gauge flip. It is still a brittle stopping rule:
    a single swapped or junk direction makes the max sine equal 1 even when
    the leading eigenpairs have converged. Prefer residual-based stopping.
    """
    X, _ = torch.linalg.qr(X, mode='reduced')
    Y, _ = torch.linalg.qr(Y, mode='reduced')
    s = torch.linalg.svdvals(X.mH @ Y).clamp(max=1.0)
    return (1.0 - s.square()).clamp(min=0.0).sqrt().max()

def lobpcg_operator(A: callable, 
                    X: torch.Tensor, 
                    precond: Optional[callable] = None, 
                    maxiter: Optional[int] = 100, 
                    tol: Optional[float] = 1e-2, 
                    largest: Optional[bool] = True,
                    verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Matrix-free LOBPCG for a symmetric operator using only matvec calls.

    Parameters:
    -----------
    A : callable
        (n --> n) function that takes a tensor with shape (n, k) and return A(X) with shape (n, k).
    X : torch.Tensor
        Initial guess for eigenvectors, shape (n, k). (rows need not be orthonormal.)
    precond : callable or None
        Function that applies a preconditioner to a tensor.
        If None, no preconditioning is applied.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the relative residual
        ``max_i ||A x_i - λ_i x_i|| / |λ_i|``. This is invariant to the
        arbitrary column sign/phase that ``eigh`` assigns each iteration,
        unlike a vector-difference or (in practice) max-principal-angle check.
    largest : bool
        If True, computes the largest eigenpairs; otherwise, the smallest.
    verbose : bool
        If True, shows progress bar.

    Returns:
    --------
    evals : torch.Tensor
        Approximated eigenvalues with shape (k,)
    evecs : torch.Tensor
        Approximated eigenvectors with shape (n, k)
    """
    
    # Ensure X has orthonormal columns
    X, _ = torch.linalg.qr(X)
    n, k = X.shape

    # Compute initial A*X and perform a Rayleigh–Ritz on the subspace spanned by X.
    AX = A(X)
    T = _hermitize(X.mH @ AX)
    evals, eigvecs = torch.linalg.eigh(T)
    
    # torch.linalg.eigh returns eigenvalues in ascending order.
    if largest:
        idx = torch.argsort(evals, descending=True)
    else:
        idx = torch.argsort(evals)
    evals = evals[idx]
    eigvecs = eigvecs[:, idx]
    X = X @ eigvecs  # new approximations for eigenvectors
    AX = AX @ eigvecs

    # Optionally store a previous search direction (for subspace expansion)
    P = None

    pbar = tqdm(range(maxiter), 'LOBPCG Iteration', disable=not verbose)
    for it in pbar:
        R = AX - X * evals.unsqueeze(0)
        rel_res = _relative_residual(R, evals)
        if verbose:
            pbar.set_postfix(resid=f'{rel_res.item():.2e}')
        if rel_res < tol:
            break

        # Apply preconditioning if available.
        if precond is not None:
            W = precond(R)
        else:
            W = R

        # Project W off the current Ritz space and drop converged (tiny) columns
        # *before* QR. Tiny columns would otherwise become random directions.
        W = W - X @ (X.mH @ W)
        W = _orth_drop(W)

        if P is not None:
            bases = X if W is None else torch.cat([X, W], dim=1)
            P = P - bases @ (bases.mH @ P)
            P = _orth_drop(P)

        blocks = [X]
        if W is not None:
            blocks.append(W)
        if P is not None:
            blocks.append(P)
        S = torch.cat(blocks, dim=1)
        S, _ = torch.linalg.qr(S, mode='reduced')

        # Compute A*S using the matvec operator.
        AS = A(S)
        T_sub = _hermitize(S.mH @ AS)
        
        # Solve the small eigenproblem.
        evals_sub, eigvecs_sub = torch.linalg.eigh(T_sub)
        if largest:
            idx = torch.argsort(evals_sub, descending=True)
        else:
            idx = torch.argsort(evals_sub)
        evals_sub = evals_sub[idx]
        eigvecs_sub = eigvecs_sub[:, idx]

        # Update our approximations: choose the first k eigenpairs.
        X_new = S @ eigvecs_sub[:, :k]
        AX_new = AS @ eigvecs_sub[:, :k]
        new_evals = evals_sub[:k]

        # Search direction: motion of the Ritz space. Drop it when it is
        # numerically zero (converged) instead of QRing a near-zero matrix.
        P = _orth_drop(X_new - X @ (X.mH @ X_new))

        X = X_new
        AX = AX_new
        evals = new_evals

    return evals, X
 
def lin_solve(AHA: torch.Tensor, 
              AHb: torch.Tensor, 
              lamda: Optional[float] = 0.0, 
              solver: Optional[int] = 'solve') -> torch.Tensor:
    """
    Solves (AHA + lamda I) @ x = AHb for x

    Args
    ----
    AHA : torch.Tensor
        square matrix with shape (..., n, n)
    AHb : torch.Tensor
        matrix with shape (..., n, m)
    lamda : float
        optional L2 regularization 
    solver : str
        'pinv' - pseudo inverse 
        'solve' - torch.linalg.solve
        'lstsq' - least squares
        'inv' - regular inverse
    
    Returns
    -------
    x : torch.Tensor
        solution with shape (..., n, m)
    """
    solver = solver.lower()
    if lamda > 0:
        I = torch.eye(AHA.shape[-1], dtype=AHA.dtype, device=AHA.device)
        tup = (AHA.ndim - 2) * (None,) + (slice(None),) * 2
        AHA += lamda * I[tup]
    if solver == 'lstsq':
        x = torch.linalg.lstsq(AHA, AHb).solution
    elif solver == 'solve':
        x = torch.linalg.solve(AHA, AHb)
    elif solver == 'pinv':
        x = torch.linalg.pinv(AHA, hermitian=True) @ AHb
    elif solver == 'pinv_noherm':
        x = torch.linalg.pinv(AHA) @ AHb
    elif solver == 'inv':
        x = torch.linalg.inv(AHA) @ AHb
    else:
        raise NotImplementedError
    return x

def conjugate_gradient(AHA: nn.Module, 
                       AHb: torch.Tensor, 
                       P: Optional[nn.Module] = None,
                       num_iters: Optional[int] = 10, 
                       lamda_l2: Optional[float] = 0.0,
                       tolerance: Optional[float] = 1e-8,
                       return_resids: Optional[bool] = False,
                       weights: Optional[torch.Tensor] = None,
                       verbose=True) -> torch.Tensor:
    """
    Conjugate gradient for complex numbers. The output is also complex. 
    Solve for argmin ||Ax - b||^2
    
    Args
    ----
    AHA : nn.Module 
        Linear operator representing the gram/normal operator of A
    AHb : torch.tensor
        The A hermitian transpose times b
    P : nn.Module
        Preconditioner 
    num_iters : int 
        Max number of iterations.
    lamda_l2 : float
        Replaces AHA with AHA + lamda_l2 * I
    tolerance : float 
        Used as stopping criteria
    return_resids : bool
        toggles return of residuals
    verbose : bool
        toggles print statements
    
    Returns
    -------
    x : torch.tensor <complex>
        least squares estimate of x, same shape as x0 if provided    
    """

    # Default preconditioner is identity matrix
    if P is None:
        P = lambda x : x
    
    # Tikonov regularization
    if weights:
        AHA_wrapper = lambda x : AHA(x) + lamda_l2 * weights * x
    else:
        AHA_wrapper = lambda x : AHA(x) + lamda_l2 * x

    # Start at AHb
    x0 = AHb.clone()
    if num_iters == 0:
        return x0

    # Define iterative vars
    r = AHb - AHA_wrapper(x0)
    z = P(r)
    p = z.clone()

    if return_resids:
        resids = []
        xs = [x0.clone()]

    # Main loop
    for i in tqdm(range(num_iters), 'CG Iterations', disable=not verbose):
        
        # Apply model
        Ap = AHA_wrapper(p)
        pAp = torch.real(torch.sum(p.conj() * Ap)).item()

        # Update x
        # assert pAp > 0, 'A is not Semi-Definite'
        rz = torch.real(torch.sum(r.conj() * z))
        alpha = rz / pAp
        x0 = x0 + alpha * p

        # Update r
        r = r - alpha * Ap
        rnrm = torch.norm(r)
        if return_resids:
            resids.append(rnrm.item())
            xs.append(x0.clone())
        if rnrm < tolerance:
            break

        # Update z
        z = P(r)

        # Update p
        beta = torch.real(torch.sum(r.conj() * z)) / rz
        p = z + beta * p
    
    if return_resids:
        return resids, xs
    else:
        return x0