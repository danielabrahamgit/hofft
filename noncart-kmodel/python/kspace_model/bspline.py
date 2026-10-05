import numpy as np
from scipy.special import factorial, comb

def bspline(p):
    """
    Returns functions that evaluate a B-spline of degree p and its inverse Fourier transform.
    
    Parameters:
    -----------
    p : int
        Degree of the B-spline.
        
    Returns:
    --------
    evaluate_basis : function
        A function that takes an array x and returns the evaluated  spline values at x.
    evaluate_basis_img : function
        A function that that takes an array x and evaluates the inverse Fourier transform of the basis functions at x.
    """
    def evaluate_basis(x):
        # Prepare polynomial coefficient matrix A
        l = np.arange(p + 1).reshape(-1, 1)
        q = np.arange(p + 1).reshape(1, -1)

        term1 = comb(p + 1, l)
        term2 = (-1.0) ** l
        term3 = (-l + (p + 1) / 2.0) ** (p - q)
        tmp = term1 * term2 * term3
        A = comb(p, q) * np.cumsum(tmp, axis=0) / factorial(p)

        nzidx = np.flatnonzero( (x < (p + 1) / 2.0) & (x >= -(p + 1) / 2.0) )
        xsub = x[nzidx]
        segn = np.floor(xsub + (p + 1) / 2.0).astype(int)
        coeff = A[segn, :]

        # Broadcast xsub^q: (N, 1) ^ (1, P) -> (N, P)
        xsub_pow_q = xsub[:, None] ** q
        y0 = np.sum(coeff * xsub_pow_q, axis=1)
        
        y = np.zeros(x.shape)
        y[nzidx] = y0

        return y

    def evaluate_basis_img(x):
        return np.power(np.sinc(x), p+1)
        
    return evaluate_basis, evaluate_basis_img

