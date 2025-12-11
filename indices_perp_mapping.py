from collections import defaultdict
import cupy

def dot_mat_pairs(X, Y, xp=None):
    """
    Vectorized dot-products over Z[√-2] between two batches of 3D vectors
    stored as pairs [a,b] meaning a + b*sqrt(-2).

    Parameters
    ----------
    X : array-like, shape (n, d, 2)
        Batch 1 of vectors. Each coord is [a,b].
    Y : array-like, shape (m, d, 2)
        Batch 2 of vectors. Each coord is [a,b].
    xp : module, optional
        Backend: numpy or cupy. Default: infer from inputs (falls back to numpy).

    Returns
    -------
    M : array, shape (n, m, 2)
        M[..., 0] = real parts, M[..., 1] = imag parts of the dot products.
        Where the dot is Σ_k (X_k * Y_k) with (a+br)*(c+dr) = (ac - 2bd) + (ad+bc)r.
    """
    # pick backend
    if xp is None:
        try:
            import cupy as cp
        except Exception:
            cp = None
        # if either input is cupy, use cupy
        if (hasattr(X, "__cuda_array_interface__") or hasattr(Y, "__cuda_array_interface__")) and cp is not None:
            xp = cp
        else:
            import numpy as np
            xp = np

    X = xp.asarray(X)
    Y = xp.asarray(Y)

    if X.ndim != 3 or Y.ndim != 3 or X.shape[2] != 2 or Y.shape[2] != 2:
        raise ValueError("X and Y must have shape (n, d, 2) and (m, d, 2).")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Feature dimension d must match: X.shape[1] == Y.shape[1].")

    # Split into a,b parts and conjugate Y
    A1, B1 = X[..., 0], X[..., 1]   # shapes (n, d)
    A2, B2 = Y[..., 0], -Y[..., 1]   # shapes (m, d)

    # Cross-gram real/imag parts using BLAS/CuBLAS matmuls
    # Re = Σ_k (a1_k*a2_k - 2*b1_k*b2_k)
    # Im = Σ_k (a1_k*b2_k + b1_k*a2_k)
    Re = A1 @ A2.T - xp.int64(2) * (B1 @ B2.T)
    Im = A1 @ B2.T + B1 @ A2.T

    return xp.stack([Re, Im], axis=-1)

def is_perp_mat(A,B):
    M = dot_mat_pairs(A,B)
    return (M[...,0] == np.int64(0)) & (M[...,1] == np.int64(0))


def zero_dict(X, Y,x_base=0,y_base=0, xp=None, d = defaultdict(set)):
    """
    Return dict: i -> set of j such that dot(X[i], Y[j]) == 0.
    """
    mask = is_perp_mat(X, Y)  # (n,m) bool
    # On GPU we must bring indices to host if we want a Python dict
    if xp is None:
        import numpy as np; xp = np
    rows, cols = xp.nonzero(mask)

    # convert to CPU ints if xp is cupy
    if xp.__name__ == "cupy":
        rows = rows.get()
        cols = cols.get()

    for i, j in zip(rows, cols):
        key = x_base + int(i)
        if key not in d.keys():
            d[key] = set()
        d[key].add(y_base + int(j))
    return dict(d)


def create_neighbors_dict_indices(n_vectors, X_batch_size, Y_batch_size):
    num_v = len(n_vectors)
    Xn_batches = num_v//X_batch_size+1
    Yn_batches = num_v//Y_batch_size+1
    d = defaultdict(set)
    print(f"{Xn_batches}x{Yn_batches}=={Xn_batches*Yn_batches} iterations")
    for n,m in itertools.product(range(Xn_batches),range(Yn_batches)):
        X = cp.array(n_vectors_np[X_batch_size*n:X_batch_size*(n+1)])
        Y = cp.array(n_vectors_np[Y_batch_size*m:Y_batch_size*(m+1)])
        d = zero_dict(X,Y,x_base=X_batch_size*n,y_base=Y_batch_size*m,d=d)
        print(n,m,len(d))
    return d