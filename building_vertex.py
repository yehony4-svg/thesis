from sage.all import *
from utils import *

def move_column_by_last_row(A, f):
    """
    Applies function `f` to each element in the last row of matrix A,
    then moves the column with the smallest f-transformed value to the last column.

    Parameters:
    A (Matrix): a SageMath matrix
    f (function): a function applied to each element of the last row

    Returns:
    Matrix: Modified matrix with reordered columns
    """
    nrows, ncols = A.nrows(), A.ncols()
    last_row = [A[nrows - 1, j] for j in range(ncols)]
    transformed = [f(x) for x in last_row]
    if min(transformed)==max(transformed):
        return A
    min_index = transformed.index(min(transformed))

    # New column order: all but min_index, then min_index
    new_order = [j for j in range(ncols) if j != min_index] + [min_index]
    return A[:, new_order]
    


def get_rep(M,E_to_Ep, Ep_to_E,p_val,p_norm):
    """
    M = [[ a, b, c],
         [ l, m, n],
         [ x, y, z]]

    returns a canonical representative for MK in G, when K = PGL_3(Z_p), G = PGL_3(Q_p)
    param M: 3*3 matrix with 
    
    """
    M = convert_matrix(M,E_to_Ep) # convert matrix to Q_p
    # sort the columns such that the least p valuation will be the last column
    M = move_column_by_last_row(M,p_val)
    x,y,z = M[2]
    k_1 = Matrix([[1,0,0],[0,1,0],[-x/z,-y/z,1]])
    M*=k_1
    
    # do the same thing for M[:2,:2]
    M[:2,:2] = move_column_by_last_row(M[:2,:2],p_val)
    l,m = M[1][:2]
    k_2 = Matrix([[1,0,0],[-l/m,1,0],[0,0,1]])
    M*=k_2

    # now we have triangular matrix, let's make the diagonal to be p powers
    a,m,z = [M[i,i] for i in range(3)]
    u,v,w = [t*p_norm(t) for t in [a,m,z]] # the invertable part in Z_p
    k_3 = diag(u**-1,v**-1,w**-1)
    M*=k_3

    n = M[1,2]
    d = (-n + (n%M[1,1]) )/M[1,1]
    k_4 = Matrix(3,[1,0,0,
                    0,1,d,
                    0,0,1])
    M*=k_4
    M[0, 1] = Ep_to_E(M[0, 1] % M[0, 0])
    M[0, 2] = Ep_to_E(M[0, 2] % M[0, 0])

    # scale to primitive matrix
    max_p = max(
        p_norm(M[i, j]) for i in range(3) for j in range(3) if M[i, j] != 0)
    M *= max_p

    #convert to rationals
    M = convert_matrix(M,Ep_to_E)

    return M

