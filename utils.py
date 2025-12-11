from sage.all import *

Qi_to_Qpi = lambda x: Q_p(x.real()) + Q_p(x.imag()) * i

def Qpi_to_Qi(z):
    """
    convert Q_p(i) to Q(i)
    """
    a = z.polynomial().constant_coefficient()
    b = z.polynomial().coefficient(1)
    return QQ(a)+QQ(b)*I


def set_im(M):
    M.set_immutable()
    return M
    
def diag(u,v,w):
    return set_im(Matrix(3,[u,0,0,
                     0,v,0,
                     0,0,w]))
Id = diag(1,1,1)

def convert_matrix(A,convert_func):
    nrows, ncols = A.nrows(), A.ncols()
    return Matrix([[convert_func(A[i, j]) for j in range(ncols)] for i in range(nrows)])

def print_mat(M):
    print('[',end='')
    for i,row in enumerate(M):
        print('[',end=' ')
        endrow_delim = '],\n' if i+1!=M.nrows() else ']'
        for j,entry in enumerate(row):
            delim = endrow_delim if j+1==len(row) else ', '
            print(entry,end=delim)
    print(']')

unitary_act = lambda A:A*A.H
unitary_cond = lambda A,n:unitary_act(A) == diag(n,n,n)