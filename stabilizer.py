import numpy as np
from utils import set_im,Id
import itertools
from sage.all import Matrix, floor, sqrt
from copy import copy

def create_stage_matrices(E):
    T = E.gen()
    stage_matrices=[]
    for val in [1,-1,T,-T]:
        for idx in [[i,j] for i,j in itertools.product(range(3),repeat=2) if i!=j]:
            U = Matrix(E,3,[1,0,0,
                          0,1,0,
                          0,0,1])
            U[*idx] = val
            stage_matrices.append(U)
    stage_matrices = [u*v for u,v in itertools.product(stage_matrices,repeat=2)]
    return stage_matrices


def min_eigenvalue(A):
    """
    A - hermitian matrix (has positive real eigenvalues)
    """
    eigenvalues = A.eigenvalues()
    eigenvalues = [float(l) for l in eigenvalues]
    return min(eigenvalues)

def reduce_trace(J,stage_matrices):
    """
    J = hermitian form
    In each step, make a row and a conjugate col elementary operation which decrease the trace the most. 
    returns Bt and W in GL_n(Z) s.t W.C.T * J * W = Bt
    """
    Bt = copy(J)
    Vt = copy(Id)
    U = copy(Id) 

    down = True
    while down:
        Vt = U*Vt
        Bt = U*Bt*U.C.T
        # print(Bt)
        cong_traces = [(U*Bt*U.H).trace() for U in stage_matrices] # traces
        idx = np.argmin(cong_traces)
        # cong_lmins = [min_eigenvalue(U*Bt*U.C.T) for U in stage_matrices] # minimal eigenvalue
        # idx = np.argmax(cong_lmins)
        U = stage_matrices[idx]
        down = np.min(cong_traces) < Bt.trace()
        # down = np.max(cong_lmins) > min_eigenvalue(Bt)
    W = Vt.H
    # print(Bt)
    return Bt, W


def vectors_solutions(B,E):
    """
    find all the vectors which holds v^*Bv=B_ii
    """
    T = E.gen()
    bound = min_eigenvalue(B)**-1 * np.max([float(B[i,i]) for i in range(3)]) 
    Z_bound = floor(sqrt(bound))
    T_bound = floor(sqrt(bound/2))
    # print(Z_bound,T_bound)
    bounds_iter = (a+b*T for a in range(-Z_bound,Z_bound+1) for b in range(-T_bound,T_bound+1)) # |a|^2<=bound 2|b|^2 <= bound

    vectors_00 = []
    vectors_11 = []
    vectors_22 = []

    for a,b,c in itertools.product(bounds_iter,repeat=3):
        v= Matrix([[a],[b],[c]])
        vsBv = v.H*B*v
        if vsBv==B[0,0]:
            vectors_00.append(v)
        if vsBv==B[1,1]:
            vectors_11.append(v)
        if vsBv==B[2,2]:
            vectors_22.append(v)
    return vectors_00,vectors_11,vectors_22


def matrix_solutions(B,E):
    vectors_00,vectors_11,vectors_22 = vectors_solutions(B,E)
    solutions=[]
    for v0 in vectors_00:
        for v1 in vectors_11:
            if v0.H*B*v1==B[0,1]:
                for v2 in vectors_22:
                    if v0.C.T*B*v2==B[0,2] and v1.C.T*B*v2==B[1,2]:
                        sol = Matrix([list(v0.T)[0],list(v1.T)[0],list(v2.T)[0]]).T
                        solutions.append(sol)
    return solutions



def get_stab(g,E,div_min_val):
    
    stage_matrices = create_stage_matrices(E)

    J = div_min_val(g.H*g)
    B,W = reduce_trace(J,stage_matrices)
    solutions = matrix_solutions(B,E)
    J_solutions = [W*sol*W**-1 for sol in solutions]
    g_stab = {set_im(div_min_val(g*h*g**-1)) for h in J_solutions}
    return g_stab
    








# other approach, doesn't work for now

def get_list(dictionary,key):
    if key in dictionary.keys():
        return dictionary[key]
    return []

norm_E = lambda x: norm(E(x)) 

def vectors_solutions_triangular(A,c):
    """
    find solutions to v.H*A.H*A*v=c which are ||Av||^2 = c
    for triangular g by solving the equation |A_00*x + A_01*y + A_02*z|^2 + |A_11*y + A_12*z|^2 + |a_22z|^2 = c 
    """
    solutions = []
    
    c_bound = c//norm_E(A[2,2])
    for z_norm in range(c_bound): # |z|^2 <= c/|a_22|^2
        for z in get_list(norms_dict, z_norm): # (|A_11||y| - |A_12*z|)^2 <=|A_11*y + A_12*z|^2 <=c by inverse triangle inequality
            cz_bond = (sqrt(c) - sqrt(norm_E(A[1,2]*z)))^2 / norm_E(A[1,1]) 
            for y_norm in range(int(cz_bond)):
                for y in get_list(norms_dict,y_norm): 
                    x_term_norm = c - norm_E(A[1,1] * y + A[1,2] * z) - norm_E(A[2,2] * z)
                    # print(x_term_norm)
                    if x_term_norm >=0:
                        for t in get_list(norms_dict,x_term_norm): # t = A_00*x + A_01*y + A_02*z
                            x = 1/A[0,0] * (t - A[0,1]*y - A[0,2]*z)
                            if p_val(x)>=0: # x is an integer 
                                v = Matrix([[x],[y],[z]])
                                solutions.append(v)
    return solutions
    
def matrix_solutions_triangular(g):
    B = g.H*g
    vectors_00 = vectors_solutions_triangular(g, int(B[0,0]))
    vectors_11 = vectors_solutions_triangular(g, int(B[1,1]))
    vectors_22 = vectors_solutions_triangular(g, int(B[2,2]))

    solutions=[]
    for v0 in vectors_00:
        for v1 in vectors_11:
            if v0.H*B*v1==B[0,1]:
                for v2 in vectors_22:
                    if v0.C.T*B*v2==B[0,2] and v1.C.T*B*v2==B[1,2]:
                        sol = Matrix([list(v0.T)[0],list(v1.T)[0],list(v2.T)[0]]).T
                        solutions.append(sol)
    return solutions
