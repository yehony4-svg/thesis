# SageMath code (tested with QuadraticField(-2))
from sage.all import *
import itertools
from collections import defaultdict
from utils import *

def Npair(pair):
    a,b = pair
    return a*a + 2*b*b

# Hermitian inner product <v,w> = sum_i v_i * conj(w_i)
# Returned as coefficients (A,B) for A + B*T so we can test zero cheaply.
def herm_inner_coeffs(v, w):
    A = 0
    B = 0
    for (a,b), (c,d) in zip(v, w):
        # (a+bT)*(c - dT) = (ac + 2bd) + (-ad + bc) T
        A += a*c + 2*b*d
        B += -a*d + b*c
    return (A, B)

def orth(v, w):
    A,B = herm_inner_coeffs(v, w)
    return (A == 0) and (B == 0)

# Optional: quick mod-p prefilter (saves time when sets are big)
def orth_modp(v, w, p=3):
    Ap = 0
    Bp = 0
    for (a,b), (c,d) in zip(v,w):
        Ap += (a*c + 2*b*d) 
        Bp += (-a*d + b*c)
    return (Ap % p == 0) and (Bp % p == 0)

# ---- Generate all (a,b) with bounded |a|,|b|, bucketed by integer norm ----
def ab_pairs_by_norm(max_abs):
    buckets = defaultdict(list)
    rng = range(-max_abs, max_abs+1)
    for a in rng:
        aa = a*a
        for b in rng:
            k = aa + 2*b*b
            buckets[k].append((a,b))
    return buckets

# ---- Build all 3-vectors (as lists of (a,b)) whose norm-sum equals n ----
def row_vectors_of_normsum(n, buckets):
    out = []
    ks = [k for k in buckets.keys() if 0 <= k <= n]
    ks.sort()
    for k1 in ks:
        for k2 in ks:
            k3 = n - k1 - k2
            if k3 < 0 or k3 not in buckets:
                continue
            for e1 in buckets[k1]:
                for e2 in buckets[k2]:
                    for e3 in buckets[k3]:
                        out.append([e1, e2, e3])
    return out

# ---- Orthogonality graph (edges between orthogonal rows) ----
def orth_graph(vecs, use_modp=True, p=3):
    N = len(vecs)
    nbrs = [set() for _ in range(N)]
    for i in range(N):
        vi = vecs[i]
        for j in range(i+1, N):
            vj = vecs[j]
            if use_modp and not orth_modp(vi, vj, p=p):
                continue
            if orth(vi, vj):
                nbrs[i].add(j)
                nbrs[j].add(i)
    return nbrs

# ---- Emit a Sage matrix over E from three (a,b) rows ----
def emit_matrix_E(rows_pairs,to_E):
    rows_E = [[to_E(e) for e in row] for row in rows_pairs]
    return matrix(rows_E)

# ---- Dedup key (canonicalize rows to avoid row-permutation duplicates) ----
def canon_key(rows_pairs):
    # flatten as tuples of ints to guarantee hashability & consistency
    flat_rows = [tuple(itertools.chain.from_iterable(row)) for row in rows_pairs]
    return tuple(sorted(flat_rows))

# ---- Triangle enumeration version ----
def find_GU_triangle(n=1, to_E=lambda x:x, max_coeff=None, use_modp=True, p=5, dedup=True):
    """
    Find all U in M_3(E) with U*U^* = n*I using triangle enumeration on the orthogonality graph.
    - n: positive integer
    - max_coeff: bound on |a|,|b|. Defaults to n (safe, you may tighten it).
    - use_modp: enable mod-p prefilter before exact integer test
    - p: small prime for the prefilter
    """
    if max_coeff is None:
        max_coeff = n

    buckets = ab_pairs_by_norm(max_coeff)
    row_vecs = row_vectors_of_normsum(n, buckets)
    if not row_vecs:
        return []

    nbrs = orth_graph(row_vecs, use_modp=use_modp, p=p)

    seen = set()
    results = []
    N = len(row_vecs)
    for i in range(N):
        Ni = nbrs[i]
        for j in [j for j in Ni if j > i]:
            common = Ni & nbrs[j]
            for k in [k for k in common if k > j]:
                rows = [row_vecs[i], row_vecs[j], row_vecs[k]]
                if dedup:
                    key = canon_key(rows)
                    if key in seen:
                        continue
                    seen.add(key)
                U = emit_matrix_E(rows, to_E)
                # sanity check (cheap and safe):
                if U * U.conjugate_transpose() == n * Id:
                    results.append(U)
    return results

# ---- Incremental backtracking version (often faster for larger pools) ----
def find_GU_backtrack(n=1,to_E=lambda x:x, max_coeff=None, use_modp=True, p=5, dedup=True):
    """
    Incrementally build orthogonal triples:
      pick v1; for v2 in N(v1); for v3 in N(v1)∩N(v2).
    """
    if max_coeff is None:
        max_coeff = n

    buckets = ab_pairs_by_norm(max_coeff)
    row_vecs = row_vectors_of_normsum(n, buckets)
    if not row_vecs:
        return []

    nbrs = orth_graph(row_vecs, use_modp=use_modp, p=p)

    seen = set()
    results = []
    N = len(row_vecs)
    for i in range(N):
        Ni = nbrs[i]
        for j in sorted(Ni):
            if j <= i:
                continue
            Nij = Ni & nbrs[j]
            for k in sorted(Nij):
                if k <= j:
                    continue
                rows = [row_vecs[i], row_vecs[j], row_vecs[k]]
                if dedup:
                    key = canon_key(rows)
                    if key in seen:
                        continue
                    seen.add(key)
                U = emit_matrix_E(rows,to_E)
                if U * U.conjugate_transpose() == n * Id:
                    results.append(U)
    return results

