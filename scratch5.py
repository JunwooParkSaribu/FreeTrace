
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import cupy as cp
import cupyx
from scipy.sparse.linalg import spsolve
import scipy
import scipy.sparse


def sparse_matrix_return(N):
    base = np.zeros([int(6*N), int(6*N)])
    add_base = np.eye(6) * 2 + np.ones([6, 6])
    for idx in range(0, base.shape[0], add_base.shape[0]):
        base[idx:idx+add_base.shape[0], idx:idx+add_base.shape[0]] = add_base

    A = base
    B = np.ones([base.shape[0], 1]) * 4
    return A, B

def seq_matrix_return():
    base = np.zeros([6, 6])
    add_base = np.eye(6) * 2 + np.ones([6, 6])
    for idx in range(0, base.shape[0], add_base.shape[0]):
        base[idx:idx+add_base.shape[0], idx:idx+add_base.shape[0]] = add_base

    A = base
    B = np.ones([base.shape[0], 1]) * 4
    return A, B   


def numpy_solve(A, B):
    result = lstsq(A.copy(), B.copy(), lapack_driver='gelsy', check_finite=False)
    return result


def numpy_overwrite_solve(A, B):
    result = lstsq(A.copy(), B.copy(), overwrite_a=True, overwrite_b=True, lapack_driver='gelsy', check_finite=False)
    return result


def cu_solve(A, B):
    #a_gpu = cp.asarray(A)
    #b_gpu = cp.asarray(B)
    #result = cp.linalg.solve(a_gpu, b_gpu)
    a_gpu = cupyx.scipy.sparse.csr_matrix(cp.asarray(A))
    b_gpu = cp.asarray(B)
    result = cupyx.scipy.sparse.linalg.spsolve(a_gpu, b_gpu)
    #a_gpu = scipy.sparse.csr_array(A)
    #b_gpu = scipy.sparse.csr_array(B)
    #print(a_gpu, b_gpu)
    #result = spsolve(a_gpu, b_gpu)
    #print(result)
    return result.get()


def sequential_solve(A, B, N):
    for _ in range(N):
        numpy_solve(A, B)

