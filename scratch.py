from timeit import default_timer as timer
from numba import jit, guvectorize, int32, int64, float64
from numba import cuda, float32
import numpy as np
import math

TPB = 32

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

@cuda.jit
def f_vec_loops2(x, y, ret, maxiter):
    nx = len(ret)
    ny = len(ret[0])

    i, j = cuda.grid(2)
    if (i < nx) & (j < ny):
        value = 0
        for k in range(maxiter):
            value += x[i, j] * y[i, j]

        ret[i, j] = value

@cuda.jit
def f_vec_loops(x, ret, maxiter):
    nx = len(ret)
    ny = len(ret[0])
    for k in range(maxiter):
        for i in range(nx):
            for j in range(ny):
                ret[i, j] += x[i, j]


x = 1024
y = 1024
a = np.random.rand(x, y).astype(np.float64)
b = np.random.rand(x, y).astype(np.float64)
nump_ret = np.zeros([x, y], dtype='float64')
cuda_ret = np.zeros([x, y], dtype='float64')
shared_ret = np.zeros([x, y], dtype='float64')

maxiter = 100000

s = timer()
for _ in range(maxiter):
    nump_ret = nump_ret + (a * b)
print('cpu:', timer() - s)
print(nump_ret)

s = timer()
a_cuda = cuda.to_device(a)
b_cuda = cuda.to_device(b)
ret_cuda = cuda.to_device(cuda_ret)

threadsperblock = (32, 32)
blockspergrid_x = math.ceil(x / threadsperblock[0])
blockspergrid_y = math.ceil(y / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

f_vec_loops2[blockspergrid, threadsperblock](a_cuda, b_cuda, ret_cuda, maxiter)
cuda.synchronize()
trt = ret_cuda.copy_to_host()
print('gpu:', timer()-s)
print(trt)

"""
s = timer()
a_cuda = cuda.to_device(a)
b_cuda = cuda.to_device(b)
shared_ret_cuda = cuda.to_device(shared_ret)

threadsperblock = (32, 32)
blockspergrid_x = math.ceil(x / threadsperblock[0])
blockspergrid_y = math.ceil(y / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
for _ in range(maxiter):
    fast_matmul[blockspergrid, threadsperblock](a_cuda, b_cuda, shared_ret_cuda)
    cuda.synchronize()
trt = shared_ret_cuda.copy_to_host()
print('shared_gpu:', timer()-s)
print(trt)
"""