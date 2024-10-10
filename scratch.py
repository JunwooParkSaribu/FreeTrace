from timeit import default_timer as timer
from numba import jit, guvectorize, int32, int64, float64
from numba import cuda, float32
import numpy as np
import math
import image_pad  # type: ignore
import cupy as cp

TPB = 16

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
def fast_matmul_3_1(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB), dtype=float32)

    x, y, z = cuda.grid(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= A.shape[0] and y >= A.shape[1] and z >= A.shape[2]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        #sA[tx, ty, tz] = A[x, y, tz + i * TPB]
        print(x, y, z)
        sA[tx, ty, tz] = A[x, y, i]
        sB[tz] = B[tz + i * TPB]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        #for j in range(TPB):
        #    tmp += sA[tx, ty, j] * sB[j]

        # Wait until all threads finish computing
        #cuda.syncthreads()

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


"""
x = 1024
y = 1024
a = np.random.rand(x, y).astype(np.float64)
b = np.random.rand(x, y).astype(np.float64)
nump_ret = np.zeros([x, y], dtype='float64')
cuda_ret = np.zeros([x, y], dtype='float64')
shared_ret = np.zeros([x, y], dtype='float64')

maxiter = 1

s = timer()
a_cuda = cuda.to_device(a)
b_cuda = cuda.to_device(b)
shared_ret_cuda = cuda.to_device(shared_ret)

threadsperblock = (TPB, TPB)
blockspergrid_x = math.ceil(x / threadsperblock[0])
blockspergrid_y = math.ceil(y / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
for _ in range(maxiter):
    fast_matmul[blockspergrid, threadsperblock](a_cuda, b_cuda, shared_ret_cuda)
    cuda.synchronize()
trt = shared_ret_cuda.copy_to_host()
print('shared_gpu:', timer()-s)
print(trt)

cp_a = cp.array(a)
cp_b = cp.array(b)
cp_ret = cp.zeros([x, y], dtype='float64')
s = timer()
for _ in range(maxiter):
    cp_ret = cp.matmul(cp_a, cp_b, cp_ret)
print('cupy_gpu:', timer()-s)
print(cp_ret)

s = timer()
for _ in range(maxiter):
    numpy_ret = a@b
print('numpy:', timer()-s)
print(numpy_ret)
"""

def likelihood(crop_imgs, gauss_grid, bg_squared_sums, bg_means, window_size1, window_size2):
    surface_window = window_size1 * window_size2
    g_mean = cp.mean(gauss_grid)
    g_bar = (gauss_grid - g_mean).reshape([window_size1 * window_size2])
    g_squared_sum = cp.sum(g_bar ** 2)
    i_hat = (crop_imgs - bg_means.reshape(crop_imgs.shape[0], 1, 1))
    i_local_mins = cp.min(i_hat, axis=(1, 2))

    for i in range(i_hat.shape[0]):
        i_hat[i,:,:] -= max(0.0, i_local_mins[i])

    i_hat = cp.matmul(i_hat, g_bar) / g_squared_sum
    i_hat = cp.maximum(cp.zeros(i_hat.shape), i_hat)
    L = ((surface_window / 2.) * cp.log(1 - (i_hat ** 2 * g_squared_sum).T /
                                        (bg_squared_sums - (surface_window * bg_means)))).T
    return L.reshape(crop_imgs.shape[0], crop_imgs.shape[1], 1)

#(50, 2128, 121) (11, 11) (50,) (50,) 11 11


a = np.random.rand(1000, 2128, 21 * 21).astype(np.float64)
b = np.random.rand(21, 21).astype(np.float64)
c = np.random.rand(1000).astype(np.float64)
d = np.random.rand(1000).astype(np.float64)

s = timer()
c_ret = np.array(image_pad.likelihood(a, b, c, d, 21, 21))
print('c_likelihood:', timer() - s)
print(c_ret.shape)


a = cp.array(a)
b = cp.array(b)
c = cp.array(c)
d = cp.array(d)
s = timer()

gpu_ret = cp.array(likelihood(a, b, c, d, 21, 21))
print('gpu_likelihood:', timer() - s)
print(gpu_ret.shape)

