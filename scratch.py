from timeit import default_timer as timer
from numba import jit, guvectorize, int32, int64, float64
from numba import cuda
import numpy as np
import math

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

maxiter = 1000000

#s = timer()
#for _ in range(maxiter):
#    nump_ret = nump_ret + a * b
#print('cpu:', timer() - s)
#print(nump_ret)

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