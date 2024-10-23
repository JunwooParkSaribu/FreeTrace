from timeit import default_timer as timer
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void saxpy(float *dest, float a, float *x, float *y) {
const int i = blockIdx.x*blockDim.x + threadIdx.x;
dest[i] = a*x[i] * y[i];
}
""")

saxpy_cuda = mod.get_function("saxpy") # Get the function pointer for the compiled kernel
a = 3.141
x = np.random.rand(512, 2048, 512).astype(np.double)
y = np.random.rand(512, 2048, 512).astype(np.double)

before_time = timer()
d_x = cuda.mem_alloc(x.nbytes) # Allocate memory for x on the GPU
d_y = cuda.mem_alloc(y.nbytes) # Allocate memory for y on the GPU
cuda.memcpy_htod(d_x, x) # Copy data from CPU to GPU
cuda.memcpy_htod(d_y, y) # Copy data from CPU to GPU
block_dim = (256, 1, 1)
grid_dim = ((512*2048*512-1) // block_dim[0] + 1, 1)
# Launch the GPU kernel
dest = np.zeros_like(y)
saxpy_cuda(cuda.Out(dest), cuda.In(np.double(a)), d_x, d_y, block=block_dim, grid=grid_dim)
cuda.memcpy_dtoh(y, d_y) # Copy the results back to the CPU
d_x.free() # Free GPU memory
d_y.free()
print(f'{"pycuda calcul":<35}:{(timer() - before_time):.2f}s')

import cupy as cp
def saxpy(a, x, y):
    return a * x + y
a = 3.141
x = cp.random.rand(512, 2048, 512).astype(np.double)
y = cp.random.rand(512, 2048, 512).astype(np.double)
before_time = timer()
result = saxpy(a, x, y)
print(f'{"cupy calcul":<35}:{(timer() - before_time):.2f}s')
