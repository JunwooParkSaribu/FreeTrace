#cython: infer_types=True
#cython: cdivision=True
from libc.math cimport sqrt, M_PI, log, pow, abs
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(False)
cpdef (double, int) predict_cauchy(double[::1] next_vec, double[::1] prev_vec, float alpha, int lag, float precision, int dimension):
    cdef int delta_t, abnormal
    cdef double log_pdf
    log_pdf = 0.0
    abnormal = 0
    delta_t = lag + 1

    for dim_i in range(dimension):
        vec1 = next_vec[dim_i]
        vec2 = prev_vec[dim_i]
        if vec1 < 0:
            vec1 = vec1 - precision
        else:
            vec1 = vec1 + precision
        if vec2 < 0:
            vec2 = vec2 - precision
        else:
            vec2 = vec2 + precision
        coord_ratio = vec1 / vec2

        if 0.95 < alpha < 1.05:
            if abs(coord_ratio) > 8: ## TODO
                abnormal = 1
            log_pdf += log( 1/M_PI * 1/((coord_ratio)*(coord_ratio) + 1) )

        else:
            rho = 1/2. * (pow((delta_t-1), alpha) - 2*pow(delta_t, alpha) + pow((delta_t+1), alpha))
            relativ_cov = 1/2. * (pow((delta_t+1), alpha) - pow((delta_t), alpha) - pow((1), alpha))
            scale = sqrt(abs(1-(rho*rho))) ## TODO
            if abs(coord_ratio-rho) > 8 * scale: ## TODO
                abnormal = 1
            log_pdf += log( 1/(M_PI * scale) * 1 / ( ((coord_ratio - relativ_cov)/scale)*((coord_ratio - relativ_cov)/scale) * (rho/relativ_cov) + (relativ_cov/rho) ) )

    return log_pdf, abnormal