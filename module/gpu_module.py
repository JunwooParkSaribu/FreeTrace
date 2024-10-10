import cupy as cp


def likelihood(crop_imgs, gauss_grid, bg_squared_sums, bg_means, window_size1, window_size2):
    crop_imgs = cp.asarray(crop_imgs)
    gauss_grid = cp.asarray(gauss_grid)
    bg_squared_sums = cp.asarray(bg_squared_sums)
    bg_means = cp.asarray(bg_means)
    surface_window = window_size1 * window_size2
    g_mean = cp.mean(gauss_grid)
    g_bar = (gauss_grid - g_mean).reshape([window_size1 * window_size2, 1])
    g_squared_sum = cp.sum(g_bar ** 2)
    i_hat = (crop_imgs - bg_means.reshape(crop_imgs.shape[0], 1, 1))
    i_local_mins = cp.min(i_hat, axis=(1, 2))

    for i in range(i_hat.shape[0]):
        i_hat[i,:,:] -= max(0.0, i_local_mins[i])

    i_hat = cp.matmul(i_hat, g_bar) / g_squared_sum
    i_hat = cp.maximum(cp.zeros(i_hat.shape), i_hat)
    L = ((surface_window / 2.) * cp.log(1 - (i_hat ** 2 * g_squared_sum).T /
                                        (bg_squared_sums - (surface_window * bg_means)))).T
    return cp.asnumpy(L.reshape(crop_imgs.shape[0], crop_imgs.shape[1], 1))
