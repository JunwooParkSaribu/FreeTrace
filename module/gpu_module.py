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


def background(imgs, window_sizes, alpha):
    bins = 0.01
    bgs = {}
    bg_means = []
    bg_stds = []
    bg_intensities = (imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2]) * 100).astype(cp.uint8) / 100
    for i in range(len(bg_intensities)):
        args = cp.arange(len(bg_intensities[i]))
        post_mask_args = args.copy()
        for _ in range(3):
            it_hist, bin_width = cp.histogram(bg_intensities[i][post_mask_args], bins=cp.arange(0, cp.max(bg_intensities[i][post_mask_args]) + bins, bins))
            if len(it_hist) < 1:
                print('Errors on images, please check images again whether it contains an empty black-image. If not, contact the author.')
                break
            mask_sums_mode = (cp.argmax(it_hist) * bins + (bins / 2))
            mask_std = cp.std(bg_intensities[i][post_mask_args])
            post_mask_args = cp.array([arg for arg, val in zip(args, bg_intensities[i]) if
                                       (mask_sums_mode - 3. * mask_std) < val < (mask_sums_mode + 3. * mask_std)])
        it_data = bg_intensities[i][post_mask_args]
        bg_means.append(cp.mean(it_data))
        bg_stds.append(cp.std(it_data))
    bg_means = cp.array(bg_means)
    bg_stds = cp.array(bg_stds)

    for window_size in window_sizes:
        bg = cp.ones((bg_intensities.shape[0], window_size[0] * window_size[1]))
        bg *= bg_means.reshape(-1, 1)
        bgs[window_size[0]] = cp.asnumpy(bg)
    return bgs, cp.asnumpy(bg_stds / bg_means / alpha)
