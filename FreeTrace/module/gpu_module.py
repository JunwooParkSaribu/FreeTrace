import cupy as cp
import numpy as np
from itertools import product, islice


CP_NP_CRITERIA = 50


def get_gpu_mem_size():
    gpu_count = cp.cuda.runtime.getDeviceCount()
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    free_mem /= 1000 * 1000 * 1000
    return int(free_mem)


def likelihood(crop_imgs, gauss_grid, bg_squared_sums, bg_means, window_size1, window_size2):
    nb_imgs = crop_imgs.shape[0]
    if nb_imgs < CP_NP_CRITERIA:
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
    i_hat -= cp.maximum(cp.zeros_like(i_local_mins), i_local_mins).reshape(-1, 1, 1)
    i_hat = cp.matmul(i_hat, g_bar) / g_squared_sum
    i_hat = cp.maximum(cp.zeros(i_hat.shape), i_hat)
    L = ((surface_window / 2.) * cp.log(1 - (i_hat ** 2 * g_squared_sum).T /
                                        (bg_squared_sums - (surface_window * bg_means)))).T
    L = cp.asnumpy(L.reshape(crop_imgs.shape[0], crop_imgs.shape[1], 1).astype(cp.float32))
    return L


def background(imgs, window_sizes, alpha):
    imgs = cp.asarray(imgs)
    imgs = imgs / cp.max(imgs, axis= (1,2)).reshape(-1, 1, 1)
    bins = 0.01
    nb_imgs = imgs.shape[0]
    img_flat_length = imgs.shape[1] * imgs.shape[2]
    bgs = {}
    bg_means = []
    bg_stds = []
    bg_intensities = (imgs.reshape(nb_imgs, img_flat_length) * 100).astype(cp.uint8) / 100

    for idx in range(nb_imgs):
        if cp.sum(bg_intensities[idx]) <= 0:
            dummy_intensities = 0.1
            bg_intensities[idx][0] = dummy_intensities

    post_mask_args = cp.array([cp.arange(img_flat_length) for _ in range(nb_imgs)], dtype=cp.int64)
    mask_sums_modes = cp.zeros(nb_imgs)
    mask_stds = cp.empty(nb_imgs)
    for repeat in range(3):
        for i in range(nb_imgs):
            if len(post_mask_args[i]) <= 0:
                print('Errors on images, please check images again whether it contains an empty black-image. If not, contact the author.')
                continue
            it_hist, bin_width = cp.histogram(bg_intensities[i, post_mask_args[i]], bins=cp.arange(0, cp.max(bg_intensities[i, post_mask_args[i]]) + bins, bins))
            if len(it_hist) < 1:
                print('Errors on images, please check images again whether it contains an empty black-image. If not, contact the author.')
                continue
            mask_sums_modes[i] = (cp.argmax(it_hist) * bins + (bins / 2))
        if repeat==0:
            mask_stds = cp.std(cp.take(bg_intensities, post_mask_args), axis=1)
        else:
            for i in range(nb_imgs):
                mask_stds[i] = cp.std(cp.take(bg_intensities[i], post_mask_args[i]))
        post_mask_args = []
        for i in range(nb_imgs):
            post_mask_args.append(cp.argwhere((bg_intensities[i] > float(mask_sums_modes[i] - 3. * mask_stds[i])) & (bg_intensities[i] < float(mask_sums_modes[i] + 3. * mask_stds[i]))).flatten())
    for i in range(nb_imgs):
        bg_means.append(cp.mean(cp.take(bg_intensities[i], post_mask_args[i])))
        bg_stds.append(cp.std(cp.take(bg_intensities[i], post_mask_args[i])))
    bg_means = cp.array(bg_means, dtype=cp.float32)
    bg_stds = cp.array(bg_stds, dtype=cp.float32)
    for window_size in window_sizes:
        bg = cp.ones((bg_intensities.shape[0], window_size[0] * window_size[1]), dtype=cp.float32)
        bg *= bg_means.reshape(-1, 1)
        bgs[window_size[0]] = cp.asnumpy(bg)

    thresholds = cp.asnumpy(1/(bg_means**2 / bg_stds**2)) * 2.0
    thresholds = np.maximum(thresholds, np.ones_like(thresholds) * 1.0)
    for th_i in range(len(thresholds)):
        if np.isnan(thresholds[th_i]):
            thresholds[th_i] = 1.0
    return bgs, thresholds / alpha


def image_cropping(extended_imgs, extend, window_size0, window_size1, shift):
    nb_imgs = extended_imgs.shape[0]
    row_size = extended_imgs.shape[1]
    col_size = extended_imgs.shape[2]
    start_row = int(extend/2 - (window_size1-1)/2)
    end_row = row_size - window_size1 - start_row + 1
    start_col = int(extend/2 - (window_size0-1)/2)
    end_col = col_size - window_size0 - start_col + 1
    row_col_comb = list(product(range(start_row, end_row, shift), range(start_col, end_col, shift)))
    if nb_imgs >= CP_NP_CRITERIA:
        extended_imgs = cp.asarray(extended_imgs)
        cropped_imgs = cp.empty([nb_imgs, len(row_col_comb), window_size0, window_size1], dtype=cp.float32)
    else:
        cropped_imgs = np.empty([nb_imgs, len(row_col_comb), window_size0, window_size1], dtype=np.float32)
    index = 0
    for r, c in row_col_comb:
        r = int(r)
        c = int(c)
        cropped_imgs[:, index] = extended_imgs[:, r:r + window_size1, c:c + window_size0]
        index += 1
    return cropped_imgs.reshape(nb_imgs, -1, window_size0 * window_size1), type(cropped_imgs)==cp.ndarray


def chunk(arr_range, arr_size):
    arr_range = iter(arr_range)
    return iter(lambda: tuple(islice(arr_range, arr_size)), ())
