import cupy as cp
import time
import threading
import numpy as np
from timeit import default_timer as timer
from itertools import product
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

from itertools import islice


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
    imgs = cp.asarray(imgs)
    bins = 0.01
    nb_imgs = imgs.shape[0]
    img_flat_length = imgs.shape[1] * imgs.shape[2]
    bgs = {}
    bg_means = []
    bg_stds = []
    bg_intensities = (imgs.reshape(nb_imgs, img_flat_length) * 100).astype(cp.uint8) / 100
    for idx in range(nb_imgs):
        if cp.sum(bg_intensities[idx]) <= 0:
            bg_intensities[idx] = bg_intensities[idx+1].copy()
    post_mask_args = cp.array([cp.arange(img_flat_length) for _ in range(nb_imgs)], dtype=cp.int32)
    mask_sums_modes = cp.zeros(nb_imgs)
    mask_stds = cp.empty(nb_imgs)
    for repeat in range(3):
        for i in range(nb_imgs):
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
    bg_means = cp.array(bg_means)
    bg_stds = cp.array(bg_stds)
    for window_size in window_sizes:
        bg = cp.ones((bg_intensities.shape[0], window_size[0] * window_size[1]))
        bg *= bg_means.reshape(-1, 1)
        bgs[window_size[0]] = cp.asnumpy(bg)
    return bgs, cp.asnumpy(bg_stds / bg_means / alpha)


def image_cropping2(extended_imgs, extend, window_size0, window_size1, shift):
    extended_imgs = cp.asarray(extended_imgs)
    nb_imgs = extended_imgs.shape[0]
    row_size = extended_imgs.shape[1]
    col_size = extended_imgs.shape[2]
    start_row = int(extend/2 - (window_size1-1)/2)
    end_row = row_size - window_size1 - start_row + 1
    start_col = int(extend/2 - (window_size0-1)/2)
    end_col = col_size - window_size0 - start_col + 1
    row_indice = np.arange(start_row, end_row, shift, dtype=int)
    col_indice = np.arange(start_col, end_col, shift, dtype=int)
    cropped_imgs = cp.empty([nb_imgs, len(row_indice) * len(col_indice), window_size0, window_size1], dtype=cp.double)
    index = 0
    for r in row_indice:
        for c in col_indice:
            r = int(r)
            c = int(c)
            cropped_imgs[:, index] = extended_imgs[:, r:r + window_size1, c:c + window_size0]
            index += 1
    return cropped_imgs.reshape(nb_imgs, -1, window_size0 * window_size1)


def image_cropping(extended_imgs, extend, window_size0, window_size1, shift):
    extended_imgs = cp.asarray(extended_imgs)
    nb_imgs = extended_imgs.shape[0]
    row_size = extended_imgs.shape[1]
    col_size = extended_imgs.shape[2]
    start_row = int(extend/2 - (window_size1-1)/2)
    end_row = row_size - window_size1 - start_row + 1
    start_col = int(extend/2 - (window_size0-1)/2)
    end_col = col_size - window_size0 - start_col + 1
    row_col_comb = list(product(range(start_row, end_row, shift), range(start_col, end_col, shift)))
    cropped_imgs = cp.empty([nb_imgs, len(row_col_comb), window_size0, window_size1], dtype=np.double)
    index = 0
    """
    before_time = timer()
    my_imgs = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        # submit tasks and collect futures
        chunk_row_col_comb = chunk(row_col_comb, len(row_col_comb)//2)
        index_chunk = chunk(list(range(len(row_col_comb))), len(row_col_comb)//2)
        futures = []
        #print(list(chunk_row_col_comb), list(index_chunk))
        #print(list(chunk_row_col_comb)[0], list(index_chunk)[0])
        for rc_chunk, idx_chunk in zip(chunk_row_col_comb, index_chunk):
            print("@@@@")
            futures.append(executor.submit(ref_cp_kernel, cropped_imgs, extended_imgs, rc_chunk, window_size0, idx_chunk))
        # process task results as they are available
        for future in as_completed(futures):
            # retrieve the result
            my_imgs.append(future.result())
    print(f'{"copying calcul1":<35}:{(timer() - before_time):.2f}s')
    """
    before_time = timer()
    for r, c in row_col_comb:
        r = int(r)
        c = int(c)
        #print(cropped_imgs[0, index], extended_imgs[0, r:r + window_size1, c:c + window_size0])
        cropped_imgs[:, index] = extended_imgs[:, r:r + window_size1, c:c + window_size0]
        #print(cropped_imgs[0, index], extended_imgs[0, r:r + window_size1, c:c + window_size0])
        index += 1
    print(f'{"copying calcul2":<35}:{(timer() - before_time):.2f}s')
    return cropped_imgs.reshape(nb_imgs, -1, window_size0 * window_size1)


def ref_cp_kernel(x, y, rc_chunk, w, index_chunk):
    for (r, c), idx in zip(rc_chunk, index_chunk):
        print(r, c, idx, w)
        x[:, idx] = y[:, r:r + w, c:c + w]



def chunk(arr_range, arr_size):
    arr_range = iter(arr_range)
    return iter(lambda: tuple(islice(arr_range, arr_size)), ())
