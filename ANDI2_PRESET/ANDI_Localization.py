import sys
sys.path.append('../')
import numpy as np
import tifffile
import concurrent.futures
import image_pad
import regression
from FileIO import write_localization, read_parameters, check_video_ext
from ImageModule import draw_cross
from timeit import default_timer as timer


def region_max_filter2(maps, window_size, thresholds, detect_range=0):
    indices = []
    if detect_range == 0:
        r_start_index = window_size[1] // 2
        col_start_index = window_size[0] // 2
    else:
        r_start_index = 1
        col_start_index = 1

    for _ in range(2):
        args_map = maps > thresholds.reshape(-1, 1, 1)
        maps = maps * args_map
        img_n, row, col = np.where(args_map == True)
        for n, r, c in zip(img_n, row, col):
            if maps[n][r][c] == np.max(maps[n, max(0, r-r_start_index):min(maps.shape[1]+1, r+r_start_index+1),
                                       max(0, c-col_start_index):min(maps.shape[2]+1, c+col_start_index+1)]) and maps[n][r][c] != 0:
                indices.append([n, r, c])
                maps[n, max(0, r - r_start_index):min(maps.shape[1] + 1, r + r_start_index + 1),
                max(0, c - col_start_index):min(maps.shape[2] + 1, c + col_start_index + 1)] = 0
    return indices


def region_max_filter(maps, window_sizes, thresholds, detect_range=0):
    indices = []
    nb_imgs = maps.shape[1]
    infos = [[] for _ in range(nb_imgs)]
    for i, (hmap, window_size) in enumerate(zip(maps, window_sizes)):
        args_map = hmap > thresholds[:, i].reshape(-1, 1, 1)
        maps[i] = hmap * args_map
        if detect_range == 0:
            r_start_index = window_size[1] // 2
            col_start_index = window_size[0] // 2
        else:
            r_start_index = detect_range
            col_start_index = detect_range
        img_n, row, col = np.where(args_map == True)
        for n, r, c in zip(img_n, row, col):
            if maps[i][n][r][c] == np.max(
                    maps[i, n, max(0, r - r_start_index):min(maps[i].shape[1] + 1, r + r_start_index + 1),
                    max(0, c - col_start_index):min(maps[i].shape[2] + 1, c + col_start_index + 1)]):
                infos[n].append([i, r, c, hmap[n][r][c]])
    maps = np.moveaxis(maps, 0, 1)
    for img_n, info in enumerate(infos):
        mask = np.zeros((maps.shape[2], maps.shape[3])).astype(np.uint8)
        if len(info) > 0:
            info = np.array(info)
            info = info[np.argsort(info[:, 3])[::-1]]
            for mol_info in info:
                if detect_range == 0:
                    extend = (window_sizes[int(mol_info[0])][0] - 1) // 2
                else:
                    extend = detect_range
                row_min = int(max(0, mol_info[1] - extend))
                row_max = int(min(mask.shape[0] - 1, mol_info[1] + extend))
                col_min = int(max(0, mol_info[2] - extend))
                col_max = int(min(mask.shape[1] - 1, mol_info[2] + extend))
                if mask[int(mol_info[1])][int(mol_info[2])] != 1:
                    indices.append([img_n, int(mol_info[1]), int(mol_info[2]), int(window_sizes[int(mol_info[0])][0])])
                    mask[row_min:row_max, col_min:col_max] = 1
    return np.array(indices)


def subtract_pdf(ext_imgs, pdfs, indices, window_size, bg_means, extend):
    for pdf, (n, r, c) in zip(pdfs, indices):
        bg = (np.ones(pdf.shape) * bg_means[n]).reshape(window_size)
        pdf = np.ascontiguousarray(pdf).reshape(window_size)
        row_indice = np.array([r - int((window_size[1]-1)/2), r + int((window_size[1]-1)/2)], dtype=np.intc) + int(extend/2)
        col_indice = np.array([c - int((window_size[0]-1)/2), c + int((window_size[0]-1)/2)], dtype=np.intc) + int(extend/2)
        ext_imgs[n][row_indice[0]:row_indice[1]+1, col_indice[0]:col_indice[1]+1] -= pdf
        ext_imgs[n][row_indice[0]:row_indice[1] + 1, col_indice[0]:col_indice[1] + 1] = (
            np.maximum(ext_imgs[n][row_indice[0]:row_indice[1]+1, col_indice[0]:col_indice[1]+1], bg))
        ext_imgs[n] = image_pad.boundary_smoothing(ext_imgs[n], row_indice, col_indice)
    return np.array(ext_imgs)


def gauss_psf(window_sizes, radi):
    gauss_grid_window = []
    for window_size, radius in zip(window_sizes, radi):
        x_subpixel = np.arange(window_size[0]) + .5
        y_subpixel = np.arange(window_size[1]) + .5
        center_x = window_size[0] / 2.
        center_y = window_size[1] / 2.
        base_vals = np.ones((window_size[1], window_size[0], 2)) * np.array([center_x, center_y])
        gauss_psf_vals = np.stack(np.meshgrid(x_subpixel, y_subpixel), -1)
        gauss_psf_vals = np.exp(-(np.sum((gauss_psf_vals - base_vals)**2, axis=2))
                                /(2*(radius**2))) / (np.sqrt(np.pi) * radius)
        gauss_grid_window.append(np.array(gauss_psf_vals))
    return gauss_grid_window


def indice_filtering(indices, window_sizes, img_shape, extend):
    masks = np.zeros(img_shape)
    win_masks = [[[[] for _ in range(img_shape[2])] for _ in range(img_shape[1])] for _ in range(img_shape[0])]
    for indexx, wins in zip(indices[::-1], window_sizes[::-1]):
        for index in indexx:
            masks[index[0], index[1], index[2]] += 1
            win_masks[index[0]][index[1]][index[2]].append(wins[0])
    ret_indices = check_masks_overlaps(masks, win_masks, extend, window_sizes)
    return ret_indices


def check_masks_overlaps(masks, window_masks, extend, window_sizes):
    nb_window_sizes = len(window_sizes)
    w_size_dict = {ws[0]: i for i, ws in enumerate(window_sizes)}
    all_groups = []
    for img_n, (mask, window_mask) in enumerate(zip(masks, window_masks)):
        groups = []
        overlay_mask = np.zeros_like(mask, dtype=np.uint8)
        rs, cs = np.where(mask >= 1)
        if len(rs) == 0:
            continue

        coords = np.vstack((rs, cs)).T
        while 1:
            group = [[] for _ in range(nb_window_sizes)]
            selected_args = []
            piv_coord = coords[0]

            for window_size in window_mask[piv_coord[0]][piv_coord[1]]:
                group[w_size_dict[window_size]].append([img_n, piv_coord[0] + extend, piv_coord[1] + extend, window_size])
                row_min = max(0, piv_coord[0] - (window_size // 2))
                row_max = min(mask.shape[0] - 1, piv_coord[0] + (window_size // 2))
                col_min = max(0, piv_coord[1] - (window_size // 2))
                col_max = min(mask.shape[1] - 1, piv_coord[1] + (window_size // 2))
                overlay_mask[row_min: row_max+1, col_min: col_max+1] += 1

            explorer_coords = coords[1:].copy()
            while 1:
                group_added = 0
                for selected_index, coord in enumerate(explorer_coords):
                    if selected_index in selected_args:
                        continue
                    if overlay_mask[coord[0]][coord[1]] >= 1:
                        selected_args.append(selected_index)
                        for window_size in window_mask[coord[0]][coord[1]]:
                            row_min = max(0, coord[0] - (window_size // 2))
                            row_max = min(mask.shape[0] - 1, coord[0] + (window_size // 2))
                            col_min = max(0, coord[1] - (window_size // 2))
                            col_max = min(mask.shape[1] - 1, coord[1] + (window_size // 2))
                            overlay_mask[row_min: row_max + 1, col_min: col_max + 1] += 1
                            group[w_size_dict[window_size]].append([img_n, coord[0] + extend, coord[1] + extend, window_size])
                            group_added += 1

                if group_added == 0:
                    groups.append(group)
                    break
            selected_args = np.array(selected_args) + 1
            selected_args = np.append(selected_args, 0).astype(np.uint32)
            coords = np.delete(coords, selected_args, axis=0)
            if len(coords) == 0:
                break
        all_groups.extend(groups)
    return all_groups


def localization(imgs: np.ndarray, bgs, f_gauss_grids, b_gauss_grids, *args):
    pass_to_multi = False
    single_winsizes = args[0]
    single_thresholds = args[2]
    multi_winsizes = args[3]
    multi_thresholds = args[5]
    p0=args[6]
    shift=args[7]
    decomp_n=args[8]
    deflation_loop_backward = args[10]
    extend = multi_winsizes[-1][0]*4

    coords = [[] for _ in range(imgs.shape[0])]
    reg_pdfs = [[] for _ in range(imgs.shape[0])]
    reg_infos = [[] for _ in range(imgs.shape[0])]
    bg_means = bgs[multi_winsizes[0][0]][:, 0]
    extended_imgs = np.zeros((imgs.shape[0], imgs.shape[1] + extend, imgs.shape[2] + extend))
    extended_imgs[:, int(extend/2):int(extend/2) + imgs.shape[1], int(extend/2):int(extend/2) + imgs.shape[2]] += imgs
    extended_imgs_copy = extended_imgs.copy()
    extended_imgs = image_pad.add_block_noise(extended_imgs, extend)

    while True:
        all_crop_imgs = {ws[0]: None for ws in single_winsizes}
        win_s_dict = {}
        for ws in single_winsizes:
            win_s_dict[ws[0]] = []

        if pass_to_multi:
            for df_loop in range(deflation_loop_backward):
                h_maps = []
                for g_grid, window_size in zip(b_gauss_grids, multi_winsizes):
                    crop_imgs = image_pad.image_cropping(extended_imgs, extend, window_size[0], window_size[1], shift=shift)
                    bg_squared_sums = window_size[0] * window_size[1] * bg_means ** 2
                    c = np.array(image_pad.likelihood(crop_imgs, g_grid, bg_squared_sums, bg_means, window_size[0], window_size[1]))
                    h_maps.append(c.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2]) * (multi_winsizes[0][0]**2 / window_size[0]**2))
                h_maps = np.array(h_maps)
                back_indices = [[] for _ in range(multi_thresholds.shape[1])]
                for backward_index in range(multi_thresholds.shape[1]-1, -1, -1):
                    back_indices[backward_index] = region_max_filter2(h_maps[backward_index], multi_winsizes[backward_index],
                                                                      multi_thresholds[:, backward_index], detect_range=1)
                reregress_indice = indice_filtering(back_indices, multi_winsizes, imgs.shape, int(extend/2))
                regress_imgs_copy = extended_imgs.copy()
                for regress_comp_set in reregress_indice:
                    loss_vals = []
                    selected_dt = []
                    for win_s_set in regress_comp_set:
                        regress_imgs = []
                        bg_regress = []
                        for regress_index in win_s_set:
                            ws = regress_index[3]
                            regress_imgs.append(regress_imgs_copy[regress_index[0],
                                                regress_index[1] - int((regress_index[3] - 1) / 2):regress_index[1] + int(
                                                    (regress_index[3] - 1) / 2) + 1,
                                                regress_index[2] - int((regress_index[3] - 1) / 2):regress_index[2] + int(
                                                    (regress_index[3] - 1) / 2) + 1])
                            bg_regress.append(bgs[ws][regress_index[0]])
                        regress_imgs = np.array(regress_imgs)

                        if len(regress_imgs) > 0:
                            min_tmp = np.min(regress_imgs, axis=(1, 2)).reshape(-1, 1)
                            regress_imgs[:,:,0] = min_tmp
                            regress_imgs[:,:,-1] = min_tmp
                            regress_imgs[:,0,:] = min_tmp
                            regress_imgs[:,-1,:] = min_tmp
                            pdfs, xs, ys, x_vars, y_vars, amps, rhos = image_regression(regress_imgs, bg_regress,
                                                                                        (ws, ws), p0=p0, decomp_n=decomp_n)

                            penalty = 0
                            for x_var, y_var, rho, dx, dy in zip(x_vars, y_vars, rhos, xs, ys):
                                if x_var < 0 or y_var < 0 or x_var > 3*ws or y_var > 3*ws or rho > 1 or rho < -1:
                                    penalty += 1e6
                            regressed_imgs = []
                            for regress_index, dx, dy in zip(win_s_set, xs, ys):
                                reged_img = regress_imgs_copy[regress_index[0],
                                            regress_index[1] - int((ws - 1) / 2) + int(np.round(dy)):
                                            regress_index[1] + int((ws - 1) / 2) + int(np.round(dy)) + 1,
                                            regress_index[2] - int((ws - 1) / 2) + int(np.round(dx)):
                                            regress_index[2] + int((ws - 1) / 2) + int(np.round(dx)) + 1]
                                if reged_img.shape == (ws, ws):
                                    regressed_imgs.append(reged_img)
                            regressed_imgs = np.array(regressed_imgs)
                            selected_dt.append([pdfs, xs, ys, x_vars, y_vars, rhos, amps])
                            if regressed_imgs.shape != (pdfs.reshape(regress_imgs.shape)).shape:
                                loss_vals.append(penalty)
                            else:
                                loss = np.mean(np.sort(np.mean(np.log(abs(regressed_imgs - pdfs.reshape(regress_imgs.shape))), axis=0).
                                                       flatten())[::-1][:multi_winsizes[0][0] * multi_winsizes[0][1]]) + penalty
                                loss_vals.append(loss)
                        else:
                            selected_dt.append([0, 0, 0, 0, 0, 0, 0])
                            loss_vals.append(1e3)
                    if np.sum(np.array(loss_vals) < 0.) >= 1:
                        selec_arg = np.argmin(loss_vals)
                        pdfs, xs, ys, x_vars, y_vars, rhos, amps = selected_dt[selec_arg]
                        infos = np.array(regress_comp_set[selec_arg])
                        ns = infos[:, 0]
                        rs = infos[:, 1]
                        cs = infos[:, 2]
                        ws = infos[:, 3][0]
                        for n, r, c, dx, dy, pdf, x_var, y_var, rho, amp in zip(ns, rs, cs, xs, ys, pdfs, x_vars, y_vars, rhos, amps):
                            r -= int(extend/2)
                            c -= int(extend/2)
                            if r+dy <= -1 or r+dy >= imgs.shape[1] or c+dx <= -1 or c+dx >= imgs.shape[2]:
                                continue
                            row_coord = max(0, min(r+dy, imgs.shape[1]-1))
                            col_coord = max(0, min(c+dx, imgs.shape[2]-1))
                            coords[n].append([row_coord, col_coord])
                            reg_pdfs[n].append(pdf)
                            reg_infos[n].append([x_var, y_var, rho, amp])
                        if df_loop < deflation_loop_backward - 1:
                            del_indices = np.round(np.array([ns, rs+ys, cs+xs])).astype(int).T
                            extended_imgs = subtract_pdf(extended_imgs, pdfs, del_indices, (ws, ws), bg_means, extend=0)
            return coords, reg_pdfs, reg_infos

        else:
            h_maps = []
            for g_grid, window_size in zip(f_gauss_grids, single_winsizes):
                crop_imgs = image_pad.image_cropping(extended_imgs, extend, window_size[0], window_size[1], shift=shift)
                all_crop_imgs[window_size[0]] = crop_imgs
                bg_squared_sums = window_size[0] * window_size[1] * bg_means**2
                c = image_pad.likelihood(crop_imgs, g_grid, bg_squared_sums, bg_means, window_size[0], window_size[1])
                h_map = mapping(c, imgs.shape, shift)
                h_map = h_map * (single_winsizes[0][0]**2 / window_size[0]**2)
                h_maps.append(h_map)
            h_maps = np.array(h_maps)      
            indices = region_max_filter(h_maps, single_winsizes, single_thresholds, detect_range=1)
            if len(indices) != 0:
                for n, r, c, ws in indices:
                    win_s_dict[ws].append([all_crop_imgs[ws][n]
                                           [int((r//shift) * (imgs.shape[2]//shift) + (c//shift))],
                                           bgs[ws][n], n, r, c])
                ws = single_winsizes[0][0]
                if len(win_s_dict[ws]) != 0:
                    err_indice = []
                    regress_imgs = []
                    bg_regress = []
                    ns = []
                    rs = []
                    cs = []
                    for i1, i2, i3, i4, i5 in win_s_dict[ws]:
                        tmp = np.array(i1).reshape(ws, ws)
                        min_tmp = np.min(tmp)
                        tmp[:,0] = min_tmp
                        tmp[:,-1] = min_tmp
                        tmp[0,:] = min_tmp
                        tmp[-1,:] = min_tmp
                        i1 = tmp.reshape(-1)
                        regress_imgs.append(i1)
                        bg_regress.append(i2)
                        ns.append(i3)
                        rs.append(i4)
                        cs.append(i5)
                    pdfs, xs, ys, x_vars, y_vars, amps, rhos = image_regression(regress_imgs, bg_regress,
                                                                                (ws, ws), p0=p0, decomp_n=decomp_n)
                    for err_i, (x_var, y_var, rho) in enumerate(zip(x_vars, y_vars, rhos)):
                        if x_var < 0 or y_var < 0 or x_var > 3*ws or y_var > 3*ws or rho > 1 or rho < -1:
                            err_indice.append(err_i)

                    if len(err_indice) == len(pdfs):
                        print(f'IMPOSSIBLE REGRESSION(MINUS VAR): {err_indice}\nWindow_size:{ws}')
                        pass_to_multi = True
                    else:
                        pdfs = np.delete(pdfs, err_indice, 0)
                        xs = np.delete(xs, err_indice, 0)
                        ys = np.delete(ys, err_indice, 0)
                        x_vars = np.delete(x_vars, err_indice, 0)
                        y_vars = np.delete(y_vars, err_indice, 0)
                        ns = np.delete(ns, err_indice, 0)
                        rs = np.delete(rs, err_indice, 0)
                        cs = np.delete(cs, err_indice, 0)
                        for n, r, c, dx, dy, pdf, x_var, y_var, rho, amp in zip(ns, rs, cs, xs, ys, pdfs, x_vars, y_vars, rhos, amps):
                            if r+dy <= -1 or r+dy >= imgs.shape[1] or c+dx <= -1 or c+dx >= imgs.shape[2]:
                                continue
                            row_coord = max(0, min(r+dy, imgs.shape[1]-1))
                            col_coord = max(0, min(c+dx, imgs.shape[2]-1))
                            coords[n].append([row_coord, col_coord])
                            reg_pdfs[n].append(pdf)
                            reg_infos[n].append([x_var, y_var, rho, amp])
                        del_indices = np.round(np.array([ns, rs+ys, cs+xs])).astype(int).T

                        extended_imgs_copy = extended_imgs.copy()
                        extended_imgs = subtract_pdf(extended_imgs, pdfs, del_indices, (ws, ws), bg_means, extend)

            if np.allclose(extended_imgs_copy, extended_imgs, atol=1e-2) or len(indices) == 0 or single_winsizes[0][0] not in indices[:, 3]:
                pass_to_multi = True


def mapping(c_likelihood, imgs_shape, shift):
    if shift == 1:
        return np.array(c_likelihood).reshape(imgs_shape[0], imgs_shape[1], imgs_shape[2])
    else:
        h_map = np.zeros_like(imgs_shape)
        index = 0
        for row in range(0, h_map.shape[1], shift):
            for col in range(0, h_map.shape[2], shift):
                h_map[:, row, col] = c_likelihood[:, index, 0]
                index += 1
        return h_map


def quantification(window_size):
    x = np.arange(-(window_size[0]-1)/2, (window_size[0]+1)/2)
    y = np.arange(-(window_size[1]-1)/2, (window_size[1]+1)/2)
    xv, yv = np.meshgrid(x, y, sparse=True)
    grid = np.stack(np.meshgrid(xv, yv), -1).reshape(window_size[0] * window_size[1], 2)
    return grid


def bi_variate_normal_pdf(xy, cov, mu, normalization=True):
    a = np.ones((cov.shape[0], xy.shape[0], xy.shape[1]), dtype=np.longdouble) * (xy - mu)
    if normalization:
        return (np.exp((-1./2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2))
                / (2 * np.pi * np.sqrt(np.linalg.det(cov).reshape(-1, 1))))
    else:
        return (np.exp((-1./2) * np.sum(a @ np.linalg.inv(cov) * a, axis=2)))


def image_regression(imgs, bgs, window_size, p0, decomp_n, amp=0, repeat=5):
    imgs = np.array(imgs).reshape([-1, window_size[0] * window_size[1]])
    bgs = np.array(bgs).reshape([-1, window_size[0] * window_size[1]])
    p0 = np.array(p0)
    x_grid = (np.array([list(np.arange(-int(window_size[0]/2), int((window_size[0]/2) + 1), 1))] * window_size[1])
              .reshape(-1, window_size[0] * window_size[1]))
    y_grid = (np.array([[y] * window_size[0] for y in range(-int(window_size[1]/2), int((window_size[1]/2) + 1), 1)])
              .reshape(-1, window_size[0] * window_size[1]))
    grid = quantification(window_size)

    coefs = regression.guo_algorithm(imgs, bgs, p0=p0, xgrid=x_grid, ygrid=y_grid, 
                                     window_size=window_size, repeat=repeat, decomp_n=decomp_n)
    variables, err_indices = regression.unpack_coefs(coefs, window_size)

    if len(err_indices) > 0:
        coefs = regression.guo_algorithm(imgs, bgs, p0=p0, xgrid=x_grid, ygrid=y_grid,
                                         window_size=window_size, repeat=repeat+1, decomp_n=decomp_n)
        variables, err_indices = regression.unpack_coefs(coefs, window_size)
    variables = np.array(variables).T
    cov_mat = np.array([variables[:, 0], variables[:, 4] * np.sqrt(variables[:, 0]) * np.sqrt(variables[:, 2]),
                        variables[:, 4] * np.sqrt(variables[:, 0] * np.sqrt(variables[:, 2])), variables[:, 2]]
                       ).T.reshape(variables.shape[0], 2, 2)
    pdfs = bi_variate_normal_pdf(grid, cov_mat, mu=np.array([0, 0]), normalization=False)
    pdfs = variables[:, 5].reshape(-1, 1) * pdfs + bgs
    for err_i in err_indices:
        variables[err_i][0] = -100
        variables[err_i][2] = -100
    return pdfs, variables[:, 1], variables[:, 3], variables[:, 0], variables[:, 2], variables[:, 5], variables[:, 4]


def make_red_circles(original_imgs, circle_imgs, localized_xys):
    stacked_imgs = []
    for img_n, coords in enumerate(localized_xys):
        xy_cum = []
        for center_coord in coords:
            x, y = int(round(center_coord[0])), int(round(center_coord[1]))
            if (x, y) in xy_cum:
                circle_imgs[img_n] = draw_cross(circle_imgs[img_n], x, y, (0, 0, 1))
            else:
                circle_imgs[img_n] = draw_cross(circle_imgs[img_n], x, y, (1, 0, 0))
            xy_cum.append((x, y))
        stacked_imgs.append(np.hstack((original_imgs[img_n], circle_imgs[img_n])))
    return stacked_imgs


def visualilzation(output_dir, images, localized_xys):
    orignal_imgs_3ch = np.array([images.copy(), images.copy(), images.copy()])
    orignal_imgs_3ch = np.ascontiguousarray(np.moveaxis(orignal_imgs_3ch, 0, 3))
    circle_imgs = orignal_imgs_3ch.copy()
    stacked_img = np.array(make_red_circles(orignal_imgs_3ch, circle_imgs, localized_xys))
    tifffile.imwrite(f'{output_dir}_locvideo.tiff', data=(stacked_img * 255).astype(np.uint8), imagej=True)


def background(imgs, window_sizes, alpha):
    bins = 0.01
    bgs = {}
    bg_means = []
    bg_stds = []
    bg_intensities = (imgs.reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2]) * 100).astype(np.uint8) / 100
    for i in range(len(bg_intensities)):
        args = np.arange(len(bg_intensities[i]))
        post_mask_args = args.copy()
        for _ in range(3):
            it_hist, bin_width = np.histogram(bg_intensities[i][post_mask_args],
                                              bins=np.arange(0, np.max(bg_intensities[i][post_mask_args]) + bins, bins))
            mask_sums_mode = (np.argmax(it_hist) * bins + (bins / 2))
            mask_std = np.std(bg_intensities[i][post_mask_args])
            post_mask_args = np.array([arg for arg, val in zip(args, bg_intensities[i]) if
                                       (mask_sums_mode - 3. * mask_std) < val < (mask_sums_mode + 3. * mask_std)])
        it_data = bg_intensities[i][post_mask_args]
        bg_means.append(np.mean(it_data))
        bg_stds.append(np.std(it_data))
    bg_means = np.array(bg_means)
    bg_stds = np.array(bg_stds)

    for window_size in window_sizes:
        bg = np.ones((bg_intensities.shape[0], window_size[0] * window_size[1]))
        bg *= bg_means.reshape(-1, 1)
        bgs[window_size[0]] = bg
    return bgs, bg_stds / bg_means / alpha


def intensity_distribution(images, reg_pdfs, xy_coords, reg_infos, sigma=3.5):
    new_pdfs = []
    new_coords = []
    new_infos = []
    for img_n, (pdfs, xy_coord, infos) in enumerate(zip(reg_pdfs, xy_coords, reg_infos)):
        if len(pdfs) < 1:
            continue
        new_pdf_tmp = pdfs.copy()
        new_xy_coord_tmp = xy_coord.copy()
        new_reg_tmp = infos.copy()
        max_pdf_vals = []

        for pdf, xy, info in zip(pdfs, xy_coord, infos):
            max_pdf_vals.append(np.max(images[img_n,
                                       max(0, int(np.round(xy[0])) - 1): min(images[img_n].shape[0], int(np.round(xy[0])) + 2),
                                       max(0, int(np.round(xy[1])) - 1): min(images[img_n].shape[1], int(np.round(xy[1])) + 2)]))

        max_pdf_vals = np.array(max_pdf_vals)
        bin_edgs = np.arange(0, np.max(max_pdf_vals) + 0.05, 0.025)
        max_pdf_vals_hist = np.histogram(max_pdf_vals, bins=bin_edgs)
        mode_sigma = (bin_edgs[:-1] + 0.025)[np.argmax(max_pdf_vals_hist[0])] + sigma * np.std(max_pdf_vals)

        for i, max_pdf_val in enumerate(max_pdf_vals):
            if max_pdf_val > mode_sigma:
                new_pdf_tmp.append(pdfs[i])
                new_xy_coord_tmp.append(xy_coord[i])
                new_reg_tmp.append(infos[i])
        new_pdfs.append(new_pdf_tmp)
        new_coords.append(new_xy_coord_tmp)
        new_infos.append(new_reg_tmp)
    return new_pdfs, new_coords, new_infos


def params_gen(win_s):
    assert type(win_s) is int
    if win_s < 5:
        win_s = 5
    if win_s % 2 == 0:
        win_s += 1

    single_winsizes = [(win_s, win_s)]
    multi_winsizes = [(win_s-2, win_s-2), (win_s, win_s), (win_s+2, win_s+2)]
    single_radius = [((r[0]//2) / 2.) for r in single_winsizes]
    multi_radius = [((r[0]//2) / 2.) for r in multi_winsizes]
    return single_winsizes, single_radius, multi_winsizes, multi_radius


def main_process(imgs, forward_gauss_grids, backward_gauss_grids, *args):
    args = list(args)
    bgs, thresholds = background(imgs, window_sizes=args[3], alpha=args[9])
    if args[2] is None:
        args[2] = np.array([thresholds for _ in range(len(args[0]))]).T
    else:
        args[2] = np.ones((len(imgs), len(args[0]))) * args[2]
    if args[5] is None:
        args[5] = np.array([thresholds for _ in range(len(args[3]))]).T
    else:
        args[5] = np.ones((len(imgs), len(args[3]))) * args[5]

    before_time = timer()
    xy_coord, pdf, info = localization(imgs, bgs, forward_gauss_grids, backward_gauss_grids, *args)
    return xy_coord, pdf, info


if __name__ == '__main__':
    params = read_parameters('./andi2_config.txt')
    images = np.array(check_video_ext(params['localization']['VIDEO'], andi2=True)[0][1:])
    
    OUTPUT_DIR = params['localization']['OUTPUT_DIR']
    OUTPUT_LOC = f'{OUTPUT_DIR}/{params["localization"]["VIDEO"].split("/")[-1].split(".tif")[0]}'

    SIGMA = params['localization']['SIGMA']
    WINSIZE = params['localization']['WINSIZE']
    THRES_ALPHA = params['localization']['THRES_ALPHA']
    DEFLATION_LOOP_IN_BACKWARD = params['localization']['DEFLATION_LOOP_IN_BACKWARD']
    BINARY_THRESHOLDS = None
    MULTI_THRESHOLDS = None

    PARALLEL = params['localization']['PARALLEL']
    CORE = params['localization']['CORE']
    DIV_Q = params['localization']['DIV_Q']
    SHIFT = params['localization']['SHIFT']
    GAUSS_SEIDEL_DECOMP = 1
    visualization = params['localization']['LOC_VISUALIZATION']
    P0 = [1.5, 0., 1.5, 0., 0., 0.5]

    xy_coords = []
    reg_pdfs = []
    reg_infos = []
    SINGLE_WINSIZES, SINGLE_RADIUS, MULTI_WINSIZES, MULTI_RADIUS = params_gen(WINSIZE)
    forward_gauss_grids = gauss_psf(SINGLE_WINSIZES, SINGLE_RADIUS)
    backward_gauss_grids = gauss_psf(MULTI_WINSIZES, MULTI_RADIUS)

    start_time = timer()
    if PARALLEL:
        for div_q in range(0, len(images), CORE * DIV_Q):
            print(f'{div_q}/{len(images)} frame (parallelized)')
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executors = {i: None for i in range(CORE)}
                for cc in range(CORE):
                    if div_q + cc*DIV_Q < len(images):
                        future = executor.submit(main_process, images[div_q + cc*DIV_Q: div_q + cc*DIV_Q + DIV_Q],
                                                 forward_gauss_grids, backward_gauss_grids,
                                                 SINGLE_WINSIZES, SINGLE_RADIUS, BINARY_THRESHOLDS,
                                                 MULTI_WINSIZES, MULTI_RADIUS, MULTI_THRESHOLDS,
                                                 P0, SHIFT, GAUSS_SEIDEL_DECOMP, THRES_ALPHA, DEFLATION_LOOP_IN_BACKWARD)
                        executors[cc] = future
                for wait_ in executors:
                    if type(executors[wait_]) is concurrent.futures.Future:
                        xy_coord, pdf, info = executors[wait_].result()
                        xy_coords.extend(xy_coord)
                        reg_pdfs.extend(pdf)
                        reg_infos.extend(info)
    else:
        for div_q in range(0, len(images), DIV_Q):
            print(f'{div_q}/{len(images)} frame (non parallelized)')
            xy_coord, pdf, info = main_process(images[div_q:div_q+DIV_Q], forward_gauss_grids, backward_gauss_grids,
                                               SINGLE_WINSIZES, SINGLE_RADIUS, BINARY_THRESHOLDS,
                                               MULTI_WINSIZES, MULTI_RADIUS, MULTI_THRESHOLDS,
                                               P0, SHIFT, GAUSS_SEIDEL_DECOMP, THRES_ALPHA, DEFLATION_LOOP_IN_BACKWARD)
            xy_coords.extend(xy_coord)
            reg_pdfs.extend(pdf)
            reg_infos.extend(info)

    reg_pdfs, xy_coords, reg_infos = intensity_distribution(images, reg_pdfs, xy_coords, reg_infos, sigma=SIGMA)
    write_localization(OUTPUT_LOC, xy_coords, reg_pdfs, reg_infos)

    if visualization:
        print(f'Visualizing localizations...')
        visualilzation(OUTPUT_LOC, images, xy_coords)
    print(f'{"Total time":<35}:{(timer() - start_time):.2f}s')