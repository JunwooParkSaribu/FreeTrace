import numpy as np
from tensorflow.keras.models import load_model # type: ignore


class RegModel:
    def __init__(self, reg_model_nums):
        self.reg_model_nums = reg_model_nums
        self.alpha_models = None
        self.k_model = None
        self.load_models(self.reg_model_nums)

    def load_models(self, model_nums):
        self.alpha_models = {n: load_model(f'{__file__.split('FreeTrace')[0]}/FreeTrace/models/reg_model_{n}.keras') for n in model_nums}
        self.k_model = load_model(f'{__file__.split('FreeTrace')[0]}/FreeTrace/models/reg_k_model.keras')

        for n in model_nums:
            self.alpha_models[n].compile()
        self.k_model.compile()

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        input_signals = []
        model_num = self.model_selection(x.shape[0])
        re_couped_x, re_couped_y = self.recoupe_trajectory(x, y, model_num)
        for r_x, r_y in zip(re_couped_x, re_couped_y):
            input_signal1, input_signal2 = self.cvt_2_signal(r_x, r_y)
            input_signals.append(input_signal1)
            input_signals.append(input_signal2)
        input_signals = np.reshape(input_signals, [-1, model_num, 1, 3])
        return input_signals, model_num

    def alpha_predict(self, inputs):
        if len(inputs) == 2:
            if len(inputs[0]) < 5:
                return 1.0
            alpha_signal, model_num = self.call(inputs)
            pred_alphas = self.alpha_models[model_num].predict_on_batch(alpha_signal)
            if pred_alphas.shape[0] > 4:
                pred_alpha = np.mean(np.quantile(pred_alphas, q=[0.25, 0.75], method='normal_unbiased'))
            else:
                pred_alpha = np.mean(pred_alphas)
            return pred_alpha
        elif len(inputs) == 3:
            if len(inputs[0]) < 5:
                return 1.0
            alpha_preds = []
            for inputs_ in inputs:
                alpha_signal, model_num = self.call(inputs_)
                pred_alphas = self.alpha_models[model_num].predict_on_batch(alpha_signal)
                if pred_alphas.shape[0] > 4:
                    pred_alpha = np.mean(np.quantile(pred_alphas, q=[0.25, 0.75], method='normal_unbiased'))
                else:
                    pred_alpha = np.mean(pred_alphas)
                alpha_preds.append(pred_alpha)
            return alpha_preds

    def k_predict(self, inputs):
        log_disps = []
        for input_ in inputs:
            log_disps.append(self.log_displacements(input_[0], input_[1]))
        k_preds = self.k_model.predict_on_batch(np.array(log_disps))
        return k_preds.flatten()

    def model_selection(self, length):
        index = 0
        crits = [8, 12, 18, 36, 72, 140]
        while True:
            if length < crits[index]:
                return self.reg_model_nums[index]
            index += 1
            if index >= len(crits):
                return self.reg_model_nums[-1]

    def log_displacements(self, xs, ys):
        disps = self.displacement(xs, ys)
        if xs.shape[0] < 10:
            return np.log10(np.mean(disps))
        else:
            return np.log10(np.mean(np.quantile(disps, q=[0.25, 0.75], method='normal_unbiased')))

    def recoupe_trajectory(self, x, y, model_num, jump=1):
        couped_x = []
        couped_y = []
        for i in range(0, x.shape[0], jump):
          tmp1 = x[i: i + model_num]
          tmp2 = y[i: i + model_num]
          if tmp1.shape[0] == model_num:
              couped_x.append(tmp1)
              couped_y.append(tmp2)
        return couped_x, couped_y

    def cvt_2_signal(self, x, y):
        rad_list, xs_raw, ys_raw = self.make_alpha_inputs(x, y)
        x = x / (np.std(x))
        x = np.cumsum(self.abs_subtraction(x)) / x.shape[0]
        y = y / (np.std(y))
        y = np.cumsum(self.abs_subtraction(y)) / y.shape[0]
        return np.transpose(np.stack((x, rad_list, xs_raw))), np.transpose(np.stack((y, rad_list, ys_raw)))

    def abs_subtraction(self, xs):
        assert xs.ndim == 1
        uncum_list = [0.]
        for i in range(1, xs.shape[0]):
            uncum_list.append(abs(xs[i] - xs[i-1]))
        return uncum_list

    def make_alpha_inputs(self, xs, ys):
        assert xs.ndim == 1 and ys.ndim == 1
        rad_list = self.radius(xs, ys)
        disp_list = self.displacement(xs, ys)
        return (rad_list / np.mean(disp_list) / xs.shape[0],
                ((xs - xs[0]) / np.mean(disp_list) / xs.shape[0]),
                ((ys - ys[0]) / np.mean(disp_list) / ys.shape[0]))

    def radius(self, xs, ys):
        rad_list = [0.]
        for i in range(1, xs.shape[0]):
            rad_list.append(np.sqrt((xs[i] - xs[0])**2 + (ys[i] - ys[0])**2))
        return rad_list

    def displacement(self, xs, ys):
        disp_list = []
        for i in range(1, xs.shape[0]):
            disp_list.append(np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2))
        return disp_list

    def get_reg_model_nums(self):
        return self.reg_model_nums
