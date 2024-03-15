import os
import sys
# sys.path.append('./')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import nn
import sympy
from sympy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 24
import matplotlib.ticker as ticker


def error(method):
    s_1 = 0.5
    s_2 = 0.5
    u_1 = 0.5
    u_2 = 1
    rho = 0.5

    if method == 'gaussian':
        alpha = 0.1
        gaussian_c = (2 * sympy.pi * s_1 * s_2 * (1 - rho ** 2) ** (0.5)) ** (-1)
        gaussian_u = (t - u_1) ** 2 / s_1 ** 2 - 2 * rho * (t - u_1) * (x - u_2) / (s_1 * s_2) + (
                x - u_2) ** 2 / s_2 ** 2
        gaussian_e = sympy.E ** (-0.5 * (1 - rho ** 2) ** (-1) * gaussian_u)
        gaussian = gaussian_c * gaussian_e
        expre_error = alpha * gaussian

    elif method == 'linear':
        alpha = 0.025
        expre_error = alpha * (t + x)

    elif method == 'hyperpara':
        alpha = 0.1
        expre_error = alpha * (t ** 2 / 1 ** 2 - x ** 2 / 2 ** 2)

    elif method == 'sin':
        alpha = 0.2
        expre_error = alpha * sympy.sin(t - 0.5) * sympy.cos(x - 1)

    elif method == 'power':
        alpha = 0.1
        expre_error = alpha * (1 - (t - 0.5) ** 2 * (x - 1) ** 2)
    return expre_error

class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        self.size = x.size()
        mymin = torch.min(x.view(self.size[0], -1), 1, keepdim=True)[0]
        mymax = torch.max(x.view(self.size[0], -1), 1, keepdim=True)[0]

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

def get_mask(loc_t, loc_x, loc_size):
    mask = torch.zeros((loc_size[-2], loc_size[-1]), dtype=torch.float64)
    loc_t = loc_t.flatten().tolist()
    loc_x = loc_x.flatten().tolist()
    mask[loc_x, loc_t] = 1
    return mask

def model_load(hf, lf, model_decomp_path, model_fusion_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loc_t = np.array([2, 13, 24, 35, 46])
    loc_x = np.array([1, 13, 25, 37, 49, 61, 73, 85, 97])
    loc_T, loc_X = np.meshgrid(loc_t, loc_x)

    hf = torch.tensor(hf).unsqueeze(0).to(device)
    lf = torch.tensor(lf).unsqueeze(0).to(device)

    lf_normalizer = RangeNormalizer(lf)
    lf_input = lf_normalizer.encode(lf)
    hf_input = lf_normalizer.encode(hf)

    sensors_mask = get_mask(loc_T, loc_X, hf.shape).to(device)
    sensors_input_hf = hf_input * sensors_mask

    torch.set_default_dtype(torch.float64)
    model_decomp = torch.load(model_decomp_path)
    model_fusion = torch.load(model_fusion_path)

    hf_pred = model_fusion(model_decomp(lf_input), sensors_input_hf)
    hf_pred = lf_normalizer.decode(hf_pred)

    return np.array(hf_pred.squeeze(0).detach().cpu())

def plot_field(data, save_path):
    m, n = np.linspace(0, 1, 50), np.linspace(0, 2, 100)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(6, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=1, vmin=-0.2)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig_1 = plt.contour(M, N, data, colors='black', levels=8, linewidth=0.2)
    plt.clabel(fig_1, inline=True, fontsize=15)

    loc_t = np.array([2, 13, 24, 35, 46])
    loc_x = np.array([1, 13, 25, 37, 49, 61, 73, 85, 97])
    loc_T, loc_X = np.meshgrid(loc_t, loc_x)
    loc_T = loc_T / 50
    loc_X = loc_X / 100 * 2
    data_max = np.max(data)
    data_min = np.min(data)
    x_max, y_max = np.where(data == np.max(data))
    x_min, y_min = np.where(data == np.min(data))
    x_max = x_max / 50
    y_max = y_max / 100 * 2
    x_min = x_min / 50
    y_min = y_min / 100 * 2
    plt.scatter(loc_T, loc_X, color='white', s=20)
    plt.scatter(y_max, x_max, color='white', s=200, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=20, xy=(y_max, x_max), xytext=(y_max + 0.2, x_max + 0.2),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))
    plt.scatter(y_min, x_min, color='white', s=200, marker='*')
    plt.annotate(f'Min: {data_min:.2f}', fontsize=20, xy=(y_min, x_min), xytext=(y_min - 0.2, x_min + 0.4),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    cb1 = plt.colorbar(im_1, fraction=0.08, pad=0.03, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    plt.xticks([0, 0.5, 1], fontsize=24)
    plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=24)

    plt.xlabel('t', fontsize=24)
    plt.ylabel('x', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_error_init(data, method, save_path):
    m, n = np.linspace(0, 1, 50), np.linspace(0, 2, 100)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(6, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    # norm_1 = mpl.colors.Normalize(vmax=1, vmin=-0.2)
    # im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    # fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100)

    cb1 = plt.colorbar(fraction=0.08, pad=0.03)
    tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    if method == 'sin':
        data_min = np.min(data)
        x_min, y_min = np.where(data == np.min(data))
        x_min = x_min[0] / 50
        y_min = y_min[0] / 100 * 2
        plt.scatter(y_min, x_min, color='white', s=200, marker='*')
        plt.annotate(f'Min: {data_min:.2f}', fontsize=20, xy=(y_min, x_min), xytext=(y_min + 0.1, x_min + 0.4),
                     arrowprops=dict(facecolor='white', shrink=0.1),
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

        data_max = np.max(data)
        x_max, y_max = np.where(data == np.max(data))
        x_max = x_max[0] / 50
        y_max = y_max[0] / 100 * 2
        plt.scatter(y_max, x_max, color='white', s=200, marker='*')
        plt.annotate(f'Max: {data_max:.2f}', fontsize=20, xy=(y_max, x_max), xytext=(y_max - 0.5, x_max - 0.4),
                     arrowprops=dict(facecolor='white', shrink=0.1),
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))
    else:
        data_max = np.max(data)
        x_max, y_max = np.where(data == np.max(data))
        x_max = x_max[0] / 50
        y_max = y_max[0] / 100 * 2
        plt.scatter(y_max, x_max, color='white', s=200, marker='*')
        plt.annotate(f'Max: {data_max:.2f}', fontsize=20, xy=(y_max, x_max), xytext=(y_max - 0.4, x_max + 0.4),
                     arrowprops=dict(facecolor='white', shrink=0.1),
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.5, 1], fontsize=24)
    plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=24)

    plt.xlabel('t', fontsize=24)
    plt.ylabel('x', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_error(data, method, save_path):
    m, n = np.linspace(0, 1, 50), np.linspace(0, 2, 100)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(6.1, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    # norm_1 = mpl.colors.Normalize(vmax=1, vmin=-0.2)
    # im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    # fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb1 = plt.colorbar(fraction=0.08, pad=0.03, format=fmt)
    tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    loc_t = np.array([2, 13, 24, 35, 46])
    loc_x = np.array([1, 13, 25, 37, 49, 61, 73, 85, 97])
    loc_T, loc_X = np.meshgrid(loc_t, loc_x)
    loc_T = loc_T / 50
    loc_X = loc_X / 100 * 2
    plt.scatter(loc_T, loc_X, color='white', s=20)

    data_max = np.max(data)
    x_max, y_max = np.where(data == np.max(data))
    x_max = x_max[0] / 50
    y_max = y_max[0] / 100 * 2
    if method == 'power':
        y_text = y_max + 0.1
        x_text = x_max - 0.4
    else:
        y_text = y_max + 0.1
        x_text = x_max + 0.4
    plt.scatter(y_max, x_max, color='white', s=200, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=20, xy=(y_max, x_max), xytext=(y_text, x_text),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.5, 1], fontsize=24)
    plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=24)

    plt.xlabel('t', fontsize=24)
    plt.ylabel('x', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_grad(data, save_path, vmax, vmin, ticks):
    m, n = np.linspace(0, 1, 48), np.linspace(0, 2, 98)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(6.2, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=vmax, vmin=vmin)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)

    cb1 = plt.colorbar(im_1, fraction=0.08, pad=0.03, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks(ticks)
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    plt.xticks([0, 0.5, 1], fontsize=24)
    plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=24)

    plt.xlabel('t', fontsize=24)
    plt.ylabel('x', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_grad_error(data, save_path):
    m, n = np.linspace(0, 1, 48), np.linspace(0, 2, 98)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(6.2, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=0.42, vmin=-0.22)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)

    cb1 = plt.colorbar(im_1, fraction=0.08, pad=0.03, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([-0.22, -0.06, 0.1, 0.26, 0.42])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    data_max = np.max(data)
    data_min = np.min(data)
    x_max, y_max = np.where(data == np.max(data))
    x_min, y_min = np.where(data == np.min(data))
    x_max = x_max / 50
    y_max = y_max / 100 * 2
    x_min = x_min / 50
    y_min = y_min / 100 * 2
    plt.scatter(y_max, x_max, color='white', s=200, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=20, xy=(y_max, x_max), xytext=(y_max - 0.2, x_max - 0.3),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))
    plt.scatter(y_min, x_min, color='white', s=200, marker='*')
    plt.annotate(f'Min: {data_min:.2f}', fontsize=20, xy=(y_min, x_min), xytext=(y_min + 0.2, x_min + 0.2),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    # fig = plt.contourf(M, N, data, cmap='jet', levels=100)
    # plt.colorbar(fraction=0.08, pad=0.03)

    plt.xticks([0, 0.5, 1], fontsize=24)
    plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=24)

    plt.xlabel('t', fontsize=24)
    plt.ylabel('x', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def func(f, num_t, num_x):
    t, x = np.linspace(0, 1, num_t, endpoint=True), np.linspace(0, 2, num_x, endpoint=True)
    T, X = np.meshgrid(t, x)

    data = f(T, X)
    return data


if __name__ == '__main__':
    my_mseloss = nn.MSELoss()
    my_l1loss = nn.L1Loss()
    x = symbols('x')
    t = symbols('t')
    num_t_hf, num_x_hf = 50, 100

    a = 1
    k = 0.2
    expre = sympy.E ** (-4 * k ** 2 * sympy.pi ** 2 * t) * a * sympy.sin(2 * sympy.pi * k * (x - t))
    expre_power = expre + error('power')
    expre_sin = expre + error('sin')
    expre_gaussian = expre + error('gaussian')

    f = lambdify([t, x], expre, "numpy")
    f_power = lambdify([t, x], expre_power, "numpy")
    f_sin = lambdify([t, x], expre_sin, "numpy")
    f_gaussian = lambdify([t, x], expre_gaussian, "numpy")

    hf = func(f, num_t_hf, num_x_hf)
    lf_power = func(f_power, num_t_hf, num_x_hf)
    lf_sin = func(f_sin, num_t_hf, num_x_hf)
    lf_gaussian = func(f_gaussian, num_t_hf, num_x_hf)

    pred_power = model_load(hf, lf_power, 'model/fine_decomp_power_fno.pth', 'model/fine_fusion_power_fno.pth')
    pred_sin = model_load(hf, lf_sin, 'model/fine_decomp_sin_fno.pth', 'model/fine_fusion_sin_fno.pth')
    pred_gaussian = model_load(hf, lf_gaussian, 'model/fine_decomp_gaussian_fno.pth', 'model/fine_fusion_gaussian_fno.pth')

    mse_power = my_mseloss(pred_power, f_power)
    l1_power = my_l1loss(pred_power, f_power)
    max_power = torch.max(torch.abs(pred_power - f_power))
    rmse_power = torch.sqrt(mse_power)
    r2_power = 1 - (torch.sum((pred_power - f_power) ** 2)) / (torch.sum((torch.mean(f_power) - f_power) ** 2))

    mse_sin = my_mseloss(pred_sin, f_sin)
    l1_sin = my_l1loss(pred_sin, f_sin)
    max_sin = torch.max(torch.abs(pred_sin - f_sin))
    rmse_sin = torch.sqrt(mse_sin)
    r2_sin = 1 - (torch.sum((pred_sin - f_sin) ** 2)) / (torch.sum((torch.mean(f_sin) - f_sin) ** 2))

    mse_gaussian = my_mseloss(pred_gaussian, f_gaussian)
    l1_gaussian = my_l1loss(pred_gaussian, f_gaussian)
    max_gaussian = torch.max(torch.abs(pred_gaussian - f_gaussian))
    rmse_gaussian = torch.sqrt(mse_gaussian)
    r2_gaussian = 1 - (torch.sum((pred_gaussian - f_gaussian) ** 2)) / (torch.sum((torch.mean(f_gaussian) - f_gaussian) ** 2))
