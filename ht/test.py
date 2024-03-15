import numpy as np
import torch
import sympy
from sympy import *
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def sensors(method):
    if method == 'uniform9':
        loc_x = np.array([24, 60, 96])
        loc_y = np.array([24, 60, 96])
        loc_X, loc_Y = np.meshgrid(loc_x, loc_y)
        loc_X = loc_X.reshape(-1, 1)
        loc_Y = loc_Y.reshape(-1, 1)
    elif method == 'uniform25':
        loc_x_0 = np.array([4, 42, 78, 116])
        loc_y_0 = np.array([4, 42, 78, 116])
        loc_x_1 = np.array([24, 60, 96])
        loc_y_1 = np.array([24, 60, 96])
        loc_X_0, loc_Y_0 = np.meshgrid(loc_x_0, loc_y_0)
        loc_X_1, loc_Y_1 = np.meshgrid(loc_x_1, loc_y_1)
        loc_X_0 = loc_X_0.reshape(-1, 1)
        loc_Y_0 = loc_Y_0.reshape(-1, 1)
        loc_X_1 = loc_X_1.reshape(-1, 1)
        loc_Y_1 = loc_Y_1.reshape(-1, 1)
        loc_X = np.concatenate((loc_X_0, loc_X_1), axis=0)
        loc_Y = np.concatenate((loc_Y_0, loc_Y_1), axis=0)
    elif method == 'uniform49':
        loc_x = np.array([4, 24, 42, 60, 78, 96, 116])
        loc_y = np.array([4, 24, 42, 60, 78, 96, 116])
        loc_X, loc_Y = np.meshgrid(loc_x, loc_y)
        loc_X = loc_X.reshape(-1, 1)
        loc_Y = loc_Y.reshape(-1, 1)
    elif method == 'selected49':
        loc_x_0 = np.array([24, 60, 96])
        loc_y_0 = np.array([24, 60, 96])
        loc_x_1 = np.array([35])
        loc_y_1 = np.array([47, 73])
        loc_x_2 = np.array([24])
        loc_y_2 = np.array([88])
        loc_x_3 = np.array([16, 32])
        loc_y_3 = np.array([16, 32])
        loc_x_4 = np.array([20, 28])
        loc_y_4 = np.array([56, 64])
        loc_x_5 = np.array([16, 32])
        loc_y_5 = np.array([88, 104])
        loc_x_6 = np.array([52, 68])
        loc_y_6 = np.array([52, 68])
        loc_x_7 = np.array([88, 104])
        loc_y_7 = np.array([88, 104])
        loc_x_8 = np.array([52, 68])
        loc_y_8 = np.array([32, 88])
        loc_x_9 = np.array([42, 78])
        loc_y_9 = np.array([42, 78])
        loc_x_10 = np.array([88])
        loc_y_10 = np.array([52, 68])
        loc_x_11 = np.array([78])
        loc_y_11 = np.array([88])
        loc_x_12 = np.array([42])
        loc_y_12 = np.array([104])
        loc_x_13 = np.array([56, 64, 92, 100])
        loc_y_13 = np.array([20])
        loc_x_14 = np.array([60])
        loc_y_14 = np.array([100])
        loc_X_0, loc_Y_0 = np.meshgrid(loc_x_0, loc_y_0)
        loc_X_1, loc_Y_1 = np.meshgrid(loc_x_1, loc_y_1)
        loc_X_2, loc_Y_2 = np.meshgrid(loc_x_2, loc_y_2)
        loc_X_3, loc_Y_3 = np.meshgrid(loc_x_3, loc_y_3)
        loc_X_4, loc_Y_4 = np.meshgrid(loc_x_4, loc_y_4)
        loc_X_5, loc_Y_5 = np.meshgrid(loc_x_5, loc_y_5)
        loc_X_6, loc_Y_6 = np.meshgrid(loc_x_6, loc_y_6)
        loc_X_7, loc_Y_7 = np.meshgrid(loc_x_7, loc_y_7)
        loc_X_8, loc_Y_8 = np.meshgrid(loc_x_8, loc_y_8)
        loc_X_9, loc_Y_9 = np.meshgrid(loc_x_9, loc_y_9)
        loc_X_10, loc_Y_10 = np.meshgrid(loc_x_10, loc_y_10)
        loc_X_11, loc_Y_11 = np.meshgrid(loc_x_11, loc_y_11)
        loc_X_12, loc_Y_12 = np.meshgrid(loc_x_12, loc_y_12)
        loc_X_13, loc_Y_13 = np.meshgrid(loc_x_13, loc_y_13)
        loc_X_14, loc_Y_14 = np.meshgrid(loc_x_14, loc_y_14)
        loc_X_0 = loc_X_0.reshape(-1, 1)
        loc_Y_0 = loc_Y_0.reshape(-1, 1)
        loc_X_1 = loc_X_1.reshape(-1, 1)
        loc_Y_1 = loc_Y_1.reshape(-1, 1)
        loc_X_2 = loc_X_2.reshape(-1, 1)
        loc_Y_2 = loc_Y_2.reshape(-1, 1)
        loc_X_3 = loc_X_3.reshape(-1, 1)
        loc_Y_3 = loc_Y_3.reshape(-1, 1)
        loc_X_4 = loc_X_4.reshape(-1, 1)
        loc_Y_4 = loc_Y_4.reshape(-1, 1)
        loc_X_5 = loc_X_5.reshape(-1, 1)
        loc_Y_5 = loc_Y_5.reshape(-1, 1)
        loc_X_6 = loc_X_6.reshape(-1, 1)
        loc_Y_6 = loc_Y_6.reshape(-1, 1)
        loc_X_7 = loc_X_7.reshape(-1, 1)
        loc_Y_7 = loc_Y_7.reshape(-1, 1)
        loc_X_8 = loc_X_8.reshape(-1, 1)
        loc_Y_8 = loc_Y_8.reshape(-1, 1)
        loc_X_9 = loc_X_9.reshape(-1, 1)
        loc_Y_9 = loc_Y_9.reshape(-1, 1)
        loc_X_10 = loc_X_10.reshape(-1, 1)
        loc_Y_10 = loc_Y_10.reshape(-1, 1)
        loc_X_11 = loc_X_11.reshape(-1, 1)
        loc_Y_11 = loc_Y_11.reshape(-1, 1)
        loc_X_12 = loc_X_12.reshape(-1, 1)
        loc_Y_12 = loc_Y_12.reshape(-1, 1)
        loc_X_13 = loc_X_13.reshape(-1, 1)
        loc_Y_13 = loc_Y_13.reshape(-1, 1)
        loc_X_14 = loc_X_14.reshape(-1, 1)
        loc_Y_14 = loc_Y_14.reshape(-1, 1)
        loc_X = np.concatenate((loc_X_0, loc_X_1, loc_X_2, loc_X_3, loc_X_4, loc_X_5, loc_X_6, loc_X_7,
                                loc_X_8, loc_X_9, loc_X_10, loc_X_11, loc_X_12, loc_X_13, loc_X_14), axis=0)
        loc_Y = np.concatenate((loc_Y_0, loc_Y_1, loc_Y_2, loc_Y_3, loc_Y_4, loc_Y_5, loc_Y_6, loc_Y_7,
                                loc_Y_8, loc_Y_9, loc_Y_10, loc_Y_11, loc_Y_12, loc_Y_13, loc_Y_14), axis=0)
    elif method == 'random49':
        np.random.seed(1)
        loc_x = np.random.randint(6, 115, size=49)
        loc_y = np.random.randint(6, 115, size=49)
        loc_X = loc_x.reshape(-1, 1)
        loc_Y = loc_y.reshape(-1, 1)
    return loc_X, loc_Y

class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        self.size = x.size()
        mymin = torch.min(x.view(self.size[0], -1), 1, keepdim=True)[0]
        mymax = torch.max(x.view(self.size[0], -1), 1, keepdim=True)[0]
        # mymin = torch.min(x.view(self.size[0], -1), 0, keepdim=True)[0]
        # mymax = torch.max(x.view(self.size[0], -1), 0, keepdim=True)[0]

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

def get_mask(loc_x, loc_y, loc_size):
    # mask = torch.zeros((loc_size[-2], loc_size[-1]), dtype=torch.float64)
    mask = torch.zeros((loc_size[-2], loc_size[-1]))
    loc_x = loc_x.flatten().tolist()
    loc_y = loc_y.flatten().tolist()
    mask[loc_x, loc_y] = 1
    return mask

def model_load(hf, lf, method, model_decomp_path, model_fusion_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loc_X, loc_Y = sensors(method)

    hf = torch.tensor(hf).float().unsqueeze(0).to(device)
    lf = torch.tensor(lf).float().unsqueeze(0).to(device)

    lf_normalizer = RangeNormalizer(lf)
    lf_input = lf_normalizer.encode(lf)
    hf_input = lf_normalizer.encode(hf)

    sensors_mask = get_mask(loc_X, loc_Y, hf.shape).to(device)
    sensors_input_hf = hf_input * sensors_mask

    model_decomp = torch.load(model_decomp_path)
    model_fusion = torch.load(model_fusion_path)

    grad_pred = model_decomp(lf_input)
    hf_pred = model_fusion(grad_pred, sensors_input_hf)
    hf_pred = lf_normalizer.decode(hf_pred)

    return np.array(hf_pred.squeeze(0).detach().cpu())

def plot_field_init(data, save_path):
    m, n = np.linspace(0, 0.12, 120), np.linspace(0, 0.12, 120)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(10, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=330, vmin=290)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig_1 = plt.contour(M, N, data, colors='black', levels=10, linewidth=0.2)
    plt.clabel(fig_1, inline=True, fontsize=15)

    data_max = np.max(data)
    x, y = np.where(data == np.max(data))
    x = x / 120 * 0.12
    y = y / 120 * 0.12
    plt.scatter(y, x, color='white', s=200, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=24, xy=(y, x), xytext=(y + 0.02, x + 0.01),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    cb1 = plt.colorbar(im_1, fraction=0.045, pad=0.03, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([290, 300, 310, 320, 330])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
    plt.yticks([0, 0.04, 0.08, 0.12], fontsize=24)

    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_field(data, method, save_path):
    m, n = np.linspace(0, 0.12, 120), np.linspace(0, 0.12, 120)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(10, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=330, vmin=290)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig_1 = plt.contour(M, N, data, colors='black', levels=10, linewidth=0.2)
    plt.clabel(fig_1, inline=True, fontsize=15)

    data_max = np.max(data)
    x, y = np.where(data == np.max(data))
    x = x / 120 * 0.12
    y = y / 120 * 0.12
    loc_X, loc_Y = sensors(method)
    loc_X = loc_X / 120 * 0.12
    loc_Y = loc_Y / 120 * 0.12
    plt.scatter(loc_X, loc_Y, color='white', s=50)
    # plt.scatter(x, y, color='white', s=50)
    plt.scatter(y, x, color='white', s=200, marker='*')
    plt.scatter(y, x, color='white', s=200, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=24, xy=(y, x), xytext=(y + 0.02, x + 0.01),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    cb1 = plt.colorbar(im_1, fraction=0.045, pad=0.03, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([290, 300, 310, 320, 330])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
    plt.yticks([0, 0.04, 0.08, 0.12], fontsize=24)

    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_error_init(data, save_path):
    m, n = np.linspace(0, 0.12, 120), np.linspace(0, 0.12, 120)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(9.5, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=16, vmin=0)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)

    x, y = [24, 60, 96], [24, 60, 96]
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    for i in range(9):
        x_text = X[i] / 120 * 0.12
        y_text = Y[i] / 120 * 0.12
        max = data[X[i], Y[i]]
        plt.scatter(y_text, x_text, color='white', s=100, marker='s')
        plt.annotate(f'T: {max:.2f}', fontsize=24, xy=(y_text, x_text), xytext=(y_text-0.011, x_text-0.018),
                     arrowprops=dict(facecolor='white', shrink=0.1),
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    cb1 = plt.colorbar(im_1, fraction=0.045, pad=0.03, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 4, 8, 12, 16])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
    plt.yticks([0, 0.04, 0.08, 0.12], fontsize=24)

    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_error(data, method, save_path):
    m, n = np.linspace(0, 0.12, 120), np.linspace(0, 0.12, 120)
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(9.5, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=3.2, vmin=0)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)

    data_max = np.max(data)
    x, y = np.where(data == np.max(data))
    x = x / 120 * 0.12
    y = y / 120 * 0.12
    loc_X, loc_Y = sensors(method)
    loc_X = loc_X / 120 * 0.12
    loc_Y = loc_Y / 120 * 0.12
    plt.scatter(loc_X, loc_Y, color='white', s=50)
    # plt.scatter(x, y, color='white', s=50)
    plt.scatter(y, x, color='white', s=200, marker='*')
    if (x >= 0.06) & (y >= 0.06):
        x_text = x - 0.02
        y_text = y - 0.05
    elif (x >= 0.06) & (y < 0.06):
        x_text = x - 0.01
        y_text = y + 0.02
    elif (x < 0.06) & (y >= 0.06):
        x_text = x + 0.02
        y_text = y - 0.05
    else:
        x_text = x + 0.01
        y_text = y + 0.02
    plt.annotate(f'Max: {data_max:.2f}', fontsize=24, xy=(y, x), xytext=(y_text, x_text),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))  # 添加注释，使用红色箭头连接注释和最大值点

    cb1 = plt.colorbar(im_1, fraction=0.045, pad=0.03, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 0.8, 1.6, 2.4, 3.2])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=24)

    plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
    plt.yticks([0, 0.04, 0.08, 0.12], fontsize=24)

    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_layout(method, save_path):
    layout = np.zeros((120, 120))
    layout[18:30, 18:30] = 1
    layout[18:30, 54:66] = 1
    layout[18:30, 90:102] = 1
    layout[54:66, 18:30] = 1
    layout[54:66, 54:66] = 1
    layout[54:66, 90:102] = 1
    layout[90:102, 18:30] = 1
    layout[90:102, 54:66] = 1
    layout[90:102, 90:102] = 1

    loc_X, loc_Y = sensors(method)
    loc_X = loc_X / 120 * 0.12
    loc_Y = loc_Y / 120 * 0.12

    m, n = np.linspace(0, 0.12, 120), np.linspace(0, 0.12, 120)
    M, N = np.meshgrid(m, n)
    plt.figure(figsize=(8, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    plt.contourf(M, N, layout, cmap='summer', levels=100)
    plt.scatter(loc_X, loc_Y, s=50, c='r')

    plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
    plt.yticks([0, 0.04, 0.08, 0.12], fontsize=24)

    plt.xlabel('x', fontsize=24)
    plt.ylabel('y', fontsize=24)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_line(data, model_name, save_path):
    x = np.linspace(0, 0.12, 120)

    plt.figure(figsize=(8, 8), dpi=100)
    for i in range(len(data)):
        plt.plot(x, data[i], lw=2, label=model_name[i])
        plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
        plt.yticks(fontsize=24)
    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.show()

def plot_line_error(data, model_name, save_path):
    x = np.linspace(0, 0.12, 120)

    fig = plt.figure(figsize=(8, 24), dpi=100)
    # ax = plt.gca()
    # ax.set_aspect(0.0015)
    # plt.plot(x, np.abs(data[1] - data[0]), label=model_name[1])  # , color='k')
    # # plt.xticks([], fontsize=24)
    # plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
    # plt.yticks([0, 5, 10, 15], fontsize=24)
    # ax.set_xticklabels([])
    # plt.grid(axis='both', ls='--')

    sns.set_palette("hls", 17)
    for i in range(len(data)):
        plt.subplot(len(data), 1, i + 1)
        ax = plt.gca()
        ax.set_aspect(0.001)
        plt.plot(x, np.abs(data[i] - data[0]), label=model_name[i])#, color='k')
        plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
        plt.yticks([0, 5, 10, 15], fontsize=24)
        if i != 16:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([0, 0.04, 0.08, 0.12])
        plt.grid(axis='both', ls='--')
    # plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_lines(data, model_name, save_path):
    x = np.linspace(0, 0.12, 120)

    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = plt.gca()
    ax.set_aspect(0.01)
    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#9467bd', '#2ca02c', '#bcbd22', '#17becf']
    for i in range(len(data)):
        plt.plot(x, np.abs(data[i] - data[0]), label=model_name[i], color=colors[i])
    plt.xticks([0, 0.04, 0.08, 0.12], fontsize=24)
    plt.yticks([0, 5, 10, 15], fontsize=24)
    plt.grid(axis='both', ls='--')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def func(f, num_x, num_y):
    x, y = np.linspace(0, 0.12, num_x, endpoint=True), np.linspace(0, 0.12, num_y, endpoint=True)
    X, Y = np.meshgrid(x, y)

    data = f(X, Y)
    return data


if __name__ == '__main__':
    num_x, num_y = 120, 120
    x = symbols('x')
    y = symbols('y')

    s_1 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    s_2 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    u_1 = [0.024, 0.024, 0.024, 0.06, 0.06, 0.06, 0.096, 0.096, 0.096]
    u_2 = [0.024, 0.06, 0.096, 0.024, 0.06, 0.096, 0.024, 0.06, 0.096]
    rho = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    alpha = [0.008, 0.002, 0.01, 0.005, 0.01, 0.008, 0.005, 0.005, 0.01]

    gaussian = 0
    for i in range(9):
        gaussian_c = (2 * sympy.pi * s_1[i] * s_2[i] * (1 - rho[i] ** 2) ** (0.5)) ** (-1)
        gaussian_u = (x - u_1[i]) ** 2 / s_1[i] ** 2 - 2 * rho[i] * (x - u_1[i]) * (y - u_2[i]) / (s_1[i] * s_2[i]) + (
                    y - u_2[i]) ** 2 / s_2[i] ** 2
        gaussian_e = sympy.E ** (-0.5 * (1 - rho[i] ** 2) ** (-1) * gaussian_u)
        gaussian += alpha[i] * gaussian_c * gaussian_e
    gaussian_error = lambdify([x, y], gaussian, "numpy")
    error = func(gaussian_error, num_x, num_y)

    data_path = 'data/size120_grid1mm_q8w_dirichlet.txt'
    hf = np.loadtxt(data_path)
    lf = hf + error

    pred_uniform9_1order = model_load(hf, lf, 'uniform9',
                                      'model/fine_ht_decomp_uniform_9sensors_1order_knone.pth',
                                      'model/fine_ht_fusion_uniform_9sensors_1order_knone.pth')
    pred_uniform25_1order = model_load(hf, lf, 'uniform25',
                                       'model/fine_ht_decomp_uniform_25sensors_1order_knone.pth',
                                       'model/fine_ht_fusion_uniform_25sensors_1order_knone.pth')
    pred_uniform49_1order = model_load(hf, lf, 'uniform49',
                                       'model/fine_ht_decomp_uniform_49sensors_1order_knone.pth',
                                       'model/fine_ht_fusion_uniform_49sensors_1order_knone.pth')
    pred_selected49_1order = model_load(hf, lf, 'selected49',
                                        'model/fine_ht_decomp_selected_49sensors_1order_knone.pth',
                                        'model/fine_ht_fusion_selected_49sensors_1order_knone.pth')
    pred_random49_1order = model_load(hf, lf, 'random49',
                                      'model/fine_ht_decomp_random_49sensors_1order_knone.pth',
                                      'model/fine_ht_fusion_random_49sensors_1order_knone.pth')
    pred_uniform9_2order = model_load(hf, lf, 'uniform9',
                                      'model/fine_ht_decomp_uniform_9sensors_2order_knone.pth',
                                      'model/fine_ht_fusion_uniform_9sensors_2order_knone.pth')
    pred_uniform25_2order = model_load(hf, lf, 'uniform25',
                                       'model/fine_ht_decomp_uniform_25sensors_2order_knone.pth',
                                       'model/fine_ht_fusion_uniform_25sensors_2order_knone.pth')
    pred_uniform49_2order = model_load(hf, lf,  'uniform49',
                                       'model/fine_ht_decomp_uniform_49sensors_2order_knone.pth',
                                       'model/fine_ht_fusion_uniform_49sensors_2order_knone.pth')
    pred_selected49_2order = model_load(hf, lf, 'selected49',
                                        'model/fine_ht_decomp_selected_49sensors_2order_knone.pth',
                                        'model/fine_ht_fusion_selected_49sensors_2order_knone.pth')
    pred_random49_2order = model_load(hf, lf, 'random49',
                                      'model/fine_ht_decomp_random_49sensors_2order_knone.pth',
                                      'model/fine_ht_fusion_random_49sensors_2order_knone.pth')
    pred_uniform9_3order = model_load(hf, lf, 'uniform9',
                                      'model/fine_ht_decomp_uniform_9sensors_3order_knone.pth',
                                      'model/fine_ht_fusion_uniform_9sensors_3order_knone.pth')
    pred_uniform25_3order = model_load(hf, lf, 'uniform25',
                                       'model/fine_ht_decomp_uniform_25sensors_3order_knone.pth',
                                       'model/fine_ht_fusion_uniform_25sensors_3order_knone.pth')
    pred_uniform49_3order = model_load(hf, lf, 'uniform49',
                                       'model/fine_ht_decomp_uniform_49sensors_3order_knone.pth',
                                       'model/fine_ht_fusion_uniform_49sensors_3order_knone.pth')
    pred_selected49_3order = model_load(hf, lf, 'selected49',
                                        'model/fine_ht_decomp_selected_49sensors_3order_knone.pth',
                                        'model/fine_ht_fusion_selected_49sensors_3order_knone.pth')
    pred_random49_3order = model_load(hf, lf, 'random49',
                                      'model/fine_ht_decomp_random_49sensors_3order_knone.pth',
                                      'model/fine_ht_fusion_random_49sensors_3order_knone.pth')



    print('done')
