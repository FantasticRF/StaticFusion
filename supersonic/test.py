import numpy as np
import torch
import sympy
from sympy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 36
import matplotlib.ticker as ticker


def plot_sensors():
    loc_x = np.array([5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124, 131, 138, 145])
    loc_y = np.array([3, 9, 15, 21, 27])
    loc_X, loc_Y = np.meshgrid(loc_x, loc_y)
    loc_X = loc_X / 150 * 0.06
    loc_Y = loc_Y / 30 * 0.012
    plt.scatter(loc_X, loc_Y, color='white', s=30)

def get_mask(loc_x, loc_y, loc_size):
    # mask = torch.zeros((loc_size[-2], loc_size[-1]), dtype=torch.float64)
    mask = torch.zeros((loc_size[-2], loc_size[-1]))
    loc_x = loc_x.flatten().tolist()
    loc_y = loc_y.flatten().tolist()
    mask[loc_y, loc_x] = 1
    return mask

class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        self.size = x.size()
        mymin = torch.min(x.contiguous().view(self.size[0], -1), 1, keepdim=True)[0]
        mymax = torch.max(x.contiguous().view(self.size[0], -1), 1, keepdim=True)[0]
        # mymin = torch.min(x.contiguous().view(self.size[0], -1), 0, keepdim=True)[0]
        # mymax = torch.max(x.contiguous().view(self.size[0], -1), 0, keepdim=True)[0]

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

def model_load(hf, lf, model_decomp_path, model_fusion_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loc_x = np.array([5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124, 131, 138, 145])
    loc_y = np.array([3, 9, 15, 21, 27])
    loc_X, loc_Y = np.meshgrid(loc_x, loc_y)

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

def model_load_couple(hf, lf, model_decomp_path, model_fusion_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loc_x = np.array([5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124, 131, 138, 145])
    loc_y = np.array([3, 9, 15, 21, 27])
    loc_X, loc_Y = np.meshgrid(loc_x, loc_y)

    hf = torch.tensor(hf).unsqueeze(0).float().to(device)
    lf = torch.tensor(lf).unsqueeze(0).float().to(device)

    u_hf = hf[:, :, 2].reshape(1, 31, 151)[:, :-1, 1:]
    p_hf = hf[:, :, 3].reshape(1, 31, 151)[:, :-1, 1:]
    t_hf = hf[:, :, 4].reshape(1, 31, 151)[:, :-1, 1:]
    u_lf = lf[:, :, 2].reshape(1, 31, 151)[:, :-1, 1:]
    p_lf = lf[:, :, 3].reshape(1, 31, 151)[:, :-1, 1:]
    t_lf = lf[:, :, 4].reshape(1, 31, 151)[:, :-1, 1:]

    u_lf_normalizer = RangeNormalizer(u_lf)
    p_lf_normalizer = RangeNormalizer(p_lf)
    t_lf_normalizer = RangeNormalizer(t_lf)

    u_lf_input = u_lf_normalizer.encode(u_lf)
    p_lf_input = p_lf_normalizer.encode(p_lf)
    t_lf_input = t_lf_normalizer.encode(t_lf)
    lf_input = torch.cat((u_lf_input, p_lf_input, t_lf_input), dim=0)

    u_hf_input = u_lf_normalizer.encode(u_hf)
    p_hf_input = p_lf_normalizer.encode(p_hf)
    t_hf_input = t_lf_normalizer.encode(t_hf)

    sensors_mask = get_mask(loc_X, loc_Y, u_hf.shape).to(device)

    sensors_u_input_lf = u_lf_input * sensors_mask
    sensors_p_input_lf = p_lf_input * sensors_mask
    sensors_t_input_lf = t_lf_input * sensors_mask
    sensors_input_lf = torch.cat((sensors_u_input_lf, sensors_p_input_lf, sensors_t_input_lf), dim=0)
    sensors_u_input_hf = u_hf_input * sensors_mask
    sensors_p_input_hf = p_hf_input * sensors_mask
    sensors_t_input_hf = t_hf_input * sensors_mask
    sensors_input_hf = torch.cat((sensors_u_input_hf, sensors_p_input_hf, sensors_t_input_hf), dim=0)

    model_decomp = torch.load(model_decomp_path)
    model_fusion = torch.load(model_fusion_path)

    grad_pred = model_decomp(lf_input)
    hf_pred = model_fusion(grad_pred, sensors_input_hf)
    u_pred = u_lf_normalizer.decode(hf_pred[[0]])
    p_pred = p_lf_normalizer.decode(hf_pred[[1]])
    t_pred = t_lf_normalizer.decode(hf_pred[[2]])

    return np.array(u_pred.squeeze(0).detach().cpu()), \
           np.array(p_pred.squeeze(0).detach().cpu()), \
           np.array(t_pred.squeeze(0).detach().cpu())

def plot_field_u(data, save_path):
    m, n = np.linspace(0, 0.06, data.shape[-1]), np.linspace(0, 0.012, data.shape[-2])
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(20, 5), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=600, vmin=0)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig_1 = plt.contour(M, N, data, colors='black', levels=10, linewidth=0.2)
    plt.clabel(fig_1, inline=True, fontsize=25)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb1 = plt.colorbar(im_1, fraction=0.01, pad=0.02, format=fmt, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 150, 300, 450, 600])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=36)

    plot_sensors()
    data_max = np.max(data)
    x_max, y_max = np.where(data == np.max(data))
    x_max = x_max / 150 * 0.06
    y_max = y_max / 30 * 0.012
    plt.scatter(y_max, x_max, color='white', s=200, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=30, xy=(y_max, x_max), xytext=(y_max + 0.002, x_max + 0.0035),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.02, 0.04, 0.06], fontsize=36)
    plt.yticks([0, 0.006, 0.012], fontsize=36)

    plt.xlabel('x', fontsize=36)
    plt.ylabel('y', fontsize=36)

    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()

def plot_error_u(data, save_path):
    m, n = np.linspace(0, 0.06, data.shape[-1]), np.linspace(0, 0.012, data.shape[-2])
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(20, 5), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=12, vmin=0)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb1 = plt.colorbar(im_1, fraction=0.01, pad=0.02, format=fmt, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 3, 6, 9, 12])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=36)

    plot_sensors()
    data_max = np.max(data)
    x_max, y_max = np.where(data == np.max(data))
    x_max = x_max / 150 * 0.06
    y_max = y_max / 30 * 0.012
    plt.scatter(y_max, x_max, color='white', s=300, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=30, xy=(y_max, x_max), xytext=(y_max + 0.008, x_max + 0.002),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))
    # data_min = np.min(data)
    # x_min, y_min = np.where(data == np.min(data))
    # x_min = x_min / 150 * 0.06
    # y_min = y_min / 30 * 0.012
    # plt.scatter(y_min, x_min, color='white', s=300, marker='*')
    # plt.annotate(f'Max: {data_min:.2f}', fontsize=30, xy=(y_min, x_min), xytext=(y_min - 0.004, x_min - 0.002),
    #              arrowprops=dict(facecolor='white', shrink=0.1),
    #              bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.02, 0.04, 0.06], fontsize=36)
    plt.yticks([0, 0.006, 0.012], fontsize=36)

    plt.xlabel('x', fontsize=36)
    plt.ylabel('y', fontsize=36)

    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()

def plot_field_p(data, save_path):
    m, n = np.linspace(0, 0.06, data.shape[-1]), np.linspace(0, 0.012, data.shape[-2])
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(20, 5), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=75000, vmin=15000)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig_1 = plt.contour(M, N, data, colors='black', levels=5, linewidth=0.2)
    plt.clabel(fig_1, inline=True, fontsize=25)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb1 = plt.colorbar(im_1, fraction=0.01, pad=0.02, format=fmt, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([15000, 30000, 45000, 60000, 75000])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=36)

    plot_sensors()
    data_max = np.max(data)
    x_max, y_max = np.where(data == np.max(data))
    x_max = x_max[0] / 150 * 0.06
    y_max = y_max[0] / 30 * 0.012
    plt.scatter(y_max, x_max, color='white', s=300, marker='*')
    plt.annotate(f'Max: {data_max:.0f}', fontsize=30, xy=(y_max, x_max), xytext=(y_max - 0.012, x_max + 0.008),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))
    data_min = np.min(data)
    x_min, y_min = np.where(data == np.min(data))
    x_min = x_min[0] / 150 * 0.06
    y_min = y_min[0] / 30 * 0.012
    plt.scatter(y_min, x_min, color='white', s=300, marker='*')
    plt.annotate(f'Min: {data_min:.0f}', fontsize=30, xy=(y_min, x_min), xytext=(y_min - 0.012, x_min + 0.008),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.02, 0.04, 0.06], fontsize=36)
    plt.yticks([0, 0.006, 0.012], fontsize=36)

    plt.xlabel('x', fontsize=36)
    plt.ylabel('y', fontsize=36)

    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()

def plot_error_p(data, save_path):
    m, n = np.linspace(0, 0.06, data.shape[-1]), np.linspace(0, 0.012, data.shape[-2])
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(20, 5), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=3000, vmin=0)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb1 = plt.colorbar(im_1, fraction=0.01, pad=0.02, format=fmt, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 1000, 2000, 3000])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=36)

    plot_sensors()
    data_max = np.max(data)
    x_max, y_max = np.where(data == np.max(data))
    x_max = x_max[0] / 150 * 0.06
    y_max = y_max[0] / 30 * 0.012
    plt.scatter(y_max, x_max, color='white', s=300, marker='*')
    plt.annotate(f'Max: {data_max:.0f}', fontsize=30, xy=(y_max, x_max), xytext=(y_max - 0.015, x_max + 0.005),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.02, 0.04, 0.06], fontsize=36)
    plt.yticks([0, 0.006, 0.012], fontsize=36)

    plt.xlabel('x', fontsize=36)
    plt.ylabel('y', fontsize=36)

    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()

def plot_field_t(data, save_path):
    m, n = np.linspace(0, 0.06, data.shape[-1]), np.linspace(0, 0.012, data.shape[-2])
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(20, 5), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=300, vmin=100)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)
    fig_1 = plt.contour(M, N, data, colors='black', levels=10, linewidth=0.2)
    plt.clabel(fig_1, inline=True, fontsize=25)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb1 = plt.colorbar(im_1, fraction=0.01, pad=0.02, format=fmt, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([100, 150, 200, 250, 300])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=36)

    plot_sensors()
    data_max = np.max(data)
    x_max, y_max = np.where(data == np.max(data))
    x_max = x_max[0] / 150 * 0.06
    y_max = y_max[0] / 30 * 0.012
    plt.scatter(y_max, x_max, color='white', s=300, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=30, xy=(y_max, x_max), xytext=(y_max - 0.012, x_max - 0.005),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))
    data_min = np.min(data)
    x_min, y_min = np.where(data == np.min(data))
    x_min = x_min[0] / 150 * 0.06
    y_min = y_min[0] / 30 * 0.012
    plt.scatter(y_min, x_min, color='white', s=300, marker='*')
    plt.annotate(f'Min: {data_min:.2f}', fontsize=30, xy=(y_min, x_min), xytext=(y_min + 0.006, x_min + 0.005),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.02, 0.04, 0.06], fontsize=36)
    plt.yticks([0, 0.006, 0.012], fontsize=36)

    plt.xlabel('x', fontsize=36)
    plt.ylabel('y', fontsize=36)

    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()

def plot_error_t(data, save_path):
    m, n = np.linspace(0, 0.06, data.shape[-1]), np.linspace(0, 0.012, data.shape[-2])
    M, N = np.meshgrid(m, n)

    plt.figure(figsize=(20, 5), dpi=100)
    ax = plt.gca()
    ax.set_aspect(1)  # 按比例绘制xy轴
    norm_1 = mpl.colors.Normalize(vmax=5, vmin=0)
    im_1 = mpl.cm.ScalarMappable(cmap='jet', norm=norm_1)
    fig = plt.contourf(M, N, data, cmap='jet', levels=100, norm=norm_1)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb1 = plt.colorbar(im_1, fraction=0.01, pad=0.02, format=fmt, ax=ax)
    tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
    cb1.locator = tick_locator
    cb1.set_ticks([0, 1, 2, 3, 4, 5])
    cb1.update_ticks()
    cb1.ax.tick_params(labelsize=36)

    plot_sensors()
    data_max = np.max(data)
    x_max, y_max = np.where(data == np.max(data))
    x_max = x_max[0] / 150 * 0.06
    y_max = y_max[0] / 30 * 0.012
    plt.scatter(y_max, x_max, color='white', s=300, marker='*')
    plt.annotate(f'Max: {data_max:.2f}', fontsize=30, xy=(y_max, x_max), xytext=(y_max - 0.015, x_max + 0.003),
                 arrowprops=dict(facecolor='white', shrink=0.1),
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='k', lw=1, alpha=0.6))

    plt.xticks([0, 0.02, 0.04, 0.06], fontsize=36)
    plt.yticks([0, 0.006, 0.012], fontsize=36)

    plt.xlabel('x', fontsize=36)
    plt.ylabel('y', fontsize=36)

    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    hf_data_path = 'data/upt_fine.txt'
    lf_data_path = 'data/upt_coarse.txt'
    num_x, num_y = 30, 150

    hf = np.loadtxt(hf_data_path)
    lf = np.loadtxt(lf_data_path)
    u_hf = (hf[:, 2].reshape(31, 151))[:-1, 1:]
    u_lf = (lf[:, 2].reshape(31, 151))[:-1, 1:]
    p_hf = (hf[:, 3].reshape(31, 151))[:-1, 1:]
    p_lf = (lf[:, 3].reshape(31, 151))[:-1, 1:]
    t_hf = (hf[:, 4].reshape(31, 151))[:-1, 1:]
    t_lf = (lf[:, 4].reshape(31, 151))[:-1, 1:]

    pred_decouple_u = model_load(u_hf, u_lf,
                                 'model/fine_decomp_decouple_u_2order.pth',
                                 'model/fine_fusion_decouple_u_2order.pth')
    pred_decouple_p = model_load(p_hf, p_lf,
                                 'model/fine_decomp_decouple_p_2order.pth',
                                 'model/fine_fusion_decouple_p_2order.pth')
    pred_decouple_t = model_load(t_hf, t_lf,
                                 'model/fine_decomp_decouple_t_2order.pth',
                                 'model/fine_fusion_decouple_t_2order.pth')
    pred_couple_u, _, _ = model_load_couple(hf, lf,
                                            'model/fine_decomp_couple_u_2order.pth',
                                            'model/fine_fusion_couple_u_2order.pth')
    _, pred_couple_p, _ = model_load_couple(hf, lf,
                                            'model/fine_decomp_couple_p_2order.pth',
                                            'model/fine_fusion_couple_p_2order.pth')
    _, _, pred_couple_t = model_load_couple(hf, lf,
                                            'model/fine_decomp_couple_t_2order.pth',
                                            'model/fine_fusion_couple_t_2order.pth')
