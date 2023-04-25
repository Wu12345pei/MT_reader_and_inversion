import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter_2D:
    def __init__(self, path):
        self.path = path
        self.y_num = None
        self.z_num = None
        self.y_grid_scale = None
        self.z_grid_scale = None
        self.flatten_rho_list = None

    def read_data(self):
        #打开文件
        with open(self.path, 'r') as f:
            lines = f.readlines()
            #读取第一行，获取网格数量
            line = lines[0]
            line = line.split()
            self.y_num = int(line[0])
            self.z_num = int(line[1])
            #将剩下的数据读取到列表中
            flatten_all_data_list = []
            for line in lines[1:]:
                line = line.split()
                for data in line:
                    flatten_all_data_list.append(float(data))
            #取出y_grid_scale和z_grid_scale
            self.y_grid_scale = flatten_all_data_list[:self.y_num]
            self.z_grid_scale = flatten_all_data_list[self.y_num:self.y_num + self.z_num]
            #取出rho
            self.flatten_rho_list = flatten_all_data_list[self.y_num + self.z_num + 1:]

    def plot_rho(self, skin_depth=0, side_grids=10, title='rho', vmin=4, vmax=5, cmap='jet', save=False, save_path=None):
        self.read_data()
        #将rho转换为二维数组
        flatten_rho_list = self.flatten_rho_list
        y_num = self.y_num
        z_num = self.z_num
        #获取y方向和z方向网格大小
        y_grid_scale = self.y_grid_scale[side_grids: -side_grids]
        z_grid_scale = self.z_grid_scale
        y_grid_scale_sum = [sum(y_grid_scale[: i]) for i in range(len(y_grid_scale))]
        z_grid_scale_sum = [sum(z_grid_scale[: i]) for i in range(len(z_grid_scale))]
        for i in z_grid_scale_sum:
            if i > skin_depth:
                z_grid_scale_sum = z_grid_scale_sum[: z_grid_scale_sum.index(i)]
                break
        #将rho转换为二维数组,y_num为列数，z_num为行数,去掉列两边的10个网格（吸收边界）
        rho = np.array(flatten_rho_list).reshape(z_num, y_num)
        rho = rho[:len(z_grid_scale_sum), side_grids: -side_grids]
        #生成网格
        y_grid_scale, z_grid_scale = np.meshgrid(y_grid_scale_sum, z_grid_scale_sum)
        #绘制图像,x轴为y方向，y轴为z方向，颜色为rho
        plt.pcolormesh(y_grid_scale, z_grid_scale, rho, cmap='jet_r', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title(title)
        #x轴置顶，y轴倒转
        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.tick_top()

class plot_dat_z_tensor:
    def __init__(self, path):
        self.path = path
        self.station_id = []
        self.station_loc = []
        self.periods = []
        self.z_tensor = []

    def read_dat(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.split()) == 11:
                    line = line.split()
                    period = float(line[0])
                    station_id = int(line[1])
                    distance = float(line[5])
                    TETM = line[7]
                    z_real = float(line[8])
                    z_imag = float(line[9])
                    error = float(line[10])
                    if period not in self.periods:
                        self.periods.append(period)
                    if station_id not in self.station_id:
                        self.station_id.append(station_id)
                        self.station_loc.append(distance)
                    self.z_tensor.append([period, station_id, TETM, z_real, z_imag, error])

    def rewrite_error(self, error=0.1):
        newlines = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                newline = line
                if len(line.split()) == 11:
                    line = line.split()
                    z_real = float(line[8])
                    z_error = error * z_real
                    newline = line[0] + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] + ' ' + line[4] + ' ' +\
                           line[5] + ' ' + line[6] + ' ' + line[7] + ' ' + line[8] + ' ' + line[9] + ' ' + \
                           str.format("{:.6E}", z_error)+'\n'
                newlines.append(newline)
        with open(self.path, 'w') as f:
            f.seek(0)
            f.truncate()
            for newline in newlines:
                f.write(newline)

    def plot_z_tensor(self, station_id, TETM='both'):
        z_tensor = self.z_tensor
        periods = self.periods
        station_loc = self.station_loc
        zTE_real_list = []
        zTE_imag_list = []
        zTE = []
        zTE_error_list = []
        zTM_real_list = []
        zTM_imag_list = []
        zTM = []
        zTM_error_list = []
        for z in z_tensor:
            if z[1] == station_id and z[2] == 'TE':
                zTE_real_list.append(z[3])
                zTE_imag_list.append(z[4])
                zTE.append(np.sqrt(z[3]**2 + z[4]**2))
                zTE_error_list.append(z[5])
            if z[1] == station_id and z[2] == 'TM':
                zTM_real_list.append(z[3])
                zTM_imag_list.append(z[4])
                zTM.append(np.sqrt(z[3]**2 + z[4]**2))
                zTM_error_list.append(z[5])
        if TETM == 'both':
            plt.scatter(periods, zTE, s=5, label='TE')
            plt.scatter(periods, zTM, s=5, label='TM')
            plt.legend()
            plt.xlabel('period')
            plt.xscale('log')
            plt.ylabel('z')
            plt.title('z_tensor')

    def get_zTE_zTM(self, station_id):
        z_tensor = self.z_tensor
        periods = self.periods
        zTE_real_list = []
        zTE_imag_list = []
        zTE = []
        zTE_error_list = []
        zTM_real_list = []
        zTM_imag_list = []
        zTM = []
        zTM_error_list = []
        for z in z_tensor:
            if z[1] == station_id and z[2] == 'TE':
                zTE_real_list.append(z[3])
                zTE_imag_list.append(z[4])
                zTE.append(z[3]+z[4]*1j)
                zTE_error_list.append(z[5])
            if z[1] == station_id and z[2] == 'TM':
                zTM_real_list.append(z[3])
                zTM_imag_list.append(z[4])
                zTM.append(z[3]+z[4]*1j)
                zTM_error_list.append(z[5])
        return periods, zTE, zTM


# pl=plot_dat_z_tensor('forward.dat')
# pl.read_dat()
# pl.plot_z_tensor(12)
# plt.show()