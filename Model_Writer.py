# 用于创建模型参数文件
# 网格数量间距示例
# 100  31 LOGE
#       8E+04      7E+04      5E+04      3E+04      2E+04      1E+04       8000       7000       5000       5000
#        3000       2000       2000       1500       1500       1500       1500       1500       1500       1500
#        1500       1500       1500       1500       1500       1500       1500       1500       1500       1500
#        1500       1500       1500       1500       1500       1500       1500       1500       1500       1500
#        1500       1500       1500       1500       1500       1500       1500       1500       1500       1500
#        1500       1500       1500       1500       1500       1500       1500       1500       1500       1500
#        1500       1500       1500       1500       1500       1500       1500       1500       1500       1500
#        1500       1500       1500       1500       1500       1500       1500       1500       1500       1500
#        1500       1500       1500       1500       1500       1500       1500       2000       2000       3000
#        5000       5000       7000       8000      1E+04      2E+04      3E+04      5E+04      7E+04      8E+04
#          10         30         60        100        200        300        500        800       1000       1200
#        1500       1800       2100       2500       3000       3600       4300       5000       6500       8000
#       1E+04      1E+04    1.5E+04      2E+04      3E+04      4E+04      5E+04      6E+04      7E+04      8E+04
#       8E+04

import numpy as np


class ModelWriter2D:
    def __init__(self, file_name, station_num, period_num, length_y, skin_depth=80000):
        self.file_name = file_name
        self.station_num = station_num
        self.length_y = length_y
        self.period_num = period_num
        self.mesh_number_y, self.mesh_number_z = self.decide_mesh_number()
        self.skin_depth = skin_depth

    def decide_mesh_number(self):
        mesh_number_y = self.station_num * 4 + 20
        mesh_number_z = self.period_num * 2
        return int(mesh_number_y), int(mesh_number_z)

    def write_mesh_number(self, mesh_number_y, mesh_number_z):
        with open(self.file_name, 'a') as f:
            #清空文件内容
            f.seek(0)
            f.truncate()
            f.write(str(mesh_number_y) + ' ' + str(mesh_number_z) + ' LOGE' + '\n')
            f.close()

    def calculate_mesh_spacing(self, mesh_number_y, mesh_number_z, length_y):
        mesh_spacing_y = []
        # 得到y方向中心网格间距
        side_mesh_num = 20
        y_center_spacing = length_y / (mesh_number_y - side_mesh_num)
        # 得到y方向边界网格间距
        station_distance = length_y / (self.station_num - 1)
        y_sider_spacing_min = station_distance
        y_sider_spacing_max = station_distance * 30
        y_sider_spacing = np.logspace(np.log10(y_sider_spacing_min), np.log10(y_sider_spacing_max), int(side_mesh_num /
                                                                                                        2))
        # 计算边界网格间距之和
        y_sider_spacing_sum = sum(y_sider_spacing)
        # 得到y方向网格间距，将边界网格间距与中心网格间距合并，边界网格对称分布
        y_sider_spacing_reverse = y_sider_spacing[::-1]
        for i in y_sider_spacing_reverse:
            mesh_spacing_y.append(int(i))
        for i in range(mesh_number_y - side_mesh_num):
            mesh_spacing_y.append(int(y_center_spacing))
        for i in y_sider_spacing:
            mesh_spacing_y.append(int(i))
        # z网格间距
        #        200        300        500        800       1000       1200
        #       1500       1800       2100       2500       3000       3600       4300       5000       6500       8000
        #      1E+04      1E+04    1.5E+04      2E+04      3E+04      4E+04      5E+04      6E+04      7E+04      8E+04
        #       8E+04
        mesh_spacing_z = np.logspace(np.log10(200), np.log10(self.skin_depth / 10), mesh_number_z)
        return mesh_spacing_y, mesh_spacing_z, y_sider_spacing_sum

    def write_mesh_spacing(self, mesh_spacing_y, mesh_spacing_z):
        with open(self.file_name, 'a') as f:
            #写入x方向网格间距，将列表mesh_spacing_x转换为字符串
            f.write(' '.join(map(str, mesh_spacing_y)) + '\n')
            #写入y方向网格间距，将列表mesh_spacing_y转换为字符串
            f.write(' '.join(map(str, mesh_spacing_z)) + '\n')
            f.close()

    def write_even_resistivity(self, resistivity):
        # 使用均匀分布的电阻率
        with open(self.file_name, 'a') as f:
            f.write('1' + '\n')
            for i in range(self.mesh_number_z):
                for j in range(self.mesh_number_y):
                    f.write(str(resistivity) + ' ')
                f.write('\n')

    def write_resistivity_block(self, resistivity, y_start, y_end, z_start, z_end, mesh_spacing_y, mesh_spacing_z,
                                y_sider_spacing_sum):
        # 使用均匀分布的电阻率
        with open(self.file_name, 'a') as f:
            f.write('1' + '\n')
            for i in range(self.mesh_number_z):
                # print(z_start<sum(mesh_spacing_z[:i])<z_end,sum(mesh_spacing_z[:i]))
                for j in range(self.mesh_number_y):
                    if z_start <= sum(mesh_spacing_z[:i]) <= z_end and y_start <= sum(mesh_spacing_y[:j]) -\
                            y_sider_spacing_sum <= y_end:
                        f.write(str(resistivity) + ' ')
                    else:
                        f.write('4.6' + ' ')
                f.write('\n')

    def write_coordinate_origin(self, y, z):
        with open(self.file_name, 'a') as f:
            f.write(str(y) + ' ' + str(z) + '\n')
            f.write('0')
            f.close()

    def write_mesh(self, resistivity=4.6, y_start=0, y_end=0, z_start=0, z_end=0):
        num_y, num_z = self.decide_mesh_number()
        self.write_mesh_number(num_y, num_z)
        spacing_y, spacing_z, _ = self.calculate_mesh_spacing(num_y, num_z, self.length_y)
        self.write_mesh_spacing(spacing_y, spacing_z)
        if sum([y_start, y_end, z_start, z_end]) == 0:
            self.write_even_resistivity(resistivity)
        else:
            self.write_resistivity_block(resistivity, y_start, y_end, z_start, z_end, spacing_y, spacing_z, _)
        # self.write_coordinate_origin(0, 0)
        return _



