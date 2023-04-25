# dat头文件样式
# Synthetic 2D MT data written in Matlab
# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error
# > TE_Impedance
# > exp(-i\omega t)
# > [V/m]/[T]
# > 0.00
# > 0.000 0.000
# > 12 40
# dat文件数据样式
# 3.000000E-01 001 0.000 0.000 0.000 302500.000 0.000 TE -3.120000E+00 1.970000E+00 2.000000E+15
# 3.000000E-01 001 0.000 0.000 0.000 302500.000 0.000 TM -3.120000E+00 1.970000E+00 2.000000E+15
import numpy as np
from data_processor import DataProcessor
from file_controller import get_start_day, cal_begin_days


class EDIReader:
    def __init__(self, path):
        self.path = path

    def get_station_id(self):
        pass

    def get_station_loc(self):
        pass

    def get_periods_and_z_tensor(self, freq_num=20):
        # 读取edi文件并获取周期和阻抗张量
        # 周期格式为
# >FREQ ORDER=INC  //   20
#   0.5859374E-04  0.7812500E-04  0.1171875E-03  0.1562500E-03  0.2343750E-03
#   0.3125000E-03  0.4687501E-03  0.6250000E-03  0.9374997E-03  0.1250000E-02
#   0.1875000E-02  0.2500000E-02  0.3750000E-02  0.5000000E-02  0.7500002E-02
#   0.1000000E-01  0.1500000E-01  0.2000000E-01  0.3000000E-01  0.4000000E-01
        z_tensor = np.zeros((freq_num, len(self.path), 2, 2), dtype=complex)
        z_tensor_error = np.zeros((freq_num, len(self.path), 2, 2))
        for p in self.path:
            index = self.path.index(p)
            freq = []
            ZxxR, ZxyR, ZyxR, ZyyR = [], [], [], []
            ZxxI, ZxyI, ZyxI, ZyyI = [], [], [], []
            Zxx_error, Zxy_error, Zyx_error, Zyy_error = [], [], [], []
            with open(p) as f:
                read_begin = 0
                for i, line in enumerate(f):
                    if line.startswith('>FREQ ORDER=INC'):
                        read_begin = 'freq'
                    if line.isspace():
                        read_begin = 0
                    if line.startswith('>ZXXR'):
                        read_begin = 'ZXXR'
                    if line.startswith('>ZXYR'):
                        read_begin = 'ZXYR'
                    if line.startswith('>ZYXR'):
                        read_begin = 'ZYXR'
                    if line.startswith('>ZYYR'):
                        read_begin = 'ZYYR'
                    if line.startswith('>ZXXI'):
                        read_begin = 'ZXXI'
                    if line.startswith('>ZXYI'):
                        read_begin = 'ZXYI'
                    if line.startswith('>ZYXI'):
                        read_begin = 'ZYXI'
                    if line.startswith('>ZYYI'):
                        read_begin = 'ZYYI'
                    if line.startswith('>ZXX.VAR'):
                        read_begin = 'ZXX.VAR'
                    if line.startswith('>ZXY.VAR'):
                        read_begin = 'ZXY.VAR'
                    if line.startswith('>ZYX.VAR'):
                        read_begin = 'ZYX.VAR'
                    if line.startswith('>ZYY.VAR'):
                        read_begin = 'ZYY.VAR'
                    if line.startswith('>T'):
                        read_begin = 'T'
                    if read_begin == 'freq' and not line.startswith('>FREQ ORDER=INC'):
                        for j in line.split():
                            freq.append(float(j))
                    if read_begin == 'ZXXR' and not line.startswith('>ZXXR'):
                        for j in line.split():
                            ZxxR.append(float(j))
                    if read_begin == 'ZXYR' and not line.startswith('>ZXYR'):
                        for j in line.split():
                            ZxyR.append(float(j))
                    if read_begin == 'ZYXR' and not line.startswith('>ZYXR'):
                        for j in line.split():
                            ZyxR.append(float(j))
                    if read_begin == 'ZYYR' and not line.startswith('>ZYYR'):
                        for j in line.split():
                            ZyyR.append(float(j))
                    if read_begin == 'ZXXI' and not line.startswith('>ZXXI'):
                        for j in line.split():
                            ZxxI.append(float(j))
                    if read_begin == 'ZXYI' and not line.startswith('>ZXYI'):
                        for j in line.split():
                            ZxyI.append(float(j))
                    if read_begin == 'ZYXI' and not line.startswith('>ZYXI'):
                        for j in line.split():
                            ZyxI.append(float(j))
                    if read_begin == 'ZYYI' and not line.startswith('>ZYYI'):
                        for j in line.split():
                            ZyyI.append(float(j))
                    if read_begin == 'ZXX.VAR' and not line.startswith('>ZXX.VAR'):
                        for j in line.split():
                            Zxx_error.append(float(j))
                    if read_begin == 'ZXY.VAR' and not line.startswith('>ZXY.VAR'):
                        for j in line.split():
                            Zxy_error.append(float(j))
                    if read_begin == 'ZYX.VAR' and not line.startswith('>ZYX.VAR'):
                        for j in line.split():
                            Zyx_error.append(float(j))
                    if read_begin == 'ZYY.VAR' and not line.startswith('>ZYY.VAR'):
                        for j in line.split():
                            Zyy_error.append(float(j))
                f.close()
            periods = 1 / np.array(freq)
            for i in range(len(freq)):
                z_tensor[i, index, 0, 0] = ZxxR[i] + 1j * ZxxI[i]
                z_tensor[i, index, 0, 1] = ZxyR[i] + 1j * ZxyI[i]
                z_tensor[i, index, 1, 0] = ZyxR[i] + 1j * ZyxI[i]
                z_tensor[i, index, 1, 1] = ZyyR[i] + 1j * ZyyI[i]
                z_tensor_error[i, index, 0, 0] = Zxx_error[i]
                z_tensor_error[i, index, 0, 1] = Zxy_error[i]
                z_tensor_error[i, index, 1, 0] = Zyx_error[i]
                z_tensor_error[i, index, 1, 1] = Zyy_error[i]
        return periods, z_tensor, z_tensor_error

class TsReader:
    def __init__(self, path):
        self.path = path

    def get_station_id(self):
        # 读取path中的编号
        station_id = []
        for p in self.path:
            station_id.append(p[9:12])
        return station_id

    def get_station_loc(self):
        # 读取ts头文件并获取经纬度
        # 头文件经纬度格式为
        # >LATITUDE  : -25.9277802
        # >LONGITUDE :  26.4511108
        station_loc = []
        for p in self.path:
            with open(p) as f:
                for i, line in enumerate(f):
                    if line.startswith('>LATITUDE  :'):
                        lat = float(line[12:])
                    if line.startswith('>LONGITUDE :'):
                        lon = float(line[13:])
                    if i > 300:
                        break
                # 将经纬度转换为xy坐标
                station_loc.append([lat, lon])
                f.close()
        return station_loc

    def cal_station_line_length(self):
        loc = self.get_station_loc()
        begin_loc = loc[0]
        end_loc = loc[-1]
        begin_lat = begin_loc[0]
        begin_lon = begin_loc[1]
        end_lat = end_loc[0]
        end_lon = end_loc[1]
        begin_x = begin_lon * 111 * np.cos(begin_lat * np.pi / 180)
        begin_y = begin_lat * 111
        end_x = end_lon * 111 * np.cos(end_lat * np.pi / 180)
        end_y = end_lat * 111
        line_length = np.sqrt((end_x - begin_x) ** 2 + (end_y - begin_y) ** 2)
        return line_length * 1000

    def get_periods_and_z_tensor(self, period_list=[]):
        # 借助DataProcessor类获取周期和阻抗张量\
        if len(period_list) == 0:
            periods_num = 20
            periods = np.logspace(1, 3.5, periods_num)
        else:
            periods_num = len(period_list)
            periods = np.array(period_list)
        z_tensor = np.zeros((periods_num, len(self.path), 2, 2), dtype=complex)
        z_tensor_error = np.zeros((periods_num, len(self.path), 2, 2))
        for p in self.path:
            sd, sh = get_start_day(p)
            s = cal_begin_days(sd, sh)
            dp = DataProcessor(p)
            dp.read_and_remove_ultimate_value(s, s + 3)
            z_tensor_by_station, z_tensor_error_by_station = dp.cal_impedance_by_rb(periods=periods)
            z_tensor[:, self.path.index(p), :, :] = z_tensor_by_station
            z_tensor_error[:, self.path.index(p), :, :] = z_tensor_error_by_station
        return periods, z_tensor, z_tensor_error


class DatWriter:
    def __init__(self, station_id, station_loc, periods, z_tensor, error):
        # station_id: 站点编号列表
        # station_loc: 站点三维坐标列表(已经将经纬度转化为xy坐标)
        # periods: 周期列表，各台站周期一致
        # z_tensor: 阻抗张量，shape为(n_period, n_station, 2, 2)
        # error: 阻抗张量的误差，shape为(n_period, n_station, 2, 2)
        self.station_id = np.array(station_id)
        self.station_loc = np.array(station_loc)
        self.periods = np.array(periods)
        self.z_tensor = z_tensor
        self.error = error

    def write_2D_dat_header(self, file, impedance_type='TE_Impedance', sign_convention='exp(-i\omega t)',
                         impedance_unit='[mV/km]/[nT]', orientation='0.00', geographic_origin='0.000 0.000',
                         n_period='100', n_station='40'):
        # 写入dat文件头
        file.write('# Synthetic 2D MT data written in Python'+'\n')
        file.write('# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) Component Real Imag Error'+ '\n')
        file.write('> ' + impedance_type + '\n')
        file.write('> ' + sign_convention + '\n')
        file.write('> ' + impedance_unit + '\n')
        file.write('> ' + orientation + '\n')
        file.write('> ' + geographic_origin + '\n')
        file.write('> ' + n_period + ' ' + n_station + '\n')

    def transfer_z_tensor_to_TE_TM(self):
        # 将经纬度转换为xy坐标
        latitudes = self.station_loc[:, 0]
        longitudes = self.station_loc[:, 1]
        x = 111.32 * (longitudes - longitudes[0]) * np.cos(latitudes * np.pi / 180)
        y = 111.32 * (latitudes - latitudes[0])
        # 计算测线方位角
        angle = np.arctan((y[-1] - y[0]) / (x[-1] - x[0]))
        # 计算阻抗的旋转矩阵
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=complex)
        # 旋转阻抗至测线方向
        z_tensor = np.zeros(self.z_tensor.shape, dtype=complex)
        for i in range(self.z_tensor.shape[0]):
            for j in range(self.z_tensor.shape[1]):
                z_tensor[i, j, :, :] = np.dot(rotation_matrix, np.dot(self.z_tensor[i, j, :, :], rotation_matrix.T))
        # 取Zxy，Zyx为TE和TM分量
        z_TE = z_tensor[:, :, 0, 1]
        z_TM = z_tensor[:, :, 1, 0]
        return z_TE, z_TM

    def calculate_distance_along_measure_line(self, origin=0):
        # 计算测线上各点的距离
        latitudes = self.station_loc[:, 0]
        longitudes = self.station_loc[:, 1]
        x = 111.32 * (longitudes - longitudes[0]) * np.cos(latitudes * np.pi / 180)
        y = 111.32 * (latitudes - latitudes[0])
        distance = np.zeros(x.shape)
        for i in range(1, x.shape[0]):
            distance[i] = distance[i - 1] + np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        return distance*1000 + origin

    def write_dat_data(self, file, impedance_type='TE_Impedance', origin=0, distance=[]):
        # 读取TE和TM分量
        z_TE, z_TM = self.z_tensor[:, :, 1, 0], self.z_tensor[:, :, 0, 1]
        # z_TE, z_TM = self.transfer_z_tensor_to_TE_TM()
        #将测线距离作为y
        if len(distance) == 0:
            distance = self.calculate_distance_along_measure_line(origin)
        # 写入dat数据
        for i in range(self.station_id.shape[0]):
            for j in range(self.periods.shape[0]):
                if z_TE[j, i].real > 10000 or z_TE[j, i].imag > 10000 or self.error[j, i, 0, 1] > 10000 or \
                        self.error[j, i, 1, 0] > 10000 or z_TM[j, i].real > 10000 or z_TM[j, i].imag > 10000 or \
                        z_TE[j, i].real == 0 or z_TE[j, i].imag == 0 or z_TM[j, i].real == 0 or z_TM[j, i].imag == 0:
                    continue
                if impedance_type == 'TE_Impedance':
                    file.write(str.format("{:.6E}", self.periods[j]) + ' ' + str(self.station_id[i]) + ' ' + '0.000'
                               + ' ' + '0.000' + ' ' + '0.000' + ' '
                               + str.format("{:.3f}", distance[i]) + ' ' + '0.000' + ' TE '
                               + str.format("{:.6E}", z_TE[j, i].real) + ' ' + str.format("{:.6E}", z_TE[j, i].imag)
                               + ' ' + str.format("{:.6E}", self.error[j, i, 0, 1]) + '\n')
                elif impedance_type == 'TM_Impedance':
                    file.write(str.format("{:.6E}", self.periods[j]) + ' ' + str(self.station_id[i]) + ' ' + '0.000'
                               + ' ' + '0.000' + ' ' + '0.000' + ' '
                               + str.format("{:.3f}", distance[i]) + ' ' + '0.000' + ' TM '
                               + str.format("{:.6E}", z_TM[j, i].real) + ' ' + str.format("{:.6E}", z_TM[j, i].imag)
                               + ' ' + str.format("{:.6E}", self.error[j, i, 1, 0]) + '\n')

    def write_template_dat_data(self, file, impedance_type='TE_Impedance', origin=0, distance=[]):
        # 读取TE和TM分量
        z_TE, z_TM = self.z_tensor[:, :, 1, 0], self.z_tensor[:, :, 0, 1]
        # z_TE, z_TM = self.transfer_z_tensor_to_TE_TM()
        #将测线距离作为y
        if len(distance) == 0:
            distance = self.calculate_distance_along_measure_line(origin)
        # 写入dat数据
        for i in range(self.station_id.shape[0]):
            for j in range(self.periods.shape[0]):
                if impedance_type == 'TE_Impedance':
                    file.write(str.format("{:.6E}", self.periods[j]) + ' ' + str(self.station_id[i]) + ' ' + '0.000'
                               + ' ' + '0.000' + ' ' + '0.000' + ' '
                               + str.format("{:.3f}", distance[i]) + ' ' + '0.000' + ' TE '
                               + str.format("{:.6E}", 0) + ' ' + str.format("{:.6E}", 0)
                               + ' ' + str.format("{:.6E}", 0) + '\n')
                elif impedance_type == 'TM_Impedance':
                    file.write(str.format("{:.6E}", self.periods[j]) + ' ' + str(self.station_id[i]) + ' ' + '0.000'
                               + ' ' + '0.000' + ' ' + '0.000' + ' '
                               + str.format("{:.3f}", distance[i]) + ' ' + '0.000' + ' TM '
                               + str.format("{:.6E}", 0) + ' ' + str.format("{:.6E}", 0)
                               + ' ' + str.format("{:.6E}", 0) + '\n')

    def write_dat(self, file_name, impedance_type='ALL_Impedance', origin=0, distance=[], template=False):
        # 写入dat文件
        if template:
            with open(file_name, 'w') as f:
                if impedance_type != 'ALL_Impedance':
                    self.write_2D_dat_header(f, impedance_type, n_period=str(self.periods.shape[0]),
                                             n_station=str(self.station_id.shape[0]))
                    self.write_template_dat_data(f, impedance_type, origin, distance)
                else:
                    self.write_2D_dat_header(f, 'TE_Impedance', n_period=str(self.periods.shape[0]),
                                             n_station=str(self.station_id.shape[0]))
                    self.write_template_dat_data(f, 'TE_Impedance', origin, distance)
                    f.write('\n')
                    self.write_2D_dat_header(f, 'TM_Impedance', n_period=str(self.periods.shape[0]),
                                             n_station=str(self.station_id.shape[0]))
                    self.write_template_dat_data(f, 'TM_Impedance', origin, distance)
        else:
            with open(file_name, 'w') as f:
                if impedance_type != 'ALL_Impedance':
                    self.write_2D_dat_header(f, impedance_type, n_period=str(self.periods.shape[0]),
                                             n_station=str(self.station_id.shape[0]))
                    self.write_dat_data(f, impedance_type, origin, distance)
                else:
                    self.write_2D_dat_header(f, 'TE_Impedance', n_period=str(self.periods.shape[0]),
                                             n_station=str(self.station_id.shape[0]))
                    self.write_dat_data(f, 'TE_Impedance', origin, distance)
                    f.write('\n')
                    self.write_2D_dat_header(f, 'TM_Impedance', n_period=str(self.periods.shape[0]),
                                             n_station=str(self.station_id.shape[0]))
                    self.write_dat_data(f, 'TM_Impedance', origin, distance)
            f.close()


def create_template_dat(station_loc_list, periods_list):
    # 创建dat模板
    station_loc = np.array(station_loc_list)
    periods = np.array(periods_list)
    z_tensor = np.zeros((periods.shape[0], station_loc.shape[0], 2, 2), dtype=complex)
    error = np.zeros((periods.shape[0], station_loc.shape[0], 2, 2), dtype=complex)
    return station_loc, periods, z_tensor, error