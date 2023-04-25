import numpy
import numpy as np

import Plotter
import data_processor
import file_controller
from data_processor import DataProcessor
import matplotlib.pyplot as plt
import DatWriter
import Model_Writer
from Plotter import Plotter_2D
import os


def process_1_forward_and_inverse_block(y_start=450000, y_end=1050000, z_start=200000, z_end=500000, inv=True,
                                        add_noise=False, data_from='kap163', data_in='storm', plot=True, block_rho=6,
                                        noise_scale=1):
    #模型文件
    length = 1500000
    T = 30000
    skin_depth = 1000 / 2 / np.pi * np.sqrt(T*1000)
    path = 'C:/Users/WuPeihong/Desktop/ModEM/examples/2D_MT/forward/'
    Mod2D = Model_Writer.ModelWriter2D(file_name=path + 'testmod.rho', station_num=25, period_num=20, length_y=1500000,
                                       skin_depth=skin_depth)
    origin_begin = Mod2D.write_mesh(resistivity=block_rho, y_start=y_start, y_end=y_end, z_start=z_start, z_end=z_end)

    #数据模板文件
    station_loc_list = np.linspace(0, length, 25) + origin_begin
    period_list = np.logspace(np.log10(20), np.log10(T), 20)
    station_loc, periods, z_tensor, error = DatWriter.create_template_dat(station_loc_list, period_list)
    Template = DatWriter.DatWriter(station_id=np.array([i for i in range(len(station_loc_list))]), station_loc=station_loc,
                                   periods=periods, z_tensor=z_tensor, error=error)
    Template.write_dat(path + 'testmod.dat', distance=station_loc_list, template=True)

    #绘图
    if plot:
        Mod_Plotter = Plotter_2D(path + 'testmod.rho')
        Mod_Plotter.plot_rho(skin_depth=skin_depth)
        plt.show()

    #在正演文件夹打开linixshell并运行./run.sh
    os.chdir(path)
    os.system('bash run.sh')

    #修改正演error
    pl = Plotter.plot_dat_z_tensor(path + 'forward.dat')
    pl.rewrite_error(error=0.03)

    #绘图正演结果
    forward_result_plot = Plotter.plot_dat_z_tensor('forward.dat')
    forward_result_plot.read_dat()
    if plot:
        forward_result_plot.plot_z_tensor(12)
        plt.show()

    #取kap163的磁信号对应频率的数据Hx, Hy
    os.chdir('C:/Users/WuPeihong/Desktop/codes/work/kap03')
    name = 'kap163as.ts'
    station_processor = DataProcessor(name)
    storm_day = file_controller.get_begin_days(name)
    if data_in == 'storm':
        start_day = storm_day
    elif data_in == 'usual':
        start_day = storm_day + 3
    station_processor.read_and_remove_ultimate_value(start_day, start_day + 3)
    if plot:
        station_processor.plot_wavelet_transform(split_line=True)
        plt.show()
    Hx_freq_domain = []
    Hy_freq_domain = []
    for period_i in period_list:
        Hx_freq_domain.append(station_processor.calculate_amplitude_at_one_frequency('Hx', 1/period_i))
        Hy_freq_domain.append(station_processor.calculate_amplitude_at_one_frequency('Hy', 1/period_i))
    if plot:
        plt.scatter(period_list, np.abs(Hx_freq_domain), label='Hx')
        plt.scatter(period_list, np.abs(Hy_freq_domain), label='Hy')
        plt.legend()
        plt.show()

    #利用E=Z*H计算Ex, Ey
    os.chdir(path)
    Ex, Ey = np.zeros(shape=(20, 25), dtype=complex), np.zeros(shape=(20, 25), dtype=complex)
    for i in range(25):
        _, zTE, zTM = forward_result_plot.get_zTE_zTM(i)
        for j in range(20):
            Ex[j, i] = zTM[j] * Hy_freq_domain[j]
            Ey[j, i] = zTE[j] * Hx_freq_domain[j]
    if plot:
        plt.scatter(period_list, np.abs(Ex[:, 12]), label='Ex')
        plt.scatter(period_list, np.abs(Ey[:, 12]), label='Ey')
        plt.legend()
        plt.show()

    #添加噪声
    if add_noise:
        length = int(3 * 24 * 3600 / 5)
        noise = np.random.normal(1, noise_scale, size=(length, 25))
        noise_spectrum = np.fft.fft(noise) / (length / 2)
        noise_spectrum[0, :] = noise_spectrum[0, :] / 2
        noise_spectrum = noise_spectrum[:length // 2, :]
        freq = np.fft.fftfreq(length, 5)
        freq = freq[:length // 2]
        for i in range(len(period_list)):
            frequency = 1 / period_list[i]
            index = np.argmin(np.abs(freq - frequency))
            Ex[i, :] += noise_spectrum[index, :]
            Ey[i, :] += noise_spectrum[index, :]
    if plot:
        plt.scatter(period_list, np.abs(Ex[:, 12]), label='Ex')
        plt.scatter(period_list, np.abs(Ey[:, 12]), label='Ey')
        plt.legend()
        plt.show()

    #重新计算各台站各频段的阻抗
    zTE, zTM = np.zeros(shape=(20, 25), dtype=complex), np.zeros(shape=(20, 25), dtype=complex)
    for i in range(25):
        for j in range(20):
            zTE[j, i] = numpy.divide(Ey[j, i], Hx_freq_domain[j], dtype=complex)
            zTM[j, i] = numpy.divide(Ex[j, i], Hy_freq_domain[j], dtype=complex)
    if plot:
        plt.scatter(period_list, np.abs(zTE[:, 12]), label='zTE')
        plt.scatter(period_list, np.abs(zTM[:, 12]), label='zTM')
        plt.legend()
        plt.show()

    #将阻抗写入新的文件
    os.chdir(path)
    Z_tensor = np.zeros(shape=(20, 25, 2, 2), dtype=complex)
    Z_tensor[:, :, 0, 1] = zTM
    Z_tensor[:, :, 1, 0] = zTE
    Zdat = DatWriter.DatWriter(station_id=np.array([i for i in range(len(station_loc_list))]), station_loc=station_loc,
                               periods=periods, z_tensor=Z_tensor, error=np.abs(Z_tensor)*0.01)
    Zdat.write_dat(path + 'testmod2.dat', distance=station_loc_list, template=False)

    if inv == False:
        return

    #创建反演文件
    inv_path = 'C:/Users/WuPeihong/Desktop/ModEM/examples/2D_MT/inversion/'
    Mod2Dinv = Model_Writer.ModelWriter2D(file_name=inv_path + 'testmod.rho', station_num=25, period_num=20,
                                          length_y=1500000, skin_depth=skin_depth)
    origin_begin = Mod2Dinv.write_mesh(resistivity=4.6)
    file_controller.copy_file(path + 'testmod2.dat', inv_path + 'forward.dat')

    #在反演文件夹打开linixshell并运行./run.sh
    os.chdir(inv_path)
    os.system('bash run.sh')

    #绘图反演结果
    #找出反演文件夹中序列号最大的文件
    inversion_file = file_controller.get_max_rho_file(inv_path)
    print(inversion_file)
    inversion_plotter = Plotter_2D(inversion_file)
    inversion_plotter.plot_rho(skin_depth=skin_depth,title=data_in)


def compare_storm_and_usual(y_start=450000, y_end=1050000, z_start=0, z_end=300000, noise_scale=1):
    plt.subplot(2, 1, 1)
    process_1_forward_and_inverse_block(y_start=y_start, y_end=y_end, z_start=z_start, z_end=z_end, inv=True,
                                        add_noise=True, data_from='kap163', data_in='storm', plot=False,
                                        block_rho=4, noise_scale=noise_scale)
    plt.subplot(2, 1, 2)
    process_1_forward_and_inverse_block(y_start=y_start, y_end=y_end, z_start=z_start, z_end=z_end, inv=True,
                                        add_noise=True, data_from='kap163', data_in='usual', plot=False,
                                        block_rho=4, noise_scale=noise_scale)
    plt.show()


def compare_noise_and_quiet():
    plt.subplot(2, 1, 1)
    process_1_forward_and_inverse_block(z_start=300000, z_end=600000, inv=True, add_noise=True, data_from='kap163',
                                        data_in='storm', plot=False, block_rho=2)
    plt.subplot(2, 1, 2)
    process_1_forward_and_inverse_block(z_start=300000, z_end=600000, inv=True, add_noise=False, data_from='kap163',
                                        data_in='storm', plot=False, block_rho=2)
    plt.show()

def inverse(use='edi'):
    # 读取edi文件，获取各频率阻抗张量及误差,读取ts文件，获取台站位置
    path = 'kap03'
    os.chdir(path)
    station_ids = []
    edi_files = []
    ts_files = []
    z_tensor_array = np.zeros(shape=(20, 25, 2, 2), dtype=complex)
    z_tensor_error_array = np.zeros(shape=(20, 25, 2, 2))
    for station_id in range(100, 176):
        if os.path.exists('kap'+str(station_id)+'.edi') and station_id != 152:
            station_ids.append(station_id)
            edi_path = 'kap'+str(station_id)+'.edi'
            ts_path = 'kap'+str(station_id)+'as.ts'
            edi_files.append(edi_path)
            ts_files.append(ts_path)
    edi_reader = DatWriter.EDIReader(edi_files)
    ts_reader = DatWriter.TsReader(ts_files)
    periods, z_tensor, z_tensor_error = edi_reader.get_periods_and_z_tensor()
    station_loc = ts_reader.get_station_loc()

    if use == 'ts':
        # 提取含有风暴期的台站
        storm_ts_files = []
        storm_station_indexes = []
        for station_id in range(127, 170):
            if station_id in station_ids and station_id != 152 and station_id != 151:
                storm_station_index = station_ids.index(station_id)
                storm_stations_ts_path = ts_files[storm_station_index]
                storm_ts_files.append(storm_stations_ts_path)
                storm_station_indexes.append(storm_station_index)
        storm_ts_readers = DatWriter.TsReader(storm_ts_files)
        periods_cal, z_tensor_cal, z_tensor_error_cal = storm_ts_readers.get_periods_and_z_tensor(period_list=periods)

        # 将风暴期的阻抗张量替换到原始阻抗张量中
        for i in range(len(storm_station_indexes)):
            z_tensor[:, storm_station_indexes[i], :, :] = z_tensor_cal[:, i, :, :]
            z_tensor_error[:, storm_station_indexes[i], :, :] = z_tensor_error_cal[:, i, :, :]

    z_tensor_error_modified = np.abs(z_tensor) * 0.1

    # 创建反演模型
    T = 18000
    skin_depth = 1000 / 2 / np.pi * np.sqrt(T*1000)
    path_for_inversion = 'C:/Users/WuPeihong/Desktop/ModEM/examples/2D_MT/kap03inversion/'
    Mod2Dinv = Model_Writer.ModelWriter2D(file_name=path_for_inversion + 'kap03.rho', station_num=25, period_num=20,
                                          length_y=1500000, skin_depth=skin_depth)
    origin_begin = Mod2Dinv.write_mesh(resistivity=4.6)

    # 将阻抗张量写入dat文件
    dat_writer = DatWriter.DatWriter(station_id=np.array(station_ids), station_loc=station_loc, periods=periods,
                                     z_tensor=z_tensor, error=z_tensor_error_modified)
    dat_writer.write_dat(path_for_inversion + 'kap03.dat', origin=origin_begin)

    # 在反演文件夹打开linixshell并运行./run.sh
    os.chdir(path_for_inversion)
    os.system('bash run.sh')

    # 绘图反演结果
    inversion_file = file_controller.get_max_rho_file(path_for_inversion)
    print(inversion_file)
    inversion_plotter = Plotter_2D(inversion_file)
    inversion_plotter.plot_rho(skin_depth=skin_depth)

T = 18000
skin_depth = 1000 / 2 / np.pi * np.sqrt(T*1000)
path_for_inversion = 'C:/Users/WuPeihong/Desktop/ModEM/examples/2D_MT/kap03inversion/'
inversion_file = file_controller.get_max_rho_file(path_for_inversion)
print(inversion_file)
inversion_plotter = Plotter_2D(inversion_file)
inversion_plotter.plot_rho(skin_depth=skin_depth, vmin=2, vmax=6)
plt.show()