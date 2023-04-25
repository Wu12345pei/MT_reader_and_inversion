import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import razorback as rb
import pycwt as wavelet


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.start_day = 0
        self.end_day = 0
        self.n_seg = 24 * (self.end_day - self.start_day)

    def read_file(self, start_day=0, end_day=0):
        self.start_day = start_day
        self.end_day = end_day
        self.n_seg = int(24 * (self.end_day - self.start_day))
        start_line = 24 * 60 * 60 / 5 * start_day
        end_line = 24 * 60 * 60 / 5 * end_day
        df = pd.read_csv(self.file_path, sep='\s+', header=None, names=['Hx', 'Hy', 'Hz', 'Ex', 'Ey'], skiprows=123)
        if start_day != 0 or end_day != 0:
            if start_day >= end_day:
                raise ValueError('start_day >= end_day')
            if start_line < 0:
                start_line = 0
            if end_line > df.shape[0]:
                end_line = df.shape[0]
            start_line = int(start_line)
            end_line = int(end_line)
            df = df.iloc[start_line:end_line, :]
        self.df = df
        return 'Read to self.df already'

    def read_and_remove_ultimate_value(self, start_day=0, end_day=0):
        self.read_file(start_day, end_day)
        df = self.df
        for channel in ['Hx', 'Hy', 'Hz', 'Ex', 'Ey']:
            line_index = df.loc[df[channel] >= 1000, ].index
            for j in line_index:
                df[channel][j] = 0
        self.df = df.interpolate()
        return self.df

    def filter_data(self, filter_type='lowpass', filter_cutoff=0.05, filter_order=4):
        df = self.df
        fs = 1/5
        nyq = 0.5*fs
        cutoff = nyq
        b, a = signal.butter(filter_order, cutoff, btype=filter_type)
        df['Hx'] = signal.filtfilt(b, a, df['Hx'])
        df['Hy'] = signal.filtfilt(b, a, df['Hy'])
        df['Hz'] = signal.filtfilt(b, a, df['Hz'])
        df['Ex'] = signal.filtfilt(b, a, df['Ex'])
        df['Ey'] = signal.filtfilt(b, a, df['Ey'])
        self.df = df
        return df

    def convert_to_frequency_domain(self, df, abs=True):
        num_samples = df.shape[0]
        freq_domain = np.fft.fft(df) / (num_samples/2)
        freq_domain[0] = freq_domain[0] / 2
        if abs:
            freq_domain = np.abs(freq_domain[:num_samples // 2])
        else:
            freq_domain = freq_domain[:num_samples // 2]
        return freq_domain

    def calculate_amplitude_at_one_frequency(self, channel='Hx', frequency=1000):
        df = self.df
        freq_data = self.convert_to_frequency_domain(df[channel], abs=False)
        freq = np.fft.fftfreq(len(df[channel]), 5)
        freq = freq[:len(freq_data)]
        index = np.argmin(np.abs(freq - frequency))
        return freq_data[index]

    def plot_data_in_time_series(self, plot_mode='all_channel', time_unit='s' ):
        df = self.df
        channel = plot_mode
        if plot_mode == 'all_channel':
            for channel in ['Hx', 'Hy', 'Hz', 'Ex', 'Ey']:
                plt.subplot(5, 1, ['Hx', 'Hy', 'Hz', 'Ex', 'Ey'].index(channel) + 1)
                plt.plot(np.arange(0, len(df[channel])*5, 5), df[channel])
        else:
            if time_unit == 's':
                plt.plot(np.arange(0, len(df[channel])*5, 5), df[channel])
            if time_unit == 'h':
                plt.plot(np.arange(0, len(df[channel])*5/3600, 5/3600), df[channel])
        #x轴起点设置为0
        if time_unit == 's':
            plt.xlim(0, len(df[channel])*5)
        if time_unit == 'h':
            plt.xlim(0, len(df[channel])*5/3600)
        plt.ylabel('nT')

    def plot_data_in_period(self, period_cut=2000):
        df = self.df
        for channel in ['Hx', 'Hy', 'Hz', 'Ex', 'Ey']:
            index = ['Hx', 'Hy', 'Hz', 'Ex', 'Ey'].index(channel)
            freq_data = self.convert_to_frequency_domain(df[channel])
            plt.subplot(3, 2, index + 1)
            periods = []
            data_list = []
            freq = np.fft.fftfreq(len(df[channel]), 5)
            for n in range(0, len(freq_data)):
                freq_n = freq[n]
                if freq_n != 0:
                    if (1 / freq_n) < period_cut:
                        periods.append(1 / freq_n)
                        data_list.append(freq_data[n])
            plt.plot(periods, data_list)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(channel)
            plt.ylabel('nT or mV/km')
            plt.tight_layout()

    def cal_hat_matrix(self):
        # data1和data2分别为两个通道的时序数据，将其分为n_seg分段
        data1 = np.array(self.df['Hx'])
        data2 = np.array(self.df['Hy'])
        seg_len = len(data1) // self.n_seg
        data1_seg = data1[:seg_len * self.n_seg].reshape(self.n_seg, seg_len)
        data2_seg = data2[:seg_len * self.n_seg].reshape(self.n_seg, seg_len)

        # 对每个片段进行fft计算
        data1_seg_fft = np.fft.fft(data1_seg, axis=1)
        data2_seg_fft = np.fft.fft(data2_seg, axis=1)

        # Hat-matrix的计算
        h_freq_list = []
        hat_matrix_list = []
        for i in range(seg_len):
            h_freq = [data1_seg_fft.T[i], data2_seg_fft.T[i]]
            h_freq_list.append(h_freq)
        for i in range(seg_len):
            h_freq = np.array(h_freq_list[i]).T
            h_matrix = np.dot(np.dot(h_freq, np.linalg.inv((np.dot(np.conj(h_freq).T, h_freq)))), np.conj(h_freq).T)
            hat_matrix_list.append(h_matrix)
        return hat_matrix_list

    def plot_hat_matrix(self):
        df = self.df
        n_seg = self.n_seg
        seg_len = df.shape[0] // n_seg
        time_list = [i for i in range(1, n_seg+1)]
        hat_matrix_list = self.cal_hat_matrix()

        n = 35
        period = 1 / (n / seg_len / 5)
        hat_matrix = np.array(hat_matrix_list[n])
        diagonal = np.diagonal(hat_matrix)
        normalized_diagonal = abs(diagonal) / (2 / n_seg)
        plt.axhline(1)
        plt.plot(time_list, normalized_diagonal, label='date{0:g}-{1:g} period {2:g}'.format(self.start_day,
                                                                                             self.end_day, period))
        plt.xlabel('hours')
        plt.ylabel('Normalized Hat-matrix')
        plt.xlim(0)
        plt.legend()

    def plot_phase_difference(self, n_seg):
        df = self.df
        num_hours = n_seg
        hour_length = df.shape[0] // num_hours

        # 分割每个小时的数据，并将它们组成新的DataFrame
        hourly_data = []
        for i in range(num_hours):
            start_index = i * hour_length
            end_index = (i + 1) * hour_length
            hourly_slice = df.iloc[start_index:end_index, :]
            hourly_data.append(hourly_slice)
        hourly_df = pd.concat(hourly_data)

        # 计算每个小时内的电场分量与磁场分量的相位差
        phase_diffs = []
        for i in range(num_hours):
            start_index = i * hour_length
            end_index = (i + 1) * hour_length
            hourly_slice = hourly_df.iloc[start_index:end_index, :]
            H = hourly_slice.loc[:, ['Hx', 'Hy']].values.T
            E = hourly_slice.loc[:, ['Ex', 'Ey']].values.T
            # 计算H和E的时频图
            f, t, Ht = signal.stft(H, nperseg=100, noverlap=50)
            f, t, Et = signal.stft(E, nperseg=100, noverlap=50)
            # 计算H和E的相位
            H_phase = np.angle(Ht).reshape(-1, 9)
            E_phase = np.angle(Et).reshape(-1, 9)
            # 计算相位差
            phase_diff = E_phase - H_phase
            phase_diffs.append(phase_diff)

        # 将相位差绘制成图像，其中每列代表一个小时
        phase_diffs = np.array(phase_diffs)
        plt.imshow(phase_diffs, aspect='auto')
        plt.colorbar()
        plt.xlabel('Hours')
        plt.ylabel('Frequency Bins')
        plt.title('Phase Difference between E and H')

    def cut_series_by_hat_matrix(self):
        df = self.df
        cut_df_list = []
        n_seg = self.n_seg
        seg_len = df.shape[0] // n_seg
        n = 35
        period = 1 / (n / seg_len / 5)
        hat_matrix_list = self.cal_hat_matrix()
        hat_matrix = np.array(hat_matrix_list[n])
        diagonal = np.diagonal(hat_matrix)
        normalized_diagonal = abs(diagonal) / (2 / n_seg)
        time_list = [i for i in range(n_seg)]
        picked_time = []
        hour_time = 0
        for diagonal_by_time in normalized_diagonal:
            if diagonal_by_time >= 1:
                picked_time.append(hour_time)
            hour_time += 1
        for i in picked_time:
            start_line = int(i * 3600 / 5)
            end_line = int((1 + i) * 3600 / 5)
            cut_df = df.iloc[start_line:end_line, :]
            cut_df_list.append(cut_df)
        return cut_df_list

    def cal_power_spectral_density(self, data1, data2, n_seg):
        # data1和data2分别为两个通道的时序数据，将其对齐
        fs = 1 / 5
        data1 = np.array(data1)
        data2 = np.array(data2)
        n = min(len(data1), len(data2))
        data1 = data1[:n]
        data2 = data2[:n]
        # 计算功率谱密度
        seg_len = n // n_seg
        freq, pxy = signal.csd(data1, data2, fs=fs, nperseg=seg_len, noverlap=seg_len // 2)

        return pxy, freq

    def cal_coherence(self, data1, data2, n_seg):
        # 计算data1和data2的自功率谱和互功率谱
        pxx, _ = self.cal_power_spectral_density(data1, data1, n_seg)
        pxy, _ = self.cal_power_spectral_density(data1, data2, n_seg)
        pyy, _ = self.cal_power_spectral_density(data2, data2, n_seg)
        # 计算相关系数
        coherence = abs(pxy) / np.sqrt(np.abs(pxx * pyy))
        # 返回相关系数和频率
        return coherence, _

    def plot_coherence_between_EH(self, n_seg=12, plot_label='storm', plot_mode='all_channel'):
        # 计算ExHx、EyHy、ExHy、EyHx的相关系数
        ex = self.df['Ex']
        ey = self.df['Ey']
        hx = self.df['Hx']
        hy = self.df['Hy']
        coherence_ExHx, freq = self.cal_coherence(ex, hx, n_seg)
        coherence_EyHy, _ = self.cal_coherence(ey, hy, n_seg)
        coherence_ExHy, _ = self.cal_coherence(ex, hy, n_seg)
        coherence_EyHx, _ = self.cal_coherence(ey, hx, n_seg)
        # 将频率转化到周期,排除掉0Hz
        freq = 1 / freq[1:]
        coherence_ExHx = coherence_ExHx[1:]
        coherence_EyHy = coherence_EyHy[1:]
        coherence_ExHy = coherence_ExHy[1:]
        coherence_EyHx = coherence_EyHx[1:]
        # 绘制相关系数图
        if plot_mode == 'all_channel':
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.scatter(freq, [coherence_ExHx, coherence_EyHy, coherence_ExHy, coherence_EyHx][i], label=plot_label, s=5
                            )
                plt.xlabel('Period(s)')
                plt.ylabel('Coherence')
                plt.title(['ExHx', 'EyHy', 'ExHy', 'EyHx'][i])
                plt.xscale('log')
                plt.legend()
            plt.tight_layout()
        if plot_mode == 'orthorhombic':
            plt.subplot(1, 2, 1)
            plt.scatter(freq, coherence_ExHy, label=plot_label, s=5)
            plt.xlabel('Period(s)')
            plt.ylabel('Coherence')
            plt.title('ExHy')
            plt.xscale('log')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.scatter(freq, coherence_EyHx, label=plot_label, s=5)
            plt.xlabel('Period(s)')
            plt.ylabel('Coherence')
            plt.title('EyHx')
            plt.xscale('log')
            plt.legend()
            plt.tight_layout()

    def cal_impedance(self, n_seg):
        hx = self.df['Hx']
        hy = self.df['Hy']
        ex = self.df['Ex']
        ey = self.df['Ey']
        spec_HxHx, _ = self.cal_power_spectral_density(hx, hx, n_seg)
        spec_HxHy, _ = self.cal_power_spectral_density(hx, hy, n_seg)
        spec_HyHy, _ = self.cal_power_spectral_density(hy, hy, n_seg)
        spec_HyHx, _ = self.cal_power_spectral_density(hy, hx, n_seg)
        spec_ExHy, _ = self.cal_power_spectral_density(ex, hy, n_seg)
        spec_ExHx, _ = self.cal_power_spectral_density(ex, hx, n_seg)
        spec_EyHy, _ = self.cal_power_spectral_density(ey, hy, n_seg)
        spec_EyHx, _ = self.cal_power_spectral_density(ey, hx, n_seg)
        Zxx = (spec_ExHy * spec_HyHx - spec_ExHx * spec_HyHy) / (spec_HxHy * spec_HyHx - spec_HxHx * spec_HyHy)
        Zxy = (spec_HxHx * spec_ExHy - spec_HxHy * spec_ExHx) / (spec_HxHx * spec_HyHy - spec_HxHy * spec_HyHx)
        Zyx = (spec_EyHy * spec_HyHx - spec_EyHx * spec_HyHy) / (spec_HxHy * spec_HyHx - spec_HxHx * spec_HyHy)
        Zyy = (spec_EyHy * spec_HxHx - spec_EyHx * spec_HyHy) / (spec_HxHx * spec_HyHy - spec_HxHy * spec_HyHx)
        return Zxx, Zxy, Zyx, Zyy, _

    def plot_all_impedance(self, n_seg):
        Zxx, Zxy, Zyx, Zyy, freq = self.cal_impedance(n_seg)
        freq = 1 / freq[1:]
        Zxx = Zxx[1:]
        Zxy = Zxy[1:]
        Zyx = Zyx[1:]
        Zyy = Zyy[1:]
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.scatter(freq, [abs(Zxx), abs(Zxy), abs(Zyx), abs(Zyy)][i], s=5)
            plt.xlabel('Period(s)')
            plt.xscale('log')
            plt.ylabel(['Zxx', 'Zxy', 'Zyx', 'Zyy'][i])

    def transfer_data_to_signal_set(self):
        hx = np.array(self.df['Hx'])
        hy = np.array(self.df['Hy'])
        hz = np.array(self.df['Hz'])
        ex = np.array(self.df['Ex'])
        ey = np.array(self.df['Ey'])
        signal_set = rb.SignalSet({'Bx': 0, 'By': 1, 'Bz': 2, 'Ex': 3, 'Ey': 4, 'B': (0, 1), 'E': (3, 4)},
                                  rb.SyncSignal([hx, hy, hz, ex, ey], 0.2, 0))
        return signal_set

    def cal_impedance_by_rb(self, periods_num=100, periods=None):
        signal_set = self.transfer_data_to_signal_set()
        if periods is None:
            periods = np.logspace(1, 3.5, periods_num)
            freq = 1 / periods
            freq = freq[::-1]
        else:
            freq = 1 / periods
            freq = freq[::-1]

        # 计算阻抗
        res = rb.utils.impedance(signal_set, freq)
        return res.impedance, res.error

    def plot_impedance_by_rb(self, plot_label='storm', periods_num=100):
        impedance_by_freq, _ = self.cal_impedance_by_rb(periods_num)
        periods = np.logspace(1, 3.5, periods_num)
        periods = periods[::-1]
        Zxy_list = []
        Zyx_list = []
        Zxx_list = []
        Zyy_list = []
        for i in range(len(periods)):
            impedance = impedance_by_freq[i]
            z_list = [abs(impedance[0][0]), abs(impedance[0][1]), abs(impedance[1][0]), abs(impedance[1][1])]
            Zxx_list.append(z_list[0])
            Zxy_list.append(z_list[1])
            Zyx_list.append(z_list[2])
            Zyy_list.append(z_list[3])
        for j in range(4):
            plt.subplot(2, 2, j + 1)
            plt.scatter(periods, [Zxx_list, Zxy_list, Zyx_list, Zyy_list][j], s=5, label=plot_label)
            plt.xlabel('Period(s)')
            plt.xscale('log')
            plt.legend()
            plt.ylabel(['Zxx', 'Zxy', 'Zyx', 'Zyy'][j])

    def wavelet_transform(self):
        hx = np.array(self.df['Hx'])
        hy = np.array(self.df['Hy'])
        ex = np.array(self.df['Ex'])
        ey = np.array(self.df['Ey'])
        time = np.arange(0, len(hx) * 5, 5)
        #小波尺度选择
        wave = 'morlet'
        scales = np.arange(1, 2000, 10)
        #计算小波变换
        cwt_hx_wave, scales, freqs, coi, fft, fftfreqs= wavelet.cwt(hx, 5, wavelet=wave)
        cwt_hy_wave, scales, freqs, coi, fft, fftfreqs= wavelet.cwt(hy, 5, wavelet=wave)
        cwt_ex_wave, scales, freqs, coi, fft, fftfreqs= wavelet.cwt(ex, 5, wavelet=wave)
        cwt_ey_wave, scales, freqs, coi, fft, fftfreqs= wavelet.cwt(ey, 5, wavelet=wave)
        period = 1 / freqs
        cut_period = 30000
        if cut_period > period[-1]:
            cut_period = period[-1] - 1
        cut_period_index = np.where(period > cut_period)[0][0]
        cwt_hx_wave = cwt_hx_wave[:cut_period_index]
        cwt_hy_wave = cwt_hy_wave[:cut_period_index]
        cwt_ex_wave = cwt_ex_wave[:cut_period_index]
        cwt_ey_wave = cwt_ey_wave[:cut_period_index]
        period = period[:cut_period_index]
        return cwt_hx_wave, cwt_hy_wave, cwt_ex_wave, cwt_ey_wave, time, period

    def plot_wavelet_transform(self, plot_mode='all_channel', split_line=False, bar=True):
        cwt_hx_wave, cwt_hy_wave, cwt_ex_wave, cwt_ey_wave, time, period = self.wavelet_transform()
        plot_list = []
        if plot_mode == 'all_channel':
            plot_list = ['hx', 'hy', 'ex', 'ey']
        elif plot_mode == 'hx':
            plot_list = ['hx']
        elif plot_mode == 'hy':
            plot_list = ['hy']
        elif plot_mode == 'ex':
            plot_list = ['ex']
        elif plot_mode == 'ey':
            plot_list = ['ey']
        period = np.array([np.log10(i) for i in period])
        time = time / 3600
        #绘图
        if 'hx' in plot_list:
            if plot_mode == 'all_channel':
                plt.subplot(2, 2, 1)
            plt.pcolormesh(time, period, abs(cwt_hx_wave), cmap='jet_r', vmin=0, vmax=4000)
            plt.xlabel('Time(h)')
            plt.ylabel('Period')
            if split_line:
                plt.axvline(x=time[-1]/2, color='black', linestyle='--')
            plt.title('Hx')
            if bar:
                plt.colorbar()
        if 'hy' in plot_list:
            if plot_mode == 'all_channel':
                plt.subplot(2, 2, 2)
            plt.pcolormesh(time, period, abs(cwt_hy_wave), cmap='jet_r', vmin=0, vmax=4000)
            plt.xlabel('Time(h)')
            plt.ylabel('Period')
            if split_line:
                plt.axvline(x=time[-1]/2, color='black', linestyle='--')
            plt.title('Hy')
            if bar:
                plt.colorbar()
        if 'ex' in plot_list:
            if plot_mode == 'all_channel':
                plt.subplot(2, 2, 3)
            plt.pcolormesh(time, period, abs(cwt_ex_wave), cmap='jet_r', vmin=0, vmax=1500)
            plt.xlabel('Time(h)')
            plt.ylabel('Period')
            if split_line:
                plt.axvline(x=time[-1]/2, color='black', linestyle='--')
            plt.title('Ex')
            if bar:
                plt.colorbar()
        if 'ey' in plot_list:
            if plot_mode == 'all_channel':
                plt.subplot(2, 2, 4)
            plt.pcolormesh(time, period, abs(cwt_ey_wave), cmap='jet_r', vmin=0, vmax=1500)
            plt.xlabel('Time(h)')
            plt.ylabel('Period')
            if split_line:
                plt.axvline(x=time[-1]/2, color='black', linestyle='--')
            plt.title('Ey')
            if bar:
                plt.colorbar()

    def plot_wavelet_transform_3D(self, plot_mode='all_channel'):
        cwt_hx_wave, cwt_hy_wave, cwt_ex_wave, cwt_ey_wave, time, period = self.wavelet_transform()
        period = [np.log10(i) for i in period]
        plot_list = []
        if plot_mode == 'all_channel':
            plot_list = ['hx', 'hy', 'ex', 'ey']
        elif plot_mode == 'hx':
            plot_list =['hx']
        elif plot_mode == 'hy':
            plot_list = ['hy']
        elif plot_mode == 'ex':
            plot_list = ['ex']
        elif plot_mode == 'ey':
            plot_list = ['ey']
        #绘图
        if 'hx' in plot_list:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(time, period)
            ax.plot_surface(X, Y, abs(cwt_hx_wave))
            ax.set_xlabel('Time')
            ax.set_ylabel('Period')
            ax.set_zlabel('Amplitude')
            ax.set_title('Hx')
        if 'hy' in plot_list:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(time, period)
            ax.plot_surface(X, Y, abs(cwt_hy_wave))
            ax.set_xlabel('Time')
            ax.set_ylabel('Period')
            ax.set_zlabel('Amplitude')
            ax.set_title('Hy')
        if 'ex' in plot_list:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(time, period)
            ax.plot_surface(X, Y, abs(cwt_ex_wave))
            ax.set_xlabel('Time')
            ax.set_ylabel('Period')
            ax.set_zlabel('Amplitude')
            ax.set_title('Ex')
        if 'ey' in plot_list:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(time, period)
            ax.plot_surface(X, Y, abs(cwt_ey_wave))
            ax.set_xlabel('Time')
            ax.set_ylabel('Period')
            ax.set_zlabel('Amplitude')
            ax.set_title('Ey')





