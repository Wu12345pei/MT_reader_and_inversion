import numpy as np
import razorback as rzb


def make_data(factor=2, nsr=1e-12, size=5000):
    rg = np.random.default_rng(12345)
    time = np.linspace(0, 10, size)
    inputs = np.sin((time[-1]-time)**2)
    outputs = factor * inputs
    inputs += rg.normal(scale=nsr, size=inputs.shape)
    outputs += rg.normal(scale=nsr, size=outputs.shape)
    data = rzb.SignalSet({'B': 0, 'E': 1}, rzb.SyncSignal([inputs, outputs], 0.2, 0))
    return data


def test_scalar_factor():
    data = make_data(2)
    freqs = np.logspace(np.log10(1e-1), np.log10(0.5), 5) * data.sampling_rates[0]
    res = rzb.utils.impedance(data, freqs)
    np.testing.assert_allclose(res.impedance, 2)
    assert [len(e) for e in res.invalid_time] == [1] * 5
    assert [[np.shape(ivt) for ivt in ivt_line] for ivt_line in res.invalid_time
    ] == [[(0,)], [(0,)], [(0,)], [(0,)], [(0,)]]

    res = rzb.utils.impedance(data, freqs, weights=rzb.weights.mest_weights)
    np.testing.assert_allclose(res.impedance, 2)
    assert [len(e) for e in res.invalid_time] == [1] * 5
    assert [[np.shape(ivt) for ivt in ivt_line] for ivt_line in res.invalid_time
    ] == [[(0,)], [(0,)], [(0,)], [(0,)], [(0,)]]

    res = rzb.utils.impedance(data, freqs, weights=rzb.weights.bi_weights(0.1, 3, 1))
    np.testing.assert_allclose(res.impedance, 2)
    assert [len(e) for e in res.invalid_time] == [1] * 5
    assert [[np.shape(ivt) for ivt in ivt_line] for ivt_line in res.invalid_time
    ] == [[(0,)], [(0,)], [(0,)], [(0,)], [(0,)]]

    res = rzb.utils.impedance(data, freqs, weights=rzb.weights.bi_weights(0.1, 3, 1),
        keep_invalid_times=True,
    )
    np.testing.assert_allclose(res.impedance, 2)
    assert [len(e) for e in res.invalid_time] == [1] * 5
    assert [[np.shape(ivt) for ivt in ivt_line] for ivt_line in res.invalid_time
    ] == [[(75,)], [(16,)], [(30,)], [(165,)], [(530,)]]

    res = rzb.utils.impedance(data, freqs, remote='B')
    np.testing.assert_allclose(res.impedance, 2)
    assert [len(e) for e in res.invalid_time] == [1] * 5
    assert [[np.shape(ivt) for ivt in ivt_line] for ivt_line in res.invalid_time
    ] == [[(0,)], [(0,)], [(0,)], [(0,)], [(0,)]]

test_scalar_factor()

def test_fail_freq_to_big():
    data = make_data(2)
    freqs = [data.sampling_rates[0]]
    res = rzb.utils.impedance(data, freqs)
    np.testing.assert_allclose(res.impedance, np.nan)
    np.testing.assert_allclose(res.error, np.nan)
    assert res.invalid_time == [()]


def test_fail_freq_to_small():
    data = make_data(2)
    freqs = [1e-5 * data.sampling_rates[0]]
    res = rzb.utils.impedance(data, freqs)
    np.testing.assert_allclose(res.impedance, np.nan)
    np.testing.assert_allclose(res.error, np.nan)
    assert res.invalid_time == [()]

def convert_to_frequency_domain(df):
    num_samples = df.shape[0]
    freq_domain = np.fft.fft(df)
    freq_domain = np.abs(freq_domain[:num_samples // 2])
    return freq_domain

import numpy as np
import matplotlib.pyplot as plt
df = np.array(np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000))
print(1/6.28)
time = np.linspace(0, 10, 1000)
plt.plot(time, df)
plt.show()
freq = np.fft.fftfreq(df.shape[0], 1/100)
freq = freq[:df.shape[0] // 2]
freq_domain = convert_to_frequency_domain(df)
print(freq[0:10], freq_domain[0:10])
plt.scatter(freq, freq_domain, s=1)
plt.show()