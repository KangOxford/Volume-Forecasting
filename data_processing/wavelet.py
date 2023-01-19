import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')



def denoise_single_column_sr(szzz_sr):

    res_4 = pywt.wavedec(np.array(szzz_sr.values), wavelet=' sym8', mode=' symmetric',level=4)
    # The four parameters correspond to the input signal,
    # the wavelet name, the signal expansion mode, and the decomposition order

    def threshold_cal(signal):
        # Acigrsuce method to calculate the threshold
        signal = abs(signal)
        signal.sort()
        signal = signal**2
        list_risk_j = []; N = len(signal)
        for j in range(N):
            if j == 0:
                risk_j = 1 + signal[N-1]
            else:
                risk_j = (N-2*j + (N-j)*(signal[N-j]) + sum(signal[:j]))/N
        list_risk_j.append(risk_j)
        k = np.array(list_risk_j).argmin()
        threshold = np.sqrt(signal[k])
        return threshold

    for j in [2,3,4]:
    # The high frequency signal is processed by the diaphragm value
        signal = np.array(res_4[j])
        threshold = threshold_cal(signal)
    # fixed threshold. Sgt(2*nR.1gg(len(signal)))
        res_4[j] = pywt.threshold(signal, threshold, "soft")
    rec_szzz = pywt.waverec(res_4, 'sym8')
    result = rec_szzz[:-1]
    return result

if __name__ == "__main__":
    pass
