import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, spectrogram as compute_spec, correlate

def load_excitation(excitation_path):
    excitation, fs = sf.read(excitation_path)
    if excitation.ndim > 1:
        excitation = excitation[:, 0]
    inv_filter = excitation[::-1]
    N = len(excitation)
    return inv_filter, N, fs

def compute_ir(data, inv_filter, N, start_sample, end_sample):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    irs = []
    for ch in range(data.shape[1]):
        ir_full = fftconvolve(data[:, ch], inv_filter, mode='full')
        ir_full = ir_full[N:N * 2]
        irs.append(ir_full[start_sample:end_sample])
    return irs

def align_signals(ref_ir, target_ir):
    ref_norm = ref_ir / (np.max(np.abs(ref_ir)) + 1e-8)
    target_norm = target_ir / (np.max(np.abs(target_ir)) + 1e-8)
    corr = correlate(target_norm, ref_norm, mode='full')
    lag = np.argmax(corr) - (len(ref_norm) - 1)
    aligned = np.roll(target_ir, -lag)
    return aligned, lag

def compute_spectrogram(ir, fs, nperseg=256, noverlap=128):
    _, _, Sxx = compute_spec(ir, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return Sxx