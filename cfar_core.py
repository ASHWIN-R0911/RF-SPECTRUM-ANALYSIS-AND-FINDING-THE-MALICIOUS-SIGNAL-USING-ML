"""
cfar_core.py - CA-CFAR detector (same algorithm and defaults as cacfar.py v2),
packaged as an importable function so classify_live.py does not need its own copy.

    from cfar_core import detect
    res = detect(samples, fs)            # N_FFT=1024, 10-frame average, 16/4 cells, Pfa=1e-4
    res["detected"]                      # False -> nothing crossed the CFAR threshold
    res["center_freq_hz"], res["bandwidth_hz"], res["peak_db"], res["noise_floor_db"]

No fallback: if no bin exceeds the threshold, detected is False and no region is invented.
"""

import numpy as np

try:
    from scipy.stats import f as f_dist
except ImportError:
    f_dist = None


def averaged_power_spectrum(x, n_fft, n_avg):
    x = np.asarray(x, float) - np.mean(x)
    n_frames = len(x) // n_fft
    if n_frames < 1:
        raise ValueError(f"signal shorter than one FFT frame ({len(x)} < {n_fft})")
    k = min(n_avg, n_frames)
    starts = np.unique(np.linspace(0, n_frames - 1, k).astype(int)) * n_fft
    win = np.hanning(n_fft)
    win_pow = np.sum(win ** 2)
    P = np.zeros(n_fft // 2 + 1)
    for s in starts:
        P += np.abs(np.fft.rfft(x[s:s + n_fft] * win)) ** 2 / win_pow
    return P / len(starts), len(starts)


def cfar_alpha(n_ref_total, k_frames, pfa):
    """Detect when P_cut > alpha * mean(reference cells). Exact for K averaged frames:
    P_cut / mean_ref ~ F(2K, 2K*N) under noise only. K = 1 gives N*(pfa^(-1/N) - 1)."""
    if f_dist is not None:
        return float(f_dist.isf(pfa, 2 * k_frames, 2 * k_frames * n_ref_total))
    return float(n_ref_total * (pfa ** (-1.0 / n_ref_total) - 1))


def detect(x, fs, n_fft=1024, n_avg=10, n_ref=16, n_guard=4, pfa=1e-4,
           dc_ignore=3, bw_drop_db=10.0, merge_gap=None):
    merge_gap = n_guard if merge_gap is None else merge_gap
    P, K = averaged_power_spectrum(x, n_fft, n_avg)
    P_db = 10 * np.log10(P + 1e-18)
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    freq_res = fs / n_fft

    n_ref_total = 2 * n_ref
    alpha = cfar_alpha(n_ref_total, K, pfa)
    half = n_ref + n_guard
    kernel = np.zeros(2 * half + 1)
    kernel[:n_ref] = 1.0
    kernel[-n_ref:] = 1.0
    noise_est = np.convolve(P, kernel, mode="same") / n_ref_total
    threshold = alpha * noise_est

    valid = np.zeros(len(P), dtype=bool)
    valid[half:len(P) - half] = True
    valid[:dc_ignore] = False
    detections = valid & (P > threshold)
    noise_floor_db = float(np.median(P_db))

    idx = np.where(detections)[0]
    regions = []
    if len(idx):
        splits = np.where(np.diff(idx) > merge_gap)[0] + 1
        for grp in np.split(idx, splits):
            lo, hi = int(grp[0]), int(grp[-1])
            pk = lo + int(np.argmax(P[lo:hi + 1]))
            regions.append({"lo": lo, "hi": hi, "peak": pk, "peak_db": float(P_db[pk])})
        regions.sort(key=lambda r: r["peak_db"], reverse=True)

    out = {"detected": bool(regions), "regions": regions, "P_db": P_db, "freqs": freqs,
           "threshold": threshold, "alpha": alpha, "K": K, "freq_res": freq_res,
           "noise_floor_db": noise_floor_db}
    if regions:
        main = regions[0]
        pk = main["peak"]
        level = main["peak_db"] - bw_drop_db
        lo, hi = pk, pk
        while lo > 0 and P_db[lo - 1] > level:
            lo -= 1
        while hi < len(P_db) - 1 and P_db[hi + 1] > level:
            hi += 1
        lo, hi = min(lo, main["lo"]), max(hi, main["hi"])
        out.update({"k_low": lo, "k_high": hi, "peak_db": main["peak_db"],
                    "peak_freq_hz": float(freqs[pk]),
                    "center_freq_hz": float(freqs[(lo + hi) // 2]),
                    "bandwidth_hz": (hi - lo + 1) * freq_res})
    return out
