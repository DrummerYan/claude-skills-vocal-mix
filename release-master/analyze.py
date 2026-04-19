#!/usr/bin/env python3
"""
混音深度分析 — 频谱 / 动态 / 瞬态 / 立体声 / LUFS / ISP / 削波

用法:
  python3 analyze.py INPUT.wav

输出结构化的诊断, 用于决定 master.py 的 EQ 和响度参数
"""
import argparse
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal


def analyze(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    L, R = x[:, 0], x[:, 1]
    mono = x.mean(axis=1)
    N = len(mono)

    # 全局
    peak_L = np.max(np.abs(L)); peak_R = np.max(np.abs(R))
    rms = np.sqrt(np.mean(mono**2))
    meter = pyln.Meter(sr)
    lufs_i = meter.integrated_loudness(x)
    clip_L = int((np.abs(L) >= 0.9999).sum())
    clip_R = int((np.abs(R) >= 0.9999).sum())
    print('=== 全局 ===')
    print(f'时长 {N/sr:.1f}s  sr {sr}  通道 {x.shape[1]}')
    print(f'peak L {20*np.log10(peak_L):+.2f} dBFS   R {20*np.log10(peak_R):+.2f} dBFS '
          f'(ΔL-R {20*np.log10(peak_L/peak_R):+.2f})')
    print(f'RMS {20*np.log10(rms):+.2f} dBFS  crest {20*np.log10(max(peak_L,peak_R)/rms):.1f} dB')
    print(f'LUFS integrated {lufs_i:+.2f}')
    print(f'真削波样本: L {clip_L}  R {clip_R}')

    # ISP (4x)
    up = signal.resample_poly(x.T, 4, 1, axis=1)
    tp = 20*np.log10(np.max(np.abs(up)) + 1e-12)
    print(f'4x True Peak: {tp:+.2f} dBTP')

    # 立体声
    M = (L+R)/2; S = (L-R)/2
    corr = float(np.corrcoef(L, R)[0, 1]) if N > 1 else 1.0
    print('\n=== 立体声 ===')
    print(f'Mid RMS {20*np.log10(np.sqrt(np.mean(M**2))+1e-12):+.2f}  '
          f'Side RMS {20*np.log10(np.sqrt(np.mean(S**2))+1e-12):+.2f}')
    print(f'L/R correlation {corr:+.3f}  (1=单声, 0=去相关, <0=反相)')

    # 频谱
    f, P = signal.welch(mono, sr, nperseg=16384, scaling='spectrum')
    P_db = 10*np.log10(P + 1e-12)
    bands = [30, 50, 80, 120, 200, 300, 500, 800, 1200, 2000,
             3000, 4500, 6500, 9000, 12000, 16000]
    vals = []
    for b in bands:
        lo, hi = b/2**(1/6), b*2**(1/6)
        sel = (f >= lo) & (f <= hi)
        vals.append(P_db[sel].mean() if sel.any() else -120)
    ref = np.interp(1000, f, P_db)
    print('\n=== 频谱 (相对 1kHz, dB) ===')
    print('freq  :', '  '.join(f'{b:>5}' for b in bands))
    print('rel dB:', '  '.join(f'{v-ref:+5.1f}' for v in vals))

    # 动态包络
    win = int(sr * 0.100)
    blocks = N // win
    env = np.array([np.sqrt(np.mean(mono[i*win:(i+1)*win]**2)) for i in range(blocks)])
    env_db = 20*np.log10(env + 1e-9)
    active = env_db[env_db > -40]
    print('\n=== 动态 (100ms RMS, active) ===')
    print(f'min {active.min():+.1f}  max {active.max():+.1f}  '
          f'p10 {np.percentile(active,10):+.1f}  p90 {np.percentile(active,90):+.1f}  '
          f'std {active.std():.1f} dB')

    # 瞬态
    w = int(sr*0.005)
    eshort = np.array([np.sqrt(np.mean(mono[i*w:(i+1)*w]**2)) for i in range(N//w)])
    odf = np.maximum(np.diff(eshort), 0)
    onsets = int((odf > odf.mean() + 2*odf.std()).sum())
    print(f'\n=== 瞬态 ===')
    print(f'onsets {onsets}  ({onsets/(N/sr)*60:.0f} /min)')

    # 三频
    def band_rms(sig, lo, hi):
        sos = signal.butter(6, [lo, hi], btype='band', fs=sr, output='sos')
        y = signal.sosfiltfilt(sos, sig)
        return np.sqrt(np.mean(y**2))
    lo_rms = band_rms(mono, 20, 250)
    mid_rms = band_rms(mono, 250, 4000)
    hi_rms = band_rms(mono, 4000, 20000)
    print('\n=== 三频平衡 (RMS dBFS) ===')
    print(f'低 (20-250) {20*np.log10(lo_rms):+.1f}  '
          f'中 (250-4k) {20*np.log10(mid_rms):+.1f}  '
          f'高 (4k-20k) {20*np.log10(hi_rms):+.1f}  '
          f'低-高差 {20*np.log10(lo_rms/hi_rms):+.1f} dB')

    # 诊断建议 (对应 master.py 的参数)
    print('\n=== 诊断建议 ===')
    if active.std() < 3:
        print('  [警告] RMS std < 3dB → 已高度压缩, 避免再叠压缩')
    lr_imb = 20*np.log10(peak_L/peak_R)
    if abs(lr_imb) > 0.3:
        print(f'  L/R 失衡 {lr_imb:+.2f} dB → master.py --lr-balance {-lr_imb:+.2f}')
    if tp > 0:
        print(f'  [警告] ISP {tp:+.2f} dBTP > 0 → master.py --headroom {max(4, tp+2):.0f}')
    if max(peak_L, peak_R) > 10**(-0.3/20):
        print('  样本 peak 接近 0dBFS → master.py --headroom 4 或更多')
    if 20*np.log10(lo_rms/hi_rms) > 6:
        print('  高频明显偏弱 → 保留 master.py 默认 high shelf 加法 EQ')
    if corr > 0.95:
        print('  立体声相关度极高, 接近单声')
    # LUFS 推荐
    crest_src = 20*np.log10(max(peak_L, peak_R) / rms)
    recommended = max(-14, -13 if crest_src < 18 else -14 if crest_src < 22 else -15)
    print(f'  源 crest {crest_src:.1f} dB → 推荐 master.py --lufs {recommended}')


def main():
    ap = argparse.ArgumentParser(description='混音深度分析 (用于 master.py 调参)')
    ap.add_argument('input', help='输入 WAV 文件路径')
    args = ap.parse_args()
    analyze(args.input)


if __name__ == '__main__':
    main()
