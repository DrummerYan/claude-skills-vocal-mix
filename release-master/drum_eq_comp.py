#!/usr/bin/env python3
"""
鼓组专业 EQ + 动态 (基于波形实测)
输入: 上一步的 boosted 文件 (-14.74 LUFS, peak -0.3 dBFS)
输出: ~/Desktop/陈童艺-drum-eq-comp.wav

策略 (只动 EQ 与压缩, 不加饱和/混响/平行压缩):
  - EQ: 针对实测频谱修 50Hz 过量 / 250·400Hz 箱感 / 抬 3.5k+6k+12k
  - 动态: SSL 式慢启动总线胶水, 克制 2:1, 最大 GR ≈ 2-3 dB
          保留 17.7 dB crest 的动态感
"""
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard, Compressor, Limiter, Gain,
    HighpassFilter, HighShelfFilter, PeakFilter,
)
from pathlib import Path

IN  = str(Path.home() / 'Desktop' / '陈童艺-drum-boosted.wav')
OUT = str(Path.home() / 'Desktop' / '陈童艺-drum-eq-comp.wav')

# ── 读取
print('读取...')
audio, sr = sf.read(IN, dtype='float32', always_2d=True)
x = audio.T.copy()
meter = pyln.Meter(sr)
lufs_in = meter.integrated_loudness(x.T)
peak_in = 20*np.log10(np.max(np.abs(x)) + 1e-12)
print(f'  输入: peak {peak_in:+.2f} dBFS · LUFS {lufs_in:+.2f}')

# 留 4dB 头室给后级动态
x = x * 10**(-4/20)

# ────────────────────────────────────────────────────────────
# EQ (参数来自波形实测)
# ────────────────────────────────────────────────────────────
print('EQ: 实测驱动的 7 段...')
eq = Pedalboard([
    HighpassFilter(cutoff_frequency_hz=25),                      # 去次声
    PeakFilter(cutoff_frequency_hz=50,   gain_db=-1.5, q=1.2),   # 整理过量 50Hz (+29.8 dB 观测)
    PeakFilter(cutoff_frequency_hz=120,  gain_db=+1.0, q=1.0),   # kick 身躯
    PeakFilter(cutoff_frequency_hz=250,  gain_db=-2.5, q=1.3),   # 箱感 (+21 dB 观测)
    PeakFilter(cutoff_frequency_hz=400,  gain_db=-1.5, q=1.4),   # 纸板 (+16 dB 观测)
    PeakFilter(cutoff_frequency_hz=3500, gain_db=+2.0, q=0.9),   # 军鼓 crack (-5 dB 观测)
    PeakFilter(cutoff_frequency_hz=6000, gain_db=+1.5, q=1.0),   # 镲片 attack
    HighShelfFilter(cutoff_frequency_hz=12000, gain_db=+2.0),    # 空气 (9-16k 偏弱)
])
x_eq = eq(x, sr)

# ────────────────────────────────────────────────────────────
# 动态: SSL 式慢启动总线胶水
# 慢 attack (30ms) → 鼓头完整穿透, 不破 transient
# 2:1 低比率, threshold 仅触发最响处, 最大 GR ≈ 2-3 dB
# ────────────────────────────────────────────────────────────
print('动态: 总线胶水压缩 2:1, 30ms attack (保留 transient)...')
comp = Pedalboard([
    Compressor(threshold_db=-14, ratio=2.0, attack_ms=30, release_ms=150),
    Gain(gain_db=+1.5),     # make-up
])
x_comp = comp(x_eq, sr)

# ────────────────────────────────────────────────────────────
# 终止峰值保护 (只抓超顶, 不做响度冲刺)
# 用户要"动态足" → 不拉 LUFS, 只保证绝对不削波
# ────────────────────────────────────────────────────────────
print('峰值保护限幅 (-0.5 dBFS)...')
safe = Pedalboard([Limiter(threshold_db=-0.8, release_ms=50)])
y = safe(x_comp, sr)

# 硬安全夹
ceiling = 10**(-0.5/20)
pk = np.max(np.abs(y))
if pk > ceiling:
    y = y * (ceiling / pk)

# ── 结果
lufs_out = meter.integrated_loudness(y.T)
peak_out = 20*np.log10(np.max(np.abs(y)) + 1e-12)
rms_out  = 20*np.log10(np.sqrt(np.mean(y**2)) + 1e-12)
crest    = peak_out - rms_out
print(f'  输出: peak {peak_out:+.2f} dBFS · RMS {rms_out:+.2f} · '
      f'crest {crest:.1f} dB · LUFS {lufs_out:+.2f} '
      f'(Δ {lufs_out - lufs_in:+.2f} dB · clip={"YES" if peak_out >= 0 else "no"})')

sf.write(OUT, y.T, sr, subtype='PCM_24')
print(f'\n✓ {OUT}')
