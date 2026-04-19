#!/usr/bin/env python3
"""
发行级母带处理 — 基于波形实测 + 每级瞬态监控

用法:
  python3 master.py INPUT [-o OUTPUT] [--lufs -13] [--ceil -1.0]
                    [--crest-guard 6] [--step 0.6]

特点:
  - 每一级打印 peak/RMS/crest/LUFS, 实时监控瞬态损失
  - 小步迭代响度 (≤0.6 dB/步), 避免一次 push 砸坏瞬态
  - 4x 上采样 hard-clip 实现 ISP-safe 限幅 (pedalboard.Limiter 不安全)
  - EQ 参数为通用模板, 建议先跑 analyze.py 再按实测调 master.py
"""
import argparse
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard, Compressor, Gain,
    HighpassFilter, HighShelfFilter, LowShelfFilter, PeakFilter,
)
from scipy.signal import resample_poly
from pathlib import Path


def monitor(tag, x, meter, base_crest=None, crest_guard=6.0):
    mono = x.mean(axis=0)
    peak = np.max(np.abs(x))
    rms  = np.sqrt(np.mean(mono**2))
    peak_db = 20*np.log10(peak + 1e-12)
    rms_db  = 20*np.log10(rms  + 1e-12)
    crest   = peak_db - rms_db
    lufs    = meter.integrated_loudness(x.T)
    msg = (f'[{tag}]'.ljust(18) +
           f' peak {peak_db:+6.2f}  RMS {rms_db:+6.2f}  '
           f'crest {crest:5.2f}  LUFS {lufs:+6.2f}')
    if base_crest is not None:
        delta = crest - base_crest
        msg += f'  Δcrest {delta:+.2f}'
        if delta < -crest_guard:
            msg += '  ⚠ 瞬态受损!'
    print(msg)
    return crest, lufs


def isp_brickwall(sig, sr, ceil_db):
    """ISP-safe brick-wall: 4x 上采样 hard-clip 再下采样"""
    ceiling = 10**(ceil_db/20)
    up = resample_poly(sig, 4, 1, axis=1).astype(np.float32)
    np.clip(up, -ceiling, ceiling, out=up)
    return resample_poly(up, 1, 4, axis=1).astype(np.float32)


def measure_tp(sig):
    up = resample_poly(sig, 4, 1, axis=1)
    return 20*np.log10(np.max(np.abs(up)) + 1e-12)


def master(input_path, output_path, target_lufs, ceil_dbtp,
           crest_guard, step_max, max_iter,
           lr_balance_db, headroom_db):
    # ── 读取
    print(f'读取 {input_path}')
    audio, sr = sf.read(input_path, dtype='float32', always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    x = audio.T.copy()
    meter = pyln.Meter(sr)
    print(f'  {sr} Hz · {x.shape[1]/sr:.1f} s · 通道 {x.shape[0]}\n')

    base_crest, _ = monitor('SOURCE', x, meter, crest_guard=crest_guard)

    # Step 1: headroom + L/R balance
    print('\nSTEP 1  预留头室 + L/R 平衡')
    x = x * 10**(-headroom_db/20)
    half = lr_balance_db / 2
    x[0] = x[0] * 10**(+half/20)
    x[1] = x[1] * 10**(-half/20)
    monitor('头室+平衡', x, meter, base_crest, crest_guard)

    # Step 2: subtractive EQ (generic template; tune from analyze.py output)
    print('\nSTEP 2  减法 EQ (低频整理 / 去浑浊)')
    x = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=28),
        PeakFilter(cutoff_frequency_hz=50,  gain_db=-2.0, q=1.2),
        PeakFilter(cutoff_frequency_hz=300, gain_db=-1.2, q=1.3),
    ])(x, sr)
    monitor('减法EQ后', x, meter, base_crest, crest_guard)

    # Step 3: additive EQ (presence + air)
    print('\nSTEP 3  加法 EQ (提亮 / 空气感)')
    x = Pedalboard([
        PeakFilter(cutoff_frequency_hz=2500,  gain_db=+1.5, q=0.9),
        PeakFilter(cutoff_frequency_hz=6000,  gain_db=+2.0, q=0.9),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=+3.5),
        LowShelfFilter(cutoff_frequency_hz=80, gain_db=+0.8),
    ])(x, sr)
    monitor('加法EQ后', x, meter, base_crest, crest_guard)

    # Step 4: SSL-style bus glue (slow attack preserves transients)
    print('\nSTEP 4  总线胶水压缩 (SSL 2:1, 30ms attack)')
    x = Pedalboard([
        Compressor(threshold_db=-16, ratio=2.0, attack_ms=30, release_ms=180),
        Gain(gain_db=+0.5),
    ])(x, sr)
    monitor('胶水压缩后', x, meter, base_crest, crest_guard)

    # Step 5: small-step iterative loudness push + ISP-safe brickwall
    print(f'\nSTEP 5  小步响度推进 → 目标 {target_lufs:+.1f} LUFS · ISP ≤ {ceil_dbtp} dBTP')
    isp_ceil = ceil_dbtp - 0.3   # 给下采样 ringing 留 0.3 dB

    for it in range(max_iter):
        cur_lufs = meter.integrated_loudness(x.T)
        mono = x.mean(axis=0)
        peak_db = 20*np.log10(np.max(np.abs(x)) + 1e-12)
        rms_db  = 20*np.log10(np.sqrt(np.mean(mono**2)) + 1e-12)
        crest   = peak_db - rms_db
        dcrest  = crest - base_crest
        tp      = measure_tp(x)
        delta   = target_lufs - cur_lufs
        print(f'  iter{it:2d}  LUFS {cur_lufs:+6.2f}  TP {tp:+5.2f}  '
              f'crest {crest:5.2f} (Δ{dcrest:+5.2f})  Δ→target {delta:+5.2f}', end='')

        if abs(delta) < 0.3:
            print('  ✓ LUFS 达标'); break
        if dcrest < -crest_guard:
            print('  ⛔ crest 保护线'); break
        push = min(delta + 0.2, step_max)
        if push <= 0.05:
            print('  ✓ 无需推进'); break
        print(f'  push {push:+.2f}')
        x = x * 10**(push/20)
        x = isp_brickwall(x, sr, isp_ceil)

    # 最终 TP 保障
    tp_final = measure_tp(x)
    if tp_final > ceil_dbtp:
        print(f'  [TP 救场] {tp_final:+.2f} > {ceil_dbtp}, 再 clip')
        x = isp_brickwall(x, sr, ceil_dbtp - 0.3)
    ceiling = 10**(ceil_dbtp/20)
    pk = np.max(np.abs(x))
    if pk > ceiling:
        x = x * (ceiling / pk)

    tp_final = measure_tp(x)
    print(f'  最终 4x True Peak: {tp_final:+.2f} dBTP')
    monitor('成品', x, meter, base_crest, crest_guard)

    # 写出
    sf.write(output_path, x.T, sr, subtype='PCM_24')
    print(f'\n✓ 输出: {output_path}')
    print(f'   目标: LUFS {target_lufs:+.1f}  ·  TP ≤ {ceil_dbtp} dBTP  ·  24-bit PCM')


def main():
    ap = argparse.ArgumentParser(description='发行级母带处理')
    ap.add_argument('input', help='输入 WAV 文件路径')
    ap.add_argument('-o', '--output', help='输出路径 (默认桌面, 原名+"-mastered")')
    ap.add_argument('--lufs', type=float, default=-13.0,
                    help='目标 LUFS (默认 -13, Spotify -14 + 1 dB)')
    ap.add_argument('--ceil', type=float, default=-1.0,
                    help='True Peak 天花板 dBTP (默认 -1.0)')
    ap.add_argument('--crest-guard', type=float, default=6.0,
                    help='允许 crest 最大下降 dB (默认 6.0)')
    ap.add_argument('--step', type=float, default=0.6,
                    help='单次推进 dB 上限 (默认 0.6)')
    ap.add_argument('--max-iter', type=int, default=15,
                    help='最大迭代次数 (默认 15)')
    ap.add_argument('--lr-balance', type=float, default=0.0,
                    help='L-R 通道平衡修正 dB (正=L升R降; 默认 0)')
    ap.add_argument('--headroom', type=float, default=2.0,
                    help='进入 EQ 前降头室 dB (默认 2.0)')
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    if args.output:
        out = Path(args.output).expanduser().resolve()
    else:
        out = Path.home() / 'Desktop' / f'{inp.stem}-mastered.wav'

    master(str(inp), str(out),
           target_lufs=args.lufs, ceil_dbtp=args.ceil,
           crest_guard=args.crest_guard, step_max=args.step,
           max_iter=args.max_iter, lr_balance_db=args.lr_balance,
           headroom_db=args.headroom)


if __name__ == '__main__':
    main()
