#!/usr/bin/env python3
"""
无损响度提升 — 不染色/不压缩/不EQ/不削波
仅做透明峰值限幅, 保留原素材全部音色与动态

用法:
  python3 lossless_boost.py INPUT.wav [-o OUTPUT] [--boost 4] [--ceil -0.3]
"""
import argparse
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import Pedalboard, Limiter, Gain
from pathlib import Path


def boost(input_path, output_path, boost_db, ceil_db):
    print(f'读取 {input_path}')
    audio, sr = sf.read(input_path, dtype='float32', always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    x = audio.T.copy()

    peak_in = np.max(np.abs(x))
    meter   = pyln.Meter(sr)
    lufs_in = meter.integrated_loudness(x.T)
    print(f'  输入: peak {20*np.log10(peak_in):+.2f} dBFS · LUFS {lufs_in:+.2f}')

    board = Pedalboard([
        Gain(gain_db=boost_db),
        Limiter(threshold_db=ceil_db - 0.5, release_ms=50),
    ])
    y = board(x, sr)

    ceiling = 10**(ceil_db/20)
    cur_peak = np.max(np.abs(y))
    if cur_peak > ceiling:
        y = y * (ceiling / cur_peak)

    lufs_out = meter.integrated_loudness(y.T)
    peak_out = 20*np.log10(np.max(np.abs(y)) + 1e-12)
    print(f'  输出: peak {peak_out:+.2f} dBFS · LUFS {lufs_out:+.2f} '
          f'(Δ{lufs_out - lufs_in:+.2f} dB · clip={"YES" if peak_out >= 0 else "no"})')

    sf.write(output_path, y.T, sr, subtype='PCM_24')
    print(f'\n✓ {output_path}')


def main():
    ap = argparse.ArgumentParser(description='无损响度提升 (仅透明限幅, 不染色)')
    ap.add_argument('input', help='输入 WAV')
    ap.add_argument('-o', '--output', help='输出 (默认桌面 + "-boosted")')
    ap.add_argument('--boost', type=float, default=4.0, help='提升 dB (默认 4)')
    ap.add_argument('--ceil', type=float, default=-0.3, help='天花板 dBFS (默认 -0.3)')
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve() if args.output else \
          Path.home() / 'Desktop' / f'{inp.stem}-boosted.wav'
    boost(str(inp), str(out), args.boost, args.ceil)


if __name__ == '__main__':
    main()
