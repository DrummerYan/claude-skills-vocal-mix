#!/usr/bin/env python3
"""
参考音高修正 v2 — Praat PSOLA 引擎
用原唱 F0 曲线作为目标，PSOLA 重合成，无电音感。

用法：
    python3 autotune_ref.py \
        --input  跑调版.wav \
        --ref    原唱参考.wav \
        --out    修音准完毕.wav \
        --strength 0.85
"""

import argparse
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

FRAME_PERIOD = 0.005  # 5ms 每帧


def load_mono(path: str):
    """读取音频，转单声道 float64 供 Praat 使用"""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float64)
    return audio, mono, sr


def extract_f0_praat(mono: np.ndarray, sr: int,
                     floor=60.0, ceiling=1100.0) -> np.ndarray:
    """用 Praat 提取 F0（比 harvest 对人声更准）"""
    snd = parselmouth.Sound(mono, sampling_frequency=sr)
    pitch = call(snd, "To Pitch (ac)", FRAME_PERIOD, floor, 15,
                 False, 0.03, 0.45, 0.01, 0.35, 0.14, ceiling)
    n_frames = call(pitch, "Get number of frames")
    f0 = np.array([
        call(pitch, "Get value in frame", i + 1, "Hertz")
        for i in range(n_frames)
    ])
    f0 = np.where(np.isnan(f0), 0.0, f0)
    return f0


def octave_align_f0(f0_src: np.ndarray, f0_ref: np.ndarray) -> np.ndarray:
    """自动检测并对齐八度差（儿童 vs 成人常见）"""
    s = np.median(f0_src[f0_src > 0]) if np.any(f0_src > 0) else 1
    r = np.median(f0_ref[f0_ref > 0]) if np.any(f0_ref > 0) else 1
    ratio = s / r
    if 1.6 < ratio < 2.5:
        print(f"  八度修正: 参考上移一个八度 (比值 {ratio:.2f})")
        return f0_ref * 2.0
    elif 0.4 < ratio < 0.65:
        print(f"  八度修正: 参考下移一个八度 (比值 {ratio:.2f})")
        return f0_ref / 2.0
    print(f"  八度检测: 无需调整 (比值 {ratio:.2f})")
    return f0_ref


def build_target_f0(f0_src: np.ndarray, f0_ref: np.ndarray,
                    strength: float) -> np.ndarray:
    """
    逐帧构建目标 F0：
    - 两文件等长时直接插值对齐
    - strength: 0 = 不修，1 = 完全跟原唱
    """
    n_src = len(f0_src)
    n_ref = len(f0_ref)

    # 把 ref 插值到与 src 等长
    if n_ref != n_src:
        x_ref = np.linspace(0, n_src - 1, n_ref)
        interp = interp1d(x_ref, f0_ref, kind="linear",
                          bounds_error=False, fill_value=0.0)
        f0_ref_aligned = interp(np.arange(n_src))
    else:
        f0_ref_aligned = f0_ref.copy()

    # 中值滤波平滑参考曲线，去掉毛刺
    f0_ref_smooth = median_filter(f0_ref_aligned, size=9)

    # 构建目标曲线
    f0_target = np.zeros(n_src)
    for i in range(n_src):
        if f0_src[i] <= 0:
            f0_target[i] = 0.0
            continue
        ref = f0_ref_smooth[i]
        if ref <= 0:
            # 参考无声：在附近找最近有声帧
            voiced_idx = np.where(f0_ref_smooth > 0)[0]
            if len(voiced_idx) == 0:
                f0_target[i] = f0_src[i]
                continue
            nearest = voiced_idx[np.argmin(np.abs(voiced_idx - i))]
            ref = f0_ref_smooth[nearest]
        # 修正幅度检查：超过 5 个半音大概率是对齐错误，跳过
        semitones = abs(12 * np.log2(ref / f0_src[i])) if ref > 0 else 99
        if semitones > 5.0:
            f0_target[i] = f0_src[i]   # 保持原音高不动
            continue

        # 柔性修正
        f0_target[i] = f0_src[i] * (1 - strength) + ref * strength

    # 最终平滑（避免相邻帧跳变）
    voiced_mask = f0_target > 0
    if voiced_mask.sum() > 10:
        f0_smooth = median_filter(f0_target, size=5)
        f0_target[voiced_mask] = f0_smooth[voiced_mask]

    return f0_target


def apply_psola(mono_src: np.ndarray, sr: int,
                f0_target: np.ndarray) -> np.ndarray:
    """
    Praat PSOLA 重合成：
    给 Manipulation 对象替换 PitchTier，再用 overlap-add 输出。
    音色/气声完全来自原始录音，只改音高。
    """
    snd = parselmouth.Sound(mono_src, sampling_frequency=sr)

    # 创建 Manipulation 对象（包含原始波形 + 音高 + 时长）
    manip = call(snd, "To Manipulation", FRAME_PERIOD, 60, 1100)

    # 提取当前 PitchTier
    pitch_tier = call(manip, "Extract pitch tier")

    # 清空原有音高点
    call(pitch_tier, "Remove points between", 0, snd.duration)

    # 写入目标 F0（每帧一个点）
    n = len(f0_target)
    times = np.arange(n) * FRAME_PERIOD + FRAME_PERIOD / 2
    for i in range(n):
        if f0_target[i] > 0:
            t = float(times[i])
            if 0 < t < snd.duration:
                call(pitch_tier, "Add point", t, float(f0_target[i]))

    # 替换 PitchTier 并重合成
    call([pitch_tier, manip], "Replace pitch tier")
    snd_out = call(manip, "Get resynthesis (overlap-add)")

    result = snd_out.values[0]  # parselmouth Sound → numpy
    return result.astype(np.float32)


def process_channel(src_mono: np.ndarray, ref_mono: np.ndarray,
                    sr: int, strength: float) -> np.ndarray:
    print("  提取源人声 F0 (Praat)...")
    f0_src = extract_f0_praat(src_mono, sr)

    print("  提取参考人声 F0 (Praat)...")
    f0_ref = extract_f0_praat(ref_mono, sr)

    f0_ref = octave_align_f0(f0_src, f0_ref)

    print("  构建目标音高曲线...")
    f0_target = build_target_f0(f0_src, f0_ref, strength)

    voiced_ratio = np.sum(f0_target > 0) / len(f0_target)
    print(f"  有声帧比例: {voiced_ratio*100:.1f}%  修正帧数: {np.sum(f0_target>0)}")

    print("  Praat PSOLA 重合成...")
    corrected = apply_psola(src_mono.astype(np.float64), sr, f0_target)

    # 对齐长度
    min_len = min(len(src_mono), len(corrected))
    return corrected[:min_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--ref",      required=True)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--strength", type=float, default=0.85)
    args = parser.parse_args()

    if args.out is None:
        base = args.input.rsplit(".", 1)[0]
        args.out = f"{base}_参考修音准.wav"

    print(f"输入:  {args.input}")
    print(f"参考:  {args.ref}")
    print(f"强度:  {args.strength}")

    src_raw, src_mono, sr = load_mono(args.input)
    _,       ref_mono, _  = load_mono(args.ref)
    n_ch = src_raw.shape[1]

    corrected_channels = []
    for ch in range(n_ch):
        print(f"\n── 声道 {ch+1}/{n_ch} ──")
        ch_src = src_raw[:, ch].astype(np.float64)
        corrected = process_channel(ch_src, ref_mono, sr, args.strength)
        corrected_channels.append(corrected)

    result = np.stack(corrected_channels, axis=-1)
    peak = np.max(np.abs(result))
    if peak > 0.999:
        result *= 0.999 / peak

    sf.write(args.out, result, sr, subtype="FLOAT")
    print(f"\n完成！输出: {args.out}")


if __name__ == "__main__":
    main()
