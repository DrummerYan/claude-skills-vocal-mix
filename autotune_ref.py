#!/usr/bin/env python3
"""
参考音高修正 (Reference-guided Pitch Correction)
用原唱的 F0 曲线作为目标，通过 DTW 对齐后纠正跑调人声。

比盲修准确得多：不猜音符，直接跟着原唱的旋律走。

用法：
    python3 autotune_ref.py \
        --input  跑调版.wav \
        --ref    原唱（参考）.wav \
        --out    修音准完毕.wav \
        --strength 0.85
"""

import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

# ── 音高提取 ─────────────────────────────────────────
def extract_f0(audio: np.ndarray, sr: int, frame_period=5.0):
    """用 harvest 提取 F0，返回 (f0, voiced_mask)"""
    audio_64 = audio.astype(np.float64)
    f0, _ = pw.harvest(audio_64, sr,
                       f0_floor=60, f0_ceil=1100,
                       frame_period=frame_period)
    voiced = f0 > 0
    return f0, voiced

def hz_to_cent(freq: np.ndarray) -> np.ndarray:
    """Hz → cent（相对 A1=55Hz），便于 DTW 距离计算"""
    out = np.zeros_like(freq)
    mask = freq > 0
    out[mask] = 1200 * np.log2(freq[mask] / 55.0)
    return out

def octave_align(f0_src: np.ndarray, f0_ref: np.ndarray) -> np.ndarray:
    """
    如果演唱者和原唱差一个八度（儿童 vs 成人很常见），自动对齐。
    """
    src_voiced = f0_src[f0_src > 0]
    ref_voiced = f0_ref[f0_ref > 0]
    if len(src_voiced) == 0 or len(ref_voiced) == 0:
        return f0_src

    ratio = np.median(src_voiced) / np.median(ref_voiced)
    if 1.6 < ratio < 2.5:          # 源比参考高一个八度
        print(f"  八度检测: 源比参考高一个八度 (比值 {ratio:.2f})，参考上移一个八度")
        f0_ref = f0_ref * 2.0
    elif 0.4 < ratio < 0.65:        # 源比参考低一个八度
        print(f"  八度检测: 源比参考低一个八度 (比值 {ratio:.2f})，参考下移一个八度")
        f0_ref = f0_ref / 2.0
    else:
        print(f"  八度检测: 正常 (比值 {ratio:.2f}，无需调整)")
    return f0_ref

# ── DTW 对齐 ─────────────────────────────────────────
def dtw_align(src_cents: np.ndarray, ref_cents: np.ndarray,
              src_voiced: np.ndarray, ref_voiced: np.ndarray):
    """
    轻量 DTW：只在有声帧上做对齐，找到 src → ref 的时间映射。
    返回：ref_aligned — 与 src 等长的参考 F0 序列
    """
    n = len(src_cents)
    m = len(ref_cents)

    # 构建代价矩阵（只用有声帧的 cent 距离）
    # 降采样加速（每隔 4 帧）
    step = 4
    src_ds = src_cents[::step]
    ref_ds = ref_cents[::step]
    n_ds, m_ds = len(src_ds), len(ref_ds)

    dtw_mat = np.full((n_ds, m_ds), np.inf)
    dtw_mat[0, 0] = abs(src_ds[0] - ref_ds[0])
    for i in range(1, n_ds):
        dtw_mat[i, 0] = dtw_mat[i-1, 0] + abs(src_ds[i] - ref_ds[0])
    for j in range(1, m_ds):
        dtw_mat[0, j] = dtw_mat[0, j-1] + abs(src_ds[0] - ref_ds[j])
    for i in range(1, n_ds):
        for j in range(1, m_ds):
            cost = abs(src_ds[i] - ref_ds[j])
            dtw_mat[i, j] = cost + min(
                dtw_mat[i-1, j],
                dtw_mat[i, j-1],
                dtw_mat[i-1, j-1]
            )

    # 回溯路径
    path_i, path_j = [n_ds-1], [m_ds-1]
    i, j = n_ds-1, m_ds-1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = np.argmin([dtw_mat[i-1,j], dtw_mat[i,j-1], dtw_mat[i-1,j-1]])
            if best == 0:   i -= 1
            elif best == 1: j -= 1
            else:           i -= 1; j -= 1
        path_i.append(i); path_j.append(j)

    path_i = np.array(path_i[::-1]) * step
    path_j = np.array(path_j[::-1]) * step

    # 插值：为 src 每一帧找到对应的 ref 帧索引
    path_i = np.clip(path_i, 0, n-1)
    path_j = np.clip(path_j, 0, m-1)
    interp = interp1d(path_i, path_j, kind='linear',
                      bounds_error=False, fill_value=(path_j[0], path_j[-1]))
    ref_idx = interp(np.arange(n)).astype(int)
    ref_idx = np.clip(ref_idx, 0, m-1)

    # 取对应的参考 F0
    ref_f0_raw, _ = None, None  # 在外部传入
    return ref_idx   # 返回映射索引


# ── 主修正函数 ────────────────────────────────────────
def pitch_correct_with_ref(src_audio: np.ndarray, ref_audio: np.ndarray,
                            sr: int, strength: float = 0.85,
                            frame_period: float = 5.0) -> np.ndarray:
    print("  提取源人声 F0...")
    f0_src, voiced_src = extract_f0(src_audio, sr, frame_period)

    print("  提取参考人声 F0...")
    f0_ref, voiced_ref = extract_f0(ref_audio, sr, frame_period)

    # 声码器完整分析（用于重合成）
    print("  声码器分析（提取音色包络）...")
    src_64 = src_audio.astype(np.float64)
    _, t_src = pw.harvest(src_64, sr, frame_period=frame_period)
    sp = pw.cheaptrick(src_64, f0_src, t_src, sr)
    ap = pw.d4c(src_64, f0_src, t_src, sr)

    # 八度对齐
    f0_ref = octave_align(f0_src, f0_ref)

    # cent 序列
    cents_src = hz_to_cent(f0_src)
    cents_ref = hz_to_cent(f0_ref)

    # DTW 对齐：找 src 每帧对应的 ref 帧
    print("  DTW 时间对齐...")
    ref_idx = dtw_align(cents_src, cents_ref, voiced_src, voiced_ref)

    # 中值滤波 ref F0，减少参考噪声
    f0_ref_smooth = median_filter(f0_ref, size=5)

    # 修正 F0
    print("  修正音高...")
    f0_corrected = f0_src.copy()
    for i in range(len(f0_src)):
        if not voiced_src[i]:
            continue
        ref_freq = f0_ref_smooth[ref_idx[i]]
        if ref_freq <= 0:
            # 参考帧无声：找最近的有声参考帧
            voiced_ref_idx = np.where(f0_ref_smooth > 0)[0]
            if len(voiced_ref_idx) == 0:
                continue
            nearest = voiced_ref_idx[np.argmin(np.abs(voiced_ref_idx - ref_idx[i]))]
            ref_freq = f0_ref_smooth[nearest]

        src_freq = f0_src[i]
        # 柔性插值修正
        f0_corrected[i] = src_freq * (1 - strength) + ref_freq * strength

    # 重合成（音色 sp/ap 完全不变）
    print("  重合成...")
    corrected = pw.synthesize(f0_corrected, sp, ap, sr,
                              frame_period=frame_period)

    min_len = min(len(src_audio), len(corrected))
    return corrected[:min_len].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="参考音高修正")
    parser.add_argument("--input",    required=True, help="需要修音准的干声 WAV")
    parser.add_argument("--ref",      required=True, help="原唱参考 WAV（音高来源）")
    parser.add_argument("--out",      default=None,  help="输出路径")
    parser.add_argument("--strength", type=float, default=0.85, help="修正强度 0~1（默认 0.85）")
    args = parser.parse_args()

    if args.out is None:
        base = args.input.rsplit(".", 1)[0]
        args.out = f"{base}_参考修音准.wav"

    print(f"输入: {args.input}")
    print(f"参考: {args.ref}")
    print(f"强度: {args.strength}")

    src_raw, sr  = sf.read(args.input, dtype="float32", always_2d=True)
    ref_raw, sr2 = sf.read(args.ref,   dtype="float32", always_2d=True)

    # 转单声道处理，分声道分别处理后合并
    n_ch = src_raw.shape[1]
    ref_mono = ref_raw.mean(axis=1)   # 参考用单声道即可

    corrected_channels = []
    for ch in range(n_ch):
        print(f"\n声道 {ch+1}/{n_ch}:")
        corrected = pitch_correct_with_ref(
            src_raw[:, ch], ref_mono, sr,
            strength=args.strength
        )
        corrected_channels.append(corrected)

    result = np.stack(corrected_channels, axis=-1)
    peak = np.max(np.abs(result))
    if peak > 0.999:
        result = result / peak * 0.999

    sf.write(args.out, result, sr, subtype="FLOAT")
    print(f"\n完成！输出: {args.out}")


if __name__ == "__main__":
    main()
