#!/usr/bin/env python3
"""
参考音高修正 v3
改进：
  1. Praat 置信度过滤 — 只修高置信度帧，气息/换气自动跳过
  2. 修正幅度封顶 — 超过 300 cent (3个半音) 的强制降强度，避免 PSOLA 失真
  3. 能量门控 — 极静帧（底噪/气声）不修
  4. 多段 DTW — 把音频切成短段分别对齐，解决节奏飘移问题
  5. 原声混合 — 修正版与原声按比例混合，保留自然质感

用法：
    python3 autotune_ref.py \
        --input  跑调版.wav \
        --ref    原唱参考.wav \
        --out    修音准完毕.wav \
        --strength 0.85 \
        --blend  0.15       # 混入 15% 原声（可选，降低电音感）
"""

import argparse
import numpy as np
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

FRAME_PERIOD  = 0.005   # 5ms/帧
F0_FLOOR      = 60.0
F0_CEIL       = 1100.0
MAX_CORR_CENT = 350.0   # 超过此幅度降低修正强度
CONF_THRESH   = 0.05    # 置信度阈值（RMS 归一后，很低即可）
ENERGY_THRESH = 0.001   # RMS 能量阈值（以下视为气息/静音）


# ── 工具函数 ──────────────────────────────────────────

def hz_to_cent(freq):
    return np.where(freq > 0, 1200 * np.log2(np.maximum(freq, 1e-6) / 55.0), 0.0)

def hz_to_semitone(f1, f2):
    """f1 → f2 需要多少半音"""
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return 12 * np.log2(f2 / f1)

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1).astype(np.float64)
    return audio, mono, sr

def rms_envelope(signal, sr, frame_period=FRAME_PERIOD):
    """每帧 RMS"""
    hop = int(sr * frame_period)
    n_frames = len(signal) // hop
    rms = np.array([
        np.sqrt(np.mean(signal[i*hop:(i+1)*hop]**2))
        for i in range(n_frames)
    ])
    return rms


# ── F0 提取（带置信度）────────────────────────────────

def extract_f0_with_confidence(mono, sr):
    """
    Praat AC 法提取 F0。
    置信度用「有声=1 / 无声=0」+ 帧内 RMS 估算（Praat 不直接暴露 strength）。
    """
    snd = parselmouth.Sound(mono, sampling_frequency=sr)
    pitch_obj = call(snd, "To Pitch (ac)",
                     FRAME_PERIOD, F0_FLOOR, 15,
                     False, 0.03, 0.45, 0.01, 0.35, 0.14, F0_CEIL)
    n = call(pitch_obj, "Get number of frames")
    f0   = np.zeros(n)
    conf = np.zeros(n)
    hop  = int(sr * FRAME_PERIOD)
    for i in range(n):
        v = call(pitch_obj, "Get value in frame", i+1, "Hertz")
        is_voiced = (v is not None and not np.isnan(v) and v > 0)
        f0[i] = v if is_voiced else 0.0
        # 用帧 RMS 作为置信度代理（越响越可信）
        s = i * hop
        e = min(s + hop, len(mono))
        rms = np.sqrt(np.mean(mono[s:e]**2)) if e > s else 0.0
        conf[i] = float(rms) if is_voiced else 0.0
    # 归一化 conf 到 0~1
    cmax = conf.max()
    if cmax > 0:
        conf /= cmax
    return f0, conf


# ── 多段 DTW 对齐 ─────────────────────────────────────

def segment_dtw_align(f0_src, f0_ref, n_segments=20):
    """
    把序列切成 n_segments 段分别做 mini-DTW，
    拼成完整的 ref_idx 映射，解决长程节奏飘移。
    """
    n = len(f0_src)
    m = len(f0_ref)
    ref_idx = np.zeros(n, dtype=int)

    seg_size_src = n // n_segments
    seg_size_ref = m // n_segments
    window = int(seg_size_ref * 0.4)   # DTW 搜索窗口 ±40%

    for seg in range(n_segments):
        s_lo = seg * seg_size_src
        s_hi = min(s_lo + seg_size_src, n) if seg < n_segments-1 else n
        r_lo = max(0, seg * seg_size_ref - window)
        r_hi = min(m, (seg+1) * seg_size_ref + window)

        src_seg = f0_src[s_lo:s_hi]
        ref_seg = f0_ref[r_lo:r_hi]
        ns, nr  = len(src_seg), len(ref_seg)
        if ns == 0 or nr == 0:
            continue

        # mini-DTW（cent 距离）
        c_src = hz_to_cent(src_seg)
        c_ref = hz_to_cent(ref_seg)
        dtw   = np.full((ns, nr), np.inf)
        dtw[0, 0] = abs(c_src[0] - c_ref[0])
        for i in range(1, ns):
            dtw[i, 0] = dtw[i-1, 0] + abs(c_src[i] - c_ref[0])
        for j in range(1, nr):
            dtw[0, j] = dtw[0, j-1] + abs(c_src[0] - c_ref[j])
        for i in range(1, ns):
            for j in range(1, nr):
                dtw[i, j] = abs(c_src[i] - c_ref[j]) + min(
                    dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

        # 回溯
        pi, pj = [ns-1], [nr-1]
        i, j = ns-1, nr-1
        while i > 0 or j > 0:
            if i == 0:               j -= 1
            elif j == 0:             i -= 1
            else:
                best = np.argmin([dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1]])
                if   best == 0: i -= 1
                elif best == 1: j -= 1
                else:           i -= 1; j -= 1
            pi.append(i); pj.append(j)

        pi = np.array(pi[::-1])
        pj = np.array(pj[::-1])
        if len(pi) < 2:
            continue
        interp_fn = interp1d(pi, pj, kind="linear",
                             bounds_error=False,
                             fill_value=(pj[0], pj[-1]))
        local_idx = np.clip(
            interp_fn(np.arange(ns)).astype(int) + r_lo, 0, m-1)
        ref_idx[s_lo:s_hi] = local_idx

    return ref_idx


# ── 目标 F0 构建 ──────────────────────────────────────

def build_target_f0(f0_src, conf_src, f0_ref,
                    ref_idx, rms_src, strength):
    n = len(f0_src)
    # ref 平滑
    f0_ref_smooth = gaussian_filter1d(
        np.where(f0_ref > 0, f0_ref, 0.0), sigma=3)

    f0_target = np.zeros(n)
    for i in range(n):
        # 能量门控：极静帧跳过
        if rms_src[min(i, len(rms_src)-1)] < ENERGY_THRESH:
            continue
        # 置信度门控
        if conf_src[i] < CONF_THRESH or f0_src[i] <= 0:
            continue

        ref_f = f0_ref_smooth[ref_idx[i]]
        if ref_f <= 0:
            # 找附近最近有声参考帧
            voiced = np.where(f0_ref_smooth > 0)[0]
            if len(voiced) == 0:
                continue
            ref_f = f0_ref_smooth[voiced[np.argmin(np.abs(voiced - ref_idx[i]))]]

        # 修正幅度计算
        semitones = abs(hz_to_semitone(f0_src[i], ref_f))
        cents     = semitones * 100

        # 超过封顶时降低强度，避免 PSOLA 失真
        effective_strength = strength
        if cents > MAX_CORR_CENT:
            # 大幅修正只做一半强度
            effective_strength = strength * (MAX_CORR_CENT / cents) * 0.6

        f0_target[i] = f0_src[i] * (1 - effective_strength) + ref_f * effective_strength

    # 最终 Gaussian 平滑（过渡自然）
    voiced_mask = f0_target > 0
    if voiced_mask.sum() > 10:
        smoothed = gaussian_filter1d(f0_target, sigma=2)
        f0_target[voiced_mask] = smoothed[voiced_mask]

    return f0_target


# ── Praat PSOLA 重合成 ────────────────────────────────

def apply_psola(mono_src, sr, f0_target):
    snd   = parselmouth.Sound(mono_src, sampling_frequency=sr)
    manip = call(snd, "To Manipulation", FRAME_PERIOD, F0_FLOOR, F0_CEIL)
    pt    = call(manip, "Extract pitch tier")
    call(pt, "Remove points between", 0, snd.duration)

    times = np.arange(len(f0_target)) * FRAME_PERIOD + FRAME_PERIOD / 2
    for i, (t, f) in enumerate(zip(times, f0_target)):
        if f > 0 and 0 < t < snd.duration:
            call(pt, "Add point", float(t), float(f))

    call([pt, manip], "Replace pitch tier")
    snd_out = call(manip, "Get resynthesis (overlap-add)")
    result  = snd_out.values[0].astype(np.float32)
    return result


# ── 主流程 ────────────────────────────────────────────

def process_channel(src_ch, ref_mono, sr, strength, blend):
    print("  [1/5] 提取源人声 F0 + 置信度...")
    f0_src, conf_src = extract_f0_with_confidence(src_ch, sr)

    print("  [2/5] 提取参考 F0...")
    f0_ref, _ = extract_f0_with_confidence(ref_mono, sr)

    # 八度对齐
    s_med = np.median(f0_src[f0_src>0]) if np.any(f0_src>0) else 1
    r_med = np.median(f0_ref[f0_ref>0]) if np.any(f0_ref>0) else 1
    ratio = s_med / r_med
    if 1.6 < ratio < 2.5:
        print(f"  八度修正: 参考×2 (比值={ratio:.2f})")
        f0_ref *= 2
    elif 0.4 < ratio < 0.65:
        print(f"  八度修正: 参考÷2 (比值={ratio:.2f})")
        f0_ref /= 2
    else:
        print(f"  八度检测: 正常 (比值={ratio:.2f})")

    print("  [3/5] 多段 DTW 时间对齐...")
    ref_idx = segment_dtw_align(f0_src, f0_ref, n_segments=30)

    print("  [4/5] 构建目标 F0（置信度+能量过滤）...")
    rms = rms_envelope(src_ch, sr)
    f0_target = build_target_f0(f0_src, conf_src, f0_ref,
                                ref_idx, rms, strength)

    active = np.sum(f0_target > 0)
    print(f"       修正帧: {active}/{len(f0_target)} ({active/len(f0_target)*100:.1f}%)")

    print("  [5/5] Praat PSOLA 重合成...")
    corrected = apply_psola(src_ch.astype(np.float64), sr, f0_target)

    min_len = min(len(src_ch), len(corrected))
    corrected = corrected[:min_len]

    # 与原声混合（降低残余电音感）
    if blend > 0:
        orig = src_ch[:min_len].astype(np.float32)
        corrected = corrected * (1 - blend) + orig * blend

    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True)
    parser.add_argument("--ref",      required=True)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--blend",    type=float, default=0.1,
                        help="混入原声比例 0~0.3（默认 0.1，降低电音感）")
    args = parser.parse_args()

    if args.out is None:
        base = args.input.rsplit(".", 1)[0]
        args.out = f"{base}_参考修音准.wav"

    print(f"输入:  {args.input}")
    print(f"参考:  {args.ref}")
    print(f"强度:  {args.strength}  原声混合: {args.blend}")

    src_raw, _,        sr = load_audio(args.input)
    _,       ref_mono, _  = load_audio(args.ref)
    n_ch = src_raw.shape[1]

    corrected_channels = []
    for ch in range(n_ch):
        print(f"\n── 声道 {ch+1}/{n_ch} ──")
        result = process_channel(
            src_raw[:, ch].astype(np.float64),
            ref_mono, sr, args.strength, args.blend
        )
        corrected_channels.append(result)

    out = np.stack(corrected_channels, axis=-1)
    peak = np.max(np.abs(out))
    if peak > 0.999:
        out *= 0.999 / peak

    sf.write(args.out, out, sr, subtype="FLOAT")
    print(f"\n完成！输出: {args.out}")


if __name__ == "__main__":
    main()
