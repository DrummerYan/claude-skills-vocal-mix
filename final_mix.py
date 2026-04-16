#!/usr/bin/env python3
"""
Professional final mix: 人声 + 伴奏 → 可发布母带
参考标准: Spotify / Apple Music -14 LUFS streaming，商业发行版 -10 LUFS

流程:
  1. 伴奏处理链 (EQ 频率让位 + 胶水压缩)
  2. 人声电平对齐
  3. 混音合并 (干湿比例精细调整)
  4. 总线母带链 (EQ + 压缩 + 立体声增强 + 限制器)
  5. LUFS 响度归一 (-10 LUFS 商业发行)
"""

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard, Compressor, Gain, Limiter,
    HighpassFilter, LowShelfFilter, HighShelfFilter, PeakFilter,
)

VOCAL_PATH = "/Users/yandrummer/Desktop/修音准完毕_1_专业混音.wav"
ACMP_PATH  = "/Users/yandrummer/Desktop/伴奏（未混音）.wav"
OUTPUT     = "/Users/yandrummer/Desktop/最终发行版_v4.wav"

sr = 44100
meter = pyln.Meter(sr)

# ── 读取 ──────────────────────────────────────────────
vocal_raw, _ = sf.read(VOCAL_PATH, dtype="float32", always_2d=True)
acmp_raw,  _ = sf.read(ACMP_PATH,  dtype="float32", always_2d=True)

print(f"人声: {vocal_raw.shape[0]} 样本 ({vocal_raw.shape[0]/sr:.2f}s) | {meter.integrated_loudness(vocal_raw):.1f} LUFS")
print(f"伴奏: {acmp_raw.shape[0]} 样本 ({acmp_raw.shape[0]/sr:.2f}s) | {meter.integrated_loudness(acmp_raw):.1f} LUFS")

# ── 对齐时长 (伴奏更长，人声末尾补零) ────────────────
max_len = max(vocal_raw.shape[0], acmp_raw.shape[0])
def pad_end(audio, target_len):
    if audio.shape[0] < target_len:
        pad = np.zeros((target_len - audio.shape[0], audio.shape[1]), dtype="float32")
        return np.concatenate([audio, pad], axis=0)
    return audio[:target_len]

vocal_raw = pad_end(vocal_raw, max_len)
acmp_raw  = pad_end(acmp_raw,  max_len)
print(f"时长对齐到: {max_len/sr:.2f}s")

# ════════════════════════════════════════════════════════
#  STEP 1 — 伴奏处理链
#  目标: 在人声频段让位，控制动态，下拉至 stem 电平
# ════════════════════════════════════════════════════════
print("\nStep 1/4: 伴奏 EQ + 压缩...")

acmp_chain = Pedalboard([
    # 低频清理
    HighpassFilter(cutoff_frequency_hz=40.0),

    # 人声频段让位 (sidechain EQ 模拟)
    # 在 1kHz~4kHz 挖 2-3dB，让人声穿透
    PeakFilter(cutoff_frequency_hz=1200.0, gain_db=-2.5, q=0.8),
    PeakFilter(cutoff_frequency_hz=2800.0, gain_db=-2.0, q=0.9),

    # 低频温暖感保留
    LowShelfFilter(cutoff_frequency_hz=100.0, gain_db=1.0, q=0.7),

    # 轻微高频修整 (防止伴奏高频盖过人声空气感)
    PeakFilter(cutoff_frequency_hz=8000.0, gain_db=-1.5, q=1.0),
    HighShelfFilter(cutoff_frequency_hz=14000.0, gain_db=-1.0, q=0.7),

    # 胶水压缩 (让伴奏动态更统一)
    Compressor(threshold_db=-20.0, ratio=2.0, attack_ms=40.0, release_ms=300.0),

    # 伴奏提升
    Gain(gain_db=3.0),
])

acmp_t = acmp_chain(acmp_raw.T.copy(), sr).T

acmp_lufs = meter.integrated_loudness(acmp_t.astype("float64"))
print(f"  伴奏处理后: {acmp_lufs:.1f} LUFS")

# ════════════════════════════════════════════════════════
#  STEP 2 — 人声电平微调 (使人声坐在伴奏之上约 4dB)
# ════════════════════════════════════════════════════════
print("Step 2/4: 人声电平对齐...")

vocal_chain = Pedalboard([
    # 人声压低
    Gain(gain_db=-17.0),
    # 去刺耳 (3-5kHz 高中频是人声"刺"的来源)
    PeakFilter(cutoff_frequency_hz=3500.0, gain_db=-3.5, q=1.2),
    PeakFilter(cutoff_frequency_hz=5000.0, gain_db=-2.5, q=1.5),
])

vocal_t = vocal_chain(vocal_raw.T.copy(), sr).T
vocal_lufs = meter.integrated_loudness(vocal_t.astype("float64"))
print(f"  人声调整后: {vocal_lufs:.1f} LUFS")

# ════════════════════════════════════════════════════════
#  STEP 3 — 混音合并
# ════════════════════════════════════════════════════════
print("Step 3/4: 混音合并...")

# 直接叠加 (各自已经电平对齐)
mix = vocal_t + acmp_t

mix_lufs_raw = meter.integrated_loudness(mix.astype("float64"))
mix_peak_raw = 20 * np.log10(np.max(np.abs(mix)) + 1e-12)
print(f"  混音前: {mix_lufs_raw:.1f} LUFS | 峰值: {mix_peak_raw:.2f} dBFS")

# ════════════════════════════════════════════════════════
#  STEP 4 — 总线母带链 (Master Bus)
#  专业流行歌母带: EQ微调 → 胶水压缩 → 限制器
# ════════════════════════════════════════════════════════
print("Step 4/4: 总线母带处理...")

master_chain = Pedalboard([
    # 总线 EQ (整体微调)
    LowShelfFilter(cutoff_frequency_hz=80.0,    gain_db=1.0,  q=0.7),   # 低频厚实
    PeakFilter(cutoff_frequency_hz=300.0,        gain_db=-1.0, q=1.2),   # 去混浊
    PeakFilter(cutoff_frequency_hz=3500.0,       gain_db=1.0,  q=1.0),   # 整体亮度/存在感
    HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.5,  q=0.7),   # 空气感

    # 胶水压缩 (让人声和伴奏粘合成一体)
    Compressor(
        threshold_db=-16.0,
        ratio=1.8,          # 极轻，只是"胶水"
        attack_ms=50.0,     # 慢攻击，保留瞬态
        release_ms=400.0,   # 慢释放，自然呼吸
    ),

    # 增益补偿
    Gain(gain_db=3.0),

    # 母带限制器 (阈值 -0.3 dBFS，留一点安全余量)
    Limiter(threshold_db=-0.3, release_ms=100.0),
])

final_t = master_chain(mix.T.copy(), sr).T

# ── LUFS 归一 (-10 LUFS 商业发行标准) ────────────────
lufs_out = meter.integrated_loudness(final_t.astype("float64"))
target_lufs = -10.0
gain_lin = 10 ** ((target_lufs - lufs_out) / 20.0)
final = final_t * gain_lin

# 最终峰值保护 (不超过 -0.3 dBFS)
peak = np.max(np.abs(final))
ceiling = 10 ** (-0.3 / 20.0)  # -0.3 dBFS
if peak > ceiling:
    final *= ceiling / peak

lufs_final = meter.integrated_loudness(final.astype("float64"))
peak_final = 20 * np.log10(np.max(np.abs(final)) + 1e-12)

print(f"\n{'='*45}")
print(f"  最终响度 : {lufs_final:.1f} LUFS")
print(f"  最终峰值 : {peak_final:.2f} dBFS")
print(f"  时长     : {final.shape[0]/sr:.2f}s")
print(f"{'='*45}")

# ── 写出 32-bit float 无损 WAV ───────────────────────
sf.write(OUTPUT, final, sr, subtype="FLOAT")
print(f"\n完成！已保存到:\n{OUTPUT}")
