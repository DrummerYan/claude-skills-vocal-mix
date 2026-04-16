#!/usr/bin/env python3
"""
Professional children's pop vocal mixing
参考：Max Martin / Wallpop / Disney / K-pop 儿童声部混音思路

处理链：
  Gain Trim → HPF → EQ清理 → EQ塑形 → De-esser → 压缩 →
  平行压缩 → 短混响(早期反射) → 大厅混响 → 1/8拍延迟 →
  Chorus宽度 → 增益补偿 → 限制器
"""

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pedalboard import (
    Pedalboard, Compressor, Gain, Limiter, Reverb, Chorus, Delay,
    HighpassFilter, LowShelfFilter, HighShelfFilter, PeakFilter,
)

INPUT  = "/Users/yandrummer/Desktop/修音准完毕_1_母带.wav"
OUTPUT = "/Users/yandrummer/Desktop/修音准完毕_1_专业混音.wav"

# ── 读取 ──────────────────────────────────────────────
audio, sr = sf.read(INPUT, dtype="float32", always_2d=True)
audio_t = audio.T.copy()
meter = pyln.Meter(sr)
print(f"读取完成 | {sr} Hz | {audio.shape[1]}ch | {meter.integrated_loudness(audio):.1f} LUFS")

# ══════════════════════════════════════════════════════
#  CHAIN 1 — EQ + 压缩 (Dry vocal treatment)
#  儿童声音特点：基频高(300-1000Hz), 泛音集中在3-8kHz,
#  天然清亮但容易"薄"或"尖", 需要温暖化处理
# ══════════════════════════════════════════════════════
chain_dry = Pedalboard([
    # 先拉低增益（母带文件已经很响）
    Gain(gain_db=-4.0),

    # ── 清理 ──────────────────────────────────────
    HighpassFilter(cutoff_frequency_hz=100.0),            # 儿童声低频不多，100Hz以下直接切
    PeakFilter(cutoff_frequency_hz=280.0,  gain_db=-2.5, q=1.2),  # 去"箱子声"/鼻腔堆积
    PeakFilter(cutoff_frequency_hz=650.0,  gain_db=-1.5, q=1.0),  # 去中低"嗡嗡"
    PeakFilter(cutoff_frequency_hz=2200.0, gain_db=-1.0, q=2.0),  # 去"硬"的金属感

    # ── 塑形 ──────────────────────────────────────
    LowShelfFilter(cutoff_frequency_hz=150.0, gain_db=1.5, q=0.7),     # 补充温暖底色
    PeakFilter(cutoff_frequency_hz=1100.0, gain_db=2.0,  q=0.9),       # 童声甜蜜区
    PeakFilter(cutoff_frequency_hz=3500.0, gain_db=2.5,  q=1.0),       # 存在感 / 穿透力
    PeakFilter(cutoff_frequency_hz=7500.0, gain_db=-2.5, q=2.5),       # De-esser (儿童齿音在7-9kHz)
    HighShelfFilter(cutoff_frequency_hz=11000.0, gain_db=3.0, q=0.7),  # 空气感/仙气

    # ── 压缩 (温柔，保留自然起伏) ─────────────────
    Compressor(
        threshold_db=-24.0,
        ratio=3.0,
        attack_ms=10.0,   # 稍慢保留字头冲击
        release_ms=120.0,
    ),
])

print("Step 1/3: EQ + 压缩...")
dry_t = chain_dry(audio_t, sr)

# ══════════════════════════════════════════════════════
#  CHAIN 2 — 平行压缩 (New York compression)
#  原信号 70% + 重压缩信号 30%，增加密度不失动态
# ══════════════════════════════════════════════════════
chain_parallel = Pedalboard([
    Compressor(threshold_db=-30.0, ratio=8.0, attack_ms=3.0, release_ms=80.0),
    Gain(gain_db=6.0),   # 补偿压缩增益损失
])

parallel_t = chain_parallel(dry_t.copy(), sr)
blended_t = dry_t * 0.70 + parallel_t * 0.30   # 混合

# ══════════════════════════════════════════════════════
#  CHAIN 3 — 空间处理
#  早期反射(室内感) + 大厅混响 + 1/8拍延迟 + 宽度
# ══════════════════════════════════════════════════════

# — 早期反射 (small room, 给声音"身体感") —
chain_room = Pedalboard([
    Reverb(
        room_size=0.15,      # 小房间
        damping=0.7,
        wet_level=0.18,
        dry_level=0.82,
        width=0.6,
    )
])

# — 大厅混响 (pop vocal plate-style) —
chain_hall = Pedalboard([
    Reverb(
        room_size=0.55,      # 中等厅堂
        damping=0.4,
        wet_level=0.22,
        dry_level=0.78,
        width=1.0,
    )
])

# — 1/8 拍延迟 (约 125ms, 适合 120bpm 流行曲) —
chain_delay = Pedalboard([
    Delay(
        delay_seconds=0.125,
        feedback=0.28,
        mix=0.18,
    )
])

# — Chorus 立体声展宽 (非常微量, 儿童声不能过宽) —
chain_chorus = Pedalboard([
    Chorus(
        rate_hz=0.6,
        depth=0.12,
        centre_delay_ms=6.0,
        feedback=0.05,
        mix=0.25,
    )
])

print("Step 2/3: 空间效果 (混响 + 延迟 + 宽度)...")
spaced_t  = chain_room(blended_t, sr)
spaced_t  = chain_hall(spaced_t,  sr)
spaced_t  = chain_delay(spaced_t, sr)
spaced_t  = chain_chorus(spaced_t, sr)

# ══════════════════════════════════════════════════════
#  CHAIN 4 — 增益补偿 + 母带限制器
# ══════════════════════════════════════════════════════
chain_master = Pedalboard([
    Gain(gain_db=5.0),
    Limiter(threshold_db=-0.5, release_ms=80.0),
])

print("Step 3/3: 增益补偿 + 限制器...")
final_t = chain_master(spaced_t, sr)
final   = final_t.T

# ── LUFS 归一到 -10 LUFS ─────────────────────────────
lufs_out = meter.integrated_loudness(final.astype("float64"))
gain_lin = 10 ** ((-10.0 - lufs_out) / 20.0)
final   *= gain_lin

peak = np.max(np.abs(final))
if peak > 0.9976:
    final *= 0.9976 / peak

lufs_final = meter.integrated_loudness(final.astype("float64"))
peak_final = 20 * np.log10(np.max(np.abs(final)) + 1e-12)
print(f"\n最终响度: {lufs_final:.1f} LUFS | 峰值: {peak_final:.2f} dBFS")

# ── 写出 ─────────────────────────────────────────────
sf.write(OUTPUT, final, sr, subtype="FLOAT")
print(f"完成！已保存到:\n{OUTPUT}")
