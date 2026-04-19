---
name: release-master
description: 将任意 WAV 混音文件处理为流媒体发行级母带 (LUFS 目标, True Peak ISP 安全, 每级实时监控瞬态 crest 损失). 用于把 Logic/DAW 导出的立体声混音 (vocal+inst, 鼓组, 或任意素材) 推到 Spotify/Apple Music/YouTube 可发行状态
---

# release-master — 发行级母带处理

## 使用时机

- 用户提供一个 WAV 文件要"混音成发行级"或"可发行版本"
- 用户要做 LUFS 归一化并保证 True Peak 安全
- 需要基于波形实测 (而非预设) 做 EQ 与动态

**不要用于**: 仅需提升音量 (用 lossless_boost.py) / 鼓干音混音 (用 drum_eq_comp.py 先做)

## 核心原则 (这次实战验证的经验)

1. **先分析再处理**: 所有 EQ/压缩参数都应从实测数据来, 不瞎套预设
2. **每级监控 crest**: 每处理一步都打印 Δcrest, 超过保护线立即停
3. **小步推响度**: 单次 push ≤ 0.6 dB, 绝不一次推到位 (会砸瞬态)
4. **ISP-safe 限幅**: `pedalboard.Limiter` **不做 inter-sample peak 安全**, 必须用 4x 上采样 hard-clip
5. **LUFS 目标按 crest 选**: 源 crest 22 想推到 -11 是物理不可能的, 现实目标 -13 ~ -14

## 工作流

### Step 1: 分析 (必做)

```bash
python3 analyze.py INPUT.wav
```

输出包含:
- 全局: peak / RMS / crest / LUFS / 真削波样本 / 4x ISP
- 立体声: L/R 相关度 / Mid-Side / 平衡
- 频谱: 16 段 1/3 oct, 相对 1kHz
- 动态: 100ms 包络 std / 瞬态 onset 密度
- 诊断建议: 直接给出对应 `master.py` 的 CLI 参数推荐

### Step 2: 母带处理

```bash
python3 master.py INPUT.wav [-o OUT] [--lufs -13] [--ceil -1.0] \
                  [--lr-balance <ΔLdB>] [--headroom 2] \
                  [--crest-guard 6] [--step 0.6]
```

五个阶段, 每级监控 peak/RMS/crest/LUFS:
1. **头室 + L/R 平衡**: 降 2 dB 给 EQ 余量, 修正声道偏移
2. **减法 EQ**: HPF 28Hz + 50Hz -2dB + 300Hz -1.2dB (清理)
3. **加法 EQ**: 2.5kHz +1.5 / 6kHz +2 / 10kHz shelf +3.5 / 80Hz shelf +0.8 (提亮空气)
4. **SSL 胶水压缩**: 2:1, 30ms attack (慢启动保瞬态), +0.5 dB make-up
5. **小步迭代响度**: 0.6 dB/步 × N 次 + 每次 4x ISP brickwall, 收敛到目标 LUFS

**关键技术**: Step 5 的 `isp_brickwall()` 函数 — `pedalboard.Limiter` 行为像激进压缩器 (1 dB push 可导致 7 dB RMS 涨), 且不做 inter-sample peak 保护. 替换方案: 4x 上采样 → `np.clip` → 下采样, 每次迭代都做, 保证 ISP ≤ ceiling.

### Step 3: 可选 — 仅音量提升

如果用户只要"更响", 不要染色:

```bash
python3 lossless_boost.py INPUT.wav --boost 4 --ceil -0.3
```

## 目标参数参考

| 源 crest | 推荐 LUFS | 理由 |
|---|---|---|
| > 22 dB | -14 (= Spotify 标准) | 高度动态素材 (如 acoustic/鼓组), 保护瞬态 |
| 18-22 | -13 | 常规流行混音 |
| 14-18 | -12 | 已有中度压缩, 可略响 |
| < 14 | -11 或不推 | 已接近砖块, 再响会严重伤动态 |

**所有场景 True Peak 天花板固定 -1.0 dBTP** (Spotify/Apple Music/YouTube 都要求 ≤ -1)

## 已知陷阱

- `pedalboard.Limiter.threshold_db=-1.6` 不等于 "输出 peak ≤ -1.6 dBTP" — 样本域是, ISP 可能 +3 dBTP
- Compressor ratio=30 作 brickwall 可以, 但仍不是 ISP-safe, 只能做 "粗限幅", ISP 保护必须另外做
- 一次性 push 大 delta (> 2 dB) 到 Limiter, RMS 会非线性飙升, crest 被砸平
- make-up gain 太猛会让 Step 5 起步就 TP 超标 (Glue 后 +1.2 make-up 实测让 TP -0.3 dBTP)

## 示例: 本次实战 (陈童艺混音.wav)

分析发现:
- crest 21.7 / LUFS -16.53 / ISP 未测 / R 比 L +0.69 dB / 高频 -15 dB below 1kHz
- 高频严重偏闷, 低频 50Hz 过强

处理结果:
- LUFS -13.19 (目标 -13) / TP -0.94 dBTP / Sample peak -1.00 / crest 18.63 (掉 3.1)
- 5 阶段全部通过 crest guard (< 6 dB)

## 依赖

```bash
pip install pedalboard pyloudnorm soundfile scipy numpy
```
