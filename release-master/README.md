# release-master

发行级母带处理工具集 — 实测驱动, 每级监控瞬态, ISP 安全.

## 快速使用

```bash
# 1. 先分析, 看诊断建议的 CLI 参数
python3 analyze.py ~/Downloads/mix.wav

# 2. 用建议参数 (或默认) 做母带
python3 master.py ~/Downloads/mix.wav --lufs -13

# 3. 成品默认输出到 ~/Desktop/<原名>-mastered.wav
```

## 三个脚本

| 脚本 | 用途 |
|---|---|
| `analyze.py` | 频谱/动态/瞬态/立体声/LUFS/ISP 分析 + 参数诊断 |
| `master.py` | 发行级母带: 5 级处理链, 小步迭代响度, ISP-safe 限幅 |
| `lossless_boost.py` | 仅提升音量, 不染色/不压缩/不 EQ |

## 关键技术

- **每级 Δcrest 监控**: 看得到瞬态损失, 超 6 dB 立刻告警
- **4x 上采样 hard-clip**: `pedalboard.Limiter` 不做 ISP 保护, 必须自己做
- **小步迭代**: ≤0.6 dB/步, 每步独立 brickwall, 避免单次 push 砸瞬态

详细文档见 `SKILL.md`.

## 安装为 Claude Code Skill

```bash
ln -s $(pwd)/release-master ~/.claude/skills/release-master
# 然后在 Claude Code 里直接说 "用 release-master 处理这个文件"
```
