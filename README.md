# claude-skills-vocal-mix

儿童童声流行歌「干声 → 混音 → 音画同步」完整工作流。

在 Claude Code 里用 `/mix-vocal`、`/sync-video`、`/iterate-mix` 调用。  
Python 脚本可直接命令行运行，不依赖 Claude。

---

## 三个脚本

### 1. `master_mix.py` — 干声处理
Logic Pro X 导出的干声（只修过音准）→ 专业人声

```bash
# 修改脚本顶部 INPUT / OUTPUT 路径后运行
python3 master_mix.py
```

处理链：频谱降噪 → 去呼吸声 → EQ（去浑浊/塑形/De-esser）→ 压缩 → 平行压缩 → 混响 → 延迟 → Chorus → LUFS -10

---

### 2. `final_mix.py` — 合并混音
处理好的人声 + 原始伴奏 → 发行级母带

```bash
# 修改脚本顶部路径后运行
python3 final_mix.py
```

当前参数：伴奏为主（+3dB），人声衬托（-17dB），总线胶水压缩，Limiter -0.3dBFS，LUFS -10

---

### 3. `sync_video.py` — 视频音画同步
原始视频 + 混音音频 → 精确对齐，视频流零损失

```bash
# 自动检测偏移（推荐）
python3 sync_video.py \
  --video  原始视频.MOV \
  --audio  混音后音频.wav \
  --acmp   伴奏（未混音）.wav \
  --output 输出_同步版.mp4

# 手动指定偏移（微调时用）
python3 sync_video.py \
  --video  原始视频.MOV \
  --audio  混音后音频.wav \
  --offset 6.760 \
  --output 输出_同步版.mp4
```

> ⚠️ 偏移检测必须用「未混音原始伴奏」，不能用混音后音频（处理过的音频相位变化会导致 2s+ 偏差）

---

## 迭代方式（不怕改坏）

```bash
# 开新分支再改
git checkout -b iterate/改动描述
# ...修改脚本，测试...

# 满意 → 合并
git add . && git commit -m "优化：描述"
git checkout main && git merge iterate/改动描述 && git push

# 不满意 → 丢弃
git checkout main && git branch -D iterate/改动描述
```

---

## 依赖安装

```bash
pip install pedalboard noisereduce pyloudnorm soundfile scipy numpy
brew install ffmpeg
```

## Claude Code Skill 命令

| 命令 | 功能 |
|------|------|
| `/mix-vocal` | 执行混音流程 |
| `/sync-video` | 音画同步 |
| `/iterate-mix` | 安全迭代（带 git 保护） |
