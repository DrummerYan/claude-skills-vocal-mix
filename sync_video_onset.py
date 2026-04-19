#!/usr/bin/env python3
"""
视频音画同步 (onset 对齐版) — 专为打击乐/鼓类内容设计

与原 sync_video.py 区别:
  - sync_video.py    : 波形互相关, 适合有持续伴奏的场景
  - sync_video_onset : onset 包络互相关, 对鼓点/重音对齐更精确,
                       不受混响/麦位变化影响 (只认攻击时刻)

原理:
  1. 从视频和目标音频各自计算 onset strength envelope
     (短时能量的半波整流一阶差分, 高通强化瞬态)
  2. 包络降采样互相关, 峰值位置 = 两者时间偏移
  3. ffmpeg 合并, 视频流 copy, 原视频音轨丢弃

用法:
  python3 sync_video_onset.py \
    --video  video.MOV \
    --audio  mixed.wav \
    --output out.mp4
"""
import argparse
import subprocess
import sys
import tempfile
import os
import numpy as np
import soundfile as sf
from scipy import signal


def extract_video_audio(video_path, out_wav):
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-map", "0:a:0", "-ac", "1", "-ar", "44100",
           "-c:a", "pcm_f32le", out_wav]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] 提取视频音轨失败:\n{r.stderr}")
        sys.exit(1)


def onset_envelope(x, sr, hop_ms=10, env_sr=400):
    """
    计算 onset strength 包络
    - 短时 RMS (hop_ms 间隔)
    - 转 dB 后一阶差分, 半波整流
    - 下采样到 env_sr (默认 400 Hz, 2.5ms 精度)
    """
    if x.ndim > 1:
        x = x.mean(axis=0) if x.shape[0] < x.shape[1] else x.mean(axis=1)
    hop = int(sr * hop_ms / 1000)
    n_frames = len(x) // hop
    rms = np.array([
        np.sqrt(np.mean(x[i*hop:(i+1)*hop]**2) + 1e-12)
        for i in range(n_frames)
    ])
    rms_db = 20 * np.log10(rms + 1e-9)
    # 差分 + 半波整流 (只看上升沿)
    odf = np.maximum(np.diff(rms_db, prepend=rms_db[0]), 0)
    # 标准化
    if odf.std() > 1e-6:
        odf = (odf - odf.mean()) / odf.std()
    # 重采样到 env_sr 供互相关
    frame_sr = 1000 / hop_ms
    if env_sr != frame_sr:
        n_target = int(len(odf) * env_sr / frame_sr)
        odf = signal.resample(odf, n_target)
    return odf, env_sr


def detect_offset_onset(video_audio, mix_audio):
    """
    返回 offset_sec (正值: 混音音频起点对应视频时间 offset_sec 处)
    """
    vid, sr_v = sf.read(video_audio, dtype="float32")
    mix, sr_m = sf.read(mix_audio, dtype="float32", always_2d=True)
    mix_mono = mix.mean(axis=1)

    print(f"  视频音轨: {len(vid)/sr_v:.1f}s @ {sr_v}Hz")
    print(f"  混音音频: {len(mix_mono)/sr_m:.1f}s @ {sr_m}Hz")

    env_v, esr = onset_envelope(vid, sr_v)
    env_m, _   = onset_envelope(mix_mono, sr_m)
    print(f"  onset 包络: 视频 {len(env_v)} / 混音 {len(env_m)} 样本 @ {esr}Hz")

    # 互相关: 找 mix 在 video 中的位置
    corr = signal.correlate(env_v, env_m, mode="full")
    lags = signal.correlation_lags(len(env_v), len(env_m), mode="full")
    peak_idx = int(np.argmax(corr))
    offset_sec = float(lags[peak_idx]) / esr

    # 置信度: peak 高度 vs 均值
    peak_val = corr[peak_idx]
    confidence = peak_val / (np.abs(corr).mean() + 1e-9)
    print(f"  offset = {offset_sec:+.4f}s  (置信度 {confidence:.1f}×, >5 = 可靠)")

    return offset_sec, confidence


def merge(video, audio, offset_sec, output):
    """合并, 视频 copy, 丢弃原视频音轨 (只 map 1:a:0)"""
    cmd = ["ffmpeg", "-y",
           "-i", video,
           "-itsoffset", f"{offset_sec:.4f}",
           "-i", audio,
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy",
           "-c:a", "aac", "-b:a", "320k",
           "-shortest", "-movflags", "+faststart",
           output]
    print(f"  ffmpeg 合并 (offset {offset_sec:+.4f}s, 视频 copy, 原声丢弃)...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] 合并失败:\n{r.stderr}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="onset 对齐视频同步 (鼓/打击乐专用)")
    ap.add_argument("--video",  required=True)
    ap.add_argument("--audio",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--offset", type=float, default=None,
                    help="手动偏移秒数, 跳过 onset 检测")
    args = ap.parse_args()

    if args.offset is not None:
        offset = args.offset
        print(f"手动偏移: {offset:+.4f}s")
    else:
        print("Step 1: 提取视频音轨...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            extract_video_audio(args.video, tmp)
            print("Step 2: onset 包络互相关...")
            offset, _ = detect_offset_onset(tmp, args.audio)
        finally:
            os.unlink(tmp)

    merge(args.video, args.audio, offset, args.output)
    print(f"\n✓ {args.output}  (offset {offset:+.4f}s)")


if __name__ == "__main__":
    main()
