#!/usr/bin/env python3
"""
视频音画同步工作流
用「未混音原始伴奏」与「原始视频音轨」做波形互相关，找到精确偏移后合并。

核心原则：不能用混音后音频做相关（EQ/压缩/混响改变相位，偏差可达 2s+）
         必须用原始未处理伴奏文件做参考。

用法：
    python3 sync_video.py \
        --video   原始视频.MOV \
        --audio   混音后音频.wav \
        --acmp    伴奏（未混音）.wav \
        --output  输出_同步版.mp4

    # 如果已知偏移，跳过检测直接合并：
    python3 sync_video.py \
        --video   原始视频.MOV \
        --audio   混音后音频.wav \
        --offset  6.760 \
        --output  输出_同步版.mp4
"""

import argparse
import subprocess
import sys
import tempfile
import os
import numpy as np
import soundfile as sf
from scipy import signal


def extract_video_audio(video_path: str, out_wav: str):
    """从视频提取单声道音轨用于相关分析"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-map", "0:a:0",
        "-ac", "1", "-ar", "44100",
        "-c:a", "pcm_f32le",
        out_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] 提取视频音轨失败:\n{result.stderr}")
        sys.exit(1)


def detect_offset(video_audio_path: str, acmp_path: str) -> float:
    """
    用降采样互相关找伴奏在视频中的起始偏移（秒）。
    返回值 > 0 表示伴奏在视频开始后第 X 秒才响起。
    """
    orig, sr = sf.read(video_audio_path, dtype="float32")
    acmp, _  = sf.read(acmp_path, dtype="float32", always_2d=True)
    acmp_mono = acmp.mean(axis=1)

    # 降采样到 4kHz 加速全局搜索
    ds      = sr // 4000
    orig_ds = orig[::ds]
    acmp_ds = acmp_mono[::ds]
    sr_ds   = sr // ds

    seg_len  = min(sr_ds * 60, len(orig_ds), len(acmp_ds))
    orig_seg = orig_ds[:seg_len];  orig_seg = orig_seg / (orig_seg.std() + 1e-8)
    acmp_seg = acmp_ds[:seg_len];  acmp_seg = acmp_seg / (acmp_seg.std() + 1e-8)

    print("  计算波形互相关...")
    corr  = signal.correlate(orig_seg, acmp_seg, mode="full")
    lags  = signal.correlation_lags(len(orig_seg), len(acmp_seg), mode="full")
    offset_sec = float(lags[np.argmax(np.abs(corr))]) / sr_ds

    return offset_sec


def merge(video_path: str, audio_path: str, offset_sec: float, output_path: str):
    """将混音音频以指定偏移合并进视频（视频流 copy，不重新编码）"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-itsoffset", f"{offset_sec:.4f}",
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k",
        "-shortest",
        "-movflags", "+faststart",
        output_path
    ]
    print(f"  合并中（视频 copy，偏移 {offset_sec:.4f}s）...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] 合并失败:\n{result.stderr}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="视频音画同步工具")
    parser.add_argument("--video",   required=True,  help="原始视频文件（MOV/MP4）")
    parser.add_argument("--audio",   required=True,  help="混音后音频（WAV）")
    parser.add_argument("--acmp",    default=None,   help="未混音原始伴奏（WAV），用于偏移检测")
    parser.add_argument("--offset",  type=float, default=None, help="手动指定偏移秒数（跳过检测）")
    parser.add_argument("--output",  required=True,  help="输出文件路径（MP4）")
    args = parser.parse_args()

    print(f"视频: {args.video}")
    print(f"音频: {args.audio}")
    print(f"输出: {args.output}")

    # 确定偏移
    if args.offset is not None:
        offset_sec = args.offset
        print(f"使用手动偏移: {offset_sec:.4f}s")
    elif args.acmp:
        print(f"伴奏参考: {args.acmp}")
        print("Step 1/2: 提取视频音轨...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            extract_video_audio(args.video, tmp_wav)
            print("Step 2/2: 波形互相关检测偏移...")
            offset_sec = detect_offset(tmp_wav, args.acmp)
        finally:
            os.unlink(tmp_wav)
        print(f"  检测结果: {offset_sec:.4f}s")
    else:
        print("[ERROR] 请提供 --acmp（伴奏文件）或 --offset（手动偏移）")
        sys.exit(1)

    # 合并
    merge(args.video, args.audio, offset_sec, args.output)

    print(f"\n完成！输出: {args.output}")
    print(f"偏移: {offset_sec:.4f}s（如需微调，加 --offset 参数重跑）")


if __name__ == "__main__":
    main()
