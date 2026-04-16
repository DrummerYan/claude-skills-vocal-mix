#!/usr/bin/env python3
"""
透明自动音准修正 (Auto Pitch Correction)
原理：pyworld 声码器分离「音高 F0」与「音色」，只修 F0，音色完全保留。
效果：自然柔性修正，无电音感，保留颤音和表情。

用法：
    python3 autotune.py 干声.wav                         # 默认修正强度 0.75
    python3 autotune.py 干声.wav --strength 0.9          # 修正更狠（0~1）
    python3 autotune.py 干声.wav --scale major --key C   # 指定调式
    python3 autotune.py 干声.wav --out 修音准完毕.wav
"""

import argparse
import numpy as np
import soundfile as sf
import pyworld as pw
import librosa

# ── 音阶定义 ──────────────────────────────────────────
SCALES = {
    "major":       [0, 2, 4, 5, 7, 9, 11],   # 大调
    "minor":       [0, 2, 3, 5, 7, 8, 10],   # 自然小调
    "pentatonic":  [0, 2, 4, 7, 9],           # 五声音阶
    "chromatic":   list(range(12)),            # 半音阶（修到最近半音）
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def build_scale_midi(scale_name: str, key: str) -> list[int]:
    """生成指定调式的所有 MIDI 音符（全音域）"""
    root = NOTE_NAMES.index(key.upper().replace("B", "A#").replace("BB", "A#"))
    intervals = SCALES[scale_name]
    midi_notes = []
    for octave in range(11):
        for interval in intervals:
            note = root + interval + octave * 12
            if 0 <= note <= 127:
                midi_notes.append(note)
    return sorted(midi_notes)

def hz_to_midi(freq: float) -> float:
    return 69 + 12 * np.log2(freq / 440.0)

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))

def nearest_scale_note(midi: float, scale_midi: list[int]) -> float:
    """找到音阶中最近的音符"""
    arr = np.array(scale_midi, dtype=float)
    return float(arr[np.argmin(np.abs(arr - midi))])

def smooth_f0(f0: np.ndarray, voiced_mask: np.ndarray,
              window_ms: int = 80, sr_frame: float = 200.0) -> np.ndarray:
    """
    对 F0 做适度平滑（保留颤音，去掉音高抖动）
    window_ms: 平滑窗口毫秒数，越大颤音越少
    """
    window = max(1, int(window_ms * sr_frame / 1000))
    f0_smooth = f0.copy()
    voiced_idx = np.where(voiced_mask)[0]
    if len(voiced_idx) < 2:
        return f0_smooth
    # 只对有声段平滑
    kernel = np.ones(window) / window
    voiced_f0 = f0[voiced_idx]
    if len(voiced_f0) >= window:
        smoothed = np.convolve(voiced_f0, kernel, mode="same")
        f0_smooth[voiced_idx] = smoothed
    return f0_smooth


def pitch_correct(audio: np.ndarray, sr: int,
                  scale_midi: list[int],
                  strength: float = 0.75,
                  smooth_window_ms: int = 60) -> np.ndarray:
    """
    核心修音准函数
    strength: 0 = 不修，1 = 完全对齐，0.75 = 自然柔性修正
    """
    # pyworld 需要 float64，单声道
    audio_64 = audio.astype(np.float64)

    # 1. 声码器分析（harvest 比 dio 对人声更精准）
    f0, t = pw.harvest(audio_64, sr, f0_floor=60, f0_ceil=1100,
                       frame_period=5.0)
    sp = pw.cheaptrick(audio_64, f0, t, sr)
    ap = pw.d4c(audio_64, f0, t, sr)

    voiced_mask = f0 > 0

    # 2. 对有声帧做音高修正
    f0_corrected = f0.copy()
    for i, (freq, voiced) in enumerate(zip(f0, voiced_mask)):
        if not voiced:
            continue
        midi_orig   = hz_to_midi(freq)
        midi_target = nearest_scale_note(midi_orig, scale_midi)
        # 柔性修正：按 strength 插值（保留颤音）
        midi_new = midi_orig + strength * (midi_target - midi_orig)
        f0_corrected[i] = midi_to_hz(midi_new)

    # 3. 适度平滑（去抖动，保留自然颤音）
    f0_corrected = smooth_f0(f0_corrected, voiced_mask,
                             window_ms=smooth_window_ms)

    # 4. 声码器重合成（音色 sp / 气声 ap 完全不变）
    corrected = pw.synthesize(f0_corrected, sp, ap, sr,
                              frame_period=5.0)

    # 对齐长度（声码器可能有轻微长度差）
    min_len = min(len(audio), len(corrected))
    return corrected[:min_len].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="透明自动音准修正")
    parser.add_argument("input",               help="输入干声 WAV")
    parser.add_argument("--out",     default=None,       help="输出路径（默认在文件名后加 _修音准）")
    parser.add_argument("--scale",   default="major",    choices=list(SCALES.keys()), help="音阶")
    parser.add_argument("--key",     default="C",        help="调性（C D E F G A B，支持 # 号）")
    parser.add_argument("--strength",type=float, default=0.75, help="修正强度 0~1（默认 0.75）")
    parser.add_argument("--smooth",  type=int,   default=60,   help="平滑窗口 ms（默认 60ms）")
    args = parser.parse_args()

    # 输出路径
    if args.out is None:
        base = args.input.rsplit(".", 1)[0]
        args.out = f"{base}_修音准.wav"

    print(f"输入: {args.input}")
    print(f"调性: {args.key} {args.scale}  强度: {args.strength}  平滑: {args.smooth}ms")

    # 读取（转单声道处理，再还原立体声）
    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    n_ch = audio.shape[1]

    scale_midi = build_scale_midi(args.scale, args.key)
    print(f"音阶音符数: {len(scale_midi)}  采样率: {sr}Hz  时长: {len(audio)/sr:.1f}s")

    corrected_channels = []
    for ch in range(n_ch):
        print(f"处理声道 {ch+1}/{n_ch}...")
        ch_audio = audio[:, ch]
        corrected = pitch_correct(ch_audio, sr, scale_midi,
                                  strength=args.strength,
                                  smooth_window_ms=args.smooth)
        corrected_channels.append(corrected)

    result = np.stack(corrected_channels, axis=-1)

    # 归一化防止轻微溢出
    peak = np.max(np.abs(result))
    if peak > 0.999:
        result = result / peak * 0.999

    sf.write(args.out, result, sr, subtype="FLOAT")
    print(f"\n完成！输出: {args.out}")


if __name__ == "__main__":
    main()
