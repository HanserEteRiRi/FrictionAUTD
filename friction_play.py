"""Play analyzed friction motion on AUTD with synchronized audio.

中文：
    这是 FrictionAUTD 的新控制入口。它读取 friction_analyze.py 生成的
    *_friction_motion.json，把每一帧里的 intensity / cycle_hz / length_mm /
    height_mm / jitter_mm 转换成 UltraSleep 18 台阵列上的左右同步 GainSTM，
    并在第一帧触觉发送成功后立刻开始播放同一段音频。

English:
    This is the compact playback controller for FrictionAUTD. It reads the
    *_friction_motion.json produced by friction_analyze.py, maps every frame's
    intensity / cycle_hz / length_mm / height_mm / jitter_mm to synchronized
    left/right GainSTM sequences on UltraSleep's 18-device geometry, and starts
    audio playback immediately after the first haptic frame is sent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    import tomli as tomllib  # type: ignore[no-redef]


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.toml"
ULTRASOUND_FREQ_HZ = 40_000


@dataclass(frozen=True)
class MotionFrame:
    """One interpolated haptic-control frame.

    中文：这是从 JSON 的 motion_curve 里按当前播放时间插值得到的一帧控制参数。
    English: Interpolated control parameters sampled from motion_curve.
    """

    time_s: float
    intensity: float
    cycle_hz: float
    length_mm: float
    height_mm: float
    jitter_mm: float
    normalized_amplitude: float
    roughness: float


@dataclass(frozen=True)
class StmTiming:
    """AUTD-compatible STM timing.

    中文：AUTD 要求 STM 采样频率能整除 40 kHz，所以不能随便发送 cycle_hz。
    English: AUTD requires STM sampling frequency to divide 40 kHz exactly.
    """

    frame_count: int
    sampling_hz: int
    divide: int
    cycle_hz: float


@dataclass(frozen=True)
class Settings:
    motion_json: Path
    audio_wav: Path
    mode: str
    start_s: float
    duration_s: float
    update_hz: float
    sync_audio: bool
    preferred_frames: int
    intensity_scale: float
    max_intensity: int
    max_consecutive_failures: int
    base_z_mm: float
    right_center_x_mm: float
    right_center_y_mm: float
    left_center_x_mm: float
    left_center_y_mm: float
    audio_backend: str
    audio_device: int | str | None
    audio_volume: float


_SEND_SLOW_WARNING_PRINTED = False


class MotionTimeline:
    """Read and sample a friction_motion.json file.

    中文：JSON 里的 motion_curve 是离散点；运行时按照当前音频时间线性插值。
    English: motion_curve is discrete; playback samples it by linear interpolation.
    """

    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        curve = sorted(data.get("motion_curve", []), key=lambda item: float(item["time_s"]))
        if not curve:
            raise ValueError(f"No motion_curve found in {path}")

        self.path = path
        self.data = data
        self.source_path = str(data.get("source_path", ""))
        self.duration_s = float(data.get("duration_s", curve[-1]["time_s"]))
        self.times = np.asarray([float(p["time_s"]) for p in curve], dtype=np.float64)
        self.values = {
            "intensity": np.asarray([float(p.get("intensity", 0.0)) for p in curve], dtype=np.float64),
            "cycle_hz": np.asarray([float(p.get("cycle_hz", 2.0 / 3.0)) for p in curve], dtype=np.float64),
            "length_mm": np.asarray([float(p.get("length_mm", 60.0)) for p in curve], dtype=np.float64),
            "height_mm": np.asarray([float(p.get("height_mm", 100.0)) for p in curve], dtype=np.float64),
            "jitter_mm": np.asarray([float(p.get("jitter_mm", 0.0)) for p in curve], dtype=np.float64),
            "normalized_amplitude": np.asarray(
                [float(p.get("normalized_amplitude", 0.0)) for p in curve],
                dtype=np.float64,
            ),
            "roughness": np.asarray([float(p.get("roughness", 0.0)) for p in curve], dtype=np.float64),
        }

    @classmethod
    def load(cls, path: Path) -> "MotionTimeline":
        with path.open("r", encoding="utf-8") as handle:
            return cls(path, json.load(handle))

    def sample(self, time_s: float) -> MotionFrame:
        """Return one interpolated frame at absolute audio time_s."""

        t = float(np.clip(time_s, self.times[0], self.times[-1]))
        sampled = {key: float(np.interp(t, self.times, values)) for key, values in self.values.items()}
        return MotionFrame(
            time_s=t,
            intensity=sampled["intensity"],
            cycle_hz=sampled["cycle_hz"],
            length_mm=sampled["length_mm"],
            height_mm=sampled["height_mm"],
            jitter_mm=sampled["jitter_mm"],
            normalized_amplitude=sampled["normalized_amplitude"],
            roughness=sampled["roughness"],
        )


class SoundDeviceAudioPlayer:
    """Small WAV player for synchronized playback.

    中文：先把 WAV 读进内存并打开 sounddevice 输出流，start() 时只启动流。
    English: Preload WAV and open sounddevice first; start() only starts the stream.
    """

    def __init__(
        self,
        wav_path: Path,
        *,
        device: int | str | None,
        volume: float,
    ) -> None:
        self.wav_path = wav_path
        self.device = normalize_audio_device(device)
        self.volume = max(0.0, float(volume))
        self.samples: np.ndarray | None = None
        self.sample_rate = 0
        self.source_channels = 0
        self.output_channels = 0
        self.position = 0
        self.stream = None
        self.finished = threading.Event()
        self.lock = threading.Lock()

    def __enter__(self) -> "SoundDeviceAudioPlayer":
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for synchronized audio playback.") from exc

        samples, self.sample_rate, self.source_channels = load_wav_pcm(self.wav_path)
        device_info = sd.query_devices(self.device, "output")
        max_channels = int(device_info["max_output_channels"])
        if max_channels <= 0:
            raise RuntimeError(f"Audio device {self.device!r} is not an output device.")

        self.output_channels = 2 if max_channels >= 2 else 1
        self.samples = adapt_channels(samples, self.output_channels)
        self.samples = np.clip(self.samples * self.volume, -1.0, 1.0).astype(np.float32)

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.output_channels,
            dtype="float32",
            device=self.device,
            blocksize=2048,
            latency="high",
            callback=self._callback,
        )
        print(
            "Audio ready:",
            f"device={device_info['name']}",
            f"sample_rate={self.sample_rate}",
            f"channels={self.source_channels}->{self.output_channels}",
            f"frames={len(self.samples)}",
        )
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.stop()

    @property
    def duration_s(self) -> float:
        if self.samples is None or self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate

    @staticmethod
    def list_devices() -> None:
        import sounddevice as sd

        print(sd.query_devices())

    def start(self, offset_s: float) -> None:
        if self.stream is None:
            raise RuntimeError("Audio stream is not prepared.")
        with self.lock:
            start_frame = int(round(max(0.0, offset_s) * self.sample_rate))
            self.position = min(start_frame, 0 if self.samples is None else len(self.samples))
            self.finished.clear()
        self.stream.start()
        print(f"Audio started at {offset_s:.3f}s.")

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.finished.set()

    def _callback(self, outdata: np.ndarray, frames: int, _time_info: object, status: object) -> None:
        if status:
            print(f"Audio status: {status}")

        samples = self.samples
        if samples is None:
            outdata.fill(0.0)
            self.finished.set()
            return

        with self.lock:
            available = len(samples) - self.position
            count = min(frames, max(0, available))
            if count > 0:
                outdata[:count] = samples[self.position : self.position + count]
                self.position += count
            if count < frames:
                outdata[count:] = 0.0
                self.finished.set()


class AfplayAudioPlayer:
    """macOS process-based audio player.

    中文：afplay 在独立进程中播放音频，不需要 Python callback，所以不会因为
    AUTD 发送阻塞 Python 而导致声音卡顿。它使用系统默认输出设备。

    English: afplay plays audio in a separate process. It does not need a Python
    callback, so AUTD sends cannot starve audio playback. It uses the system
    default output device.
    """

    def __init__(self, wav_path: Path, *, volume: float, duration_s: float) -> None:
        self.wav_path = wav_path
        self.volume = max(0.0, float(volume))
        self.requested_duration_s = max(0.0, float(duration_s))
        self.process: subprocess.Popen[bytes] | None = None
        self.finished = threading.Event()
        self._duration_s = read_wav_duration_s(wav_path)

    def __enter__(self) -> "AfplayAudioPlayer":
        if shutil.which("afplay") is None:
            raise RuntimeError("afplay was not found. Use audio.backend='sounddevice' instead.")
        print(
            "Audio ready:",
            "backend=afplay",
            f"source={self.wav_path.name}",
            f"duration={self._duration_s:.3f}s",
        )
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.stop()

    @property
    def duration_s(self) -> float:
        return self._duration_s

    def start(self, offset_s: float) -> None:
        if offset_s > 0.001:
            raise RuntimeError("afplay does not support start offsets; use audio.backend='sounddevice'.")

        command = ["/usr/bin/afplay", "-v", f"{self.volume:g}"]
        if self.requested_duration_s > 0:
            command.extend(["-t", f"{self.requested_duration_s:g}"])
        command.append(str(self.wav_path))

        # 中文：如果主程序用 sudo 跑 AUTD，尽量切回原登录用户启动 afplay。
        # English: If AUTD runs under sudo, start afplay as the original login user.
        sudo_user = os.environ.get("SUDO_USER", "")
        if os.geteuid() == 0 and sudo_user and sudo_user != "root" and shutil.which("sudo"):
            command = ["/usr/bin/sudo", "-u", sudo_user, *command]

        self.finished.clear()
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(0.02)
        if self.process.poll() is not None:
            stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
            self.finished.set()
            raise RuntimeError(f"afplay exited immediately. stderr={stderr.strip()!r}")
        print("Audio started by afplay.")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        self.process = None
        self.finished.set()


class SimulatedFrictionAUTD:
    """Terminal-only playback target.

    中文：没有 AUTD 时使用，单行动态打印当前焦点参数。
    English: Use without AUTD hardware; prints current focus parameters on one line.
    """

    def __enter__(self) -> "SimulatedFrictionAUTD":
        print("SIM mode: AUTD hardware is not opened.")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def send(self, frame: MotionFrame, _audio_time_s: float) -> bool:
        line = (
            f"\rSIM t={frame.time_s:6.2f}s "
            f"I={round(frame.intensity):3d} "
            f"cycle={frame.cycle_hz:4.2f}Hz "
            f"length={frame.length_mm:5.1f}mm "
            f"height={frame.height_mm:5.1f}mm "
            f"jitter={frame.jitter_mm:4.1f}mm"
        )
        print(line, end="", flush=True)
        return True

    def close(self) -> None:
        print("\nSIM stopped.")


class RealFrictionAUTD:
    """Drive UltraSleep's unchanged 18-device geometry from friction frames.

    Devices 0..8 render the right trajectory and devices 9..17 render the left
    trajectory. Both GainSTM sequences share timing and phase, so the two
    friction strokes start and move together.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.autd = None
        self.last_frame: MotionFrame | None = None
        self.last_timing: StmTiming | None = None

    def __enter__(self) -> "RealFrictionAUTD":
        import config
        from autd_manager import AUTDManager

        self.autd = AUTDManager(link=config.LINK)
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def send(self, frame: MotionFrame, audio_time_s: float) -> bool:
        if self.autd is None:
            raise RuntimeError("AUTD controller is not opened.")

        timing = choose_stm_timing(frame.cycle_hz, self.settings.preferred_frames)
        gain_right, gain_left = self._make_gain_stms(frame, timing, audio_time_s)
        ok = self._send_gains(
            gain_right,
            gain_left,
            label=f"t={frame.time_s:.2f}s I={round(frame.intensity)}",
        )
        if ok:
            self.last_frame = frame
            self.last_timing = timing
        return ok

    def close(self) -> None:
        if self.autd is None:
            return
        try:
            self.autd.stop()
        finally:
            self.autd.close()
            self.autd = None
            print("AUTD stopped.")

    def _make_gain_stms(self, frame: MotionFrame, timing: StmTiming, audio_time_s: float) -> tuple[Any, Any]:
        import config
        from pyautd3 import Focus, FocusOption, GainSTM, GainSTMMode, GainSTMOption, SamplingConfig
        try:
            from pyautd3 import EmitIntensity
        except ImportError:  # EmitIntensity was renamed to Intensity in pyautd3 >= 35.
            from pyautd3 import Intensity as EmitIntensity

        intensity = int(round(frame.intensity * self.settings.intensity_scale))
        intensity = min(self.settings.max_intensity, max(0, intensity))

        z = self.settings.base_z_mm + max(1.0, frame.height_mm)
        half_length = max(1.0, frame.length_mm) / 2.0
        jitter = max(0.0, frame.jitter_mm)
        phase0 = 2.0 * math.pi * timing.cycle_hz * max(0.0, audio_time_s)

        gains_right = []
        gains_left = []
        for i in range(timing.frame_count):
            theta = phase0 + 2.0 * math.pi * i / timing.frame_count

            # 中文：cos 轨迹在左右端点速度为 0，模拟摩擦来回运动的加减速。
            # English: Cosine motion has zero velocity at endpoints, matching rubbing strokes.
            x = -half_length * math.cos(theta)

            # 中文：jitter 用平滑正弦扰动表示粗糙摩擦，不使用逐帧随机跳变。
            # English: Smooth sinusoidal jitter represents roughness without random frame jumps.
            y = jitter * (0.65 * math.sin(3.0 * theta) + 0.35 * math.sin(7.0 * theta + 0.7))
            point_right = (
                np.array(
                    [
                        self.settings.right_center_x_mm + x,
                        self.settings.right_center_y_mm + y,
                        z,
                    ],
                    dtype=np.float64,
                )
                + config.DEVICE_CENTER_RIGHT
            )
            point_left = (
                np.array(
                    [
                        self.settings.left_center_x_mm + x,
                        self.settings.left_center_y_mm + y,
                        z,
                    ],
                    dtype=np.float64,
                )
                + config.DEVICE_CENTER_LEFT
            )
            option = FocusOption(intensity=EmitIntensity(intensity))
            gains_right.append(Focus(point_right, option=option))
            gains_left.append(Focus(point_left, option=option))

        stm_option = GainSTMOption(mode=GainSTMMode.PhaseIntensityFull)
        sampling = SamplingConfig(timing.divide)
        return (
            GainSTM(gains=gains_right, config=sampling, option=stm_option),
            GainSTM(gains=gains_left, config=sampling, option=stm_option),
        )

    def _send_gains(self, gain_right: Any, gain_left: Any, *, label: str) -> bool:
        from pyautd3 import Static

        assert self.autd is not None
        for attempt in range(1, 4):
            try:
                self.autd.perform_double_irradiate(g1=gain_right, g2=gain_left, m=Static())
                return True
            except Exception as exc:  # noqa: BLE001 - pyautd3 raises several transport exceptions.
                print(f"\nAUTD send failed ({attempt}/3) {label}: {exc}")
                time.sleep(0.03)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read friction_motion.json, drive AUTD focus motion, and play audio in sync.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="TOML config path.")
    parser.add_argument("--mode", choices=["real", "simulate"], default="", help="Override config playback.mode.")
    parser.add_argument("--json", default="", help="Override config paths.motion_json.")
    parser.add_argument("--wav", default="", help="Override config paths.audio_wav or JSON source_path.")
    parser.add_argument("--start-s", type=float, default=None, help="Override playback.start_s.")
    parser.add_argument("--duration-s", type=float, default=None, help="Override playback.duration_s.")
    parser.add_argument("--no-audio", action="store_true", help="Run haptics without synchronized audio.")
    parser.add_argument("--list-audio-devices", action="store_true", help="Print sounddevice output devices.")
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> Settings:
    config_path = resolve_path(Path(args.config), base=SCRIPT_DIR)
    config = read_toml(config_path)
    paths = config.get("paths", {})
    playback = config.get("playback", {})
    autd = config.get("autd", {})
    audio = config.get("audio", {})

    motion_json_value = args.json or str(paths.get("motion_json", ""))
    if not motion_json_value:
        raise SystemExit("No motion JSON configured. Set paths.motion_json in config.toml.")
    motion_json = resolve_path(Path(motion_json_value), base=SCRIPT_DIR)

    with motion_json.open("r", encoding="utf-8") as handle:
        motion_data = json.load(handle)

    audio_value = args.wav or str(paths.get("audio_wav", "")) or str(motion_data.get("source_path", ""))
    if not audio_value:
        raise SystemExit("No audio WAV configured and JSON has no source_path.")
    audio_wav = resolve_existing_path(Path(audio_value), bases=[Path.cwd(), SCRIPT_DIR, PROJECT_ROOT])

    mode = args.mode or str(playback.get("mode", "real"))
    if mode not in {"real", "simulate"}:
        raise SystemExit(f"Invalid playback.mode={mode!r}; use real or simulate.")

    sync_audio = bool(playback.get("sync_audio", True)) and not args.no_audio
    return Settings(
        motion_json=motion_json,
        audio_wav=audio_wav,
        mode=mode,
        start_s=float(playback.get("start_s", 0.0) if args.start_s is None else args.start_s),
        duration_s=float(playback.get("duration_s", 0.0) if args.duration_s is None else args.duration_s),
        update_hz=max(0.5, float(playback.get("update_hz", 8.0))),
        sync_audio=sync_audio,
        preferred_frames=max(16, int(autd.get("preferred_frames", 96))),
        intensity_scale=max(0.0, float(autd.get("intensity_scale", 1.0))),
        max_intensity=min(255, max(0, int(autd.get("max_intensity", 255)))),
        max_consecutive_failures=max(1, int(autd.get("max_consecutive_failures", 5))),
        base_z_mm=float(autd.get("base_z_mm", 250.0)),
        right_center_x_mm=float(autd.get("right_center_x_mm", 0.0)),
        right_center_y_mm=float(autd.get("right_center_y_mm", 100.0)),
        left_center_x_mm=float(autd.get("left_center_x_mm", 0.0)),
        left_center_y_mm=float(autd.get("left_center_y_mm", -100.0)),
        audio_device=normalize_audio_device(audio.get("device", None)),
        audio_backend=str(audio.get("backend", "auto")).strip().lower() or "auto",
        audio_volume=max(0.0, float(audio.get("volume", 1.0))),
    )


def read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(path: Path, *, base: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_existing_path(path: Path, *, bases: list[Path]) -> Path:
    if path.is_absolute():
        return path
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def normalize_audio_device(device: int | str | None) -> int | str | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    value = str(device).strip()
    if value == "":
        return None
    if value.isdecimal():
        return int(value)
    return value


def load_wav_pcm(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = bytes_[:, 0].astype(np.int32) | (bytes_[:, 1].astype(np.int32) << 8) | (bytes_[:, 2].astype(np.int32) << 16)
        values = np.where(values & 0x800000, values | ~0xFFFFFF, values)
        data = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    return data.astype(np.float32), sample_rate, channels


def read_wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
    return frames / sample_rate if sample_rate > 0 else 0.0


def adapt_channels(samples: np.ndarray, output_channels: int) -> np.ndarray:
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    source_channels = samples.shape[1]
    if source_channels == output_channels:
        return samples
    if source_channels == 1 and output_channels == 2:
        return np.repeat(samples, 2, axis=1)
    if source_channels > output_channels:
        return samples[:, :output_channels]
    padding = np.zeros((len(samples), output_channels - source_channels), dtype=np.float32)
    return np.concatenate([samples, padding], axis=1)


def choose_stm_timing(requested_cycle_hz: float, preferred_frames: int) -> StmTiming:
    """Find a close STM loop frequency whose sampling rate divides 40 kHz."""

    requested = max(0.05, float(requested_cycle_hz))
    min_frames = max(16, preferred_frames - 64)
    max_frames = max(min_frames, preferred_frames + 96)
    valid_sampling_hz = [hz for hz in range(1, ULTRASOUND_FREQ_HZ + 1) if ULTRASOUND_FREQ_HZ % hz == 0]

    best: tuple[float, int, int, float] | None = None
    for frame_count in range(min_frames, max_frames + 1):
        for sampling_hz in valid_sampling_hz:
            cycle_hz = sampling_hz / frame_count
            if cycle_hz < 0.05 or cycle_hz > 8.0:
                continue
            relative_error = abs(cycle_hz - requested) / requested
            frame_penalty = abs(frame_count - preferred_frames) / max(1, preferred_frames) * 0.01
            score = relative_error + frame_penalty
            if best is None or score < best[0]:
                best = (score, frame_count, sampling_hz, cycle_hz)

    if best is None:
        sampling_hz = 64
        return StmTiming(
            frame_count=preferred_frames,
            sampling_hz=sampling_hz,
            divide=ULTRASOUND_FREQ_HZ // sampling_hz,
            cycle_hz=sampling_hz / preferred_frames,
        )

    _, frame_count, sampling_hz, cycle_hz = best
    return StmTiming(
        frame_count=frame_count,
        sampling_hz=sampling_hz,
        divide=ULTRASOUND_FREQ_HZ // sampling_hz,
        cycle_hz=cycle_hz,
    )


def compute_run_duration(settings: Settings, timeline: MotionTimeline, audio_duration_s: float | None) -> float:
    remaining_timeline = max(0.0, timeline.duration_s - settings.start_s)
    candidates = [remaining_timeline]
    if audio_duration_s is not None and audio_duration_s > 0:
        candidates.append(max(0.0, audio_duration_s - settings.start_s))
    natural_duration = min(candidates)
    if settings.duration_s > 0:
        return min(settings.duration_s, natural_duration)
    return natural_duration


def run_playback(
    settings: Settings,
    timeline: MotionTimeline,
    audio_player: SoundDeviceAudioPlayer | AfplayAudioPlayer | None,
) -> None:
    controller: RealFrictionAUTD | SimulatedFrictionAUTD
    controller = SimulatedFrictionAUTD() if settings.mode == "simulate" else RealFrictionAUTD(settings)

    audio_duration = None if audio_player is None else audio_player.duration_s
    run_duration = compute_run_duration(settings, timeline, audio_duration)
    if run_duration <= 0:
        raise SystemExit("Nothing to play: start_s is beyond the audio or timeline duration.")

    update_period = 1.0 / settings.update_hz
    consecutive_failures = 0
    global _SEND_SLOW_WARNING_PRINTED

    with controller:
        first_frame = timeline.sample(settings.start_s)
        if not timed_send(controller, first_frame, settings.start_s, update_period):
            raise RuntimeError("Initial AUTD frame failed; audio was not started.")

        if audio_player is not None:
            audio_player.start(settings.start_s)
        start_wall = time.perf_counter()
        next_update_s = update_period

        print(
            "Synchronized playback:",
            f"mode={settings.mode}",
            f"motion={settings.motion_json.name}",
            f"audio={settings.audio_wav.name if audio_player is not None else 'disabled'}",
            f"start={settings.start_s:.3f}s",
            f"duration={run_duration:.3f}s",
            f"update_hz={settings.update_hz:.2f}",
        )

        try:
            while True:
                elapsed = time.perf_counter() - start_wall
                if elapsed >= run_duration:
                    break
                if audio_player is not None and audio_has_finished(audio_player):
                    break

                if elapsed >= next_update_s:
                    audio_time = settings.start_s + elapsed
                    frame = timeline.sample(audio_time)
                    ok = timed_send(controller, frame, audio_time, update_period)
                    if ok:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= settings.max_consecutive_failures:
                            raise RuntimeError(
                                f"AUTD send failed {consecutive_failures} times in a row; stopping playback."
                            )
                    next_update_s += update_period

                time.sleep(0.002)
        except KeyboardInterrupt:
            print("\nStopping synchronized playback...")


def main() -> None:
    args = parse_args()
    if args.list_audio_devices:
        SoundDeviceAudioPlayer.list_devices()
        return

    settings = load_settings(args)
    timeline = MotionTimeline.load(settings.motion_json)

    print(
        "Loaded motion:",
        f"json={settings.motion_json}",
        f"audio={settings.audio_wav}",
        f"duration={timeline.duration_s:.3f}s",
    )

    if settings.sync_audio:
        audio_player: SoundDeviceAudioPlayer | AfplayAudioPlayer
        run_duration = compute_run_duration(settings, timeline, read_wav_duration_s(settings.audio_wav))
        backend = choose_audio_backend(settings)
        if backend == "afplay":
            audio_player = AfplayAudioPlayer(
                settings.audio_wav,
                volume=settings.audio_volume,
                duration_s=run_duration,
            )
        elif backend == "sounddevice":
            audio_player = SoundDeviceAudioPlayer(
                settings.audio_wav,
                device=settings.audio_device,
                volume=settings.audio_volume,
            )
        else:
            raise SystemExit(f"Invalid audio.backend={settings.audio_backend!r}; use afplay, sounddevice, or auto.")

        with audio_player:
            run_playback(settings, timeline, audio_player)
    else:
        run_playback(settings, timeline, None)


def timed_send(
    controller: RealFrictionAUTD | SimulatedFrictionAUTD,
    frame: MotionFrame,
    audio_time_s: float,
    update_period_s: float,
) -> bool:
    global _SEND_SLOW_WARNING_PRINTED

    start = time.perf_counter()
    ok = controller.send(frame, audio_time_s)
    elapsed = time.perf_counter() - start
    if (
        ok
        and not _SEND_SLOW_WARNING_PRINTED
        and not isinstance(controller, SimulatedFrictionAUTD)
        and elapsed > update_period_s * 0.8
    ):
        print(
            "\nWarning: AUTD send is slow:",
            f"send_time={elapsed:.3f}s",
            f"update_period={update_period_s:.3f}s.",
            "Lower playback.update_hz in config.toml if this repeats.",
        )
        _SEND_SLOW_WARNING_PRINTED = True
    return ok


def audio_has_finished(audio_player: SoundDeviceAudioPlayer | AfplayAudioPlayer) -> bool:
    if isinstance(audio_player, AfplayAudioPlayer) and audio_player.process is not None:
        return_code = audio_player.process.poll()
        if return_code is not None:
            if return_code != 0 and audio_player.process.stderr is not None:
                stderr = audio_player.process.stderr.read().decode("utf-8", errors="replace").strip()
                if stderr:
                    print(f"\nafplay exited with code {return_code}: {stderr}")
            audio_player.finished.set()
            return True
    return audio_player.finished.is_set()


def choose_audio_backend(settings: Settings) -> str:
    backend = settings.audio_backend
    if backend == "auto":
        backend = "afplay" if shutil.which("afplay") is not None else "sounddevice"
    if backend == "afplay" and settings.start_s > 0.001:
        print("audio.backend=afplay cannot start from offset; using sounddevice instead.")
        return "sounddevice"
    if backend == "afplay" and settings.audio_device is not None:
        print("audio.device is set; using sounddevice so the requested device can be used.")
        return "sounddevice"
    return backend


if __name__ == "__main__":
    main()
