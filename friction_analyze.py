"""Analyze rubbing/friction audio into focus-motion parameters.

This script is independent from AudioAUTD and intentionally skips AudioSep.
It uses librosa to extract framewise loudness, onset strength, tempo, and
spectral descriptors, then writes a compact JSON timeline for the friction
focus controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a friction/rubbing WAV file into AUTD focus-motion parameters.",
    )
    parser.add_argument("--wav", required=True, help="Input WAV file.")
    parser.add_argument("--output", default="", help="Output JSON path.")
    parser.add_argument("--plot", action="store_true", help="Also write a PNG summary plot.")
    parser.add_argument("--plot-output", default="", help="Output PNG path.")
    parser.add_argument("--sr", type=int, default=22_050, help="Analysis sample rate.")
    parser.add_argument("--frame-ms", type=float, default=100.0, help="Feature frame size.")
    parser.add_argument("--hop-ms", type=float, default=50.0, help="Feature hop size.")
    parser.add_argument("--floor-db", type=float, default=-55.0, help="Amplitude normalization floor.")
    parser.add_argument("--ceiling-db", type=float, default=-12.0, help="Amplitude normalization ceiling.")
    parser.add_argument("--min-cycle-hz", type=float, default=0.25, help="Slowest allowed focus cycle.")
    parser.add_argument("--max-cycle-hz", type=float, default=3.0, help="Fastest allowed focus cycle.")
    parser.add_argument("--default-cycle-hz", type=float, default=2.0 / 3.0, help="Fallback focus cycle.")
    parser.add_argument("--min-length-mm", type=float, default=30.0, help="Shortest stroke length.")
    parser.add_argument("--max-length-mm", type=float, default=90.0, help="Longest stroke length.")
    parser.add_argument("--max-jitter-mm", type=float, default=4.0, help="Maximum roughness jitter.")
    parser.add_argument(
        "--onset-divisor",
        type=float,
        default=2.0,
        help="Convert onset pulse rate to full back-and-forth cycle rate. Use 2 if one rub pulse is a half-cycle.",
    )
    return parser.parse_args()


def clamp(value: np.ndarray | float, low: float, high: float) -> np.ndarray | float:
    return np.minimum(high, np.maximum(low, value))


def normalize_db(db: np.ndarray, floor_db: float, ceiling_db: float) -> np.ndarray:
    span = max(1e-6, ceiling_db - floor_db)
    return clamp((db - floor_db) / span, 0.0, 1.0)


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def estimate_cycle_curve(
    onset_env: np.ndarray,
    hop_length: int,
    sample_rate: int,
    frame_count: int,
    min_cycle_hz: float,
    max_cycle_hz: float,
    default_cycle_hz: float,
    onset_divisor: float,
) -> np.ndarray:
    """Estimate a local rubbing cycle curve from onset strength.

    The local pulse rate is estimated from distances between onset peaks. Because
    many rubbing sounds produce one acoustic pulse per half stroke, the default
    mapping divides the pulse rate by two.
    """

    try:
        import librosa
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("librosa is required. Install it with: pip install librosa") from exc

    cycle = np.full(frame_count, default_cycle_hz, dtype=np.float32)
    if frame_count == 0 or len(onset_env) == 0:
        return cycle

    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=6,
        post_avg=6,
        delta=max(0.02, float(np.std(onset_env)) * 0.2),
        wait=2,
    )
    if len(peaks) < 2:
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sample_rate, hop_length=hop_length)
        if len(tempo):
            cycle[:] = float(clamp(float(tempo[0]) / 60.0 / max(1e-6, onset_divisor), min_cycle_hz, max_cycle_hz))
        return cycle

    peak_times = librosa.frames_to_time(peaks, sr=sample_rate, hop_length=hop_length)
    pulse_hz = 1.0 / np.maximum(np.diff(peak_times), 1e-6)
    local_cycle_hz = pulse_hz / max(1e-6, onset_divisor)

    frame_times = librosa.frames_to_time(np.arange(frame_count), sr=sample_rate, hop_length=hop_length)
    mid_times = (peak_times[:-1] + peak_times[1:]) / 2.0
    interpolated = np.interp(
        frame_times,
        mid_times,
        local_cycle_hz,
        left=float(local_cycle_hz[0]),
        right=float(local_cycle_hz[-1]),
    )
    return np.asarray(clamp(smooth(interpolated, 5), min_cycle_hz, max_cycle_hz), dtype=np.float32)


def analyze_friction(args: argparse.Namespace) -> dict[str, object]:
    try:
        import librosa
    except Exception as exc:
        raise SystemExit("librosa is required. Install it with: pip install librosa") from exc

    wav_path = Path(args.wav)
    y, sr = librosa.load(wav_path, sr=args.sr, mono=True)
    if len(y) == 0:
        raise SystemExit(f"Empty audio: {wav_path}")

    frame_length = max(8, int(round(sr * args.frame_ms / 1000.0)))
    hop_length = max(1, int(round(sr * args.hop_ms / 1000.0)))
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    db = librosa.amplitude_to_db(np.maximum(rms, EPSILON), ref=1.0, amin=EPSILON, top_db=None)
    amplitude = normalize_db(db, args.floor_db, args.ceiling_db).astype(np.float32)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = np.asarray(onset_env[: len(rms)], dtype=np.float32)
    if len(onset_env) < len(rms):
        onset_env = np.pad(onset_env, (0, len(rms) - len(onset_env)))
    onset_norm = min_max_normalize(smooth(onset_env, 3))

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=max(16, frame_length),
        hop_length=hop_length,
        center=False,
    )[0][: len(rms)]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=max(16, frame_length),
        hop_length=hop_length,
        center=False,
    )[0][: len(rms)]
    centroid_norm = min_max_normalize(centroid)
    bandwidth_norm = min_max_normalize(bandwidth)
    roughness = clamp(0.65 * centroid_norm + 0.35 * bandwidth_norm, 0.0, 1.0)

    cycle_hz = estimate_cycle_curve(
        onset_env=onset_env,
        hop_length=hop_length,
        sample_rate=sr,
        frame_count=len(rms),
        min_cycle_hz=args.min_cycle_hz,
        max_cycle_hz=args.max_cycle_hz,
        default_cycle_hz=args.default_cycle_hz,
        onset_divisor=args.onset_divisor,
    )
    length_mm = args.min_length_mm + amplitude * (args.max_length_mm - args.min_length_mm)
    jitter_mm = roughness * args.max_jitter_mm
    intensity = np.rint(amplitude**0.7 * 255.0).astype(int)

    points = [
        {
            "time_s": float(times[i]),
            "rms": float(rms[i]),
            "db": float(db[i]),
            "normalized_amplitude": float(amplitude[i]),
            "onset_strength": float(onset_env[i]),
            "onset_normalized": float(onset_norm[i]),
            "spectral_centroid_hz": float(centroid[i]) if i < len(centroid) else 0.0,
            "spectral_bandwidth_hz": float(bandwidth[i]) if i < len(bandwidth) else 0.0,
            "roughness": float(roughness[i]) if i < len(roughness) else 0.0,
            "intensity": int(clamp(intensity[i], 0, 255)),
            "cycle_hz": float(cycle_hz[i]),
            "length_mm": float(length_mm[i]),
            "height_mm": 100.0,
            "jitter_mm": float(jitter_mm[i]) if i < len(jitter_mm) else 0.0,
        }
        for i in range(len(rms))
    ]

    active = amplitude > 0.03
    return {
        "source_path": str(wav_path),
        "duration_s": float(librosa.get_duration(y=y, sr=sr)),
        "sample_rate": int(sr),
        "backend": "librosa",
        "audiosep": "disabled",
        "frame_ms": float(args.frame_ms),
        "hop_ms": float(args.hop_ms),
        "motion_mapping": {
            "intensity": "normalized_amplitude ** 0.7 * 255",
            "cycle_hz": "onset peak interval / onset_divisor, clamped",
            "length_mm": "normalized_amplitude mapped to min/max length",
            "jitter_mm": "spectral centroid/bandwidth roughness mapped to max jitter",
            "height_mm": "fixed at 100 mm for first implementation",
        },
        "summary": {
            "rms_mean": float(np.mean(rms)),
            "db_mean": float(np.mean(db)),
            "normalized_mean": float(np.mean(amplitude)),
            "normalized_max": float(np.max(amplitude)),
            "intensity_mean": float(np.mean(intensity)),
            "intensity_max": int(np.max(intensity)),
            "cycle_hz_mean": float(np.mean(cycle_hz)),
            "cycle_hz_min": float(np.min(cycle_hz)),
            "cycle_hz_max": float(np.max(cycle_hz)),
            "length_mm_mean": float(np.mean(length_mm)),
            "jitter_mm_mean": float(np.mean(jitter_mm)),
            "active_ratio": float(np.mean(active)),
        },
        "motion_curve": points,
    }


def default_output_path(wav_path: str | Path) -> Path:
    source = Path(wav_path)
    return Path(__file__).resolve().parent / "analysis" / f"{source.stem}_friction_motion.json"


def clip_figure_dir_from_json(json_path: Path) -> Path:
    stem = json_path.stem
    if stem.endswith("_friction_motion"):
        stem = stem[: -len("_friction_motion")]
    return json_path.parent / "figures" / stem


def write_line_figure(
    output_path: Path,
    times: np.ndarray,
    series: list[tuple[np.ndarray, str, str]],
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.2))
    for values, label, color in series:
        ax.plot(times, values, label=label, color=color, linewidth=1.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (s)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def write_plot(result: dict[str, object], output_dir: Path) -> list[Path]:
    """Write separate analysis figures instead of one combined subplot image."""

    curve = result["motion_curve"]
    times = np.asarray([p["time_s"] for p in curve], dtype=float)
    amplitude = np.asarray([p["normalized_amplitude"] for p in curve], dtype=float)
    intensity = np.asarray([p["intensity"] for p in curve], dtype=float)
    cycle_hz = np.asarray([p["cycle_hz"] for p in curve], dtype=float)
    length_mm = np.asarray([p["length_mm"] for p in curve], dtype=float)
    jitter_mm = np.asarray([p["jitter_mm"] for p in curve], dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_line_figure(
            output_dir / "01_normalized_amplitude.png",
            times,
            [(amplitude, "normalized amplitude", "#2563eb")],
            title="Normalized Audio Amplitude",
            ylabel="amplitude",
            ylim=(-0.05, 1.05),
        ),
        write_line_figure(
            output_dir / "02_autd_intensity.png",
            times,
            [(intensity, "AUTD intensity", "#dc2626")],
            title="Mapped AUTD Intensity",
            ylabel="intensity",
            ylim=(-5, 260),
        ),
        write_line_figure(
            output_dir / "03_cycle_hz.png",
            times,
            [(cycle_hz, "cycle_hz", "#16a34a")],
            title="Estimated Back-and-Forth Cycle Speed",
            ylabel="cycle Hz",
        ),
        write_line_figure(
            output_dir / "04_stroke_length_mm.png",
            times,
            [(length_mm, "length_mm", "#9333ea")],
            title="Mapped Focus Stroke Length",
            ylabel="length (mm)",
        ),
        write_line_figure(
            output_dir / "05_jitter_mm.png",
            times,
            [(jitter_mm, "jitter_mm", "#ea580c")],
            title="Mapped Roughness Jitter",
            ylabel="jitter (mm)",
        ),
    ]


def main() -> None:
    args = parse_args()
    result = analyze_friction(args)
    output_path = Path(args.output) if args.output else default_output_path(args.wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Friction analysis written to:", output_path)
    summary = result["summary"]
    print(
        "Summary:",
        f"duration={result['duration_s']:.3f}s",
        f"points={len(result['motion_curve'])}",
        f"intensity_avg={summary['intensity_mean']:.1f}",
        f"intensity_max={summary['intensity_max']}",
        f"cycle_hz={summary['cycle_hz_min']:.2f}-{summary['cycle_hz_max']:.2f}",
    )
    if args.plot:
        plot_dir = Path(args.plot_output) if args.plot_output else clip_figure_dir_from_json(output_path)
        plot_paths = write_plot(result, plot_dir)
        print("Friction analysis plots written to:", plot_dir)
        for path in plot_paths:
            print("  ", path.name)


if __name__ == "__main__":
    main()
