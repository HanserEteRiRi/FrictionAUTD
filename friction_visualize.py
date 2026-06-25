"""Visualize friction analysis results as separate per-feature figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate intuitive plots from a friction_motion.json file.",
    )
    parser.add_argument("--json", required=True, help="friction_motion.json path.")
    parser.add_argument("--output-dir", default="", help="Output directory for per-feature PNG files.")
    parser.add_argument(
        "--output",
        default="",
        help="Backward-compatible alias for --output-dir. If a PNG file path is passed, its parent is used.",
    )
    return parser.parse_args()


def moving_focus_position(times: np.ndarray, cycle_hz: np.ndarray, length_mm: np.ndarray) -> np.ndarray:
    """Integrate cycle frequency into an eased left-right-left focus position."""

    if len(times) == 0:
        return np.asarray([], dtype=float)
    phase = np.zeros_like(times, dtype=float)
    for i in range(1, len(times)):
        dt = max(0.0, times[i] - times[i - 1])
        phase[i] = phase[i - 1] + 2.0 * math.pi * cycle_hz[i] * dt
    return -0.5 * length_mm * np.cos(phase)


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def clip_name_from_json(json_path: Path) -> str:
    stem = json_path.stem
    if stem.endswith("_friction_motion"):
        stem = stem[: -len("_friction_motion")]
    return stem


def default_output_dir(json_path: Path) -> Path:
    return json_path.parent / "figures" / clip_name_from_json(json_path)


def prepare_output_dir(json_path: Path, output_arg: str, output_dir_arg: str) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    if output_arg:
        output_path = Path(output_arg)
        return output_path.parent if output_path.suffix.lower() == ".png" else output_path
    return default_output_dir(json_path)


def save_single_axis_figure(
    output_path: Path,
    times: np.ndarray,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    lines: list[tuple[np.ndarray, str, str, float]] | None = None,
    fills: list[tuple[np.ndarray, str, float]] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.4))
    if fills:
        for values, color, alpha in fills:
            ax.fill_between(times, values, color=color, alpha=alpha)
    if lines:
        for values, label, color, width in lines:
            ax.plot(times, values, color=color, linewidth=width, label=label)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (s)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if lines:
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.24)
    ax.set_xlim(times[0], times[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize(json_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    data = json.loads(json_path.read_text(encoding="utf-8"))
    curve = data["motion_curve"]
    if not curve:
        raise SystemExit(f"No motion_curve in {json_path}")

    times = np.asarray([p["time_s"] for p in curve], dtype=float)
    amplitude = np.asarray([p["normalized_amplitude"] for p in curve], dtype=float)
    onset = np.asarray([p["onset_normalized"] for p in curve], dtype=float)
    intensity = np.asarray([p["intensity"] for p in curve], dtype=float)
    cycle_hz = np.asarray([p["cycle_hz"] for p in curve], dtype=float)
    length_mm = np.asarray([p["length_mm"] for p in curve], dtype=float)
    roughness = np.asarray([p["roughness"] for p in curve], dtype=float)
    jitter_mm = np.asarray([p["jitter_mm"] for p in curve], dtype=float)
    centroid = np.asarray([p["spectral_centroid_hz"] for p in curve], dtype=float)
    focus_x = moving_focus_position(times, cycle_hz, length_mm)
    focus_speed = np.gradient(focus_x, times, edge_order=1) if len(times) > 1 else np.zeros_like(times)

    summary = data.get("summary", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(
        save_single_axis_figure(
            output_dir / "06_audio_amplitude_and_onset.png",
            times,
            title="Audio Amplitude and Friction Onset Pulses",
            ylabel="normalized value",
            ylim=(-0.05, 1.05),
            fills=[(amplitude, "#93c5fd", 0.62)],
            lines=[
                (amplitude, "normalized amplitude", "#2563eb", 1.2),
                (onset, "friction pulses / onset", "#f97316", 1.4),
            ],
        )
    )
    written.append(
        save_single_axis_figure(
            output_dir / "07_visual_autd_intensity.png",
            times,
            title=f"AUTD Intensity Mapping, avg/max={summary.get('intensity_mean', 0):.1f}/{summary.get('intensity_max', 0)}",
            ylabel="intensity",
            ylim=(0, 260),
            fills=[(intensity, "#fecaca", 0.35)],
            lines=[(intensity, "AUTD intensity", "#dc2626", 1.7)],
        )
    )
    written.append(
        save_single_axis_figure(
            output_dir / "08_cycle_speed_hz.png",
            times,
            title=f"Back-and-Forth Cycle Speed, {summary.get('cycle_hz_min', 0):.2f}-{summary.get('cycle_hz_max', 0):.2f} Hz",
            ylabel="cycle Hz",
            lines=[(cycle_hz, "cycle speed", "#16a34a", 1.7)],
        )
    )
    written.append(
        save_single_axis_figure(
            output_dir / "09_stroke_length_mm.png",
            times,
            title=f"Focus Stroke Length, avg={summary.get('length_mm_mean', 0):.1f} mm",
            ylabel="length (mm)",
            lines=[(length_mm, "stroke length", "#7c3aed", 1.6)],
        )
    )
    written.append(
        save_single_axis_figure(
            output_dir / "10_roughness_and_spectral_centroid.png",
            times,
            title="Spectral Roughness and Centroid",
            ylabel="normalized value",
            ylim=(-0.05, 1.05),
            fills=[(normalize(centroid), "#99f6e4", 0.35)],
            lines=[
                (roughness, "roughness", "#0f766e", 1.5),
                (normalize(centroid), "spectral centroid (normalized)", "#14b8a6", 1.1),
            ],
        )
    )
    written.append(
        save_single_axis_figure(
            output_dir / "11_path_jitter_mm.png",
            times,
            title=f"Focus Path Jitter, avg={summary.get('jitter_mm_mean', 0):.2f} mm",
            ylabel="jitter (mm)",
            lines=[(jitter_mm, "path jitter", "#ea580c", 1.4)],
        )
    )

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.fill_between(times, -0.5 * length_mm, 0.5 * length_mm, color="#e5e7eb", alpha=0.6, label="stroke range")
    ax.plot(times, focus_x, color="#111827", linewidth=1.6, label="estimated focus x")
    ax.set_title("Estimated Focus X Motion", fontsize=14, fontweight="bold")
    ax.set_ylabel("focus x (mm)")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.24)
    ax.set_xlim(times[0], times[-1])
    ax.legend(loc="upper right")
    fig.tight_layout()
    focus_path = output_dir / "12_estimated_focus_x_motion.png"
    fig.savefig(focus_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(focus_path)

    written.append(
        save_single_axis_figure(
            output_dir / "13_estimated_focus_speed.png",
            times,
            title="Estimated Focus Speed",
            ylabel="speed (mm/s)",
            lines=[(focus_speed, "focus speed", "#64748b", 1.2)],
        )
    )
    return written


def main() -> None:
    args = parse_args()
    json_path = Path(args.json)
    output_dir = prepare_output_dir(json_path, args.output, args.output_dir)
    paths = visualize(json_path, output_dir)
    print("Friction feature visualizations written to:", output_dir)
    for path in paths:
        print("  ", path.name)


if __name__ == "__main__":
    main()
