"""Friction-like motion on UltraSleep's 18-device AUTD geometry.

This standalone script is intentionally small and independent from AudioAUTD.
It sends synchronized right/left focal paths through UltraSleep's unchanged
9+9 device grouping. The trajectory uses sinusoidal easing, so each focus slows
down at both endpoints and accelerates away from them like a rubbing stroke.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import config
from autd_manager import AUTDManager
from pyautd3 import Focus, FocusOption, GainSTM, GainSTMMode, GainSTMOption, SamplingConfig, Static
try:
    from pyautd3 import EmitIntensity
except ImportError:  # EmitIntensity was renamed to Intensity in pyautd3 >= 35.
    from pyautd3 import Intensity as EmitIntensity


ULTRASOUND_FREQ_HZ = 40_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a maximum-intensity eased back-and-forth focus for friction/rubbing tests.",
    )
    parser.add_argument("--height-mm", type=float, default=100.0, help="Focus height above the AUTD plane.")
    parser.add_argument("--base-z-mm", type=float, default=250.0, help="UltraSleep global z before adding height-mm.")
    parser.add_argument("--length-mm", type=float, default=60.0, help="Back-and-forth stroke length.")
    parser.add_argument(
        "--cycle-hz",
        type=float,
        default=2.0 / 3.0,
        help="Full left->right->left cycles per second.",
    )
    parser.add_argument("--frames", type=int, default=96, help="STM frames per full cycle.")
    parser.add_argument("--intensity", type=int, default=255, help="AUTD intensity, 0..255.")
    return parser.parse_args()


def compatible_cycle_hz(frame_count: int, requested_cycle_hz: float) -> float:
    """Adjust the STM sampling frequency so it divides the 40 kHz ultrasound."""

    target_sampling_hz = max(1.0, frame_count * requested_cycle_hz)
    valid_sampling_hz = [
        hz for hz in range(1, ULTRASOUND_FREQ_HZ + 1) if ULTRASOUND_FREQ_HZ % hz == 0
    ]
    selected_sampling_hz = min(valid_sampling_hz, key=lambda hz: abs(hz - target_sampling_hz))
    return selected_sampling_hz / frame_count


def eased_back_and_forth_points(center: np.ndarray, length_mm: float, frames: int) -> list[np.ndarray]:
    """Return a seamless sinusoidal back-and-forth path.

    x = -A * cos(theta)

    theta = 0 starts at the left endpoint. The derivative is zero at both
    endpoints, so the focus decelerates before reversal and accelerates after it.
    """

    if frames < 8:
        raise ValueError("--frames should be at least 8.")
    half_length = length_mm / 2.0
    points: list[np.ndarray] = []
    for i in range(frames):
        theta = 2.0 * math.pi * i / frames
        x = -half_length * math.cos(theta)
        points.append(center + np.array([x, 0.0, 0.0]))
    return points


def focus(point: np.ndarray, intensity_value: int) -> Focus:
    return Focus(
        point,
        option=FocusOption(intensity=EmitIntensity(min(255, max(0, int(intensity_value))))),
    )


def main() -> None:
    args = parse_args()
    z = args.base_z_mm + args.height_mm
    center_right = np.array([0.0, 100.0, z]) + config.DEVICE_CENTER_RIGHT
    center_left = np.array([0.0, -100.0, z]) + config.DEVICE_CENTER_LEFT
    path_right = eased_back_and_forth_points(center_right, args.length_mm, args.frames)
    path_left = eased_back_and_forth_points(center_left, args.length_mm, args.frames)
    cycle_hz = compatible_cycle_hz(len(path_right), args.cycle_hz)
    sampling_hz = int(round(len(path_right) * cycle_hz))
    sampling = SamplingConfig(ULTRASOUND_FREQ_HZ // sampling_hz)
    option = GainSTMOption(mode=GainSTMMode.PhaseIntensityFull)
    gain_right = GainSTM(
        gains=[focus(point, args.intensity) for point in path_right],
        config=sampling,
        option=option,
    )
    gain_left = GainSTM(
        gains=[focus(point, args.intensity) for point in path_left],
        config=sampling,
        option=option,
    )

    autd = AUTDManager(link=config.LINK)
    try:
        print(
            "Friction focus on UltraSleep 18-device geometry:",
            f"height={args.height_mm:.1f}mm",
            f"global_z={z:.1f}mm",
            f"length={args.length_mm:.1f}mm",
            f"intensity={args.intensity}{' (MAX)' if int(args.intensity) >= 255 else ''}",
            f"frames={len(path_right)}",
            f"cycle_hz={cycle_hz:.6f}",
            f"sampling_hz={sampling_hz}",
        )
        print("Right endpoints:", path_right[0], path_right[len(path_right) // 2])
        print("Left endpoints:", path_left[0], path_left[len(path_left) // 2])

        autd.perform_double_irradiate(g1=gain_right, g2=gain_left, m=Static())
        try:
            input("Running friction back-and-forth STM. Press Enter to stop...")
        except KeyboardInterrupt:
            print("\nStopping friction STM...")
    finally:
        autd.stop()
        autd.close()
        print("Friction STM stopped.")


if __name__ == "__main__":
    main()
