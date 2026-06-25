import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from pyautd3.link.simulator import Simulator


class LinkType(Enum):
    TC = "tc"
    SIM = "sim"


class SkinDetectionMethod(Enum):
    YCRCB = "YCrCb"
    DEPTH_ONLY = "DepthOnly"
    KOLKUR = "Kolkur"
    COMBINE = "Combine"


class ArmSTMCalcurationMethod(Enum):
    WRIST_DEPTH = "WristDepth"
    ARM_DEPTH = "ArmDepth"
    LINEARFIT_DEPTH = "LinearFitDepth"


# RealSense設定
REALSENSE_SERIAL_NUMBER_CENTER = "250122077255"
REALSENSE_SERIAL_NUMBER_RIGHT = "317422071322"
REALSENSE_SERIAL_NUMBER_LEFT = "246422070754"
HAND_DETECTION_DEPTH_MIN_MM = 50
HAND_DETECTION_DEPTH_MAX_MM = 450
SKIN_DETECTION_YCRCB_LOWER = np.array([0, 120, 77])
SKIN_DETECTION_YCRCB_UPPER = np.array([255, 173, 127])
WHITE_BALANCE = 4500  # 5500
EXPOSURE = 200  # 10000
GAIN = 32  # 128

DEVICE_CENTER_RIGHT = np.array([0, 0, 0])
DEVICE_CENTER_LEFT = np.array([0, -151.4 - 33, 0])
REALSENSE_OFFSET_RIGHT = np.array([40.1, 85.6, -5])
REALSENSE_OFFSET_LEFT = np.array([40.1, -85.6, -5])

# 触覚刺激設定
N_DEVICE = 18
FREQ_STM = 0.5
FREQ_UPDATE_STM_POS = 1
LENGTH_STM_MM = 150
Z_OFFSET = 5
N_STM = 100

LINK = LinkType.TC
SKIN_DETECTION_METHOD = SkinDetectionMethod.KOLKUR
ARM_STM_CALCURATION_METHOD = ArmSTMCalcurationMethod.WRIST_DEPTH


@dataclass
class WindowConfig:
    show_input_rgb: bool = False
    show_input_depth: bool = False
    show_depth_mask: bool = False
    show_skin_mask: bool = False
    show_combined_mask: bool = False
    show_result_region_mask: bool = False
    show_arm_direction: bool = False
    show_mediapipe: bool = False
    show_stm_start_end: bool = False
    show_stm_trajectory: bool = False

    @property
    def any_enabled(self) -> bool:
        """いずれかのデバッグウィンドウが有効かどうか"""
        return any(
            [
                self.show_input_rgb,
                self.show_input_depth,
                self.show_depth_mask,
                self.show_skin_mask,
                self.show_arm_direction,
                self.show_mediapipe,
                self.show_stm_start_end,
                self.show_stm_trajectory,
            ]
        )


REALSENSE_PLOT_CONFIG = WindowConfig(
    show_input_rgb=False,
    show_input_depth=False,
    show_depth_mask=True,
    show_skin_mask=True,
    show_result_region_mask=True,
    show_arm_direction=True,
    show_mediapipe=False,
    show_stm_start_end=False,
    show_stm_trajectory=True,
)

REALSENSE_PLOT_CONFIG_ALLTRUE = WindowConfig(
    show_input_rgb=True,
    show_input_depth=True,
    show_depth_mask=True,
    show_skin_mask=True,
    show_result_region_mask=True,
    show_arm_direction=True,
    show_mediapipe=True,
    show_stm_start_end=True,
    show_stm_trajectory=True,
)
