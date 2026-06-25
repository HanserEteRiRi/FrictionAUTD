import numpy as np
import cv2
import pyrealsense2 as rs
import config
from realsense_base import DepthCameraBase
from arm_detector import ArmDetector
import threading
import time


def _display_stm_trajectory(color_frame, depth_map, window_name):
    """STMの軌跡をカラー画像上に描画"""
    img = np.asanyarray(color_frame.get_data()).copy()
    z_min, z_max = 100, 300
    z_range = max(z_max - z_min, 1e-5)

    for px, py, z_mm in depth_map:
        ratio = np.clip((z_mm - z_min) / z_range, 0, 1)
        b = int(255 * (1 - ratio))
        r = int(255 * ratio)
        cv2.circle(img, (int(px), int(py)), 5, (b, 0, r), -1)

    cv2.imshow(f"stm trajectory {window_name}", img)


def _calc_stm_trajectory_px(pt_start, pt_last, n_stm):
    """2点間のピクセル軌跡を計算"""
    if pt_start is None or pt_last is None:
        return []

    ts = np.linspace(0, 1, n_stm)
    trajectory = []
    for t in ts:
        pt = (1 - t) * np.array(pt_start) + t * np.array(pt_last)
        trajectory.append(tuple(np.round(pt).astype(int)))
    return trajectory


class RealSenseForSleep:
    def __init__(self, window_config: config.WindowConfig, n_stm=25, realsense_serial_number=None):
        self.depth_camera = DepthCameraBase(serial_number=realsense_serial_number)
        self.window_config = window_config
        self.is_enabled = False
        self.is_running = True
        self.n_stm = n_stm
        self.stm_real_pos_list = [np.zeros(3) for _ in range(4)]  # 初期値
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)

        # ハンドの識別
        self.which_hand = "right" if realsense_serial_number == config.REALSENSE_SERIAL_NUMBER_RIGHT else "left"
        self.arm_detector = ArmDetector(window_config=window_config, which_hand=self.which_hand)

    def get_stm_positions(self):
        return self.stm_real_pos_list

    def run(self):
        try:
            while self.is_running:
                frames = self._acquire_frames()
                if frames is None:
                    continue
                color_f, depth_f, color_img, depth_img = frames
                stm_px_list = self._get_trajectory_pixels(color_img, depth_img, color_f, depth_f)
                if not stm_px_list:
                    self._handle_no_detection(color_f)
                    if self._check_exit():
                        break
                    continue
                real_pos_list, depth_map = self._process_coordinate_conversion(stm_px_list, color_f, depth_f)

                if real_pos_list:
                    self.stm_real_pos_list = real_pos_list

                # print("wrist pos?", real_pos_list[0])
                if self.window_config.show_stm_trajectory:
                    _display_stm_trajectory(color_f, depth_map, self.which_hand)
                if self._check_exit():
                    break
        finally:
            self.stop()

    def _acquire_frames(self):
        color_f, depth_frame = self.depth_camera.get_frames()
        if color_f is None or depth_frame is None:
            return None

        self.is_enabled = True
        color_img = np.asanyarray(color_f.get_data())
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_camera.depth_scale * 1000
        return color_f, depth_frame, color_img, depth_img

    def _get_trajectory_pixels(self, color_img, depth_img, color_f, depth_f):
        mask, pt_start, pt_last_tmp, direction = self.arm_detector.process_image(color_img, depth_img)
        if pt_start is None or pt_last_tmp is None:
            print(f"Arm not detected ({self.which_hand})")
            return []

        pt_last = self._find_last_point_from_distance(pt_start, direction, depth_f, color_f, max_distance_mm=config.LENGTH_STM_MM)
        if pt_last is None:
            print(f"Target length point not found ({self.which_hand})")
            return []

        return _calc_stm_trajectory_px(pt_start, pt_last, self.n_stm)

    def _process_coordinate_conversion(self, px_list, color_f, depth_f):
        real_pos_list = []
        depth_map = []
        w, h = color_f.get_width(), color_f.get_height()

        for px, py in px_list:
            # 境界チェック
            if not (0 <= px < w and 0 <= py < h):
                continue

            try:
                x_mm, y_mm, z_mm = DepthCameraBase.get_real_pos_from_images(depth_f, color_f, int(px), int(py))

                if self.which_hand == "right":
                    x_mm, y_mm = -x_mm, -y_mm  # REALSENSE座標系とAUTD座標系が逆なので
                    x_mm += config.REALSENSE_OFFSET_RIGHT[0]
                    y_mm += config.REALSENSE_OFFSET_RIGHT[1]
                    z_mm += config.REALSENSE_OFFSET_RIGHT[2]

                elif self.which_hand == "left":
                    x_mm, y_mm = -x_mm, -y_mm
                    x_mm += config.REALSENSE_OFFSET_LEFT[0]
                    y_mm += config.REALSENSE_OFFSET_LEFT[1]
                    z_mm += config.REALSENSE_OFFSET_LEFT[2]
                # Depth 0 (欠損値) チェック
                if z_mm == 0:
                    continue
                real_pos_list.append([x_mm, y_mm, z_mm])
                depth_map.append((px, py, z_mm))
            except Exception as e:
                print(f"Error at ({px}, {py}): {e}")
                continue

        return real_pos_list, depth_map

    def _find_last_point_from_distance(self, pt_start, unit_vec, depth_frame, color_frame, max_distance_mm=200):
        """腕に沿って進み、一定距離(mm)離れたピクセル座標を探す"""
        px_pos = np.array(pt_start, dtype=np.float32)
        try:
            start_pos_mm = np.array(DepthCameraBase.get_real_pos_from_images(depth_frame, color_frame, int(px_pos[0]), int(px_pos[1])))
        except:
            return None

        w, h = color_frame.get_width(), color_frame.get_height()
        for _ in range(300):
            px_pos += unit_vec
            px_int = np.round(px_pos).astype(int)
            # 画像外に出たら終了
            if not (0 <= px_int[0] < w and 0 <= px_int[1] < h):
                break
            try:
                cur_pos_mm = np.array(DepthCameraBase.get_real_pos_from_images(depth_frame, color_frame, px_int[0], px_int[1]))
                if cur_pos_mm[2] == 0:
                    continue  # Depthエラーはスキップ
                distance = np.linalg.norm(cur_pos_mm - start_pos_mm)
                if distance >= max_distance_mm:
                    return tuple(px_int)
            except:
                continue
        return None

    def _handle_no_detection(self, color_f):
        """検出失敗時にメッセージ付きで映像を表示"""
        if self.window_config.show_stm_trajectory:
            img = np.asanyarray(color_f.get_data()).copy()
            cv2.putText(img, f"No Detection ({self.which_hand})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(f"stm trajectory {self.which_hand}", img)

    def _check_exit(self):
        key = cv2.waitKey(1)
        should_exit = key & 0xFF == ord("q") or key == 27
        if should_exit:
            self.is_running = False
        return should_exit

    def stop(self):
        self.is_running = False
        self.is_enabled = False
        self.depth_camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import config

    integration = RealSenseForSleep(
        window_config=config.REALSENSE_PLOT_CONFIG_ALLTRUE, n_stm=config.N_STM, realsense_serial_number=config.REALSENSE_SERIAL_NUMBER_LEFT
    )
    integration.run()
