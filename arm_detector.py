import cv2
import mediapipe as mp
import numpy as np
import config


class ArmDetector:
    def __init__(self, window_config: config.WindowConfig, which_hand=None):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.which_hand = which_hand
        self.window_config = window_config

    def _hand_region_mask(self, rgb_image, depth_image):
        if config.SKIN_DETECTION_METHOD == config.SkinDetectionMethod.KOLKUR:
            color_mask = self._apply_advanced_skin_detection(rgb_image)
        elif config.SKIN_DETECTION_METHOD == config.SkinDetectionMethod.YCRCB:
            color_mask = self._apply_easy_skin_detection(rgb_image)
        depth_condition = np.logical_and(depth_image >= config.HAND_DETECTION_DEPTH_MIN_MM, depth_image <= config.HAND_DETECTION_DEPTH_MAX_MM)
        depth_mask = (depth_condition * 255).astype(np.uint8)
        if config.SKIN_DETECTION_METHOD == config.SkinDetectionMethod.COMBINE:
            mask = cv2.bitwise_and(color_mask, depth_mask)
        else:
            mask = color_mask

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask)

        largest_contour = max(contours, key=cv2.contourArea)
        result_region_mask = np.zeros_like(mask)
        cv2.drawContours(result_region_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        img_YCrCb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)
        h, w = img_YCrCb.shape[:2]
        center_y = h // 2
        center_x = w // 2
        Y, Cr, Cb = img_YCrCb[center_y, center_x]
        # print(f"Center YCrCb: Y={Y}, Cr={Cr}, Cb={Cb}")

        if self.window_config.show_input_rgb:
            cv2.imshow("Input RGB {}".format(self.which_hand), rgb_image)
        if self.window_config.show_input_depth:
            d_min = config.HAND_DETECTION_DEPTH_MIN_MM
            d_max = config.HAND_DETECTION_DEPTH_MAX_MM

            depth_clipped = np.clip(depth_image, d_min, d_max)
            depth_norm = ((depth_clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            masked_depth = cv2.bitwise_and(depth_norm, depth_norm, mask=result_region_mask)
            cv2.imshow("Input Depth {}".format(self.which_hand), depth_image)
            cv2.imshow("Masked Depth Image{}".format(self.which_hand), masked_depth)

        if self.window_config.show_depth_mask:
            d_min, d_max = config.HAND_DETECTION_DEPTH_MIN_MM, config.HAND_DETECTION_DEPTH_MAX_MM
            depth_norm = np.clip((1 - (depth_image - d_min) / (d_max - d_min)) * 255, 0, 255).astype(np.uint8)
            depth_gray_v = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            depth_jet = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

            combined_vis = depth_gray_v.copy()
            combined_vis[depth_mask == 255] = depth_jet[depth_mask == 255]
            cv2.imshow(f"Visualized Depth {self.which_hand}", combined_vis)

        if self.window_config.show_result_region_mask:
            rgb_copy = rgb_image.copy()
            cv2.drawContours(rgb_copy, [largest_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Result Region RGB {}".format(self.which_hand), cv2.bitwise_and(rgb_image, rgb_image, mask=result_region_mask))
            cv2.imshow("Result Region Mask {}".format(self.which_hand), result_region_mask)
            cv2.waitKey(1)

        return result_region_mask

    def _apply_easy_skin_detection(self, rgb_image):
        img_YCrCb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)
        lower = config.SKIN_DETECTION_YCRCB_LOWER
        upper = config.SKIN_DETECTION_YCRCB_UPPER
        color_mask = cv2.inRange(img_YCrCb, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        return color_mask

    def _apply_advanced_skin_detection(self, rgb_image):
        """
        Human Skin Detection Using RGB, HSV and YCbCr Color Models
        parameterは若干違う
        """
        ALPHA = 1.2
        BETA = 30
        R_MIN, G_MIN, B_MIN = 95, 40, 20
        RG_DIFF_MIN = 15
        H_MIN, H_MAX = 0, 50
        S_MIN, S_MAX = 10, 173  # 58,173
        Y_MIN = 5  # 80

        bgr = cv2.convertScaleAbs(rgb_image, alpha=ALPHA, beta=BETA)

        # 1. RGB 条件の計算
        B = bgr[:, :, 0].astype(np.int32)
        G = bgr[:, :, 1].astype(np.int32)
        R = bgr[:, :, 2].astype(np.int32)

        rgb_cond = (R > R_MIN) & (G > G_MIN) & (B > B_MIN) & (R > G) & (R > B) & (np.abs(R - G) > RG_DIFF_MIN)

        # 2. HSV 条件の計算
        bgr_hsv = cv2.GaussianBlur(bgr, (7, 7), 0)
        hsv = cv2.cvtColor(bgr_hsv, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].astype(np.int32)
        S = hsv[:, :, 1].astype(np.int32)
        hsv_cond = (H >= H_MIN) & (H <= H_MAX) & (S >= S_MIN) & (S <= S_MAX)

        # 3. YCbCr 条件の計算
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)
        Cr = ycrcb[:, :, 1].astype(np.float32)
        Cb = ycrcb[:, :, 2].astype(np.float32)

        ycrcb_cond = (
            (Cr > 135)
            & (Cb > 85)
            & (Y > Y_MIN)
            & (Cr <= (1.5862 * Cb) + 20)
            & (Cr >= (0.3448 * Cb) + 76.2069)
            & (Cr >= (-4.5652 * Cb) + 234.5652)
            & (Cr <= (-1.15 * Cb) + 301.75)
            & (Cr <= (-2.2857 * Cb) + 432.85)
        )
        cond_1 = rgb_cond & hsv_cond
        cond_2 = rgb_cond & ycrcb_cond
        final_mask = cond_1 | cond_2

        # 5. マスク化とノイズ除去
        combined_skin = (final_mask * 255).astype(np.uint8)

        # ゴマ塩ノイズや小さな穴を埋める処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_skin = cv2.morphologyEx(combined_skin, cv2.MORPH_OPEN, kernel)
        combined_skin = cv2.morphologyEx(combined_skin, cv2.MORPH_CLOSE, kernel)

        def _plot():
            rgb_vis = rgb_cond.astype("uint8") * 255
            hsv_vis = hsv_cond.astype("uint8") * 255
            ycrcb_vis = ycrcb_cond.astype("uint8") * 255
            combined_vis = combined_skin.copy()
            cv2.putText(rgb_vis, "RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.putText(hsv_vis, "HSV", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.putText(ycrcb_vis, "YCbCr", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.putText(combined_vis, "Combined", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            top = cv2.hconcat([rgb_vis, hsv_vis])
            bottom = cv2.hconcat([ycrcb_vis, combined_vis])
            grid = cv2.vconcat([top, bottom])
            cv2.imshow("KOLKUR Debug {}".format(self.which_hand), grid)

        if config.WindowConfig.show_skin_mask:
            _plot()
        return combined_skin

    def _calc_arm_direction(self, mask, wrist_pos, line_length=200):
        if wrist_pos is not None:
            wrist_x = int(wrist_pos[0])
            mask[:, :wrist_x] = 0
        mask_255 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return (np.array([0, 0]), np.array([100, 100]), np.array([-1, 0]))
        contour = max(contours, key=cv2.contourArea)
        points = contour.reshape(-1, 2)
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        direction = np.array([vx, vy]).flatten()
        direction = direction / np.linalg.norm(direction)
        pt_center = np.array([x0, y0]).flatten()
        half_vec = direction * (line_length / 2)
        pt1 = (pt_center - half_vec).astype(int)
        pt2 = (pt_center + half_vec).astype(int)

        if self.window_config.show_arm_direction:
            distance = np.linalg.norm(pt2 - pt1)
            cv2.line(mask, tuple(pt1), tuple(pt2), (0, 0, 255), 2)
            cv2.circle(mask, tuple(pt1), 4, (255, 0, 0), -1)
            cv2.circle(mask, tuple(pt2), 4, (255, 0, 0), -1)
            cv2.putText(mask, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Arm Direction {}".format(self.which_hand), mask)
            cv2.waitKey(1)
        return pt1, pt2, direction

    def _get_wrist_from_mediapipe(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            h, w = image.shape[:2]
            wrist = results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.WRIST]
            x = int(wrist.x * w)
            y = int(wrist.y * h)
            return np.array([x, y])
        return None

    def process_image(self, rgb_image, depth_image):
        """
        肌色マスク，深度マスクを使って手の領域を作る．また，mediapipeで腕付け根位置を取得する．
        最終的にはSTM開始位置と終了位置（未スケール）の位置を返す．

        """
        # time
        hand_mask = self._hand_region_mask(rgb_image, depth_image)

        wrist = self._get_wrist_from_mediapipe(rgb_image)
        orig_pt1, orig_pt2, direction = self._calc_arm_direction(hand_mask, wrist)
        if wrist is not None:
            pt2 = wrist
            if self.which_hand == "left":
                pt1 = (pt2 + orig_pt2 - orig_pt1).astype(int)  # pt1 = (pt2 - orig_pt2 + orig_pt1).astype(int)
            elif self.which_hand == "right":
                pt1 = (pt2 + orig_pt2 - orig_pt1).astype(int)

        else:
            # print("wrist is none")
            pt1, pt2 = orig_pt1, orig_pt2
        # clipping
        h, w = depth_image.shape[:2]
        pt1[0] = np.clip(pt1[0], 0, w - 1)
        pt1[1] = np.clip(pt1[1], 0, h - 1)
        pt2[0] = np.clip(pt2[0], 0, w - 1)
        pt2[1] = np.clip(pt2[1], 0, h - 1)
        pt_start = pt2
        pt_last = pt1

        if self.window_config.show_stm_start_end:
            img = rgb_image.copy()
            if wrist is not None:
                cv2.circle(img, tuple(pt1), 4, (255, 0, 0), -1)  # 青
                cv2.circle(img, tuple(pt2), 4, (0, 255, 0), -1)  # 緑
            cv2.imshow("STM Start and End {}".format(self.which_hand), img)
            cv2.waitKey(1)

        return hand_mask.astype(np.uint8), pt_start, pt_last, direction


if __name__ == "__main__":
    from realsense_base import DepthCameraBase
    import time

    win_cfg = config.WindowConfig(
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
    cam = DepthCameraBase(serial_number="246422070754")
    detector = ArmDetector(window_config=win_cfg, which_hand="right")

    print("Starting Arm Detection... Press 'ESC' to exit.")

    try:
        while True:
            color_frame, depth_frame = cam.get_frames()
            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_gray = cv2.convertScaleAbs(depth_image, alpha=0.03)

            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", depth_gray)
            mask, pt_start, pt_end, direction = detector.process_image(color_image, depth_image)
            if pt_start is not None:
                x, y = pt_start
                X, Y, Z = cam.get_real_pos_from_images(depth_frame, color_frame, x, y)
            if cv2.waitKey(1) == 27:  # ESC key
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
