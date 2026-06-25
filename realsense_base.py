import pyrealsense2 as rs
import cv2
import config as CFG


class DepthCameraBase:
    def __init__(self, serial_number: str = None):
        self.is_running = False
        self._init_realsense_config(serial_number)

    def _init_realsense_config(self, serial_number: str = None):
        self.pipeline = rs.pipeline()
        config = rs.config()
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        self.depth_scale = None

        if len(devices) == 0:
            raise RuntimeError("No RealSense devices connected.")
        if len(devices) == 1:
            print("Only one RealSense device connected.")
        if len(devices) >= 2 and serial_number is None:
            raise RuntimeError(f"More than 2 RealSense devices are connected. Please specify serial number. Serial numbers: {serials}")

        if serial_number:
            if serial_number not in serials:
                raise RuntimeError(f"Specified serial number {serial_number} not found. Available: {serials}")
            config.enable_device(serial_number)
            print(f"Using specified RealSense device with serial number: {serial_number}")
        else:
            serial_number = serials[0]
            config.enable_device(serial_number)
            print(f"No serial specified. Using default RealSense device: {serial_number}")

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        sensors = device.query_sensors()
        self.color_sensor = None

        for s in sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                self.color_sensor = s
                break

        if self.color_sensor is None:
            raise RuntimeError("The device does not have an RGB camera.")

        # self.color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        self.color_sensor.set_option(rs.option.exposure, CFG.EXPOSURE)
        self.color_sensor.set_option(rs.option.gain, CFG.GAIN)
        self.color_sensor.set_option(rs.option.white_balance, CFG.WHITE_BALANCE)

        device_product_line = device.get_info(rs.camera_info.product_line)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == "L500":
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        self.is_running = True
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        if not self.is_running:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(500)
        except RuntimeError:
            return None, None

        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        return color_frame, depth_frame

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass

    @staticmethod
    def get_real_pos_from_images(depth_frame, color_frame, px, py):
        depth_m = depth_frame.get_distance(px, py)

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        X_m, Y_m, Z_m = rs.rs2_deproject_pixel_to_point(color_intrinsics, [px, py], depth_m)

        X_mm = round(X_m * 1000.0, 1)
        Y_mm = round(Y_m * 1000.0, 1)
        Z_mm = round(Z_m * 1000.0, 1)

        return X_mm, Y_mm, Z_mm


def sample_calibrate_color():
    import numpy as np
    import config

    cam = DepthCameraBase()

    clicked_point = [None, None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = x
            clicked_point[1] = y

    cv2.namedWindow("Color")
    cv2.setMouseCallback("Color", mouse_callback)
    cv2.namedWindow("Depth")

    try:
        while True:
            color_frame, depth_frame = cam.get_frames()

            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_gray = cv2.convertScaleAbs(depth_image, alpha=0.03)

            if clicked_point[0] is not None:
                x, y = clicked_point

                X_mm, Y_mm, Z_mm = cam.get_real_pos_from_images(depth_frame, color_frame, x, y)

                # print(f"Pixel ({x}, {y}) -> X:{X_mm}mm Y:{Y_mm}mm Z:{Z_mm}mm")

                cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(depth_gray, (x, y), 5, (255,), -1)

                text = f"{Z_mm}mm"

                cv2.putText(
                    color_image,
                    text,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                cv2.putText(
                    depth_gray,
                    text,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,),
                    1,
                )

            # 現在の設定表示
            exposure = config.EXPOSURE
            white_balance = config.WHITE_BALANCE
            gain = config.GAIN

            # gain = cam.color_sensor.get_option(rs.option.gain)
            # wb = cam.color_sensor.get_option(rs.option.white_balance)

            info = f"EXP:{int(exposure)} GAIN:{int(gain)} WB:{int(white_balance)}"
            print(info)
            cv2.putText(color_image, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", depth_gray)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == ord("1"):
                exposure = exposure - 100
                cam.color_sensor.set_option(rs.option.exposure, exposure)
                print(exposure)

            elif key == ord("2"):
                exposure = min(exposure + 100, 5000)
                cam.color_sensor.set_option(rs.option.exposure, exposure)

            elif key == ord("3"):
                gain = gain - 10
                cam.color_sensor.set_option(rs.option.gain, gain)

            elif key == ord("4"):
                gain = min(gain + 10, 128)
                cam.color_sensor.set_option(rs.option.gain, gain)

            elif key == ord("5"):
                white_balance = white_balance - 200
                cam.color_sensor.set_option(rs.option.white_balance, white_balance)

            elif key == ord("6"):
                white_balance = min(white_balance + 200, 6000)
                cam.color_sensor.set_option(rs.option.white_balance, white_balance)

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np

    cam = DepthCameraBase("246422070754")

    clicked_point = [None, None]

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point[0] = x
            clicked_point[1] = y

    cv2.namedWindow("Color")
    cv2.setMouseCallback("Color", mouse_callback)
    cv2.namedWindow("Depth")

    try:
        while True:
            color_frame, depth_frame = cam.get_frames()
            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_gray = cv2.convertScaleAbs(depth_image, alpha=0.03)
            if clicked_point[0] is not None:
                x, y = clicked_point
                X_mm, Y_mm, Z_mm = cam.get_real_pos_from_images(depth_frame, color_frame, x, y)
                print(f"Pixel ({x}, {y}) -> X:{X_mm}mm Y:{Y_mm}mm Z:{Z_mm}mm")

                cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(depth_gray, (x, y), 5, (255,), -1)

                text = f"{Z_mm}mm"
                cv2.putText(color_image, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(depth_gray, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)

            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", depth_gray)

            if cv2.waitKey(1) == 27:
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
