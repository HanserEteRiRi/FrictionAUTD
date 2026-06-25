import time
import os
import numpy as np
from pyautd3 import (
    AUTD3,
    Silencer,
    Controller,
    EulerAngles,
    rad,
    Hz,
    FixedCompletionTime,
    FixedUpdateRate,
    FociSTM,
    ForceFan,
    FocusOption,
    SineOption,
    SquareOption,
)
try:
    from pyautd3 import EmitIntensity
except ImportError:  # EmitIntensity was renamed to Intensity in pyautd3 >= 35.
    from pyautd3 import Intensity as EmitIntensity
from pyautd3.gain import Focus, Null

# from pyautd3_link_soem import SOEM, SOEMOption, Status
try:
    from pyautd3.link.twincat import TwinCAT, RemoteTwinCAT
except ImportError:  # RemoteTwinCAT was removed in pyautd3 >= 38 and is unused here.
    from pyautd3.link.twincat import TwinCAT

    RemoteTwinCAT = TwinCAT
from pyautd3.utils import Duration
from pyautd3.modulation import Sine, Static, Square
from pyautd3 import Group, Device
from dataclasses import dataclass
try:
    from pyautd3.link.simulator import Simulator
except ModuleNotFoundError:  # pyautd3 >= 38 moved the simulator to a separate package.
    Simulator = None
import config
import math


# def err_handler(slave: int, status: Status) -> None:
#     print(f"slave [{slave}]: {status}")
#     if status == Status.Lost():
#         os._exit(-1)


class AUTDManager:
    def __init__(self, link="TwinCAT", enable_silencer=False):
        self.n_device = config.N_DEVICE
        w, h, l = AUTD3.DEVICE_WIDTH, AUTD3.DEVICE_HEIGHT, 5.08

        # dis, dis1, dis2, dis01, dis02 = 33, 35, 500, 33, 36  # distance of 2 sides
        dis, dis1, dis2, dis01, dis02 = 33, 29.5, 37.5, 33, 36  # distance of 2 sides
        x0, x1, x2 = -3 * w / 2, -w / 2, w / 2
        y0, y1, y2 = -h / 2, h / 2 + dis1, h / 2 + dis1 + h / np.sqrt(2) - l
        y00, y01 = -h / 2 - dis - h, -h / 2 - dis - h - dis01 - (h - l) / np.sqrt(2)
        y02 = y01 + l
        z00 = 4.1
        z0, z1, z01, z02 = 0, dis2 + h / np.sqrt(2), (h - l) / np.sqrt(2), h / np.sqrt(2) + dis02 + h
        print(z1, y2)
        theta0, theta1, theta2 = 0 * rad, np.pi / 4 * rad, np.pi / 2 * rad
        theta01, theta02 = -np.pi / 4 * rad, -np.pi / 2 * rad
        geometry = [
            AUTD3([x0, y0, z0]),
            AUTD3([x1, y0, z0]),
            AUTD3([x2, y0, z0]),
            AUTD3([x0, y1, z00], EulerAngles.XYZ(theta1, theta0, theta0)),
            AUTD3([x1, y1, z00], EulerAngles.XYZ(theta1, theta0, theta0)),
            AUTD3([x2, y1, z00], EulerAngles.XYZ(theta1, theta0, theta0)),
            AUTD3([x0, y2, z1], EulerAngles.XYZ(theta2, theta0, theta0)),
            AUTD3([x1, y2, z1], EulerAngles.XYZ(theta2, theta0, theta0)),
            AUTD3([x2, y2, z1], EulerAngles.XYZ(theta2, theta0, theta0)),
            AUTD3([x0, y00, z0]),
            AUTD3([x1, y00, z0]),
            AUTD3([x2, y00, z0]),
            AUTD3([x0, y01, z01], EulerAngles.XYZ(theta01, theta0, theta0)),
            AUTD3([x1, y01, z01], EulerAngles.XYZ(theta01, theta0, theta0)),
            AUTD3([x2, y01, z01], EulerAngles.XYZ(theta01, theta0, theta0)),
            AUTD3([x0, y02, z02], EulerAngles.XYZ(theta02, theta0, theta0)),
            AUTD3([x1, y02, z02], EulerAngles.XYZ(theta02, theta0, theta0)),
            AUTD3([x2, y02, z02], EulerAngles.XYZ(theta02, theta0, theta0)),
        ]
        if link == config.LinkType.TC:
            self.autd: Controller = Controller.open(geometry, TwinCAT())
            print(self.autd.center)
        elif link == config.LinkType.SIM:
            if Simulator is None:
                raise RuntimeError("Simulator link is not installed for this pyautd3 version.")
            self.autd: Controller = Controller[Simulator].open(
                geometry,
                Simulator("127.0.0.1:8080"),
            )
            print("specified simulator")
        else:
            print("------no link specified for autd----------")

        if enable_silencer:
            silencer = Silencer(
                config=FixedCompletionTime(
                    intensity=Duration.from_micros(25 * 2),
                    phase=Duration.from_micros(25 * 3),
                )
            )
            self.autd.send(silencer)
        else:
            self.autd.send(Silencer.disable())

    def key_map(self, dev: Device) -> str | None:
        if 0 <= dev.idx() < self.n_device:
            half = self.n_device // 2
            return "device_1" if dev.idx() < half else "device_2"
        return None

    def perform_irradiate(self, g, m):
        self.autd.send((g, m))

    def perform_double_irradiate(
        self,
        g1=Focus(np.array([0, 0, 200]), option=FocusOption(intensity=EmitIntensity(255))),
        g2=Focus(np.array([0, 50, 200]), option=FocusOption(intensity=EmitIntensity(255))),
        # g2=Null(),
        m=Static(),
    ):
        g = Group(
            key_map=self.key_map,
            data_map={"device_1": g1, "device_2": g2},
        )
        self.autd.send((g, m))

    def stop(self):
        self.autd.send(Null())

    def close(self):
        self.autd.close()

    def fan(self, enable=True):
        self.autd.send(ForceFan(lambda _: enable))


def main():
    import config
    from sleep_conditions import SleepCondition, HoloConfig, STMConfig, create_sleep_gain

    AUTDExp = AUTDManager(link=config.LINK, enable_silencer=True)
    AUTDExp.autd.send(Silencer())

    for _ in range(10):
        holo_cfg = HoloConfig(div_x=1, div_y=1, length_x=0, length_y=0)
        stm_pos_list_right = [(-200.0 * i / 100 + 100, 100, 420) + config.DEVICE_CENTER_RIGHT for i in range(100)]
        stm_pos_list_left = [(-200.0 * i / 100 + 100, -100, 420) + config.DEVICE_CENTER_LEFT for i in range(100)]

        g1 = Focus(np.array([0, 50, 231] + config.DEVICE_CENTER_RIGHT), option=FocusOption(intensity=EmitIntensity(255)))
        g2 = Focus(np.array([0, -50, 239] + config.DEVICE_CENTER_LEFT), option=FocusOption(intensity=EmitIntensity(255)))
        m = Static()  # Sine(30 * Hz, SineOption())
        # g1 = create_sleep_gain(
        #     holo_config=holo_cfg,
        #     stm_config=STMConfig(positions=stm_pos_list_right, frequency=config.FREQ_STM, z_offset=config.Z_OFFSET),
        # )
        # g2 = create_sleep_gain(
        #     holo_config=holo_cfg,
        #     stm_config=STMConfig(positions=stm_pos_list_left, frequency=config.FREQ_STM, z_offset=config.Z_OFFSET),
        # )
        AUTDExp.perform_double_irradiate(g1, g2, m)
        input()
        AUTDExp.stop()
        time.sleep(1)  # 休憩時間

    AUTDExp.close()


if __name__ == "__main__":
    # main_simulator()
    main()
