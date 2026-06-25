# FrictionAUTD

这是一个和 `AudioAUTD/` 隔离的摩擦声触觉项目。设备层已经重构为与
`UltraSleep` 完全相同的 18 台 AUTD 几何、TwinCAT 链路和 9+9 分组。
`friction_analyze.py` 的音频分析逻辑没有改动。

当前有两个入口：

```text
1. friction_play.py
   读取分析后的 friction_motion.json，
   控制 AUTD 焦点运动，
   并同步播放原始音频。

2. friction_back_and_forth.py
   不读取音频，只运行一个固定参数的来回运动焦点，
   用于快速确认 AUTD 是否能产生摩擦式触感。
```

固定焦点测试的基础运动是：

```text
在 UltraSleep 工作区域生成左右两个同步焦点，
分别由 0–8 和 9–17 号设备驱动并沿 x 方向来回运动，
并在左右端点自然减速、反向后自然加速。
```

运动轨迹使用：

```text
x = -A * cos(theta)
```

所以焦点在端点处速度为 0，中间速度最大，适合模拟“摩擦 / 抚摸”的周期性来回动作。

## 运行

### 根据分析 JSON 同步播放音频和 AUTD

这是现在推荐使用的控制入口。它会读取 `config.toml`：

```text
friction_motion.json
  -> intensity / cycle_hz / length_mm / height_mm / jitter_mm
  -> 左右同步 GainSTM 焦点运动
  -> UltraSleep AUTDManager（18 台，TwinCAT，9+9 分组）
  -> 第一帧触觉发送成功后立刻播放音频
```

真实设备运行前，需要像 UltraSleep 一样启动 AUTD Server/TwinCAT，并确保
EtherCAT 设备顺序与 `autd_manager.py` 的 18 台 `geometry` 顺序一致。进入本文件夹后运行：

```bash
cd FrictionAUTD
python friction_play.py
```

没有 AUTD 时可以模拟运行：

```bash
python friction_play.py --mode simulate
```

如果需要换分析文件或音频文件，优先改 `config.toml`：

```toml
[paths]
motion_json = "analysis/asmr_friction_01h22m01s_01h22m20s_friction_motion.json"
audio_wav = ""
```

`audio_wav` 留空时，会自动使用 JSON 里的 `source_path`。

### 直接运行固定摩擦焦点

进入本文件夹运行：

```bash
cd FrictionAUTD
python friction_back_and_forth.py
```

## 常用参数

```bash
python friction_back_and_forth.py \
  --height-mm 100 \
  --base-z-mm 250 \
  --length-mm 60 \
  --cycle-hz 0.6667 \
  --frames 96 \
  --intensity 255
```

- `--height-mm`：分析结果提供的相对高度，默认 `100 mm`。
- `--base-z-mm`：UltraSleep 全局 Z 基准，默认 `250 mm`，所以最终 Z 为 `350 mm`。
- `--length-mm`：往返线段总长度，默认 `60 mm`。
- `--cycle-hz`：完整左到右再回到左的周期频率，默认 `0.6667 Hz`，也就是原来 `2 Hz` 的三分之一速度。
- `--frames`：一个周期内的 STM 帧数，默认 `96`。这样 `0.6667 Hz * 96 = 64 Hz`，满足 AUTD 采样频率约束。
- `--intensity`：AUTD 强度，默认最大值 `255`。
如果触感太弱，应在 UltraSleep 原本的左右工作区域寻找焦点路径；默认中心分别为
`[0, 100, 350]` 和 `[0, -284.4, 350] mm`。
如果触感太快或太慢，优先调整 `--cycle-hz`。

## 同步控制逻辑

`friction_play.py` 的运行链路是：

```text
config.toml
  -> 读取 motion_json
  -> 找到对应 audio_wav
  -> 预加载音频并打开 sounddevice 输出流
  -> 用 UltraSleep AUTDManager 打开 18 台 AUTD
  -> 为左右区域生成同相位、同采样配置的 GainSTM
  -> Group 一次发送给 0–8 和 9–17 号设备
  -> 立刻 start 音频
  -> 按 update_hz 持续读取 JSON 当前时间点参数
  -> 动态更新 AUTD 焦点运动
```

每个时间点传给 AUTD 的信息：

- `intensity`：当前摩擦声音强度，映射为 AUTD 发射强度。
- `cycle_hz`：焦点完整左-右-左往返的速度。
- `length_mm`：焦点往返距离。
- `height_mm`：与 `[autd].base_z_mm` 相加得到全局焦点 Z。
- `jitter_mm`：根据频谱粗糙度生成的平滑路径扰动。

程序会自动选择合法的 STM 分频，避免出现：

```text
Sampling frequency (...) must divide the ultrasound frequency
```

如果需要指定音频输出设备，先查看设备：

```bash
../.venv/bin/python friction_play.py --list-audio-devices
```

然后在 `config.toml` 中修改，并把后端切到 `sounddevice`：

```toml
[audio]
backend = "sounddevice"
device = 4
```

默认配置使用自动选择：

```toml
[audio]
backend = "auto"
```

在 macOS 上会优先使用 `afplay`，在 UltraSleep 的 TwinCAT/Windows 环境会使用
`sounddevice`。如果指定了 `audio.device`，也会强制使用 `sounddevice`。

如果终端出现：

```text
Warning: AUTD send is slow
```

说明真实设备发送耗时已经接近当前更新周期。优先在 `config.toml` 里继续降低：

```toml
[playback]
update_hz = 2.0
```

## 分析摩擦音频

`friction_analyze.py` 会把一段摩擦 / 抚摸 / 刮擦 WAV 分析成焦点运动参数。

它不使用 AudioSep，只使用 librosa：

```text
摩擦 WAV
  -> RMS / dB / normalized amplitude
  -> onset strength / local tempo
  -> spectral centroid / bandwidth
  -> intensity / cycle_hz / length_mm / jitter_mm
  -> friction_motion.json
```

运行：

```bash
cd /Users/mikieteriri/proj/AUTDtest/FrictionAUTD
../.venv/bin/python friction_analyze.py --wav path/to/friction.wav --plot
```

默认输出：

```text
analysis/friction_friction_motion.json
analysis/figures/friction/
  01_normalized_amplitude.png
  02_autd_intensity.png
  03_cycle_hz.png
  04_stroke_length_mm.png
  05_jitter_mm.png
```

也就是说，每个 clip 会有自己的图片文件夹，不会再把多张图合并成一个大图。

JSON 中最重要的是 `motion_curve`：

```json
{
  "time_s": 1.25,
  "normalized_amplitude": 0.62,
  "intensity": 183,
  "cycle_hz": 0.74,
  "length_mm": 67.2,
  "height_mm": 100.0,
  "jitter_mm": 1.8
}
```

字段含义：

- `intensity`：摩擦越响，AUTD 强度越大。
- `cycle_hz`：根据 onset 周期估计的来回运动速度。
- `length_mm`：根据响度估计的往返距离。
- `jitter_mm`：根据频谱粗糙度估计的路径抖动量。
- `height_mm`：第一版固定为 `100 mm`。

如果焦点运动速度明显过快，可以调大：

```bash
--onset-divisor 3
```

如果速度过慢，可以调小：

```bash
--onset-divisor 1
```

## 画出摩擦声音特征

如果已经有 `friction_motion.json`，可以生成更直观的特征图：

```bash
../.venv/bin/python friction_visualize.py \
  --json analysis/asmr_friction_01h22m01s_01h22m20s_friction_motion.json
```

默认输出到：

```text
analysis/figures/asmr_friction_01h22m01s_01h22m20s/
```

每一种特征会单独生成一张图片：

- 音频强度和摩擦 onset 脉冲；
- 映射后的 AUTD intensity；
- 估计的焦点来回速度 `cycle_hz`；
- 焦点往返长度 `length_mm`；
- 频谱粗糙度和路径抖动 `jitter_mm`；
- 估计的焦点 x 方向运动轨迹。

当前图片文件命名为：

```text
06_audio_amplitude_and_onset.png
07_visual_autd_intensity.png
08_cycle_speed_hz.png
09_stroke_length_mm.png
10_roughness_and_spectral_centroid.png
11_path_jitter_mm.png
12_estimated_focus_x_motion.png
13_estimated_focus_speed.png
```
