[English Readme](./Readme_en.md)
# 雷达-相机外参标定流程
本项目用于等矩形投影全景图或针孔相机图像与雷达之间的外参标定，可在统一流程下切换成像模型。

两套传感器同时观察同一矩形标定板，通过手动拾取板上四个角点完成外参优化。整套流程分为三步：

1. `get_point.py` —— 标注每张图像里的四个角点
2. `pcd_show.py` —— 标注对应雷达点云中的四个角点
3. `main_yuyan.py` —— 根据两套标注求解相机-雷达外参并做可视化验证

下面列出了运行前的配置方式、每一步的详细操作以及常用快捷键。

## 目录
- [环境与依赖](#环境与依赖)
- [配置文件说明](#配置文件说明)
- [第一步：图像角点标注 (`get_point.py`)](#第一步图像角点标注-get_pointpy)
- [第二步：点云角点标注 (`pcd_show.py`)](#第二步点云角点标注-pcd_showpy)
- [第三步：求解外参 (`main_yuyan.py`)](#第三步求解外参-main_yuyanpy)
- [常见问题排查](#常见问题排查)

## 环境与依赖

- Python 3.10（或更高）
- 依赖包：`open3d`, `numpy`, `opencv-python`, `scipy`, `pyyaml`

```bash
pip install open3d numpy opencv-python scipy pyyaml
```

## 配置文件说明

项目根目录下的 `config/config.yaml` 只负责“指向”真实的配置文件：

```yaml
# config/config.yaml
config_path: config/911.yaml
```

各数据集的具体参数写在 `config/*.yaml` 中（例如 `config/911.yaml`）：

```yaml
# 标注结果输出路径
lidar_out: "/path/to/lidar_OUT911.txt"      # 雷达角点标注输出
photo_out: "/path/to/cam_point911.txt"     # 图像角点标注输出

# 输入数据
lidar_dir: "/path/to/lidar_pcd_dir"         # 存放 .pcd 的目录
image_dir: "/path/to/image_dir"             # 存放图像的目录

# 外参输出
extrinsic_out: "./sign/extrinsic_911_py.txt"

# 成像模型：equirectangular（等矩形全景）或 pinhole（针孔畸变）
projection_model: "pinhole"

# 针孔模型下的内参与畸变文件（至少 9 个数，后续可附加最多 5 个畸变系数）
intrinsics_path: "/path/to/int_pianzhen.txt"
```

若使用全景等矩形模型，将 `projection_model` 设为 `equirectangular` 或省略该参数，并省略 `intrinsics_path`；针孔模型需提供对应的内参文件（矩阵 3×3 展平 + 畸变）。配置修改后，`main_yuyan.py` 会自动读取新的模型与参数。

修改流程：

1. 新建/调整某个数据集的 YAML（如上例中的 `config/911.yaml`）。
2. 在 `config/config.yaml` 中把 `config_path` 指向该 YAML。
3. 重新运行标注或求解脚本即可读到最新配置。

> **提示**：所有脚本都会在启动时清空对应输出文件。如果需要保留旧结果，请先做备份。

## 第一步：图像角点标注 (`get_point.py`)

```bash
python get_point.py
```

脚本启动后会读取 `photo_out` 指定的 TXT 文件，并在开始前清空内容。每张图像需要按照固定顺序（通常为顺时针或逆时针）拾取 **四个角点**。每拾取一张图像后，脚本会立即把数据写入 `photo_out`。

### 常用快捷键

- `鼠标左键`：拾取角点，最多四个
- `Enter / 回车`：输出当前图片的角点并切换到下一张
- `n` / `p`：切换下一张 / 上一张图片
- `b`：撤销当前图像上的最后一个角点
- `r`：清空当前图像上的所有角点
- `j` / `h`：放大 / 缩小
- `W` `S` `A` `D`：视图平移
- `q` / `Esc`：保存当前图像的角点后退出

### 其他行为

- 所有 JPEG/JPG/PNG/BMP 等图片会按**自然排序**加载（`1, 2, …, 10`），避免 `3.jpg` 排在 `29.jpg` 之后。
- 支持固定窗口（默认 1600×900）或按原图大小显示，可在脚本顶部修改 `FIXED_WINDOW`。

## 第二步：点云角点标注 (`pcd_show.py`)

```bash
python pcd_show.py
```

脚本读取 `lidar_dir` 下的全部 `.pcd` 文件，并在开始前清空 `lidar_out` 指定的标注文件。需要同样按照固定顺序拾取标定板的四个角点（单位：毫米）。

### 常用快捷键

- `Shift + 鼠标左键`：拾取点云角点（一次四个）
- `Enter / 回车`：保存当前点云标注并自动跳转下一帧
- `n` / `Shift + n`：下一帧 / 上一帧
- `b`：撤销上一个角点
- `p` / `l`：保存 / 加载当前相机视角到 `saved_cam.cam`
- `r`：重置视角到点云包围盒
- `q`：退出程序

### 其他行为

- 点云颜色会根据距离自动着色：0–20 米经历多次冷暖色循环，超过阈值的点固定为暖色。
- 同一张标定板的四个角点保存后不会插入空行，方便与旧工具对比。

## 第三步：求解外参 (`main_yuyan.py`)

```bash
python main_yuyan.py
```

脚本会按照配置文件读取：

- `photo_out`：图像角点 TXT
- `lidar_out`：点云角点 TXT
- `image_dir`：对应的原始图像目录（用于可视化）
- `extrinsic_out`：计算得到的外参输出文件（4×4 格式）

执行流程：

1. 读取角点配对数据，并生成 90° 旋转的初值集合。
2. 对每个初值执行非线性最小二乘（四元数 + 平移）。
3. 选择误差最小的解并写入 `extrinsic_out`。
4. 若开启 `visualize=True`（默认），按照 `photo_out` 中的顺序逐张图像展示理论投影与手工标注点，以便人工检查。

### 终端输出

- 每个初值会输出 `(u_err, v_err)` 平均误差及其和(和小于10为优)。
- 训练完成后会提示最优误差，并在窗口中展示匹配情况。

## 常见问题排查

- **找不到配置文件**：确认 `config/config.yaml` 指向的路径存在且拼写正确。
- **加载不到图片/点云**：检查 YAML 中的路径是否带有特殊字符或空格，必要时使用绝对路径。
- **标注顺序不一致**：务必保证图像与点云的角点拾取顺序一致（同一块板，同一旋转方向）。
- **窗口过大或过小**：可调整 `get_point.py` 中的 `WINDOW_W/H` 以及 `FIXED_WINDOW`，或在 `pcd_show.py` 中保存自定义相机视角。

完成以上三步，即可得到与原 C++ 流程等价的外参结果。祝标定顺利！