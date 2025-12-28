# LiDAR-Camera Extrinsic Calibration Pipeline

This project calibrates LiDAR-to-camera extrinsics for pinhole/fisheye cameras, equirectangular (ERP) images, and panoramic annular (PAL) cameras. The workflow uses manual selection of four corresponding corners, followed by optimization and visual validation.

**Data Preparation (Required)**
Sample images and point clouds are stored in `example/`. Download the zip from the [Google Drive Link](https://drive.google.com/file/d/1UM3ZfcYInfgu6ADCPa5w8MAr3hAx7B-p/view?usp=sharing), unzip it, and place the contents in that folder before running examples.

## Table of Contents

- [Supported Camera Types](#supported-camera-types)
- [Environment](#environment)
- [Project Layout](#project-layout)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pinhole/Fisheye/ERP Workflow](#pinholefisheyeerp-workflow)
- [PAL (Panoramic Annular) Workflow](#pal-panoramic-annular-workflow)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)

## Supported Camera Types

- Pinhole / fisheye
- Multi-fisheye stitched panorama (ERP)
- Panoramic annular camera (PAL), with ERP unfolding

## Environment

- Python 3.10 or later
- Packages: `open3d`, `numpy`, `opencv-python`, `scipy`, `pyyaml`

```bash
pip install open3d numpy opencv-python scipy pyyaml
```

## Project Layout

- `config/`: configuration files
- `example/`: sample data
- `sign/`: annotations and intrinsics samples
- `result/`: extrinsics and refined intrinsics outputs
- `label_image_corners.py`: image annotation tool
- `label_lidar_corners.py`: LiDAR annotation tool
- `lidar_image_overlay.py`: pinhole/fisheye/ERP calibration and visualization
- `extrinsics_PAL.cpp`: PAL calibration and visualization

## Quick Start

1. Shallow clone the repo (no history to keep data size small):

```bash
git clone --depth 1 https://github.com/losehu/CameraLiDAR-Calib.git
```

2. Choose a dataset config in `config/config.yaml`.
3. Annotate image corners with `label_image_corners.py`.
4. Annotate LiDAR corners with `label_lidar_corners.py`.
5. Run the calibration/visualization step for your camera type.

## Configuration

`config/config.yaml` points to the actual dataset configuration:

```yaml
config_path: config/911.yaml
```

Dataset configuration example (pinhole/fisheye/ERP):

```yaml
# output annotation files
lidar_out: "./sign/lidar_point_hongwai.txt"
photo_out: "./sign/photo_point_hongwai.txt"
# input data folders
lidar_dir: "./example/hongwai/lidar"
image_dir: "./example/hongwai/photo"
# output extrinsics
extrinsic_out: "./result/extrinsic_hongwai.txt"
# intrinsics (not required for ERP)
intrinsics_path: "./sign/int_hongwai.txt"
# projection model (not required for ERP)
projection_model: "pinhole"
# image size
image_width: 640
image_height: 512
```

Notes:
- `lidar_out` / `photo_out` / `extrinsic_out` are output paths and can be customized.
- `lidar_dir` / `image_dir` / `intrinsics_path` / `image_width` / `image_height` must match your data.
- Intrinsics format examples: `sign/int_hongwai.txt` and `sign/int_pianzhen.txt`.

## Pinhole/Fisheye/ERP Workflow

1. Point `config/config.yaml` to the dataset config.
   - `config/911.yaml`, `config/913.yaml`, `config/923.yaml` are ERP examples.
   - `config/hongwai.yaml`, `config/pianzhen.yaml` are pinhole/fisheye examples.
2. Annotate image corners:

```bash
python label_image_corners.py
```

Left-click four corners (e.g., calibration board corners).

3. Annotate LiDAR corners:

```bash
python label_lidar_corners.py
```

Use Shift + left-click to select four LiDAR points.

4. Run calibration and visualization:

```bash
python lidar_image_overlay.py
```

## PAL (Panoramic Annular) Workflow

1. Obtain initial intrinsics using [PanoRing-Calib](https://github.com/losehu/PanoRing-Calib). Example: `sign/int_huandai.txt`.
2. Fill the PAL config file (see `config/huandai.yaml`):

```yaml
lidar_out: "./sign/lidar_point_huandai.txt"
photo_out: "./sign/photo_point_huandai.txt"
lidar_dir: "./example/huandai/lidar"
image_dir: "./example/huandai/photo"
erp_image_dir: "./example/huandai/new_unfold"
extrinsic_out: "./result/extrinsic_huandai.txt"
intrinsics_path: "./sign/int_huandai.txt"
intrinsics_better: "./result/ocam_refined_huandai.txt"
```

3. Annotate image and LiDAR corners:

```bash
python label_image_corners.py
python label_lidar_corners.py
```

4. Install the C++ dependencies listed in `CMakeLists.txt`, then build and run `extrinsics_PAL.cpp` to calibrate and visualize.

5. The calibration outputs a `flip` status; update the `flip` parameter in the follow-up Python scripts (`PAL2ERP.py`, `lidar_PAL_ERP_overlay.py`, `lidar_PAL_overlay.py`).

6. After calibration:

- Unfold PAL images to ERP:

```bash
python PAL2ERP.py
```

- Overlay LiDAR on ERP:

```bash
python lidar_PAL_ERP_overlay.py
```

- Overlay LiDAR on PAL:

```bash
python lidar_PAL_overlay.py
```

## Outputs

- Extrinsics: `result/extrinsic_*.txt`
- Refined PAL intrinsics: `result/ocam_refined_*.txt`
- Annotations: `sign/lidar_point_*.txt` and `sign/photo_point_*.txt`

## Troubleshooting

- Annotation files are overwritten on each run; back up if needed.
- Corner ordering must match between image and LiDAR.
- If paths include non-ASCII characters, prefer absolute paths.
