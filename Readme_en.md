# LiDAR-Camera Extrinsic Calibration Pipeline

This project calibrates LiDAR against either panoramic (equirectangular) images produced by stitched fisheye cameras or standard pinhole cameras by manually picking four rectangle corners per frame.

## Workflow Overview

1. `get_point.py` – pick four image corners per frame.
2. `pcd_show.py` – pick the corresponding four LiDAR points.
3. `main_yuyan.py` – solve the extrinsic parameters and optionally visualize reprojection.

Each script wipes its output file before writing new annotations, so back up previous results if needed.

## Environment

- Python ≥ 3.10
- Packages: `open3d`, `numpy`, `opencv-python`, `scipy`, `pyyaml`

```bash
pip install open3d numpy opencv-python scipy pyyaml
```

## Configuration Files

`config/config.yaml` only redirects to the actual dataset configuration:

```yaml
config_path: config/911.yaml
```

A dataset configuration (e.g., `config/911.yaml`) contains paths and projection options:

```yaml
lidar_out: "./sign/lidar_point911.txt"
photo_out: "./sign/photo_point911.txt"
lidar_dir: "./example/911/lidar"
image_dir: "./example/911/photo"
extrinsic_out: "./result/extrinsic_911.txt"

# Choose between "equirectangular" and "pinhole"
projection_model: "pinhole"

# Required when projection_model == "pinhole":
# first 9 numbers form K (row major), optional next 5 are distortion [k1, k2, p1, p2, k3]
intrinsics_path: "./sign/int_pianzhen.txt"

image_width: 5188.0
image_height: 2594.0
```

To switch models:
- `equirectangular`: omit `intrinsics_path`; the solver uses spherical projection.
- `pinhole`: provide `intrinsics_path` pointing to a text file with camera matrix and optional distortion.

Update `config/config.yaml` to point to the dataset file you want to process, then run the scripts in order.

## Running the Steps

```bash
python get_point.py   # annotate image corners
python pcd_show.py    # annotate LiDAR corners
python main_yuyan.py  # optimize extrinsics and visualize
```

`main_yuyan.py` iterates over 90° rotation seeds, optimizes quaternion + translation with Levenberg–Marquardt, saves the best extrinsics, and shows per-frame overlays if visualization is enabled.

## Tips

- Ensure image and LiDAR corner ordering matches.
- Use absolute paths if folders contain non-ASCII characters.
- Visualization windows open sequentially; press any key to advance.
