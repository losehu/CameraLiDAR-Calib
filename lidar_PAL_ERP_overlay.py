#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project LiDAR PCD points onto unfolded equirectangular images.

Dependencies:
  pip install numpy opencv-python open3d
"""

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import cv2
import yaml

try:
    import open3d as o3d
except Exception as exc:
    raise RuntimeError("Requires open3d: pip install open3d") from exc


def numeric_key(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem)
    except Exception:
        pass
    best = None
    for i in range(len(stem) - 1, -1, -1):
        if stem[i].isdigit():
            j = i
            while j >= 0 and stem[j].isdigit():
                j -= 1
            try:
                best = int(stem[j + 1:i + 1])
            except Exception:
                best = None
            break
    return best if best is not None else (2 ** 31 - 1)


def list_images_sorted_by_number(dir_path: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    paths = [p for p in Path(dir_path).iterdir()
             if p.is_file() and p.suffix.lower() in exts]
    paths.sort(key=lambda p: (numeric_key(p), p.name))
    return paths


def list_pcd_sorted_by_number(dir_path: str) -> List[Path]:
    paths = [p for p in Path(dir_path).iterdir()
             if p.is_file() and p.suffix.lower() == ".pcd"]
    paths.sort(key=lambda p: (numeric_key(p), p.name))
    return paths


def load_extrinsic_4x4(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            low = ln.lower()
            if low.startswith("extrinsic"):
                continue
            parts = ln.split()
            if len(parts) >= 4:
                vals.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    if len(vals) < 3:
        raise ValueError(f"Invalid extrinsic file: {path}")
    mat = np.eye(4, dtype=float)
    mat[0, :4] = vals[0][:4]
    mat[1, :4] = vals[1][:4]
    mat[2, :4] = vals[2][:4]
    return mat


def project_to_equirect_uv(pc: np.ndarray,
                           out_w: int,
                           out_h: int,
                           lon_min_deg: float,
                           lon_max_deg: float,
                           lat_min_deg: float,
                           lat_max_deg: float):
    norm = np.linalg.norm(pc)
    if norm <= 0.0:
        return None
    x, y, z = pc / norm
    lon = math.degrees(math.atan2(y, x))
    lat = math.degrees(math.asin(z))

    lon_low = -lon_max_deg
    lon_high = -lon_min_deg
    lat_low = -lat_max_deg
    lat_high = -lat_min_deg
    if lon < lon_low or lon > lon_high or lat < lat_low or lat > lat_high:
        return None

    lon_range = lon_max_deg - lon_min_deg
    lat_range = lat_max_deg - lat_min_deg
    uf = (-lon_min_deg - lon) / lon_range * out_w
    vf = (lat + lat_max_deg) / lat_range * out_h

    ui = int(round(uf))
    vi = int(round(vf))
    if ui == out_w:
        ui = out_w - 1
    if vi == out_h:
        vi = out_h - 1
    if ui < 0 or ui >= out_w or vi < 0 or vi >= out_h:
        return None
    return ui, vi


def hsv_color_from_depth(depth: float, p90: float):
    tnorm = depth / p90 if p90 > 1e-12 else 1.0
    if tnorm > 1.0:
        tnorm = 1.0
    hue = (1.0 - tnorm) * 270.0
    hsv = np.uint8([[[int(hue / 2.0 + 0.5), 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def load_config(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of keys to values")

    redirect = data.get("config_path")
    if redirect:
        nested_path = (path.parent / str(redirect)).resolve()
        try:
            with open(nested_path, "r", encoding="utf-8") as stream:
                nested = yaml.safe_load(stream)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Nested config file not found: {nested_path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse nested YAML config: {exc}") from exc
        if not isinstance(nested, dict):
            raise ValueError("Nested config file must contain a mapping of keys to values")
        return nested

    return data


def _parse_flip(value) -> List[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    if isinstance(value, str):
        parts = [p for p in value.replace(",", " ").split() if p]
        if len(parts) >= 3:
            return [float(parts[0]), float(parts[1]), float(parts[2])]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--img_dir", default=None)
    parser.add_argument("--pcd_dir", default=None)
    parser.add_argument("--extrinsic", default=None)
    parser.add_argument("--lon_min", type=float, default=None)
    parser.add_argument("--lon_max", type=float, default=None)
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--flip", type=float, nargs=3, default=(-1,-1,-1))
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--pcd_m_to_mm", action="store_true", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    img_dir = args.img_dir or config.get("erp_image_dir") or config.get("image_dir")
    pcd_dir = args.pcd_dir or config.get("pcdlidar_dir") or config.get("lidar_dir")
    extrinsic = args.extrinsic or config.get("extrinsic_out")
    lon_min = args.lon_min if args.lon_min is not None else config.get("erp_lon_min", -180.0)
    lon_max = args.lon_max if args.lon_max is not None else config.get("erp_lon_max", 180.0)
    lat_min = args.lat_min if args.lat_min is not None else config.get("erp_lat_min", -6.0)
    lat_max = args.lat_max if args.lat_max is not None else config.get("erp_lat_max", 39.0)
    step = args.step if args.step is not None else config.get("pcd_project_step", 5)
    radius = args.radius if args.radius is not None else config.get("pcd_project_radius", 1)
    flip = args.flip if args.flip is not None else _parse_flip(config.get("pcd_project_flip"))
    min_depth = args.min_depth if args.min_depth is not None else config.get("pcd_project_min_depth", 1e-6)
    if args.pcd_m_to_mm is not None:
        pcd_m_to_mm = args.pcd_m_to_mm
    else:
        pcd_m_to_mm = bool(config.get("pcd_m_to_mm", True))

    if not img_dir or not pcd_dir or not extrinsic:
        raise ValueError("Config missing required paths for img_dir/pcd_dir/extrinsic")

    if flip is None:
        flip = [-1.0, -1.0, -1.0]

    img_paths = list_images_sorted_by_number(str(img_dir))
    pcd_paths = list_pcd_sorted_by_number(str(pcd_dir))
    if not img_paths or not pcd_paths:
        raise RuntimeError("No images or pcds found.")

    n = min(len(img_paths), len(pcd_paths))
    if len(img_paths) != len(pcd_paths):
        print(f"[Warn] image count={len(img_paths)} pcd count={len(pcd_paths)} => process first {n} pairs")

    ext = load_extrinsic_4x4(extrinsic)
    Rm = ext[:3, :3]
    tv = ext[:3, 3]
    A = np.array(flip, dtype=float)


    for i in range(n):
        img = cv2.imread(str(img_paths[i]))
        if img is None or img.size == 0:
            print(f"Failed to load image: {img_paths[i]}")
            continue

        cloud = o3d.io.read_point_cloud(str(pcd_paths[i]))
        pts = np.asarray(cloud.points, dtype=np.float64)
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.size == 0:
            print("No valid points in cloud")
            continue

        s = max(1, int(step))
        depths = np.linalg.norm(pts, axis=1)
        depths_sorted = np.sort(depths)
        idx90 = int(math.floor(0.9 * len(depths_sorted)))
        idx90 = min(idx90, len(depths_sorted) - 1)
        p90 = depths_sorted[idx90]
        if p90 <= 0:
            p90 = depths_sorted[-1]

        drawn = 0
        for k in range(0, len(pts), s):
            pt = pts[k]
            depth = float(np.linalg.norm(pt))
            pl = pt.astype(float)
            if pcd_m_to_mm:
                pl *= 1000.0
            pc = Rm @ pl + tv
            pc = pc * A
            if np.linalg.norm(pc) < min_depth:
                continue

            uv = project_to_equirect_uv(pc, img.shape[1], img.shape[0],
                                        lon_min, lon_max,
                                        lat_min, lat_max)
            if uv is None:
                continue
            u, v = uv
            color = hsv_color_from_depth(depth, p90)
            cv2.circle(img, (u, v), int(radius), color, -1, cv2.LINE_AA)
            drawn += 1

        out_path =  (img_paths[i].stem + "_pcdproj" + img_paths[i].suffix)
        cv2.imshow(str(out_path), img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
