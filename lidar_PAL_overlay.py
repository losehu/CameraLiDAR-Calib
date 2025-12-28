#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project LiDAR PCD points onto original panoramic ring images using ocam intrinsics.

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


class OcamModel:
    def __init__(self):
        self.pol = []
        self.invpol = []
        self.length_pol = 0
        self.length_invpol = 0
        self.xc = 0.0  # column center
        self.yc = 0.0  # row center
        self.c = 1.0
        self.d = 0.0
        self.e = 0.0
        self.width = 0
        self.height = 0


def _polyval_horner(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    for a in coeffs:
        y = y * x + a
    return y


def load_ocam_model_auto(path: str) -> OcamModel:
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f.readlines()]
    raw = [ln for ln in raw if ln]

    m = OcamModel()

    if raw[0].lower().startswith("invpol_len"):
        k = int(raw[0].split()[1])
        idx = 1
        inv = []
        while idx < len(raw):
            low = raw[idx].lower()
            if low.startswith("xc yc c d e") or low.startswith("w h"):
                break
            inv.extend([float(x) for x in raw[idx].split()])
            idx += 1
        inv = inv[:k]
        if len(inv) != k:
            raise ValueError(f"[simple] invpol_len={k} but got {len(inv)} numbers")
        m.invpol = inv
        m.length_invpol = k

        while idx < len(raw) and not raw[idx].lower().startswith("xc yc c d e"):
            idx += 1
        if idx >= len(raw):
            raise ValueError("[simple] missing 'xc yc c d e'")
        idx += 1
        aff = [float(x) for x in raw[idx].split()]
        m.xc, m.yc, m.c, m.d, m.e = aff[:5]

        idx += 1
        while idx < len(raw) and not raw[idx].lower().startswith("w h"):
            idx += 1
        if idx >= len(raw):
            raise ValueError("[simple] missing 'W H'")
        idx += 1
        wh = [float(x) for x in raw[idx].split()]
        m.width = int(round(wh[0]))
        m.height = int(round(wh[1]))
        return m

    def is_comment(ln: str) -> bool:
        return ln.startswith("#")

    def first_numeric_line(start: int) -> int:
        i = start
        while i < len(raw):
            ln = raw[i]
            if is_comment(ln):
                i += 1
                continue
            ch = ln[0]
            if ch.isdigit() or ch in "+-.":
                return i
            i += 1
        return -1

    def parse_coeff_line(ln: str):
        parts = ln.split()
        length = int(float(parts[0]))
        coeffs = [float(x) for x in parts[1:1 + length]]
        return length, coeffs

    idx = first_numeric_line(0)
    if idx < 0:
        raise ValueError("[ocamcalib] cannot find pol line")
    m.length_pol, m.pol = parse_coeff_line(raw[idx])

    idx = first_numeric_line(idx + 1)
    if idx < 0:
        raise ValueError("[ocamcalib] cannot find invpol line")
    m.length_invpol, m.invpol = parse_coeff_line(raw[idx])

    idx = first_numeric_line(idx + 1)
    if idx < 0:
        raise ValueError("[ocamcalib] cannot find center line")
    row, col = [float(x) for x in raw[idx].split()[:2]]
    m.yc = row
    m.xc = col

    idx = first_numeric_line(idx + 1)
    if idx < 0:
        raise ValueError("[ocamcalib] cannot find affine line")
    m.c, m.d, m.e = [float(x) for x in raw[idx].split()[:3]]

    idx = first_numeric_line(idx + 1)
    if idx < 0:
        raise ValueError("[ocamcalib] cannot find size line")
    m.height = int(round(float(raw[idx].split()[0])))
    m.width = int(round(float(raw[idx].split()[1])))
    return m


def world2cam(point3d: np.ndarray, m: OcamModel):
    n = float(np.linalg.norm(point3d))
    if n <= 0.0:
        return m.xc, m.yc
    X, Y, Z = (point3d / n).astype(float)
    norm_xy = math.sqrt(X * X + Y * Y)
    if norm_xy > 1e-12:
        theta = math.atan2(Z, norm_xy)
        rho = _polyval_horner(np.asarray(m.invpol, dtype=np.float64), np.array([theta]))[0]
        invn = 1.0 / norm_xy
        xx = X * invn * rho
        yy = Y * invn * rho
        u = xx * m.c + yy * m.d + m.xc
        v = xx * m.e + yy + m.yc
    else:
        u = m.xc
        v = m.yc
    return float(u), float(v)


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
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--extrinsic", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--flip", type=float, nargs=3, default=(-1,-1,-1))
    parser.add_argument("--min_depth", type=float, default=None)
    parser.add_argument("--pcd_m_to_mm", action="store_true", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    img_dir = args.img_dir or config.get("image_dir")
    pcd_dir = args.pcd_dir or config.get("pcdlidar_dir") or config.get("lidar_dir")
    intrinsics = args.intrinsics or config.get("intrinsics_better") or config.get("intrinsics_path")
    extrinsic = args.extrinsic or config.get("extrinsic_out")
    step = args.step if args.step is not None else config.get("pcd_project_step", 5)
    radius = args.radius if args.radius is not None else config.get("pcd_project_radius", 1)
    flip = args.flip if args.flip is not None else _parse_flip(config.get("pcd_project_flip"))
    min_depth = args.min_depth if args.min_depth is not None else config.get("pcd_project_min_depth", 1e-6)
    if args.pcd_m_to_mm is not None:
        pcd_m_to_mm = args.pcd_m_to_mm
    else:
        pcd_m_to_mm = bool(config.get("pcd_m_to_mm", True))

    if not img_dir or not pcd_dir or not intrinsics or not extrinsic:
        raise ValueError("Config missing required paths for img_dir/pcd_dir/intrinsics/extrinsic")

    if flip is None:
        flip = [-1.0, -1.0, -1.0]

    ocam = load_ocam_model_auto(intrinsics)
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

        depths = np.linalg.norm(pts, axis=1)
        depths_sorted = np.sort(depths)
        idx90 = int(math.floor(0.9 * len(depths_sorted)))
        idx90 = min(idx90, len(depths_sorted) - 1)
        p90 = depths_sorted[idx90]
        if p90 <= 0:
            p90 = depths_sorted[-1]

        s = max(1, int(step))
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

            u, v = world2cam(pc, ocam)
            ui = int(round(u))
            vi = int(round(v))
            if ui < 0 or ui >= img.shape[1] or vi < 0 or vi >= img.shape[0]:
                continue
            color = hsv_color_from_depth(depth, p90)
            cv2.circle(img, (ui, vi), int(radius), color, -1, cv2.LINE_AA)
            drawn += 1

        cv2.imshow("show", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
