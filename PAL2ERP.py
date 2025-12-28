#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unfold ocam fisheye ring images to equirectangular (lon/lat) using refined intrinsics.

Dependencies:
  pip install numpy opencv-python
"""

import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import cv2
import yaml


class OcamModel:
    def __init__(self):
        self.pol = []
        self.invpol = []
        self.length_pol = 0
        self.length_invpol = 0
        # xc=col, yc=row
        self.xc = 0.0
        self.yc = 0.0
        # affine
        self.c = 1.0
        self.d = 0.0
        self.e = 0.0
        # image size
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


def world2cam_batch(X, Y, Z, ocam: OcamModel, eps=1e-12):
    X, Y, Z = np.broadcast_arrays(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
    )

    norm_xy = np.sqrt(X * X + Y * Y)
    u = np.full_like(norm_xy, ocam.xc, dtype=np.float64)
    v = np.full_like(norm_xy, ocam.yc, dtype=np.float64)

    mask = norm_xy > eps
    if np.any(mask):
        theta = np.arctan2(Z[mask], norm_xy[mask])
        rho = _polyval_horner(np.asarray(ocam.invpol, dtype=np.float64), theta)
        invn = 1.0 / norm_xy[mask]
        xx = X[mask] * invn * rho
        yy = Y[mask] * invn * rho
        u[mask] = xx * ocam.c + yy * ocam.d + ocam.xc
        v[mask] = xx * ocam.e + yy + ocam.yc
    return u, v


def create_equirect_remap(ocam: OcamModel,
                          out_w: int,
                          out_h: int,
                          lon_min_deg: float,
                          lon_max_deg: float,
                          lat_min_deg: float,
                          lat_max_deg: float):
    lon_range = lon_max_deg - lon_min_deg
    lat_range = lat_max_deg - lat_min_deg
    yy = np.arange(out_h, dtype=np.float64)[:, None]
    xx = np.arange(out_w, dtype=np.float64)[None, :]
    v = yy / float(out_h)
    u = xx / float(out_w)

    lat_deg = -lat_max_deg + v * lat_range
    lon_deg = -lon_min_deg - u * lon_range

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat)

    X = cos_lat * np.cos(lon)
    Y = cos_lat * np.sin(lon)
    Z = np.sin(lat) * np.ones_like(lon)

    u_map, v_map = world2cam_batch(X, Y, Z, ocam)
    mapx = u_map.astype(np.float32)
    mapy = v_map.astype(np.float32)
    return mapx, mapy


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input_dir", default=None, help="Folder with raw ring images")
    parser.add_argument("--output_dir", default=None, help="Folder to save unfolded images")
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--lon_min", type=float, default=None)
    parser.add_argument("--lon_max", type=float, default=None)
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    input_dir = args.input_dir or config.get("image_dir")
    output_dir = args.output_dir or config.get("erp_image_dir")
    intrinsics = args.intrinsics or config.get("intrinsics_better") or config.get("intrinsics_path")
    out_h = args.height if args.height is not None else config.get("erp_out_height", 512)
    lon_min = args.lon_min if args.lon_min is not None else config.get("erp_lon_min", -180.0)
    lon_max = args.lon_max if args.lon_max is not None else config.get("erp_lon_max", 180.0)
    lat_min = args.lat_min if args.lat_min is not None else config.get("erp_lat_min", -6.0)
    lat_max = args.lat_max if args.lat_max is not None else config.get("erp_lat_max", 39.0)

    if not input_dir or not output_dir or not intrinsics:
        raise ValueError("Config missing required paths for input_dir/output_dir/intrinsics")

    ocam = load_ocam_model_auto(intrinsics)

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    out_h = int(out_h)
    out_w = int(round(out_h * lon_range / lat_range))

    mapx, mapy = create_equirect_remap(
        ocam, out_w, out_h,
        lon_min, lon_max,
        lat_min, lat_max
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images_sorted_by_number(str(input_dir))
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    for path in images:
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            print(f"Failed to load image: {path}")
            continue
        pano = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), pano)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
