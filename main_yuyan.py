#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_yuyan.py
Python port of main_yuyan.cpp: generate 90-degree rotations, run nonlinear
optimization (quaternion + translation) for each initialization, keep the
best solution, and finally visualize the best result.

Dependencies:
  pip install numpy scipy opencv-python

Usage: adjust paths inside `main()` or pass via CLI enhancements.
"""

import os
import math
import glob
from pathlib import Path
from typing import List

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import yaml


class PnPData:
    def __init__(self, x, y, z, u, v):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.u = float(u)
        self.v = float(v)


def rotation_x(deg):
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(deg):
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(deg):
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def generate_all_90_degree_rotations():
    angles = [0, 90, 180, 270]
    mats = []
    for x in angles:
        Rx = rotation_x(x)
        for y in angles:
            Ry = rotation_y(y)
            for z in angles:
                Rz = rotation_z(z)
                M = Rz @ Ry @ Rx
                mats.append(M)
    # Remove duplicates (by rounding)
    unique = []
    for M in mats:
        found = False
        for U in unique:
            if np.allclose(M, U, atol=1e-6):
                found = True
                break
        if not found:
            unique.append(M)
    return unique


def read_points_pair(lidar_path, photo_path):
    """Read lidar and photo text files line-by-line.
    Only lines containing spaces (coordinate data) are parsed. Returns list of PnPData.
    """
    pData = []
    with open(lidar_path, 'r') as fl, open(photo_path, 'r') as fp:
        for l_line, p_line in zip(fl, fp):
            l_line = l_line.strip()
            p_line = p_line.strip()
            has_l = (len(l_line.split()) == 3)
            has_p = (len(p_line.split()) >= 2)
            if has_l and has_p:
                # lidar line: x y z (maybe as strings)
                parts_l = l_line.split()
                parts_p = p_line.split()
                try:
                    x = float(parts_l[0])
                    y = float(parts_l[1])
                    z = float(parts_l[2])
                    u = float(parts_p[0])
                    v = float(parts_p[1])
                    pData.append(PnPData(x, y, z, u, v))
                except Exception:
                    # skip malformed
                    continue
            else:
                # skip lines without coordinate data
                continue
    return pData


def load_intrinsics_file(path: str):
    values: List[float] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            for token in stripped.split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue
    if len(values) < 9:
        raise ValueError(f"Intrinsics file requires at least 9 numeric values: {path}")
    K = np.array(values[:9], dtype=float).reshape(3, 3)
    distortion = np.zeros(5, dtype=float)
    remaining = values[9:]
    for idx in range(min(5, len(remaining))):
        distortion[idx] = remaining[idx]
    return K, distortion


def make_projector(model: str, img_width: float, img_height: float, intrinsics=None, distortion=None):
    model_lower = (model or 'equirectangular').lower()
    if model_lower == 'pinhole':
        if intrinsics is None:
            raise ValueError("Pinhole projection requires intrinsics.")
        K = np.asarray(intrinsics, dtype=float)
        dist = np.zeros(5, dtype=float)
        if distortion is not None:
            dist[:min(len(distortion), 5)] = np.asarray(distortion, dtype=float)[:min(len(distortion), 5)]
        def projector(Rmat, tvec, point):
            p_c = Rmat @ point + tvec
            z = p_c[2]
            if abs(z) < 1e-9:
                z = 1e-9 if z >= 0 else -1e-9
            x = p_c[0] / z
            y = p_c[1] / z
            r2 = x * x + y * y
            k1, k2, p1, p2, k3 = dist
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            uv_h = K @ np.array([x_distorted, y_distorted, 1.0], dtype=float)
            denom = uv_h[2]
            if abs(denom) < 1e-9:
                denom = 1e-9 if denom >= 0 else -1e-9
            return float(uv_h[0] / denom), float(uv_h[1] / denom)
        return projector, model_lower
    def projector(Rmat, tvec, point):
        return project_point_uv_from_ext(Rmat, tvec, point[0], point[1], point[2], img_width, img_height)
    return projector, 'equirectangular'


def project_point_uv_from_ext(Rmat, tvec, x, y, z, img_width, img_height):
    """Implements getTheoreticalUV_yuyan logic (same mapping as C++)."""
    matrix2 = np.zeros((3, 4), dtype=float)
    matrix2[:, :3] = Rmat
    matrix2[:, 3] = tvec
    coord = np.array([x, y, z, 1.0], dtype=float)
    result = matrix2 @ coord
    u = float(result[0])
    v = float(result[1])
    depth = float(result[2])
    n = math.sqrt(u * u + v * v + depth * depth)
    if n > 0:
        u /= n
        v /= n
        depth /= n
    lon = math.atan2(v, u)
    lat = math.atan2(depth, math.sqrt(u * u + v * v))
    uv0 = (math.pi - lon) * img_width / (2.0 * math.pi)
    uv1 = (0.5 * math.pi - lat) * img_height / math.pi
    return uv0, uv1


def residuals_func(params, points, projector):
    """params: length-7 vector [qx,qy,qz,qw, tx,ty,tz]"""
    q = np.array(params[0:4], dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        q = np.array([0, 0, 0, 1.0], dtype=float)
    else:
        q = q / q_norm
    rot = R.from_quat([q[0], q[1], q[2], q[3]])
    Rmat = rot.as_matrix()
    t = np.array(params[4:7], dtype=float)
    res = []
    for pd in points:
        p_l = np.array([pd.x, pd.y, pd.z], dtype=float)
        u, v = projector(Rmat, t, p_l)
        res.append(u - pd.u)
        res.append(v - pd.v)
    return np.array(res, dtype=float)


def write_ext_file(path, Rmat, tvec):
    """Write extrinsic matrix in legacy 4x4 text format.

    The output matches existing C++ tools:

    extrinsic
    r00 r01 r02 tx
    r10 r11 r12 ty
    r20 r21 r22 tz
    0 0 0 1
    """
    rows = [
        f"{Rmat[0, 0]:.9g} {Rmat[0, 1]:.9g} {Rmat[0, 2]:.9g} {tvec[0]:.9g}",
        f"{Rmat[1, 0]:.9g} {Rmat[1, 1]:.9g} {Rmat[1, 2]:.9g} {tvec[1]:.9g}",
        f"{Rmat[2, 0]:.9g} {Rmat[2, 1]:.9g} {Rmat[2, 2]:.9g} {tvec[2]:.9g}",
        "0 0 0 1",
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('extrinsic\n')
        for line in rows:
            f.write(line + '\n')


def load_config(path: Path) -> dict:
    """Read YAML configuration with required path parameters."""
    try:
        with open(path, 'r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of keys to values")

    redirect = data.get('config_path')
    if redirect:
        nested_path = (path.parent / str(redirect)).resolve()
        try:
            with open(nested_path, 'r', encoding='utf-8') as stream:
                nested = yaml.safe_load(stream)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Nested config file not found: {nested_path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse nested YAML config: {exc}") from exc
        if not isinstance(nested, dict):
            raise ValueError("Nested config file must contain a mapping of keys to values")
        return nested

    return data


def _parse_extrinsic_lines(lines):
    """Parse extrinsic text (supports legacy single-line and new 4x4 layouts)."""
    numeric_lines: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if stripped.startswith('#'):
            continue
        if lowered.startswith('extrinsic'):
            continue
        numeric_lines.append(stripped)

    if not numeric_lines:
        raise ValueError('extrinsic file contains no numeric rows')

    # Support legacy format: single line with 12 floats
    if len(numeric_lines) == 1:
        parts = numeric_lines[0].split()
        if len(parts) < 12:
            raise ValueError('extrinsic row requires at least 12 floats')
        vals = [float(x) for x in parts[:12]]
    else:
        rows: List[List[float]] = []
        for entry in numeric_lines:
            nums = [float(x) for x in entry.split()]
            if len(nums) >= 4:
                rows.append(nums[:4])
        if len(rows) < 3:
            raise ValueError('extrinsic matrix needs three rows of four floats')
        vals = []
        for row in rows[:3]:
            vals.extend(row[:3])
            vals.append(row[3])

    Rmat = np.array([[vals[0], vals[1], vals[2]],
                     [vals[4], vals[5], vals[6]],
                     [vals[8], vals[9], vals[10]]], dtype=float)
    tvec = np.array([vals[3], vals[7], vals[11]], dtype=float)
    return Rmat, tvec


def list_images_sorted_by_number(dirpath):
    p = Path(dirpath)
    imgs = []
    for entry in p.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() in ['.jpg', '.jpeg']:
            imgs.append(entry)
    def numeric_key(pth):
        stem = pth.stem
        try:
            val = int(stem)
            return (0, val, pth.name)
        except Exception:
            return (1, stem, pth.name)
    imgs.sort(key=numeric_key)
    return [str(x) for x in imgs]


def extract_image_order_from_txt(photo_path: str) -> List[str]:
    order: List[str] = []
    try:
        with open(photo_path, 'r', encoding='utf-8') as pf:
            for line in pf:
                stripped = line.strip()
                if not stripped:
                    continue
                if ' ' in stripped:
                    continue
                if '.' not in stripped:
                    continue
                order.append(stripped)
    except OSError as exc:
        print(f"Failed to read photo txt for image order: {exc}")
    return order


def compute_uv_error_and_visualize(extrinsic_path, lidar_path, photo_path, image_dir, projector, threshold=12, visualize=False):
    with open(extrinsic_path, 'r', encoding='utf-8') as f:
        Rmat, tvec = _parse_extrinsic_lines(f.readlines())
    # open lidar/photo and images list
    imgs = list_images_sorted_by_number(image_dir)
    image_lookup = {Path(p).name: p for p in imgs}
    ordered_names = extract_image_order_from_txt(photo_path)
    in_lidar = open(lidar_path, 'r')
    in_photo = open(photo_path, 'r')
    # `count` used previously counted all lines (including non-data), which
    # misaligns grouping of 4 valid correspondences per image when there are
    # skipped/malformed lines. Use `valid_idx` which increments only when a
    # valid lidar-photo correspondence is processed. This ensures each image
    # receives exactly up to 4 drawn points grouped together.
    total_lines = 0
    valid_idx = 0
    errorU_total = 0.0
    errorV_total = 0.0
    useful = 0
    image = None
    image_path = None
    while True:
        lline = in_lidar.readline()
        pline = in_photo.readline()
        total_lines += 1
        if not lline or not pline:
            break
        lline = lline.strip()
        pline = pline.strip()
        has_l = (len(lline.split()) == 3)
        has_p = (len(pline.split()) >= 2)
        if has_l and has_p:
            # parse
            parts_l = lline.split()
            parts_p = pline.split()
            try:
                x = float(parts_l[0]); y = float(parts_l[1]); z = float(parts_l[2])
                dataU = float(parts_p[0]); dataV = float(parts_p[1])
            except Exception:
                # malformed numeric values: skip but do not advance valid_idx
                continue
            # load image for this group at the first valid point of the group
            if (valid_idx % 4 == 0) and visualize:
                idx = valid_idx // 4
                image = None
                image_path = None
                candidate_paths: List[str] = []
                name_candidate = ordered_names[idx] if idx < len(ordered_names) else None
                if name_candidate:
                    candidate = Path(name_candidate)
                    if candidate.is_absolute() and candidate.exists():
                        candidate_paths.append(str(candidate))
                    else:
                        from_order = Path(image_dir) / candidate
                        candidate_paths.append(str(from_order))
                        base = candidate.name
                        if base:
                            from_base = Path(image_dir) / base
                            candidate_paths.append(str(from_base))
                            lookup = image_lookup.get(base)
                            if lookup:
                                candidate_paths.append(str(lookup))
                if idx < len(imgs):
                    candidate_paths.append(imgs[idx])
                seen_paths = set()
                for cand in candidate_paths:
                    cand_str = str(Path(cand).resolve())
                    if cand_str in seen_paths:
                        continue
                    seen_paths.add(cand_str)
                    img = cv2.imread(cand_str)
                    if img is not None:
                        image = img
                        image_path = cand_str
                        break
            point = np.array([x, y, z], dtype=float)
            theoryU, theoryV = projector(Rmat, tvec, point)
            if not (math.isfinite(theoryU) and math.isfinite(theoryV)):
                continue
            if visualize and image is not None:
                # draw circles as C++: theory green, data colors vary by (valid_idx%4)
                cv2.circle(image, (int(round(theoryU)), int(round(theoryV))), 20, (0,255,0), -1)
                color_map = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
                color = color_map[valid_idx % 4]
                cv2.circle(image, (int(round(dataU)), int(round(dataV))), 20, color, -1)
                # show after the 4th valid point in this group
                if (valid_idx + 1) % 4 == 0:
                    idx_show = valid_idx // 4
                    if image_path is not None:
                        winname = os.path.basename(image_path)
                    elif name_candidate:
                        winname = name_candidate
                    else:
                        winname = f"image_{idx_show}"
                    try:
                        cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
                    except Exception:
                        pass
                    cv2.imshow(winname, image)
                    # Small non-blocking wait to let the GUI update the window
                    cv2.waitKey(1)
                    # Now block until user presses a key
                    cv2.waitKey(0)
                    try:
                        cv2.destroyWindow(winname)
                    except Exception:
                        pass
            errorU = abs(dataU - theoryU)
            errorV = abs(dataV - theoryV)
            errorU_total += errorU
            errorV_total += errorV
            useful += 1
            # advance valid correspondence counter only when we processed a valid pair
            valid_idx += 1
    in_lidar.close()
    in_photo.close()
    avgU = errorU_total / useful if useful>0 else 0.0
    avgV = errorV_total / useful if useful>0 else 0.0
    return avgU, avgV


def optimize_for_initial_rotation(points, R_init, projector, max_nfev=2000):
    rot = R.from_matrix(R_init)
    q0 = rot.as_quat()
    x0 = np.zeros(7, dtype=float)
    x0[0:4] = q0
    x0[4:7] = 0.0
    res = least_squares(
        lambda p: residuals_func(p, points, projector),
        x0,
        method='lm',
        max_nfev=max_nfev,
    )
    q = res.x[0:4]
    q = q / np.linalg.norm(q)
    t = res.x[4:7]
    Rmat = R.from_quat(q).as_matrix()
    r = residuals_func(res.x, points, projector)
    u_err = np.mean(np.abs(r[0::2]))
    v_err = np.mean(np.abs(r[1::2]))
    return Rmat, t, u_err, v_err, res


def main():
    config_path = Path(__file__).resolve().parent / 'config/config.yaml'
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return

    required_keys = ('lidar_out', 'photo_out', 'image_dir', 'extrinsic_out')
    missing = [key for key in required_keys if not config.get(key)]
    if missing:
        print(f"Config missing required keys: {', '.join(missing)}")
        return

    lidar_path = str(config['lidar_out'])
    photo_path = str(config['photo_out'])
    image_dir = str(config['image_dir'])
    extrinsic_out = str(config['extrinsic_out'])

    img_width = config.get('image_width', 5188.0)
    img_height = config.get('image_height', 2594.0)
    try:
        img_width = float(img_width)
    except (TypeError, ValueError):
        img_width = 5188.0
    try:
        img_height = float(img_height)
    except (TypeError, ValueError):
        img_height = 2594.0
    projection_model = str(config.get('projection_model', 'equirectangular'))
    intrinsics_path = config.get('intrinsics_path')
    intrinsics = None
    distortion = None
    if projection_model.lower() == 'pinhole':
        if not intrinsics_path:
            print("Config missing intrinsics_path for pinhole projection")
            return
        try:
            intrinsics, distortion = load_intrinsics_file(str(intrinsics_path))
        except (OSError, ValueError) as exc:
            print(exc)
            return
    try:
        projector, projection_model = make_projector(
            projection_model,
            img_width,
            img_height,
            intrinsics=intrinsics,
            distortion=distortion,
        )
    except ValueError as exc:
        print(exc)
        return
    print("Loading data...")
    points = read_points_pair(lidar_path, photo_path)
    print(f"Loaded {len(points)} correspondences")
    if len(points) == 0:
        print("No data, exiting")
        return

    rotations = generate_all_90_degree_rotations()
    print(f"Generated {len(rotations)} unique init rotations")

    best_score = float('inf')
    best_R = None
    best_t = None
    best_uv = (0.0, 0.0)

    for i, R_init in enumerate(rotations):
        print(f"Trying init {i+1}/{len(rotations)}")
        try:
            Rm, tv, u_err, v_err, res = optimize_for_initial_rotation(points, R_init, projector)
        except Exception as e:
            print(f"Optimization failed for init {i}: {e}")
            continue
        score = float(u_err + v_err)
        print(f"-> u_err={u_err:.4f} v_err={v_err:.4f} sum={score:.4f}")
        if score < best_score:
            best_score = score
            best_R = Rm
            best_t = tv
            best_uv = (u_err, v_err)
            # write best extrinsic to file (4x4 text layout)
            os.makedirs(os.path.dirname(extrinsic_out), exist_ok=True)
            write_ext_file(extrinsic_out, best_R, best_t)
            print("  -> new best, written to", extrinsic_out)

    print("Optimization done. Best score:", best_score, "(u+v)")
    print("Best per-channel:", best_uv)

    # visualize using the best extrinsic (display)
    if best_R is not None:
        avgU, avgV = compute_uv_error_and_visualize(
            extrinsic_out,
            lidar_path,
            photo_path,
            image_dir,
            projector,
            threshold=12,
            visualize=True,
        )
        print(f"Displayed best result. Average U error={avgU:.3f}, V error={avgV:.3f}, sum={avgU+avgV:.3f}")
    else:
        print("No valid solution found.")


if __name__ == '__main__':
    main()
