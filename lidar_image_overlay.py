#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projectCloud_yuyan.py - Python 版本的点云投影到图像的交互式外参标定工具
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import open3d as o3d
import sys
import math
import yaml

# 尝试导入获取屏幕大小的库
try:
    import tkinter as tk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# ====== 全局参数 ======
POINT_ALPHA = 0.60  # 雷达点的不透明度（0=全透明，1=不透明）
POINT_RADIUS = 2  # 点半径（像素）
THRESHOLD_LIDAR = 300000000000
DEG_STEP = 1.0  # 每次角度改变量（度）
TRANS_STEP_MM = 100.0  # 每次平移改变量（毫米）
VOXEL_DOWNSAMPLE_M = 0.03  # 体素降采样尺寸（米），0 表示不降采样
MAX_POINTS_PER_PCD = 0  # 每帧点数上限，0 表示不限制

width = None
height = None
# ============ HSV -> RGB（用于深度着色） ============
def hsv_to_rgb(h, s, v):
    """
    h: 0-360
    s: 0-1
    v: 0-1
    返回: (r, g, b) 0-255
    """
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c
    
    if h < 60:
        r1, g1, b1 = c, x, 0
    elif h < 120:
        r1, g1, b1 = x, c, 0
    elif h < 180:
        r1, g1, b1 = 0, c, x
    elif h < 240:
        r1, g1, b1 = 0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    
    r = int((r1 + m) * 255)
    g = int((g1 + m) * 255)
    b = int((b1 + m) * 255)
    
    return r, g, b


def hsv_to_bgr_batch(h, s, v):
    """批量将 HSV 转为 BGR，h: [N], s/v: 标量或 [N]"""
    h = np.asarray(h, dtype=np.float32) % 360.0
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    c = v * s
    h_div = h / 60.0
    x = c * (1.0 - np.abs(h_div % 2 - 1.0))
    m = v - c

    zeros = np.zeros_like(c)
    r = np.zeros_like(c)
    g = np.zeros_like(c)
    b = np.zeros_like(c)

    mask0 = (h_div >= 0) & (h_div < 1)
    mask1 = (h_div >= 1) & (h_div < 2)
    mask2 = (h_div >= 2) & (h_div < 3)
    mask3 = (h_div >= 3) & (h_div < 4)
    mask4 = (h_div >= 4) & (h_div < 5)
    mask5 = (h_div >= 5) & (h_div < 6)

    r[mask0], g[mask0], b[mask0] = c[mask0], x[mask0], zeros[mask0]
    r[mask1], g[mask1], b[mask1] = x[mask1], c[mask1], zeros[mask1]
    r[mask2], g[mask2], b[mask2] = zeros[mask2], c[mask2], x[mask2]
    r[mask3], g[mask3], b[mask3] = zeros[mask3], x[mask3], c[mask3]
    r[mask4], g[mask4], b[mask4] = x[mask4], zeros[mask4], c[mask4]
    r[mask5], g[mask5], b[mask5] = c[mask5], zeros[mask5], x[mask5]

    r = ((r + m) * 255.0).clip(0, 255).astype(np.uint8)
    g = ((g + m) * 255.0).clip(0, 255).astype(np.uint8)
    b = ((b + m) * 255.0).clip(0, 255).astype(np.uint8)

    return np.stack([b, g, r], axis=1)


# ============ 点云加载 ============
def load_pcd(pcd_path):
    """加载单个 PCD 文件，返回点云数据 (N, 3)"""
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if VOXEL_DOWNSAMPLE_M and VOXEL_DOWNSAMPLE_M > 0:
            try:
                pcd = pcd.voxel_down_sample(VOXEL_DOWNSAMPLE_M)
            except Exception as down_err:
                print(f"Voxel down sample failed ({pcd_path}): {down_err}")
        points = np.asarray(pcd.points)
        if MAX_POINTS_PER_PCD and MAX_POINTS_PER_PCD > 0 and len(points) > MAX_POINTS_PER_PCD:
            idx = np.random.choice(len(points), MAX_POINTS_PER_PCD, replace=False)
            points = points[idx]
        print(f"Loaded {Path(pcd_path).name} points={len(points)}")
        return points
    except Exception as e:
        print(f"Failed to load {pcd_path}: {e}")
        return np.array([])


def load_pointcloud_from_path(path_str):
    """
    从文件或目录加载点云
    返回: numpy array (N, 3)
    """
    path = Path(path_str)
    points_list = []
    
    if not path.exists():
        print(f"Path not exists: {path}")
        return np.array([])
    
    if path.is_file():
        if path.suffix.lower() == '.pcd':
            pts = load_pcd(str(path))
            if len(pts) > 0:
                points_list.append(pts)
        else:
            print(f"Not a .pcd file: {path}")
    elif path.is_dir():
        pcd_files = sorted(path.glob('*.pcd'))
        for pcd_file in pcd_files:
            pts = load_pcd(str(pcd_file))
            if len(pts) > 0:
                points_list.append(pts)
        print(f"Done. total files loaded: {len(pcd_files)}")
    
    if points_list:
        all_points = np.vstack(points_list)
        print(f"Total points: {len(all_points)}")
        return all_points
    return np.array([])


# ============ 外参工具：兼容 3x4(12) 与 4x4(16) ============
def vec_to_rt(ext):
    """
    将外参向量（12或16维）转换为 R(3x3) 和 t(3,)
    假设行主序：[r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz  (若有则) 0 0 0 1]
    """
    if len(ext) not in [12, 16]:
        raise ValueError("extrinsic vector must have 12 (3x4) or 16 (4x4) floats.")
    
    R = np.array([
        [ext[0], ext[1], ext[2]],
        [ext[4], ext[5], ext[6]],
        [ext[8], ext[9], ext[10]]
    ], dtype=np.float32)
    
    t = np.array([ext[3], ext[7], ext[11]], dtype=np.float32)
    
    return R, t


def rt_to_vec(R, t, want16=False):
    """
    将 R(3x3) 和 t(3,) 转换为外参向量
    """
    if want16:
        ext = np.zeros(16, dtype=np.float32)
    else:
        ext = np.zeros(12, dtype=np.float32)
    
    ext[0], ext[1], ext[2], ext[3] = R[0, 0], R[0, 1], R[0, 2], t[0]
    ext[4], ext[5], ext[6], ext[7] = R[1, 0], R[1, 1], R[1, 2], t[1]
    ext[8], ext[9], ext[10], ext[11] = R[2, 0], R[2, 1], R[2, 2], t[2]
    
    if want16:
        ext[15] = 1.0
    
    return ext


# ============ 欧拉角（度） -> 旋转矩阵 ============
def deg2rad(d):
    return d * np.pi / 180.0


def rx_deg(deg):
    """绕 X 轴旋转"""
    a = deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=np.float32)


def ry_deg(deg):
    """绕 Y 轴旋转"""
    a = deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)


def rz_deg(deg):
    """绕 Z 轴旋转"""
    a = deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def orthonormalize(R):
    """使用 SVD 正交化旋转矩阵"""
    U, _, Vt = np.linalg.svd(R)
    return (U @ Vt).astype(np.float32)


# ============ 内参读取 ============
def load_intrinsics_file(path: str):
    """从文件读取内参矩阵和任意长度的畸变系数（兼容旧格式）。

    返回：K（3x3 numpy array），distortion（numpy array，可为空，长度可变）
    """
    values = []
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
    remaining = values[9:]
    distortion = np.array(remaining, dtype=float) if remaining else np.array([], dtype=float)
    return K, distortion


# ============ 等距圆柱投影函数 ============
def project_point_uv_from_ext(Rmat, tvec, x, y, z, img_width, img_height, up_degree: float = 90.0, low_degree: float = -90.0):
    """等距圆柱投影（对应 getTheoreticalUV_yuyan 逻辑）。

    支持通过 up_degree / low_degree（单位：度）自定义垂直映射范围，默认 90 / -90。
    """
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
    # 使用可配置的 up/low 度数计算 uv1
    uv1 = (up_degree * math.pi / 180.0 - lat) * img_height / (math.pi * (up_degree - low_degree) / 180.0)
    return uv0, uv1


# ============ 投影器创建 ============
def make_projector(model: str, img_width: float, img_height: float, intrinsics=None, distortion=None, up_degree: float = 90.0, low_degree: float = -90.0):
    """创建投影器函数，支持针孔和等距圆柱投影"""
    model_lower = (model or 'equirectangular').lower()
    if model_lower == 'pinhole':
        if intrinsics is None:
            raise ValueError("Pinhole projection requires intrinsics.")
        # Follow C++ getTheoreticalUV exactly: uv = K * (R * point + t); then u = uv[0]/uv[2], v = uv[1]/uv[2]
        K = np.asarray(intrinsics, dtype=float)
        dist = np.zeros(5, dtype=float)
        if distortion is not None:
            dist[:min(len(distortion), 5)] = np.asarray(distortion, dtype=float)[:min(len(distortion), 5)]
        def projector(Rmat, tvec, point):
            # point is expected in the same units as extrinsic (e.g., mm)
            p_c = Rmat @ point + tvec
            X = float(p_c[0]); Y = float(p_c[1]); Z = float(p_c[2])
            # match C++ behaviour: only consider points with cam_z > 1e-6 (points behind camera are skipped)
            if Z <= 1e-6:
                return (float('nan'), float('nan'))
            # normalized coordinates (x = X/Z, y = Y/Z)
            x = X / Z
            y = Y / Z
            # apply radial/tangential distortion (k1,k2,p1,p2,k3)
            k1, k2, p1, p2, k3 = dist
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)
            x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
            u = fx * x_dist + cx
            v = fy * y_dist + cy
            return float(u), float(v)
        return projector, model_lower
    # 等距圆柱投影
    def projector(Rmat, tvec, point):
        return project_point_uv_from_ext(Rmat, tvec, point[0], point[1], point[2], img_width, img_height, up_degree, low_degree)
    return projector, 'equirectangular' 


# ============ 外参读取（需要根据实际格式调整） ============
def get_extrinsic(extrinsic_path):
    """
    读取外参文件，返回 numpy array
    兼容旧格式（单行 12 个浮点数）与新格式（4x4 文本矩阵）
    """
    try:
        with open(extrinsic_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

        numeric_lines = []
        for raw in raw_lines:
            stripped = raw.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if stripped.startswith('#'):
                continue
            if lower.startswith('extrinsic'):
                continue
            numeric_lines.append(stripped)

        if not numeric_lines:
            raise ValueError('extrinsic file missing numeric rows')

        if len(numeric_lines) == 1:
            values = [float(x) for x in numeric_lines[0].split()]
            if len(values) < 12:
                raise ValueError('extrinsic row does not contain 12 floats')
            values = values[:12]
        else:
            rows = []
            for line in numeric_lines:
                nums = [float(x) for x in line.split()]
                if len(nums) >= 4:
                    rows.append(nums[:4])
            if len(rows) < 3:
                raise ValueError('extrinsic matrix must provide at least 3 rows')
            values = []
            for row in rows[:3]:
                values.extend(row[:3])
                values.append(row[3])

        return np.array(values, dtype=np.float32)
    except Exception as e:
        print(f"Failed to read extrinsic: {e}")
        return None


# ============ 深度范围计算 ============
def compute_depth_range(pointcloud):
    """计算点云的深度范围"""
    if len(pointcloud) == 0:
        return 0.0, 1.0
    
    depths = np.linalg.norm(pointcloud, axis=1) * 1000.0
    dmin = np.min(depths)
    dmax = np.max(depths)
    
    if dmax <= dmin:
        dmax = dmin + 1e-6
    
    return dmin, dmax


# ============ 渲染投影 ============
def render_projection(base_img, pointcloud, depth_range, Rmat, tvec, projector):
    """
    将点云投影到图像上
    projector: 投影器函数，接受 (Rmat, tvec, point) 返回 (u, v)
    颜色按能投影到图像的点的深度范围归一化，使用彩虹色映射
    """
    overlay = base_img.copy()
    img_h, img_w = overlay.shape[:2]

    # 先筛选出能投到图像上的点，并收集其深度用于归一化
    valid_points = []
    depths_valid = []
    for pt in pointcloud:
        x, y, z = pt[0], pt[1], pt[2]
        point_mm = np.array([x * 1000.0, y * 1000.0, z * 1000.0], dtype=np.float64)
        try:
            u, v = projector(Rmat, tvec, point_mm)
        except Exception:
            continue
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        u_int, v_int = int(round(u)), int(round(v))
        if u_int < 0 or u_int >= img_w or v_int < 0 or v_int >= img_h:
            continue

        try:
            p_c = Rmat.astype(np.float64) @ point_mm + tvec.astype(np.float64)
            depth_cam = float(abs(p_c[2]))  # 使用相机坐标系的 Z 轴距离（毫米）
        except Exception:
            depth_cam = float(abs(point_mm[2]))

        valid_points.append((u_int, v_int, depth_cam))
        depths_valid.append(depth_cam)

    if len(depths_valid) == 0:
        return base_img.copy()

    dmin = float(depth_range[0])
    dmax = float(depth_range[1])
    if dmax <= dmin:
        dmax = dmin + 1e-6

    # 按深度排序后均匀映射到 0~360 色相，避免深度分布不均造成颜色跳变
    n = len(valid_points)
    order = np.argsort(depths_valid)
    hue_by_idx = np.zeros(n, dtype=np.float64)
    if n > 1:
        hue_by_idx[order] = (np.arange(n, dtype=np.float64) / (n - 1)) * 360.0
    else:
        hue_by_idx[0] = 0.0

    for idx, (u_int, v_int, depth_cam) in enumerate(valid_points):
        hue = hue_by_idx[idx]
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        cv2.circle(overlay, (u_int, v_int), POINT_RADIUS, (b, g, r), -1, cv2.LINE_AA)

    out_img = cv2.addWeighted(overlay, POINT_ALPHA, base_img, 1.0 - POINT_ALPHA, 0.0)
    return out_img


# ============ 屏幕自适应 ============
def get_screen_size():
    """获取屏幕大小，返回 (width, height)"""
    if HAS_TKINTER:
        try:
            root = tk.Tk()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except Exception:
            pass
    # 默认值（如果无法获取）
    return 1920, 1080


def resize_to_fit_screen(img, max_scale=0.9):
    """
    调整图像大小以适应屏幕
    max_scale: 最大占用屏幕的比例（默认90%）
    返回调整后的图像和缩放比例
    """
    screen_w, screen_h = get_screen_size()
    img_h, img_w = img.shape[:2]
    
    # 计算缩放比例
    scale_w = (screen_w * max_scale) / img_w
    scale_h = (screen_h * max_scale) / img_h
    scale = min(scale_w, scale_h, 1.0)  # 不超过原始大小，只缩小不放大
    
    if scale < 1.0:
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    else:
        return img, 1.0


# ============ HUD 显示 ============
def draw_hud(img, pitch_deg, roll_deg, yaw_deg, t_mm):
    """在图像上绘制 HUD 信息"""
    text1 = f"Pitch(x): {pitch_deg:.3f} deg | Roll(y): {roll_deg:.3f} deg | Yaw(z): {yaw_deg:.3f} deg"
    text2 = f"t (mm): x={t_mm[0]:.2f}  y={t_mm[1]:.2f}  z={t_mm[2]:.2f}  R={POINT_RADIUS}px"
    text3 = "[u/j]=+/- pitch(x)  [n/m]=-/+ roll(y)  [h/k]=-/+ yaw(z)"
    text4 = "[w/s]=+/- x  [a/d]=-/+ y  [z/x]=-/+ z   [r]=reset  [p]=print"
    text5 = "[SPACE/ENTER]=next image  [q/ESC]=quit"
    
    cv2.putText(img, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, text3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(img, text4, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(img, text5, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def print_state(pitch_deg, roll_deg, yaw_deg, t_mm):
    """打印当前状态"""
    print(f"STATE  d_pitch_x={pitch_deg:.3f} deg  d_roll_y={roll_deg:.3f} deg  "
          f"d_yaw_z={yaw_deg:.3f} deg  |  t(mm)=[{t_mm[0]:.2f}, {t_mm[1]:.2f}, {t_mm[2]:.2f}]")


# ============ 主程序 ============
def main():
    # 读取配置（与 main_yuyan.py 相同风格）
    def load_config(path: Path) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as stream:
                data = yaml.safe_load(stream)
        except FileNotFoundError as exc:
            print(f"Config file not found: {path}")
            return None
        except yaml.YAMLError as exc:
            print(f"Failed to parse YAML config: {exc}")
            return None

        if not isinstance(data, dict):
            print("Config file must contain a mapping of keys to values")
            return None

        redirect = data.get('config_path')
        if redirect:
            nested_path = (path.parent / str(redirect)).resolve()
            try:
                with open(nested_path, 'r', encoding='utf-8') as stream:
                    nested = yaml.safe_load(stream)
            except FileNotFoundError:
                print(f"Nested config file not found: {nested_path}")
                return None
            except yaml.YAMLError as exc:
                print(f"Failed to parse nested YAML config: {exc}")
                return None
            if not isinstance(nested, dict):
                print("Nested config file must contain a mapping of keys to values")
                return None
            return nested

        return data

    config_path = Path(__file__).resolve().parent / 'config/config.yaml'
    config = load_config(config_path)

    # 默认值（当配置缺失时作为回退）
    img_dir = None
    pcd_dir = None
    extrinsic_path = None
    global width, height

    if config:
        # 支持以下键：image_dir, pcd_dir 或 lidar_dir, extrinsic_out, output_path
        img_dir = str(config.get('image_dir') or '') or None
        pcd_dir = str(config.get('pcd_dir') or config.get('lidar_dir') or '') or None
        extrinsic_path = str(config.get('extrinsic_out') or '') or None
        width = config.get('image_width') or config.get('width')
        height = config.get('image_height') or config.get('height')

    # 若仍为空，则使用旧的硬编码路径（确保可运行）
    if not img_dir:
        print("Warning: image_dir not set in config, using default path")
        exit(1)
    if not pcd_dir:
        print("Warning: pcd_dir not set in config, using default path")
        exit(1)
    if not extrinsic_path:
        print("Warning: extrinsic_out not set in config, using default path")
        exit(1)
    if not width or not height:
        print("Warning: width/height not set in config")
        exit(1)
    
    # 读取投影模型和内参
    projection_model = str(config.get('projection_model', 'equirectangular')) if config else 'equirectangular'
    intrinsics_path = config.get('intrinsics_path') if config else None
    intrinsics = None
    distortion = None
    
    if projection_model.lower() == 'pinhole':
        if not intrinsics_path:
            print("Config missing intrinsics_path for pinhole projection")
            exit(1)
        try:
            intrinsics, distortion = load_intrinsics_file(str(intrinsics_path))
            print(f"Loaded intrinsics from {intrinsics_path}")
        except (OSError, ValueError) as exc:
            print(f"Failed to load intrinsics: {exc}")
            exit(1)
    
    # 读取可配置的垂直映射范围（up_degree / low_degree）
    up_degree = config.get('up_degree', 90.0) if config else 90.0
    low_degree = config.get('low_degree', -90.0) if config else -90.0
    try:
        up_degree = float(up_degree)
    except (TypeError, ValueError):
        up_degree = 90.0
    try:
        low_degree = float(low_degree)
    except (TypeError, ValueError):
        low_degree = -90.0
    if not (up_degree > low_degree):
        print(f"Warning: up_degree ({up_degree}) <= low_degree ({low_degree}), swapping values.")
        up_degree, low_degree = max(up_degree, low_degree), min(up_degree, low_degree)

    # 创建投影器
    try:
        projector, projection_model_used = make_projector(
            projection_model,
            float(width),
            float(height),
            intrinsics=intrinsics,
            distortion=distortion,
            up_degree=up_degree,
            low_degree=low_degree,
        )
        print(f"Using projection model: {projection_model_used}")
    except ValueError as exc:
        print(f"Failed to create projector: {exc}")
        exit(1)

    # 获取图片列表
    img_extensions = ['*.jpg', '*.jpeg', '*.png','*.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    img_files = sorted(img_files)

    # 获取 PCD 列表
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))

    num_pairs = min(len(img_files), len(pcd_files))
    print(f"Found {len(img_files)} images and {len(pcd_files)} PCD files")
    print(f"Processing up to {num_pairs} pairs")

    if num_pairs == 0:
        print("No image-PCD pairs found!")
        return
    
    # 读取初始外参
    extrinsic_any = get_extrinsic(extrinsic_path)
    if extrinsic_any is None:
        print("Failed to load extrinsic!")
        return
    
    if len(extrinsic_any) not in [12, 16]:
        print(f"Extrinsic must have 12 (3x4) or 16 (4x4) floats, got {len(extrinsic_any)}")
        return
    
    want16 = (len(extrinsic_any) == 16)

    # 拆分为 R0, t0（基准）
    R0, t0 = vec_to_rt(extrinsic_any)

    window_name = "Lidar->Image Projection (interactive extrinsic tuning)"
    window_created = False

    # 预加载图像和点云，避免在图像切换时频繁读盘导致卡顿
    data_pairs = []
    for idx in range(num_pairs):
        img_path = img_files[idx]
        pcd_path = pcd_files[idx]

        src_img = cv2.imread(img_path)
        if src_img is None:
            print(f"Skip {Path(img_path).name}: failed to load image")
            continue

        pointcloud = load_pointcloud_from_path(pcd_path)
        if len(pointcloud) == 0:
            print(f"Skip {Path(pcd_path).name}: empty point cloud")
            continue

        depth_range = compute_depth_range(pointcloud)
        data_pairs.append((img_path, src_img, pcd_path, pointcloud, depth_range))

    if not data_pairs:
        print("No valid image-PCD pairs after loading")
        return

    total_pairs = len(data_pairs)

    # 遍历处理每对图像和点云
    for i, (input_photo_path, src_img, input_pcd_path, pointcloud, depth_range) in enumerate(data_pairs, start=1):
        print(f"\nProcessing image {i}/{total_pairs}: {Path(input_photo_path).name} + {Path(input_pcd_path).name}")

        # 当前真状态
        R_cur = R0.copy()
        t_cur = t0.copy()
        
        # HUD 显示的累计角度
        cum_pitch_x = 0.0
        cum_roll_y = 0.0
        cum_yaw_z = 0.0
        
        print("Controls:")
        print("  u/j : + / - Pitch (around X axis)")
        print("  n/m : - / + Roll  (around Y axis)")
        print("  h/k : - / + Yaw   (around Z axis)")
        print("  w/s : + / - translate X (mm)")
        print("  a/d : - / + translate Y (mm)")
        print("  z/x : - / + translate Z (mm)")
        print("  r   : reset to initial extrinsic")
        print("  p   : print current extrinsic (always shows 4x4)")
        print("  SPACE/ENTER: next image")
        print("  q/ESC: quit and save")
        
        while True:
            # 渲染
            canvas = render_projection(src_img, pointcloud, depth_range, R_cur, t_cur, projector)
            draw_hud(canvas, cum_pitch_x, cum_roll_y, cum_yaw_z, t_cur)
            
            # 自适应调整图像大小以适应屏幕
            display_img, scale = resize_to_fit_screen(canvas, max_scale=0.9)
            
            # 确保窗口存在并更新标题（复用同一个窗口避免频繁销毁）
            if not window_created:
                try:
                    # 使用 WINDOW_NORMAL 允许调整窗口大小
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    # 设置窗口大小为显示图像的大小
                    h, w = display_img.shape[:2]
                    cv2.resizeWindow(window_name, w, h)
                except Exception:
                    pass
                window_created = True

            title_suffix = f"{Path(input_photo_path).name}"
            if hasattr(cv2, "setWindowTitle"):
                try:
                    cv2.setWindowTitle(window_name, f"{window_name} - {title_suffix}")
                except Exception:
                    pass

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(0) & 0xFF
            
            changed = False
            
            # 下一张图片
            if key == 32 or key == 13:  # SPACE or ENTER
                print("Moving to next image...")
                break
            
            # 退出程序
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or q
                print("Quitting...")
                cv2.destroyAllWindows()
                return
            
            # 重置
            if key == ord('r') or key == ord('R'):
                R_cur = R0.copy()
                t_cur = t0.copy()
                cum_pitch_x = cum_roll_y = cum_yaw_z = 0.0
                print_state(cum_pitch_x, cum_roll_y, cum_yaw_z, t_cur)
                continue
            
            # 打印
            if key == ord('p') or key == ord('P'):
                print("extrinsic (4x4 print)")
                print(f"{R_cur[0,0]:9.6f} {R_cur[0,1]:9.6f} {R_cur[0,2]:9.6f} {t_cur[0]:9.6f}")
                print(f"{R_cur[1,0]:9.6f} {R_cur[1,1]:9.6f} {R_cur[1,2]:9.6f} {t_cur[1]:9.6f}")
                print(f"{R_cur[2,0]:9.6f} {R_cur[2,1]:9.6f} {R_cur[2,2]:9.6f} {t_cur[2]:9.6f}")
                print("0 0 0 1")
                continue
            
            # ===== 旋转 =====
            # u/j: +/- pitch (绕 X)
            if key == ord('u') or key == ord('U'):
                R_cur = rx_deg(DEG_STEP) @ R_cur
                R_cur = orthonormalize(R_cur)
                cum_pitch_x += DEG_STEP
                changed = True
            if key == ord('j') or key == ord('J'):
                R_cur = rx_deg(-DEG_STEP) @ R_cur
                R_cur = orthonormalize(R_cur)
                cum_pitch_x -= DEG_STEP
                changed = True
            
            # n/m: -/+ roll (绕 Y)
            if key == ord('n') or key == ord('N'):
                R_cur = ry_deg(-DEG_STEP) @ R_cur
                R_cur = orthonormalize(R_cur)
                cum_roll_y -= DEG_STEP
                changed = True
            if key == ord('m') or key == ord('M'):
                R_cur = ry_deg(DEG_STEP) @ R_cur
                R_cur = orthonormalize(R_cur)
                cum_roll_y += DEG_STEP
                changed = True
            
            # h/k: -/+ yaw (绕 Z)
            if key == ord('h') or key == ord('H'):
                R_cur = rz_deg(-DEG_STEP) @ R_cur
                R_cur = orthonormalize(R_cur)
                cum_yaw_z -= DEG_STEP
                changed = True
            if key == ord('k') or key == ord('K'):
                R_cur = rz_deg(DEG_STEP) @ R_cur
                R_cur = orthonormalize(R_cur)
                cum_yaw_z += DEG_STEP
                changed = True
            
            # ===== 平移 =====
            if key == ord('w') or key == ord('W'):
                t_cur[0] += TRANS_STEP_MM
                changed = True
            if key == ord('s') or key == ord('S'):
                t_cur[0] -= TRANS_STEP_MM
                changed = True
            
            if key == ord('a') or key == ord('A'):
                t_cur[1] -= TRANS_STEP_MM
                changed = True
            if key == ord('d') or key == ord('D'):
                t_cur[1] += TRANS_STEP_MM
                changed = True
            
            if key == ord('z') or key == ord('Z'):
                t_cur[2] -= TRANS_STEP_MM
                changed = True
            if key == ord('x') or key == ord('X'):
                t_cur[2] += TRANS_STEP_MM
                changed = True
            
            if changed:
                print_state(cum_pitch_x, cum_roll_y, cum_yaw_z, t_cur)
    
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
