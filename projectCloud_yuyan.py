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

# ====== 全局参数 ======
POINT_ALPHA = 0.60  # 雷达点的不透明度（0=全透明，1=不透明）
THRESHOLD_LIDAR = 3000
DEG_STEP = 1.0  # 每次角度改变量（度）
TRANS_STEP_MM = 100.0  # 每次平移改变量（毫米）
VOXEL_DOWNSAMPLE_M = 0.03  # 体素降采样尺寸（米），0 表示不降采样
MAX_POINTS_PER_PCD = 150000  # 每帧点数上限，0 表示不限制


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


# ============ 投影函数（基于 getTheoreticalUV_yuyan） ============
def get_theoretical_uv_yuyan(ext, x_mm, y_mm, z_mm):
    """
    将雷达点 (x, y, z) 毫米坐标投影到全景图像 UV
    对应 C++ 中的 getTheoreticalUV_yuyan 函数
    使用等距圆柱投影（经纬度映射）
    """
    # 构建外参矩阵 3x4
    matrix2 = np.array([
        [ext[0], ext[1], ext[2], ext[3]],
        [ext[4], ext[5], ext[6], ext[7]],
        [ext[8], ext[9], ext[10], ext[11]]
    ], dtype=np.float64)
    
    # 齐次坐标
    coordinate = np.array([x_mm, y_mm, z_mm, 1.0], dtype=np.float64)
    
    # 变换到相机坐标系
    result = matrix2 @ coordinate
    
    u = result[0]
    v = result[1]
    depth = result[2]
    
    # 归一化
    n = np.sqrt(u * u + v * v + depth * depth)
    if n > 0:
        u /= n
        v /= n
        depth /= n
    
    # 计算经纬度
    # 经度：从 +Z 轴（前方）开始，沿 +Z→+X 为正
    lon = np.arctan2(v, u)  # [-π, π]
    lat = np.arctan2(depth, np.sqrt(u * u + v * v))  # [-π/2, π/2]
    
    # 转为像素坐标（全景图尺寸：5188 x 2594）
    uv_0 = (np.pi - lon) * 5188 / (2 * np.pi)
    uv_1 = (np.pi / 2 - lat) * 2594 / np.pi
    
    return uv_0, uv_1


# ============ 深度范围计算 ============
def compute_depth_range(pointcloud):
    """计算点云的深度范围"""
    if len(pointcloud) == 0:
        return 0.0, 1.0
    
    depths = np.linalg.norm(pointcloud, axis=1)
    dmin = np.min(depths)
    dmax = np.max(depths)
    
    if dmax <= dmin:
        dmax = dmin + 1e-6
    
    return dmin, dmax


# ============ 渲染投影 ============
def render_projection(base_img, pointcloud, depth_range, extrinsic, threshold_lidar):
    """
    将点云投影到图像上
    """
    overlay = base_img.copy()
    dmin, dmax = depth_range
    
    for pt in pointcloud:
        x, y, z = pt[0], pt[1], pt[2]
        depth = np.sqrt(x**2 + y**2 + z**2)
        
        # 深度归一化
        t = np.arctan(((depth - dmin) / (dmax - dmin)) * 10)
        t = np.clip(t, 0.0, 1.0)
        
        # 投影到图像
        u, v = get_theoretical_uv_yuyan(extrinsic, x * 1000, y * 1000, z * 1000)
        
        u_int, v_int = int(round(u)), int(round(v))
        
        # 检查是否在图像范围内
        if u_int < 0 or u_int >= overlay.shape[1] or v_int < 0 or v_int >= overlay.shape[0]:
            continue
        
        # 颜色编码（深度）
        hue = (1.0 - t) * 270.0
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        
        # 画点
        cv2.circle(overlay, (u_int, v_int), 5, (b, g, r), -1, cv2.LINE_AA)
    
    # 混合
    out_img = cv2.addWeighted(overlay, POINT_ALPHA, base_img, 1.0 - POINT_ALPHA, 0.0)
    
    return out_img


# ============ HUD 显示 ============
def draw_hud(img, pitch_deg, roll_deg, yaw_deg, t_mm):
    """在图像上绘制 HUD 信息"""
    text1 = f"Pitch(x): {pitch_deg:.3f} deg | Roll(y): {roll_deg:.3f} deg | Yaw(z): {yaw_deg:.3f} deg"
    text2 = f"t (mm): x={t_mm[0]:.2f}  y={t_mm[1]:.2f}  z={t_mm[2]:.2f}"
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

    if config:
        # 支持以下键：image_dir, pcd_dir 或 lidar_dir, extrinsic_out, output_path
        img_dir = str(config.get('image_dir') or '') or None
        pcd_dir = str(config.get('pcd_dir') or config.get('lidar_dir') or '') or None
        extrinsic_path = str(config.get('extrinsic_out') or '') or None

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
    # 获取图片列表
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
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
            ext_now = rt_to_vec(R_cur, t_cur, want16)
            canvas = render_projection(src_img, pointcloud, depth_range, ext_now, THRESHOLD_LIDAR)
            draw_hud(canvas, cum_pitch_x, cum_roll_y, cum_yaw_z, t_cur)
            
            # 确保窗口存在并更新标题（复用同一个窗口避免频繁销毁）
            if not window_created:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                except Exception:
                    pass
                window_created = True

            title_suffix = f"{Path(input_photo_path).name}"
            if hasattr(cv2, "setWindowTitle"):
                try:
                    cv2.setWindowTitle(window_name, f"{window_name} - {title_suffix}")
                except Exception:
                    pass

            cv2.imshow(window_name, canvas)
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
