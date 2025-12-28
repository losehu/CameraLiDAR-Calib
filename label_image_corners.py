import os
import re
import csv
import cv2
import glob
import numpy as np
import argparse
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"


def load_config(path: Path) -> dict:
    """Load YAML configuration shared across tools."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        print(f"Failed to parse YAML config: {exc}")
        return {}

    if not isinstance(data, dict):
        print("Config file must contain key/value pairs; ignoring config contents.")
        return {}

    redirect = data.get("config_path")
    if redirect:
        nested_path = (path.parent / str(redirect)).resolve()
        try:
            with open(nested_path, "r", encoding="utf-8") as stream:
                nested = yaml.safe_load(stream)
        except FileNotFoundError:
            print(f"Nested config file not found: {nested_path}")
            return {}
        except yaml.YAMLError as exc:
            print(f"Failed to parse nested YAML config: {exc}")
            return {}
        if not isinstance(nested, dict):
            print("Nested config file must contain key/value pairs; ignoring config contents.")
            return {}
        return nested

    return data


CONFIG = load_config(CONFIG_PATH)


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


# ===== Configuration =====
DEFAULT_FOLDER = str(CONFIG.get(
    "image_dir",
    '/Users/losehu/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_y3fi4f9tcu0322_11eb/msg/file/2025-11/911/image',
))  # Default image folder (can be overridden by CLI)
OUTPUT_FILE = str(CONFIG.get("photo_out", 'cam_point.txt'))  # Output filename
OUTPUT_PATH = Path(OUTPUT_FILE)
MAX_POINTS = 4
ZOOM_STEP = 1.25
MIN_ZOOM = 0.2
MAX_ZOOM = 20.0
PAN_STEP_DISPLAY = 50
FIXED_WINDOW = True          # True: use fixed window size (smoother); False: use original image size
WINDOW_W, WINDOW_H = 1600, 900

# OpenCV 优化
cv2.setUseOptimized(True)


def reset_output_file() -> None:
    """Clear the destination text file before annotation starts."""
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"Cleared previous annotations in {OUTPUT_FILE}.")
    except OSError as exc:
        print(f"Failed to reset output file {OUTPUT_FILE}: {exc}")


# ===== 工具函数 =====
def natsort_key(s: str):
    """Natural sort key: splits digits and non-digits."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def save_all_points(show_summary: bool = True) -> None:
    """Save all annotated points to the output file."""
    if not points_per_image:
        if show_summary:
            print("No annotations to save.")
        return

    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if show_summary:
            print(f"Failed to ensure output directory {OUTPUT_PATH.parent}: {exc}")
        return

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for img_path in images:
            pts = points_per_image.get(img_path, [])
            if pts:  # only save images that have annotations
                name = os.path.basename(img_path)
                f.write(f"{name}\n")
                for point_idx, (x, y) in enumerate(pts):
                    f.write(f"{point_idx}\n")
                    # keep the original spacing format between x and y
                    f.write(f"{x}                {y}\n")
    if show_summary:
        total_pts = sum(len(pts) for pts in points_per_image.values() if pts)
        print(f"\nSaved annotations to: {OUTPUT_FILE}")
        print(f"Total annotated points: {total_pts}")

# —— 打印当前图片的文件名和所有点 ——
def dump_current_points(idx):
    """Print current image name and its annotated points."""
    name = os.path.basename(images[idx])
    pts = points_per_image.get(images[idx], [])
    print(f"{name}")
    for i, point in enumerate(pts):
        print(i)
        print(f"{point[0]}                {point[1]}")
    save_all_points(show_summary=False)
def parse_args():
    p = argparse.ArgumentParser(description='Annotate points on images')
    p.add_argument('-f', '--folder', default=DEFAULT_FOLDER, help='Image folder path (default: %(default)s)')
    return p.parse_args()


args = parse_args()
FOLDER = args.folder

if not os.path.isdir(FOLDER):
    raise RuntimeError(f'Folder not found: {FOLDER}')

reset_output_file()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def list_images(folder):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff','*.webp')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files, key=natsort_key)
    return files

# ===== 全局状态（随当前图片变化）=====
points_per_image = {}   # filename -> [(x,y),...]
images = list_images(FOLDER)
if not images:
    raise RuntimeError(f'No images found in: {FOLDER}')

idx = 0  # 当前图片索引
base = None
H = W = 0

zoom = 1.0
view_x = view_y = 0
scaled_full = None
SW = SH = 0
VIEW_W = VIEW_H = 0
win_name = "Annotator"
def load_image(i):
    """加载第 i 张图片。i==0 时重置为适配窗口+居中；否则沿用上一张的 zoom / view_x / view_y。"""
    global base, H, W, zoom, view_x, view_y, scaled_full, SW, SH, VIEW_W, VIEW_H

    fp = images[i]
    base = cv2.imread(fp)
    if base is None:
        raise FileNotFoundError(f"Failed to read: {fp}")
    H, W = base.shape[:2]

    # 窗口大小
    if FIXED_WINDOW:
        VIEW_W, VIEW_H = WINDOW_W, WINDOW_H
        cv2.resizeWindow(win_name, VIEW_W, VIEW_H)
    else:
        VIEW_W, VIEW_H = W, H
        cv2.resizeWindow(win_name, min(W, 2000), min(H, 1200))  # 防止超大窗口

    if i == 0:
        # —— 第1张：自适应窗口缩放 + 居中显示 ——
        fit_zoom = min(VIEW_W / W, VIEW_H / H)
        zoom = clamp(fit_zoom, MIN_ZOOM, MAX_ZOOM)

        # 清零位移，再按当前 zoom 重建
        view_x, view_y = 0, 0
        rebuild_scaled_full(1.0, zoom, center_from_vxvy=False)

        # 居中视口
        view_w = min(VIEW_W, SW)
        view_h = min(VIEW_H, SH)
        view_x = max(0, (SW - view_w) // 2)
        view_y = max(0, (SH - view_h) // 2)
    else:
        # —— 后续图片：沿用上一张的 zoom / view_x / view_y ——
        zoom = clamp(zoom, MIN_ZOOM, MAX_ZOOM)
        # 不根据旧中心复位，保持现有 view_x/view_y，只重建并自动夹取边界
        rebuild_scaled_full(zoom, zoom, center_from_vxvy=False)

    # 确保存在 points 列表
    if images[i] not in points_per_image:
        points_per_image[images[i]] = []

def rebuild_scaled_full(old_zoom, new_zoom, center_from_vxvy=True):
    """缩放变化时重建 scaled_full，并根据旧中心进行复位。"""
    global scaled_full, SW, SH, view_x, view_y, zoom
    zoom = new_zoom

    SW = max(1, int(round(W * zoom)))
    SH = max(1, int(round(H * zoom)))
    interp = cv2.INTER_LINEAR if zoom >= 1.0 else cv2.INTER_AREA
    scaled_full = cv2.resize(base, (SW, SH), interpolation=interp)

    if center_from_vxvy:
        cx_scaled = view_x + min(VIEW_W, SW) // 2
        cy_scaled = view_y + min(VIEW_H, SH) // 2
        ox = cx_scaled / old_zoom if old_zoom != 0 else 0
        oy = cy_scaled / old_zoom if old_zoom != 0 else 0
        cx_new = int(round(ox * zoom))
        cy_new = int(round(oy * zoom))
        view_x = cx_new - min(VIEW_W, SW) // 2
        view_y = cy_new - min(VIEW_H, SH) // 2

    view_w = min(VIEW_W, SW)
    view_h = min(VIEW_H, SH)
    view_x = clamp(view_x, 0, max(0, SW - view_w))
    view_y = clamp(view_y, 0, max(0, SH - view_h))

def redraw():
    """裁剪 ROI，绘制点和 HUD。"""
    view_w = min(VIEW_W, SW)
    view_h = min(VIEW_H, SH)
    vx = clamp(view_x, 0, max(0, SW - view_w))
    vy = clamp(view_y, 0, max(0, SH - view_h))

    roi = scaled_full[vy:vy+view_h, vx:vx+view_h if False else vx+view_w]  # 防御性写法
    img_view = roi.copy()

    # 画点（原图 -> 缩放后 -> 视图）
    pts = points_per_image[images[idx]]
    for idx2,(ox, oy) in enumerate(pts):
        sx = int(round(ox * zoom))
        sy = int(round(oy * zoom))
        dx = sx - vx
        dy = sy - vy
        if 0 <= dx < view_w and 0 <= dy < view_h:
            if idx2 == 0:
                cv2.circle(img_view, (dx, dy), 3, (0, 0, 255), -1)
            elif idx2 == 1:
                cv2.circle(img_view, (dx, dy), 3, (0, 255, 0), -1)
            elif idx2 == 2:
                cv2.circle(img_view, (dx, dy), 3, (255, 0, 0), -1)
            elif idx2 == 3:
                cv2.circle(img_view, (dx, dy), 3, (255, 255, 0), -1)

    # HUD
    name = os.path.basename(images[idx])
    txt1 = f"[{idx+1}/{len(images)}] {name}"
    txt2 = f"{len(pts)}/{MAX_POINTS} pts | zoom:{zoom:.2f} | b:undo r:reset  j/h:zoom  WSAD:pan  n/p:next/prev  Enter:finish  q/ESC:quit"
    cv2.putText(img_view, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img_view, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(img_view, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_view, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow(win_name, img_view)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 视图 -> 缩放后 -> 原图
        sx = view_x + x
        sy = view_y + y
        ox = int(round(sx / zoom))
        oy = int(round(sy / zoom))
        ox = clamp(ox, 0, W-1)
        oy = clamp(oy, 0, H-1)
        pts = points_per_image[images[idx]]
        if len(pts) < MAX_POINTS:
            pts.append((ox, oy))
            redraw()



# ===== 主程序 =====

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
if FIXED_WINDOW:
    cv2.resizeWindow(win_name, WINDOW_W, WINDOW_H)

load_image(idx)

redraw()
cv2.setMouseCallback(win_name, mouse_callback)
clear()

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):
        pts = points_per_image[images[idx]]
        if pts:
            pts.pop()
        redraw()

    elif key == ord('r'):
        points_per_image[images[idx]] = []
        redraw()

    elif key == ord('j'):
        old = zoom
        new = clamp(zoom * ZOOM_STEP, MIN_ZOOM, MAX_ZOOM)
        if abs(new - old) > 1e-6:
            rebuild_scaled_full(old, new, center_from_vxvy=True)
        redraw()

    elif key == ord('h'):
        old = zoom
        new = clamp(zoom / ZOOM_STEP, MIN_ZOOM, MAX_ZOOM)
        if abs(new - old) > 1e-6:
            rebuild_scaled_full(old, new, center_from_vxvy=True)
        redraw()

    elif key in (ord('w'), ord('W')):
        view_y = clamp(view_y - PAN_STEP_DISPLAY, 0, max(0, SH - min(VIEW_H, SH)))
        redraw()
    elif key in (ord('s'), ord('S')):
        view_y = clamp(view_y + PAN_STEP_DISPLAY, 0, max(0, SH - min(VIEW_H, SH)))
        redraw()
   
    elif key in (ord('a'), ord('A')):
        view_x = clamp(view_x - PAN_STEP_DISPLAY, 0, max(0, SW - min(VIEW_W, SW)))
        redraw()
    elif key in (ord('d'), ord('D')):
        view_x = clamp(view_x + PAN_STEP_DISPLAY, 0, max(0, SW - min(VIEW_W, SW)))
        redraw()

    elif key == ord('n'):  # 下一张
        if idx < len(images) - 1:
            idx += 1
            load_image(idx)
            redraw()

    elif key == ord('p'):  # 上一张
        if idx > 0:
            idx -= 1
            load_image(idx)
            redraw()

    elif key in (13, 10):  # Enter：先输出，再切到下一张（若有）/或结束
        dump_current_points(idx)
        if idx < len(images) - 1:
            idx += 1
            load_image(idx)
            redraw()
        else:
            break

    elif key in (27, ord('q')):  # q/ESC：先输出，再退出
        dump_current_points(idx)
        break

# 保存所有标注结果
save_all_points()
cv2.destroyAllWindows()
