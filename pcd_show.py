#!/usr/bin/env python3
"""Interactive LiDAR marker tool rewritten with Open3D GUI.

This script mirrors the behaviour of the legacy PCL-based tool:
- Load single pcd file or a directory of pcd files.
- Skip clouds that already have measurements in the output txt.
- Overlay LiDAR reference spheres pulled from an existing txt file.
- Hold Shift and left-click to pick up to four points (stored in millimetres).
- b deletes the last picked point; Enter saves/overwrites the record and advances.
- n / N switch forward/backward between clouds.
- p stores the current camera pose to saved_cam.cam; l reloads the pose.

Requirements: Open3D >= 0.17 (GUI module) and NumPy.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering
import yaml

# ---------- constants & globals ----------

MAX_POINTS = 4
CONFIG_NAME = "config/config.yaml"
OUTPUT_FILE = ""
CAM_PATH = "saved_cam.cam"
PICKED_RADIUS_MM = 60.0
LIDAR_RADIUS_MM = 100.0
MAX_PICK_DISTANCE_M = 0.5

POINT_SHADER = "defaultUnlit"
SPHERE_SHADER_SOLID = "defaultLit"
SPHERE_SHADER_ALPHA = "defaultLitTransparency"

LIDAR_COLOR_MAP = {
    0: (1.0, 1.0, 0.0),
    1: (1.0, 0.0, 0.0),
    2: (0.0, 1.0, 0.0),
    3: (0.0, 0.0, 1.0),
}

PICK_COLOR_MAP = {
    0: (1.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
    2: (0.0, 0.0, 1.0),
    3: (1.0, 1.0, 0.0),
}


def load_config(path: Path) -> Dict[str, object]:
    """Load shared YAML configuration values."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config file must contain key/value pairs")

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
            raise ValueError("Nested config file must contain key/value pairs")
        return nested

    return data


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def parse_record_header(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("# "):
        stripped = stripped[2:].strip()
    if stripped.lower().endswith(".pcd"):
        return stripped
    return None


def silence_framework_logs() -> None:
    """Silence Apple unified logging to reduce console noise."""
    if sys.platform == "darwin":
        os.environ["OS_ACTIVITY_MODE"] = "disable"


def ensure_numpy(vec: Tuple[float, float, float]) -> np.ndarray:
    return np.asarray(vec, dtype=np.float64)


def create_sphere(radius_m: float, color: Tuple[float, float, float], opacity: float) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius_m)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    if opacity < 1.0:
        mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(mesh.triangles))
    return mesh


def sphere_material(color: Tuple[float, float, float], opacity: float) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = SPHERE_SHADER_ALPHA if opacity < 1.0 else SPHERE_SHADER_SOLID
    mat.base_color = (*color, opacity)
    mat.base_metallic = 0.0
    mat.base_roughness = 0.4
    mat.transmission = 0.0
    return mat


def point_material(point_size: float = 1.0) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = POINT_SHADER
    mat.point_size = point_size
    mat.line_width = 1.0
    return mat


def hsv_to_rgb_array(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = (h % 360.0) / 60.0
    c = v * s
    x = c * (1.0 - np.abs(h % 2 - 1.0))
    m = v - c

    r = np.zeros_like(c)
    g = np.zeros_like(c)
    b = np.zeros_like(c)

    mask = (0 <= h) & (h < 1)
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0

    mask = (1 <= h) & (h < 2)
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0

    mask = (2 <= h) & (h < 3)
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]

    mask = (3 <= h) & (h < 4)
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]

    mask = (4 <= h) & (h < 5)
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]

    mask = (5 <= h) & (h < 6)
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]

    r += m
    g += m
    b += m

    return np.column_stack((r, g, b))


def colorize_by_distance(cloud: o3d.geometry.PointCloud) -> None:
    if len(cloud.points) == 0:
        return

    points = np.asarray(cloud.points)
    dists = np.linalg.norm(points, axis=1)
    d_min = float(np.min(dists))
    d_max = float(np.max(dists))

    if not math.isfinite(d_min) or not math.isfinite(d_max) or math.isclose(d_min, d_max):
        cloud.colors = o3d.utility.Vector3dVector(np.full_like(points, 0.5))
        return

    threshold = 300.0
    cycle_count = 1.0  # number of rainbow cycles within the threshold distance
    norm = np.clip(dists, 0.0, threshold) / max(threshold, 1e-6)
    norm = np.power(norm, 0.8)  # bias toward near distances for stronger contrasts
    hues = (240.0 - norm * cycle_count * 360.0) % 360.0
    s = np.ones_like(hues)
    v = np.ones_like(hues)
    colors = hsv_to_rgb_array(hues, s, v)

    far_mask = dists > threshold
    if np.any(far_mask):
        warm_rgb = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        colors[far_mask] = warm_rgb

    cloud.colors = o3d.utility.Vector3dVector(colors)


def extract_need_id_from_path(pcd_path: str) -> Tuple[bool, int]:
    stem = Path(pcd_path).stem
    match = re.search(r"(?:pcd)?[_\- ]*(\d+)", stem, re.IGNORECASE)
    if match:
        return True, int(match.group(1))
    return False, 0


def pcd_result_exists(pcd_path: str) -> bool:
    target_name = Path(pcd_path).name
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as inp:
            for line in inp:
                header = parse_record_header(line)
                if header == target_name:
                    return True
    except OSError:
        return False
    return False


def read_lidar_markers(need_id: int) -> List[Tuple[float, float, float]]:
    markers: List[Tuple[float, float, float]] = []
    target_block = need_id - 1
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as infile:
            lines = [line.strip() for line in infile if line.strip()]
    except OSError:
        return markers

    current_block = -1
    idx_in_block = 0
    for line in lines:
        header = parse_record_header(line)
        if header is not None:
            current_block += 1
            idx_in_block = 0
            continue
        if current_block != target_block:
            idx_in_block += 1
            continue
        if idx_in_block % 2 == 1:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, z = map(float, parts[:3])
                    markers.append((x, y, z))
                except ValueError:
                    pass
        idx_in_block += 1
    return markers


def overwrite_pcd_results(pcd_path: str, picked_pts_mm: List[Tuple[float, float, float]]) -> None:
    if not picked_pts_mm:
        return

    pcd_filename = Path(pcd_path).name

    try:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
            out.write(pcd_filename + "\n")
            for idx, point in enumerate(picked_pts_mm, start=1):
                out.write(f"{idx}\n")
                out.write(f"{point[0]} {point[1]} {point[2]}\n")
    except OSError as exc:
        print(f"Failed to open output file {OUTPUT_FILE}: {exc}")
        return

    print(f"Results for {Path(pcd_path).name} appended to {OUTPUT_FILE}.")


def snapshot_camera_pose(cam: rendering.Camera) -> Optional[Dict[str, np.ndarray]]:
    if cam is None:
        return None

    # Preferred API (Open3D 0.18+)
    if all(hasattr(cam, attr) for attr in ("get_position", "get_forward_vector", "get_up_vector")):
        try:
            pos = ensure_numpy(cam.get_position())
            forward = _normalize(ensure_numpy(cam.get_forward_vector()))
            up = _normalize(ensure_numpy(cam.get_up_vector()))
            return {"position": pos, "forward": forward, "up": up}
        except Exception:
            pass

    # Fallback for versions that only expose the view matrix
    view_getter = getattr(cam, "get_view_matrix", None)
    if callable(view_getter):
        try:
            view = np.asarray(view_getter(), dtype=np.float64)
            if view.shape == (4, 4):
                inv = np.linalg.inv(view)
                pos = inv[:3, 3]
                forward = (inv @ np.array([0.0, 0.0, -1.0, 0.0]))[:3]
                up = (inv @ np.array([0.0, 1.0, 0.0, 0.0]))[:3]
                return {
                    "position": np.asarray(pos, dtype=np.float64),
                    "forward": _normalize(forward),
                    "up": _normalize(up),
                }
        except Exception:
            pass

    return None


def save_camera_to_file(cam: rendering.Camera, window: gui.Window, path: str) -> bool:
    pose = snapshot_camera_pose(cam)
    if pose is None:
        print("Failed to save camera parameters: unable to snapshot camera pose")
        return False

    try:
        data: Dict[str, List[float]] = {
            "position": list(map(float, pose["position"])),
            "forward": list(map(float, pose["forward"])),
            "up": list(map(float, pose["up"])),
            "fov": [float(cam.get_field_of_view())] if hasattr(cam, "get_field_of_view") else [60.0],
            "near": [float(cam.get_near())] if hasattr(cam, "get_near") else [0.1],
            "far": [float(cam.get_far())] if hasattr(cam, "get_far") else [1000.0],
            "window_size": [window.content_rect.width, window.content_rect.height],
        }
        with open(path, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2)
        print(f"Camera parameters saved to: {path}")
        return True
    except (OSError, AttributeError, TypeError, ValueError) as exc:
        print(f"Failed to save camera parameters: {exc}")
        return False


def load_camera_from_file(cam: rendering.Camera, window: gui.Window, path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as inp:
            data = json.load(inp)
    except OSError:
        return False

    try:
        position = ensure_numpy(tuple(data["position"]))
        forward = ensure_numpy(tuple(data["forward"]))
        up = ensure_numpy(tuple(data["up"]))
        target = position + forward
        cam.look_at(target, position, up)
        if "far" in data and hasattr(cam, "set_far"):
            try:
                cam.set_far(float(data["far"][0]))
            except Exception as exc:
                print(f"Failed to set far plane: {exc}")
        if "near" in data and hasattr(cam, "set_near"):
            try:
                cam.set_near(float(data["near"][0]))
            except Exception as exc:
                print(f"Failed to set near plane: {exc}")
        if "fov" in data and hasattr(cam, "set_field_of_view"):
            try:
                cam.set_field_of_view(float(data["fov"][0]))
            except Exception as exc:
                print(f"Failed to set field of view: {exc}")
        w, h = data.get("window_size", [window.content_rect.width, window.content_rect.height])
        window.size = gui.Size(int(w), int(h))
        return True
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        print(f"Failed to load camera parameters: {exc}")
        return False


class PCDLabelWindow:
    def __init__(self, pcd_paths: List[str]):
        self.pcds = pcd_paths
        self.cur_idx = 0
        self.current_cloud = o3d.geometry.PointCloud()
        self.current_cloud_np: Optional[np.ndarray] = None
        self.current_kd_tree: Optional[o3d.geometry.KDTreeFlann] = None
        self.current_path: Optional[str] = None
        self._current_bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = None

        self.picked_ids: List[str] = []
        self.picked_pts_mm: List[Tuple[float, float, float]] = []
        self.lidar_ids: List[str] = []
        self._mouse_down_supported: Optional[bool] = None  # Detect missing BUTTON_DOWN callbacks

        self.app = gui.Application.instance
        self.window = self.app.create_window("PCD Labeler (Open3D)", 1440, 900)
        self.window.set_on_layout(self._on_layout)

        self.panel = gui.Vert()
        if hasattr(self.panel, "spacing"):
            self.panel.spacing = 4
        if hasattr(self.panel, "padding"):
            self.panel.padding = gui.Margins(6, 6, 6, 6)
        self.info_label = gui.Label(
            "Shift+左键点选; b 撤销; Enter 保存并下一帧; n/N 前后切换; p/l 保存/加载相机; r 恢复视角; q 退出"
        )
        self.panel.add_child(self.info_label)
        self.status_label = gui.Label("")
        self.panel.add_child(self.status_label)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_on_mouse(self._on_mouse_event)
        self.scene_widget.scene.set_background([0, 0, 0, 1])
        self.scene_widget.scene.show_skybox(False)

        self._saved_camera_state: Optional[Dict[str, np.ndarray]] = None

        if hasattr(self.window, "set_on_key_event"):
            self.window.set_on_key_event(self._on_key_event)
        else:
            self.scene_widget.set_on_key(self._on_key_event)

        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)

        gui.Application.instance.post_to_main_thread(self.window, self._initialize_scene)

    def _on_layout(self, ctx: gui.LayoutContext) -> None:
        content = self.window.content_rect
        panel_width = min(420, content.width // 3)
        self.panel.frame = gui.Rect(content.get_right() - panel_width, content.y, panel_width, content.height)
        self.scene_widget.frame = gui.Rect(content.x, content.y, content.width - panel_width, content.height)

    def _set_status(self, text: str) -> None:
        self.status_label.text = text

    def _initialize_scene(self) -> None:
        cam = self.scene_widget.scene.camera
        if not load_camera_from_file(cam, self.window, CAM_PATH):
            bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
            self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())
        self._set_status("> Point picking enabled. [n/N] cycle, [b] undo, [Enter] save.")
        self._skip_labeled_pcds()
        self._load_current_pcd()

    def _clear_scene(self) -> None:
        self.scene_widget.scene.clear_geometry()
        self.picked_ids.clear()
        self.picked_pts_mm.clear()
        self.lidar_ids.clear()

    def _add_point_cloud(self, cloud: o3d.geometry.PointCloud) -> None:
        self.scene_widget.scene.add_geometry("cloud", cloud, point_material(1.5))

    def _add_sphere_mm(
        self,
        name: str,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        radius_mm: float,
        color: Tuple[float, float, float],
        opacity: float,
    ) -> None:
        radius_m = radius_mm * 0.001
        sphere = create_sphere(radius_m, color, opacity)
        sphere.translate((x_mm * 0.001, y_mm * 0.001, z_mm * 0.001))
        self.scene_widget.scene.add_geometry(name, sphere, sphere_material(color, opacity))

    def _draw_lidar_markers(self, need_id: int) -> None:
        markers = read_lidar_markers(need_id)
        for idx, (x, y, z) in enumerate(markers):
            color = LIDAR_COLOR_MAP.get(idx, LIDAR_COLOR_MAP[0])
            name = f"lidar_marker_{idx}"
            self._add_sphere_mm(name, x, y, z, LIDAR_RADIUS_MM, color, 0.8)
            self.lidar_ids.append(name)

    def _snapshot_camera(self, cam: rendering.Camera) -> Optional[Dict[str, np.ndarray]]:
        if cam is None:
            return None

        if hasattr(cam, "get_position") and hasattr(cam, "get_forward_vector") and hasattr(cam, "get_up_vector"):
            try:
                pos = ensure_numpy(cam.get_position())
                forward = ensure_numpy(cam.get_forward_vector())
                up = ensure_numpy(cam.get_up_vector())
                center = pos + forward
                return {"eye": pos, "center": center, "up": up}
            except Exception:
                pass

        get_view_matrix = getattr(cam, "get_view_matrix", None)
        if callable(get_view_matrix):
            try:
                view = np.asarray(get_view_matrix(), dtype=np.float64)
                if view.shape == (4, 4):
                    inv = np.linalg.inv(view)
                    pos = inv[:3, 3]
                    forward = (inv @ np.array([0.0, 0.0, -1.0, 0.0]))[:3]
                    up = (inv @ np.array([0.0, 1.0, 0.0, 0.0]))[:3]
                    center = pos + forward
                    return {
                        "eye": np.array(pos, dtype=np.float64),
                        "center": np.array(center, dtype=np.float64),
                        "up": np.array(up, dtype=np.float64),
                    }
            except Exception:
                pass

        return None

    def _reset_camera_to_bbox(self, bbox: o3d.geometry.AxisAlignedBoundingBox) -> None:
        cam = self.scene_widget.scene.camera

        if self._saved_camera_state is not None:
            data = self._saved_camera_state
            eye = data.get("eye")
            center = data.get("center")
            up = data.get("up")
            if eye is not None and center is not None and up is not None:
                cam.look_at(center, eye, up)
        else:
            center = bbox.get_center()
            extent = bbox.get_extent()
            diag = float(np.linalg.norm(extent))
            if not math.isfinite(diag) or diag <= 1e-6:
                diag = 1.0
            self.scene_widget.setup_camera(60.0, bbox, center)

        if hasattr(cam, "get_near") and hasattr(cam, "get_far"):
            try:
                near = float(cam.get_near())
                far = float(cam.get_far())
            except Exception:
                near = 0.1
                far = 1000.0
        else:
            near = 0.1
            far = 1000.0

        if hasattr(cam, "set_near"):
            try:
                cam.set_near(float(max(near, 0.01)))
            except Exception:  # pragma: no cover
                pass
        if hasattr(cam, "set_far"):
            try:
                cam.set_far(float(max(far, near + 1.0)))
            except Exception:  # pragma: no cover
                pass

        app = gui.Application.instance

        def _trigger_redraw() -> None:
            self._saved_camera_state = None
            if hasattr(self.scene_widget, "force_redraw"):
                self.scene_widget.force_redraw()
            elif hasattr(self.window, "post_redraw"):
                self.window.post_redraw()

        app.post_to_main_thread(self.window, _trigger_redraw)

    def _refresh_user_markers(self) -> None:
        for name in list(self.picked_ids):
            self.scene_widget.scene.remove_geometry(name)
        self.picked_ids.clear()
        for idx, (x_mm, y_mm, z_mm) in enumerate(self.picked_pts_mm):
            color = PICK_COLOR_MAP.get(idx, PICK_COLOR_MAP[0])
            name = f"picked_ball{idx + 1}"
            self._add_sphere_mm(name, x_mm, y_mm, z_mm, PICKED_RADIUS_MM, color, 1.0)
            self.picked_ids.append(name)

    def _skip_labeled_pcds(self) -> None:
        return

    def _load_current_pcd(self) -> None:
        if not (0 <= self.cur_idx < len(self.pcds)):
            return
        pcd_path = self.pcds[self.cur_idx]
        self.current_path = pcd_path
        cloud = o3d.io.read_point_cloud(pcd_path)
        if cloud.is_empty():
            print(f"Failed to load or empty cloud: {pcd_path}")
            return

        colorize_by_distance(cloud)
        self.current_cloud = cloud
        self.current_cloud_np = np.asarray(cloud.points)
        self.current_kd_tree = o3d.geometry.KDTreeFlann(cloud)

        self._clear_scene()
        self._add_point_cloud(cloud)

        found, need_id = extract_need_id_from_path(pcd_path)
        if not found:
            print("[id] not found")
            need_id = 1
        self._draw_lidar_markers(need_id)

        bbox = cloud.get_axis_aligned_bounding_box()
        self._current_bbox = bbox
        self._reset_camera_to_bbox(bbox)

        self.info_label.text = (
            f"当前: {Path(pcd_path).name} ({self.cur_idx + 1}/{len(self.pcds)})"
        )
        print(Path(pcd_path).name)

    def _on_key_event(self, event: gui.KeyEvent) -> bool:
        key_enter = getattr(gui.KeyName, "KEY_ENTER", getattr(gui.KeyName, "ENTER", None))
        key_kp_enter = getattr(gui.KeyName, "KEY_KP_ENTER", None)
        key_b = getattr(gui.KeyName, "KEY_B", getattr(gui.KeyName, "B", None))
        key_n = getattr(gui.KeyName, "KEY_N", getattr(gui.KeyName, "N", None))
        key_l = getattr(gui.KeyName, "KEY_L", getattr(gui.KeyName, "L", None))
        key_p = getattr(gui.KeyName, "KEY_P", getattr(gui.KeyName, "P", None))
        key_r = getattr(gui.KeyName, "KEY_R", getattr(gui.KeyName, "R", None))
        key_q = getattr(gui.KeyName, "KEY_Q", getattr(gui.KeyName, "Q", None))

        if event.type == gui.KeyEvent.Type.DOWN:
            if self.scene_widget.scene is not None:
                cam = self.scene_widget.scene.camera
                self._saved_camera_state = self._snapshot_camera(cam)
            if event.is_repeat:
                return False
            if key_p is not None and event.key == key_p:
                cam = self.scene_widget.scene.camera
                save_camera_to_file(cam, self.window, CAM_PATH)
                return True
            if key_l is not None and event.key == key_l:
                cam = self.scene_widget.scene.camera
                if not load_camera_from_file(cam, self.window, CAM_PATH):
                    print("Failed to load camera parameters from file")
                return True
            if key_r is not None and event.key == key_r:
                if self._current_bbox is not None:
                    self._saved_camera_state = None
                    self._reset_camera_to_bbox(self._current_bbox)
                return True
            if key_q is not None and event.key == key_q:
                gui.Application.instance.quit()
                return True
            if key_b is not None and event.key == key_b:
                if self.picked_pts_mm:
                    self.picked_pts_mm.pop()
                    self._refresh_user_markers()
                return True
            if (key_enter is not None and event.key == key_enter) or (
                key_kp_enter is not None and event.key == key_kp_enter
            ):
                if self.current_path:
                    overwrite_pcd_results(self.current_path, self.picked_pts_mm)
                if self.cur_idx + 1 < len(self.pcds):
                    self.cur_idx = (self.cur_idx + 1) % len(self.pcds)
                    self._skip_labeled_pcds()
                    self._load_current_pcd()
                else:
                    gui.Application.instance.quit()
                return True
            if key_n is not None and event.key == key_n:
                shift_mask = int(gui.KeyModifier.SHIFT) if hasattr(gui.KeyModifier, "SHIFT") else 0
                modifiers = int(event.modifiers) if hasattr(event, "modifiers") else 0
                if modifiers & shift_mask:
                    if self.pcds:
                        self.cur_idx = (self.cur_idx - 1 + len(self.pcds)) % len(self.pcds)
                        self._load_current_pcd()
                else:
                    if self.pcds:
                        self.cur_idx = (self.cur_idx + 1) % len(self.pcds)
                        self._load_current_pcd()
                return True
        return False

    def _pick_point(self, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        if self.current_kd_tree is None or self.current_cloud_np is None:
            return None

        frame = self.scene_widget.frame
        local_x = int(round(x - frame.x))
        local_y = int(round(y - frame.y))
        if frame.width <= 0 or frame.height <= 0:
            return None
        local_x = max(0, min(local_x, frame.width - 1))
        local_y = max(0, min(local_y, frame.height - 1))


        scene_widget_scene = self.scene_widget.scene
        scene = scene_widget_scene.scene

        pick_options_type = getattr(rendering.Scene, "PickOptions", None)
        pick_method = getattr(scene, "pick", None)
        pick_point_method = getattr(scene, "pick_point", None)

        hit_point = None

        if pick_options_type is not None and callable(pick_method):
            options = pick_options_type()
            if hasattr(options, "max_distance"):
                options.max_distance = MAX_PICK_DISTANCE_M
            if hasattr(options, "pick_point_cloud"):
                options.pick_point_cloud = True
            result = pick_method(options, local_x, local_y)
            if getattr(result, "is_hittable", False):
                hit_point = np.asarray(result.point)
        elif callable(pick_point_method):
            picked = pick_point_method(local_x, local_y)
            if picked is not None:
                hit_point = np.asarray(picked)

        if hit_point is None:
            hit_point = self._pick_point_by_projection(local_x, local_y, frame)

        if hit_point is None:
            hit_point = self._pick_point_by_ray(local_x, local_y, frame)

        if hit_point is None:
            return None

        _, idxs, _ = self.current_kd_tree.search_knn_vector_3d(hit_point, 1)
        if not idxs:
            return None
        point = self.current_cloud_np[idxs[0]]
        return float(point[0]), float(point[1]), float(point[2])

    def _pick_point_by_projection(self, local_x: int, local_y: int, frame: gui.Rect) -> Optional[np.ndarray]:
        if self.current_cloud_np is None or self.current_cloud_np.size == 0:
            return None

        cam = self.scene_widget.scene.camera
        convert = getattr(cam, "convert_to_pinhole_camera_parameters", None)
        if convert is None:
            return None

        params = convert()
        intrinsic = np.asarray(params.intrinsic.intrinsic_matrix)
        width = getattr(params.intrinsic, "width", self.scene_widget.frame.width)
        height = getattr(params.intrinsic, "height", self.scene_widget.frame.height)
        extrinsic = np.asarray(params.extrinsic)

        if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
            return None

        rel_x = min(max(local_x, 0), frame.width)
        rel_y = min(max(local_y, 0), frame.height)

        scale_x = width / max(frame.width, 1)
        scale_y = height / max(frame.height, 1)
        target_x = rel_x * scale_x
        target_y = rel_y * scale_y


        pts = self.current_cloud_np
        pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
        cam_pts = (extrinsic @ pts_h.T).T
        z = cam_pts[:, 2]
        mask = z > 0.0
        if not np.any(mask):
            return None
        cam_pts = cam_pts[mask]
        z = z[mask]
        pixels = (intrinsic @ cam_pts[:, :3].T).T
        px = pixels[:, 0] / z
        py = pixels[:, 1] / z

        diff_x = px - target_x
        diff_y = py - target_y
        dist_sq = diff_x * diff_x + diff_y * diff_y
        min_idx = int(np.argmin(dist_sq))

        min_dist = math.sqrt(dist_sq[min_idx]) if dist_sq.size else float("inf")
        if math.isinf(min_dist):
            return None

        selected_idx = np.nonzero(mask)[0][min_idx]
        return self.current_cloud_np[selected_idx]

    def _pick_point_by_ray(self, local_x: int, local_y: int, frame: gui.Rect) -> Optional[np.ndarray]:
        if self.current_cloud_np is None or self.current_cloud_np.size == 0:
            return None

        width = max(frame.width, 1)
        height = max(frame.height, 1)

        px = (local_x + 0.5) / width
        py = (local_y + 0.5) / height
        norm_x = px * 2.0 - 1.0
        norm_y = 1.0 - py * 2.0

        cam = self.scene_widget.scene.camera
        try:
            position = ensure_numpy(cam.get_position())
            forward = ensure_numpy(cam.get_forward_vector())
            up = ensure_numpy(cam.get_up_vector())
        except AttributeError:
            position = None
            forward = None
            up = None

        if position is None or forward is None or up is None:
            get_view_matrix = getattr(cam, "get_view_matrix", None)
            if callable(get_view_matrix):
                view_matrix = np.asarray(get_view_matrix(), dtype=np.float64)
                if view_matrix.shape == (4, 4):
                    try:
                        cam_to_world = np.linalg.inv(view_matrix)
                    except np.linalg.LinAlgError:
                        return None
                    position = np.array(cam_to_world[:3, 3], dtype=np.float64)
                    forward = np.array(
                        (cam_to_world @ np.array([0.0, 0.0, -1.0, 0.0]))[:3], dtype=np.float64
                    )
                    up = np.array(
                        (cam_to_world @ np.array([0.0, 1.0, 0.0, 0.0]))[:3], dtype=np.float64
                    )
                else:
                    return None
            else:
                return None

        try:
            fov_deg = float(cam.get_field_of_view())
        except AttributeError:
            fov_deg = 60.0

        if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(up) < 1e-9:
            return None

        forward = forward / np.linalg.norm(forward)
        up = up / np.linalg.norm(up)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-9:
            return None
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        fov_rad = math.radians(max(min(fov_deg, 179.0), 1.0))
        aspect = width / height
        dx = norm_x * math.tan(fov_rad / 2.0) * aspect
        dy = norm_y * math.tan(fov_rad / 2.0)
        ray_dir = forward + dx * right + dy * up
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-9:
            return None
        ray_dir /= norm

        pts = self.current_cloud_np
        vecs = pts - position
        with np.errstate(all="ignore"):
            proj = vecs @ ray_dir
        proj = np.where(np.isfinite(proj), proj, 0.0)
        mask = proj > 0.0
        if not np.any(mask):
            return None

        vecs = vecs[mask]
        proj = proj[mask]
        perp = vecs - np.outer(proj, ray_dir)
        dist_sq = np.sum(perp * perp, axis=1)
        min_idx = int(np.argmin(dist_sq))
        min_dist = math.sqrt(dist_sq[min_idx]) if dist_sq.size else float("inf")
        if min_dist > MAX_PICK_DISTANCE_M:
            return None

        selected_indices = np.nonzero(mask)[0]
        return pts[selected_indices[min_idx]]

    def _on_mouse_event(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        if hasattr(event, "is_shift_down"):
            shift_held = event.is_shift_down
        else:
            modifiers = event.modifiers if hasattr(event, "modifiers") else 0
            shift_mask = int(gui.KeyModifier.SHIFT) if hasattr(gui.KeyModifier, "SHIFT") else 0
            shift_held = bool(int(modifiers) & shift_mask)

        button = getattr(event, "button", None)
        buttons_mask = getattr(event, "buttons", 0)

        left_button = None
        for candidate in ("LEFT", "BUTTON_PRIMARY", "PRIMARY"):
            lb = getattr(gui.MouseButton, candidate, None)
            if lb is not None:
                left_button = lb
                break

        left_pressed = False
        if button is not None:
            if left_button is not None:
                left_pressed = button == left_button
            else:
                left_pressed = bool(button)
        else:
            if left_button is not None:
                left_pressed = bool(int(buttons_mask) & int(left_button))
            else:
                left_pressed = bool(buttons_mask)
        down_types = {
            getattr(gui.MouseEvent.Type, name, None)
            for name in ("BUTTON_DOWN", "LEFT_BUTTON_DOWN")
        }
        down_types.discard(None)
        up_types = {
            getattr(gui.MouseEvent.Type, name, None)
            for name in ("BUTTON_UP", "LEFT_BUTTON_UP")
        }
        up_types.discard(None)

        if self._mouse_down_supported is None:
            if event.type in down_types:
                self._mouse_down_supported = True
            elif event.type in up_types:
                self._mouse_down_supported = False

        should_trigger = event.type in down_types
        if not should_trigger and event.type in up_types:
            fallback_allowed = (self._mouse_down_supported is False) or not down_types
            if fallback_allowed:
                should_trigger = True

        if should_trigger and shift_held and (left_pressed or event.type in up_types):
            if len(self.picked_pts_mm) >= MAX_POINTS:
                return gui.Widget.EventCallbackResult.CONSUMED
            point_m = self._pick_point(event.x, event.y)
            if point_m is None:
                return gui.Widget.EventCallbackResult.CONSUMED
            x_mm = point_m[0] * 1000.0
            y_mm = point_m[1] * 1000.0
            z_mm = point_m[2] * 1000.0
            self.picked_pts_mm.append((x_mm, y_mm, z_mm))
            self._refresh_user_markers()
            print(f"[Pick] ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm")
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED


def collect_pcds(input_arg: str) -> List[str]:
    path = Path(input_arg)
    pcds: List[str] = []
    if path.is_dir():
        for entry in sorted(path.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".pcd":
                pcds.append(str(entry))
    elif path.is_file() and path.suffix.lower() == ".pcd":
        pcds.append(str(path))
    else:
        print(f"No PCD files found at {input_arg}")
    return pcds


def main() -> int:
    silence_framework_logs()

    config_path = Path(__file__).resolve().parent / CONFIG_NAME
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    lidar_dir = config.get("lidar_dir")
    lidar_out = config.get("lidar_out")

    if len(sys.argv) >= 2:
        lidar_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        lidar_out = sys.argv[2]

    if not lidar_dir:
        print("Config missing 'lidar_dir' and no CLI override provided.")
        return 1

    if not lidar_out:
        print("Config missing 'lidar_out' and no CLI override provided.")
        return 1

    global OUTPUT_FILE
    OUTPUT_FILE = str(lidar_out)

    pcds = collect_pcds(str(lidar_dir))
    if not pcds:
        print("No PCD files found.")
        return 1

    try:
        out_path = Path(OUTPUT_FILE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as cleared:
            cleared.write("")
        print(f"Cleared previous annotations in {OUTPUT_FILE}.")
    except OSError as exc:
        print(f"Failed to reset output file {OUTPUT_FILE}: {exc}")

    app = gui.Application.instance
    app.initialize()
    _viewer = PCDLabelWindow(pcds)
    print("> Point picking enabled.  [n] next, [N] prev, [b] undo, [Enter] save to file and next")
    app.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())
