"""Dataset adapters used by the first Opti3R experiments.

The adapters deliberately expose the same dictionaries expected by Wat3R's
ComposedDataset.  Real underwater frames never expose their depth files to
the training loop; the DTU-Haze adapter is the only labelled branch.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from PIL import Image

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import read_image_cv2


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _camera_matrix(camera) -> np.ndarray:
    name = str(getattr(camera, "model_name", getattr(camera, "model", ""))).upper()
    p = np.asarray(camera.params, dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    if "SIMPLE" in name:
        K[0, 0] = K[1, 1] = p[0]
        K[0, 2], K[1, 2] = p[1], p[2]
    elif "PINHOLE" in name:
        K[0, 0], K[1, 1] = p[0], p[1]
        K[0, 2], K[1, 2] = p[2], p[3]
    else:
        K[0, 0] = p[0]
        K[1, 1] = p[1] if len(p) > 3 else p[0]
        offset = 2 if len(p) > 3 else 1
        K[0, 2], K[1, 2] = p[offset], p[offset + 1]
    return K


def _cam_from_world(image) -> np.ndarray:
    transform = getattr(image, "cam_from_world", None)
    if transform is not None:
        if callable(transform):
            transform = transform()
        matrix = getattr(transform, "matrix", None)
        if matrix is not None:
            if callable(matrix):
                matrix = matrix()
            return np.asarray(matrix, dtype=np.float32)
    q = np.asarray(image.qvec, dtype=np.float64)
    t = np.asarray(image.tvec, dtype=np.float32)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y + 2 * z * w, 2 * x * z - 2 * y * w],
        [2 * x * y - 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z + 2 * x * w],
        [2 * x * z + 2 * y * w, 2 * y * z - 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float32)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3], out[:3, 3] = R, t
    return out


class Opti3rRealWaterDataset(BaseDataset):
    """Unlabelled sequence sampler for the FLSea-VI ``canyons`` layout."""

    no_label_dataset = True

    def __init__(self, common_conf, DATASET_DIR, split="train", heldout_scenes="u_canyon",
                 len_train=100000, len_test=10000, min_num_images=24,
                 max_teacher_images=8, **kwargs):
        super().__init__(common_conf)
        self.training = bool(common_conf.training)
        self.inside_random = bool(common_conf.inside_random)
        self.get_nearby = bool(common_conf.get_nearby)
        self.allow_duplicate_img = bool(common_conf.allow_duplicate_img)
        self.min_num_images = int(min_num_images)
        self.max_teacher_images = int(max_teacher_images)
        self.root = Path(DATASET_DIR).expanduser()
        heldout = {x.strip() for x in str(heldout_scenes).split(",") if x.strip()}
        self.sequence_dict = {}
        for imgs_root in sorted(self.root.rglob("imgs"), key=lambda p: _natural_key(str(p))):
            files = sorted(
                [p for p in imgs_root.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}],
                key=lambda p: _natural_key(p.name),
            )
            if len(files) < self.min_num_images:
                continue
            scene = imgs_root.parent.name
            if (split == "test") != (scene in heldout):
                continue
            key = scene
            self.sequence_dict[key] = [(scene, scene, str(path)) for path in files]
        if not self.sequence_dict:
            raise RuntimeError(f"No usable real underwater sequences found under {self.root}")
        self.seq_keys = list(self.sequence_dict)
        self.len_train = int(len_train if split == "train" else len_test)

    @staticmethod
    def _teacher_ids(ids, n, total):
        required = np.unique(np.asarray(ids, dtype=np.int64))
        anchors = np.linspace(0, total - 1, max(n, len(required)), dtype=np.int64)
        extras = np.setdiff1d(np.unique(anchors), required, assume_unique=False)
        values = np.concatenate([required, extras[:max(0, n - len(required))]])
        if len(values) < n:
            extra = np.random.randint(0, total, size=max(0, n - len(values)))
            values = np.unique(np.concatenate([values, extra]))
        while len(values) < n:
            values = np.append(values, values[-1])
        values = values[:n]
        np.random.shuffle(values)
        return values.astype(np.int64)

    @staticmethod
    def _mapping(ids, teacher_ids):
        return np.asarray([int(np.flatnonzero(teacher_ids == i)[0]) for i in ids], dtype=np.int64)

    @staticmethod
    def _resize_center_crop(image, target_h, target_w):
        h, w = image.shape[:2]
        scale = max(target_w / w, target_h / h)
        resized = cv2.resize(image, (max(target_w, int(round(w * scale))), max(target_h, int(round(h * scale)))))
        y0 = (resized.shape[0] - target_h) // 2
        x0 = (resized.shape[1] - target_w) // 2
        return resized[y0:y0 + target_h, x0:x0 + target_w]

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0):
        key = np.random.choice(self.seq_keys) if self.inside_random else self.seq_keys[int(seq_index)]
        sequence = self.sequence_dict[key]
        total = len(sequence)
        n = max(2, min(int(img_per_seq or 2), total))
        ids = np.asarray(ids if ids is not None else np.random.choice(total, n, replace=self.allow_duplicate_img), dtype=np.int64)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, total, expand_ratio=8)
        ids = np.asarray(ids, dtype=np.int64)
        teacher_n = min(total, max(4, self.max_teacher_images, len(ids)))
        teacher_ids = self._teacher_ids(ids, teacher_n, total)
        mapping = self._mapping(ids, teacher_ids)
        target_h, target_w = map(int, self.get_target_shape(float(aspect_ratio)))
        images, teacher_images, rotations, thetas = [], [], [], []
        for idx in ids:
            image = read_image_cv2(sequence[int(idx)][2], rgb=True)
            teacher_image = self._resize_center_crop(image, target_h, target_w)
            theta = int(np.random.choice([0, 90, 180, 270]))
            student = cv2.rotate(teacher_image, {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}.get(theta, cv2.ROTATE_180)) if theta else teacher_image.copy()
            student = cv2.resize(student, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            images.append(student)
            rotations.append({
                0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
                90: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32),
                180: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),
                270: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            }[theta])
            thetas.append(theta)
        for idx in teacher_ids:
            image = read_image_cv2(sequence[int(idx)][2], rgb=True)
            teacher_images.append(self._resize_center_crop(image, target_h, target_w))
        return {
            "seq_name": "optir3r_real_" + str(key), "ids": ids, "teacher_ids": teacher_ids,
            "mapping": mapping, "frame_num": len(images), "images": images,
            "teacher_images": teacher_images, "R_matrixs": rotations, "thetas": thetas,
            "image_paths": [sequence[int(i)][2] for i in ids],
        }


class Opti3rDTUHazeDataset(BaseDataset):
    """Labelled hazy DTU sequences with COLMAP cameras and depth maps."""

    def __init__(self, common_conf, DATASET_DIR, split="train", heldout_scenes="scan37,scan40",
                 len_train=100000, len_test=1000, depth_scale=1000.0, **kwargs):
        super().__init__(common_conf)
        self.training = bool(common_conf.training)
        self.inside_random = bool(common_conf.inside_random)
        self.get_nearby = bool(common_conf.get_nearby)
        self.allow_duplicate_img = bool(common_conf.allow_duplicate_img)
        self.depth_scale = float(depth_scale)
        self.root = Path(DATASET_DIR).expanduser()
        heldout = {x.strip() for x in str(heldout_scenes).split(",") if x.strip()}
        root = self.root
        self.sequence_list = {}
        for scene_root in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name)):
            is_test = scene_root.name in heldout
            if (split == "test") != is_test:
                continue
            sparse = scene_root / "colmap" / "sparse" / "0"
            hazy, depth = scene_root / "hazy", scene_root / "depth"
            if not sparse.is_dir() or not hazy.is_dir() or not depth.is_dir():
                continue
            reconstruction = pycolmap.Reconstruction(str(sparse))
            frames = []
            for image in reconstruction.images.values():
                name = Path(image.name).name
                image_path, depth_path = hazy / name, depth / name
                if not image_path.is_file() or not depth_path.is_file():
                    continue
                camera = reconstruction.cameras[image.camera_id]
                frames.append({"image": str(image_path), "depth": str(depth_path),
                               "K": _camera_matrix(camera), "extrinsic": _cam_from_world(image), "name": name})
            frames.sort(key=lambda item: _natural_key(item["name"]))
            if len(frames) >= 2:
                self.sequence_list[scene_root.name] = frames
        if not self.sequence_list:
            raise RuntimeError(f"No usable DTU-Haze sequences found under {root}")
        self.seq_keys = list(self.sequence_list)
        self.len_train = int(len_train if split == "train" else len_test)

    def get_data(self, seq_index=None, img_per_seq=None, seq_name=None, ids=None, aspect_ratio=1.0):
        key = np.random.choice(self.seq_keys) if self.inside_random else self.seq_keys[int(seq_index)]
        sequence = self.sequence_list[key]
        scene_root = Path(self.root) / key
        total = len(sequence)
        n = max(2, min(int(img_per_seq or 2), total))
        ids = np.asarray(ids if ids is not None else np.random.choice(total, n, replace=self.allow_duplicate_img), dtype=np.int64)
        if self.get_nearby:
            ids = np.asarray(self.get_nearby_ids(ids, total, expand_ratio=8), dtype=np.int64)
        target_shape = self.get_target_shape(float(aspect_ratio))
        images, depths, extrinsics, intrinsics, cam_points, world_points, point_masks, optics_targets = [], [], [], [], [], [], [], []
        image_paths, original_sizes = [], []
        for idx in ids:
            frame = sequence[int(idx)]
            image = read_image_cv2(frame["image"], rgb=True)
            optics_target = np.zeros(8, dtype=np.float32)
            clear_path = scene_root / "clear" / frame["name"]
            trans_path = scene_root / "trans" / frame["name"]
            if clear_path.is_file() and trans_path.is_file():
                clear = read_image_cv2(str(clear_path), rgb=True).astype(np.float32) / 255.0
                hazy_float = image.astype(np.float32) / 255.0
                trans = cv2.imread(str(trans_path), cv2.IMREAD_UNCHANGED)
                if trans is not None:
                    trans = trans.astype(np.float32)
                    if trans.ndim == 2:
                        trans = trans[..., None]
                    trans = trans / max(float(np.nanmax(trans)), 1.0)
                    if trans.shape[:2] != image.shape[:2]:
                        trans = cv2.resize(trans, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                    trans_mean = trans.mean((0, 1)).reshape(-1)
                    if trans_mean.size == 1:
                        trans_mean = np.repeat(trans_mean, 3)
                    else:
                        trans_mean = np.pad(trans_mean[:3], (0, max(0, 3 - trans_mean.size)))[:3]
                    optics_target = np.concatenate([
                        hazy_float.mean((0, 1)), trans_mean,
                        np.array([np.abs(hazy_float - clear).mean(), hazy_float.std()], dtype=np.float32),
                    ]).astype(np.float32)
            depth = cv2.imread(frame["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32) / self.depth_scale
            if depth.shape[:2] != image.shape[:2]:
                depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth[~np.isfinite(depth) | (depth <= 0) | (depth >= 65.534)] = 0.0
            original_size = np.array(image.shape[:2])
            out = self.process_one_image(image, depth, frame["extrinsic"][:3], frame["K"], original_size, target_shape, filepath=frame["image"])
            image, depth, extrinsic, intrinsic, world, cam, mask, _ = out
            images.append(image); depths.append(depth); extrinsics.append(extrinsic); intrinsics.append(intrinsic)
            world_points.append(world); cam_points.append(cam); point_masks.append(mask)
            optics_targets.append(optics_target)
            image_paths.append(frame["image"]); original_sizes.append(original_size)
        return {
            "seq_name": "optir3r_dtu_" + str(key), "ids": ids, "frame_num": len(images), "images": images,
            "depths": depths, "extrinsics": extrinsics, "intrinsics": intrinsics,
            "cam_points": cam_points, "world_points": world_points, "point_masks": point_masks,
            "optics_targets": optics_targets,
            "image_paths": image_paths, "original_sizes": original_sizes,
        }
