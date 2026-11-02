# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
# Modified for DehazePhys-NeRF: independent test-camera ray generation.
"""Camera-to-world rays for the released test views."""
import numpy as np


def generate_rays(intrinsic, extrinsic, image_size):
    h, w = (int(v) for v in image_size)
    i, j = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5,
                       np.arange(h, dtype=np.float32) + 0.5, indexing='xy')
    directions = np.stack([(i - intrinsic[0, 2]) / intrinsic[0, 0],
                           (j - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(i)], -1)
    origins = np.tile(extrinsic[np.newaxis, :3, 3], (1, h * w, 1)).reshape(-1, 3).astype(np.float32)
    rays_d = np.einsum('hwc, rc -> hwr', directions, extrinsic[:3, :3]).reshape(-1, 3).astype(np.float32)
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)
    viewdirs = rays_d.copy()
    viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=1, keepdims=True)
    original_directions = np.einsum('hwc, rc -> hwr', directions, extrinsic[:3, :3])
    dx = np.sqrt(np.sum((original_directions[:-1] - original_directions[1:]) ** 2, -1))
    dx = np.concatenate([dx, dx[-2:-1]], 0)
    radii = (dx[..., None] * 2 / np.sqrt(12)).reshape(-1, 1).astype(np.float32)
    return {'rays_o': origins, 'rays_d': rays_d, 'viewdirs': viewdirs, 'radii': radii}
