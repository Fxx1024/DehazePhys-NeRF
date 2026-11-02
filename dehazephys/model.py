# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
# Modified for DehazePhys-NeRF: deterministic inference and object radiance rendering.
"""Scene-specific radiance field used by the released models."""
import torch
from torch import nn
from . import math_utils as helper


class RadianceMLP(nn.Module):
    def __init__(self, netdepth=8, netwidth=256, disable_rgb=False, possibility=False):
        super().__init__()
        self.netdepth = netdepth
        self.skip_layer = 4
        self.min_deg_point = 0
        self.max_deg_point = 12
        self.deg_view = 4
        self.disable_rgb = disable_rgb
        self.possibility = possibility
        self.register_buffer('pos_basis_t', helper.generate_basis('icosahedron', 2))
        pos_size = 24 * self.pos_basis_t.shape[-1]
        layers = [nn.Linear(pos_size, netwidth)]
        for idx in range(netdepth - 1):
            size = netwidth + pos_size if idx % self.skip_layer == 0 and idx > 0 else netwidth
            layers.append(nn.Linear(size, netwidth))
        self.pts_linear = nn.ModuleList(layers)
        self.density_layer = nn.Linear(netwidth, 1)
        if not disable_rgb:
            self.bottleneck_layer = nn.Linear(netwidth, 256)
            self.views_linear = nn.ModuleList([nn.Linear(256 + 27, 128)])
            self.rgb_layer = nn.Linear(128, 3)
        self.net_activation = nn.ReLU()
        self.density_activation = nn.Softplus()
        self.rgb_activation = nn.Sigmoid()
        self.possibility_activation = nn.Sigmoid()

    def predict_density(self, means, covs):
        means, covs = helper.contract(means, covs)
        lifted_means, lifted_vars = helper.lift_and_diagonalize(means, covs, self.pos_basis_t)
        x = helper.integrated_pos_enc(lifted_means, lifted_vars, self.min_deg_point, self.max_deg_point)
        inputs = x
        for idx in range(self.netdepth):
            x = self.net_activation(self.pts_linear[idx](x))
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)
        return self.density_layer(x)[..., 0], x

    def forward(self, gaussians, viewdirs):
        means, covs = gaussians
        raw_density, x = self.predict_density(means, covs)
        if self.possibility:
            return self.possibility_activation(raw_density)
        density = self.density_activation(raw_density - 1.0)
        if self.disable_rgb:
            return {'density': density}
        bottleneck = self.bottleneck_layer(x)
        dir_enc = helper.pos_enc(viewdirs, 0, self.deg_view, True)
        dir_enc = torch.broadcast_to(dir_enc[..., None, :], bottleneck.shape[:-1] + (dir_enc.shape[-1],))
        x = torch.cat([bottleneck, dir_enc], dim=-1)
        x = self.net_activation(self.views_linear[0](x))
        rgb = self.rgb_activation(self.rgb_layer(x))
        rgb = rgb * (1 + 2 * 0.001) - 0.001
        return {'density': density, 'rgb': rgb}


class DehazePhysNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlps = nn.ModuleList([RadianceMLP(4, 256, disable_rgb=True),
                                   RadianceMLP(4, 256, disable_rgb=True),
                                   RadianceMLP(8, 1024)])
        self.pmlps = nn.ModuleList([RadianceMLP(4, 128, disable_rgb=True, possibility=True)])
        self.register_buffer('sampling_exponent', torch.ones((), dtype=torch.float32))

    def forward(self, batch, near, far):
        bsz = batch['rays_o'].shape[0]
        device = batch['rays_o'].device
        _, s_to_t = helper.construct_ray_warps(near, far)
        sdist = torch.cat([torch.zeros((bsz, 1), device=device),
                           torch.ones((bsz, 1), device=device)], dim=-1)
        weights = torch.ones(bsz, 1, device=device)
        prod_num_samples = 1
        for level in range(3):
            num_samples = 64 if level < 2 else 32
            dilation = 0.0025 + 0.5 / prod_num_samples
            prod_num_samples *= num_samples
            if level > 0:
                sdist, weights = helper.max_dilate_weights(sdist, weights, dilation,
                                                         domain=(0.0, 1.0), renormalize=True)
                sdist, weights = sdist[..., 1:-1], weights[..., 1:-1]
            logits = torch.where(sdist[..., 1:] > sdist[..., :-1],
                                 self.sampling_exponent * torch.log(weights + 0.0),
                                 torch.full_like(weights, -torch.inf))
            sdist = helper.sample_intervals(sdist, logits, num_samples,
                                            single_jitter=True, domain=(0.0, 1.0)).detach()
            tdist = s_to_t(sdist)
            gaussians = helper.cast_rays(tdist, batch['rays_o'], batch['rays_d'],
                                        batch['radii'], 'cone', diag=False)
            radiance = self.mlps[level](gaussians, batch['viewdirs'])
            weights = helper.compute_alpha_weights(radiance['density'], tdist,
                                                     batch['rays_d'], opaque_background=True)[0]
            if level == 2:
                probability = self.pmlps[0](gaussians, batch['viewdirs'])
                density = radiance['density'] * probability
                rgb = radiance['rgb'] * probability.unsqueeze(-1)
                object_weights = helper.compute_alpha_weights(density, tdist,
                                                                batch['rays_d'], opaque_background=True)[0]
                return helper.volumetric_rendering(rgb, object_weights, tdist, 1.0, far, False)['rgb']


def load_model(path, device):
    """Load an array-only model archive with strict tensor shape validation."""
    import numpy as np
    model = DehazePhysNeRF()
    with np.load(str(path), allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)
