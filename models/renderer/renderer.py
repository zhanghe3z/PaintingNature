import torch
import torch.nn as nn
# from torchsearchsorted import searchsorted

__all__ = ['render_rays']
"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    # (N_rays, N_samples_)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above],
                               -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    denom[denom < eps] = 1
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u-cdf_g[..., 0]) / \
        denom * (bins_g[..., 1]-bins_g[..., 0])
    return samples


def eval_points(points, models, embeddings):
    # points = points.unsqueeze(0)
    N_rays = points.shape[0]
    model_fine = models[-1]

    embedding_xyz = embeddings[0]

    def inference(model, xyz_, weights_only=True):
        """
        Helper function that performs model inference.

        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not

        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        # N_samples_ = xyz_.shape[1]
        # Embed directions
        # xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        chunk = 1024 * 32
        for i in range(0, B, chunk):
            # Embed positions by chunk
            # xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            xyz_embedded = (xyz_[i:i + chunk])
            xyzdir_embedded = xyz_embedded
            # print(xyzdir_embedded.shape)
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)
        # if weights_only:
        # sigmas = out.view(N_rays, 1)
        # sigmas = out.view(N_rays, N_samples_)
        # print(out.shape)
        return out

    points_enc = embedding_xyz(points)
    # print(points.shape, points_enc.shape)
    sigmas = \
        inference(model_fine, points_enc, weights_only=True)
    return sigmas


def render_rays(models,
                embeddings,
                rays,
                alphas,
                bbx,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False,
                detach_coarse=False,
                noisy_coarse=True,
                render_semantic=False,
                z=None,
                sky_mask=None,
                uv=None,
                masks=None,
                depth=None,
                label=None,
                points=None):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    def inference_coarse(model,
                         embedding_xyz,
                         xyz_,
                         dir_,
                         depth,
                         z_vals,
                         bbx,
                         weights_only=False):
        """
        Helper function that performs model inference.
        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i:i + chunk])
            xyzdir_embedded = xyz_embedded
            out_chunks += [
                model(xyzdir_embedded,
                      render_rgb=False,
                      render_semantic=True,
                      sigma_only=weights_only)
            ]
        out = torch.cat(out_chunks, 0)
        rgbsigma = out.view(N_rays, N_samples_, out.shape[-1])
        sigmas = rgbsigma[..., 0]  # (N_rays, N_samples_)
        if weights_only == False:
            semantics = rgbsigma[..., 1:]  # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        # (N_rays, 1) the last delta is infinity
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        # noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # compute alpha by the formula (3)
        # (N_rays, N_samples_)
        alphas = 1 - torch.exp(-deltas * 10 * sigmas)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 -
                       alphas+1e-10], -1)  # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -
                                   1)[:, :-1]  # (N_rays, N_samples_)
        # (N_rays), the accumulated opacity along the rays
        weights_sum = weights.sum(1)
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        # compute final weighted outputs
        # rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2)  # (N_rays, 3)
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if weights_only == False:
            semantics_final = torch.sum(weights.unsqueeze(-1) * semantics,
                                        -2)  # (N_rays, 3)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        if weights_only == True:
            return depth_final, weights
        else:
            return depth_final, weights, semantics_final

    def inference_fine(model,
                       embedding_xyz,
                       xyz_,
                       dir_,
                       z_vals,
                       bbx,
                       z,
                       label,
                       weights_only=False,
                       output_semantic=False):
        """
        Helper function that performs model inference.
        Inputs:
            model: NeRF model (coarse or fine)
            embedding_xyz: embedding module for xyz
            xyz_: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_: (N_rays, 3) ray directions
            dir_embedded: (N_rays, embed_dir_channels) embedded directions
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            weights_only: do inference on sigma only or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        b_ = xyz_.shape[0]
        N_samples_ = xyz_.shape[1]
        # Embed directions
        # xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        # Embed positions by chunk
        xyz_embedded = embedding_xyz(xyz_)
        xyzdir_embedded = xyz_embedded
        out = model(xyzdir_embedded,
                        z=z,
                        render_rgb=True,
                        render_semantic=output_semantic,
                        sigma_only=weights_only)

        out = out
        rgbs = out[..., :-1]  # (N_rays, N_samples_, 3)
        weights = out[..., -1]  # (N_rays, N_samples_)
        # Convert these values using volume rendering (Section 4)
        return rgbs, weights

    # Extract models from lists

    with torch.no_grad():
        model_coarse = models[0]
        embedding_xyz = embeddings[0]
        embedding_dir = embeddings[1]
        model_background = models[1]
        # Decompose the inputs
        B = rays.shape[0]
        rays_o, rays_d = rays[..., 0:3], rays[..., 3:6]  # both (N_rays, 3)
        near, far = rays[..., 6:7], rays[..., 7:8]  # both (N_rays, 1)

        # perturb_rand = torch.linspace(
        # -0.01, 0.01, 3, device=rays.device)  # (N_samples)
        # perturb_rand[..., -1] = 0
    zvals_depth = depth

    # if perturb > 0:  # perturb sampling depths (z_vals)
    # (N_rays, N_samples-1) interval mid points
    # z_vals_mid = 0.5 * (zvals_depth[:, :-1] + zvals_depth[:, 1:])
    # get intervals between samples
    # upper = torch.cat([z_vals_mid, zvals_depth[:, -1:]], -1)
    # lower = torch.cat([zvals_depth[:, :1], z_vals_mid], -1)
    # perturb_rand = perturb * torch.rand(zvals_depth.shape, device=rays.device)
    # zvals_depth = lower + (upper - lower) * perturb_rand
    # zvals_depth, _ = torch.sort(zvals_depth, -1)
    result = {}
    mask = (points > 0).reshape(points.shape[0], -1, 1)
    zvals_depth[mask] = points.unsqueeze(-1)[mask]
    depth_guided_sample = rays_o + \
            rays_d * zvals_depth.detach()
    # depth_guided_sample[mask] = points.reshape(depth_guided_sample.shape[0],
                                               # -1, 3)[mask]
    # depth_guided_sample += torch.randn(depth_guided_sample.shape,
                                           # device=rays.device).clip(min=-1, max=1) * 0.002
    z = z.float()
    rgb_fine, weights  = \
            inference_fine(model_coarse, embedding_xyz, depth_guided_sample, rays_d, zvals_depth, bbx, z, label, weights_only=False, output_semantic=False)
    sky_color = model_background(uv, z=z)  #, rgb_net = model_fine.rgb_net)
    # rgb_fine = weights.unsqueeze(-1) * rgb_fine + sky_color * (1-weights).unsqueeze(-1)
    # sky_mask = (~sky_mask).float()
    # sky_mask = (points<19).float()
    # sky_mask = alphas.float().to(rgb_fine)
    # rgb_fine = sky_mask.unsqueeze(-1) * rgb_fine + sky_color * (
        # 1 - sky_mask).unsqueeze(-1)
    rgb_fine = rgb_fine
    if rgb_fine.shape[2] > 3:
        z = model_coarse.map_net(z)
        conv = model_coarse.triplane_density
        rgb_fine = rgb_fine.reshape(-1, 256, 256, conv.out_in_channels).permute(0, 3, 1, 2)
        # rgb_fine = conv.conv3(
            # conv.act(
                # conv.conv2(
                    # conv.act(conv.conv1(rgb_fine)))))
        rgb_fine = conv.conv_out(rgb_fine, z)
        m = nn.Sigmoid()
        rgb_fine = m(rgb_fine.reshape(-1, 3, 256 * 256).permute(0, 2, 1))

    result['rgb_fine'] = rgb_fine
    result['weights'] = weights
    result['sky_color'] = sky_color
    return result
