import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import tinycudann as tcnn
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from models.nets.generator import Generator as Generator_small
from models.nets.generator_stylegan import Generator


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply


class NeRF_background(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4], use_new_activation=False, tensorf=False, semantic_nc = 184, triplane=False, z_dim = 256):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_background, self).__init__()
        # color network
        self.num_layers = D
        self.triplane = Generator_small(z_dim, output_rgb=True)

        # rgb_input = self.triplane.triplane_feature_all

        # self.rgb_act = 'Sigmoid'
        # self.rgb_net = \
            # tcnn.Network(
                # n_input_dims=rgb_input, n_output_dims=3,
                # network_config={
                    # "otype": "FullyFusedMLP",
                    # "activation": "ReLU",
                    # "output_activation": self.rgb_act,
                    # "n_neurons": 64,
                    # "n_hidden_layers": 2,
                # }
            # )


    def forward(self, uv, z=None, rgb_net = None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        z = z.float()
        color = self.triplane(z)
        m = nn.Sigmoid()
        color = m(color)
        color = F.grid_sample(color, uv[:,None]).view(uv.shape[0], -1, uv.shape[1]).permute(0,2,1)
        return color

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        # if x.max()>3:
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

def latent_to_w(style_vectorizer, latent_descr, num_layer):
    return [(style_vectorizer(latent_descr), num_layer) for z in latent_descr]

def image_noise(n, im_size):
    return nn.Parameter(torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.))

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

class NeRF_1p_style(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4], use_new_activation=False, tensorf=False, semantic_nc = 184, triplane=False, z_dim = 256, L=16, embed_s=False):
        """
        D: number of ldef styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)ayers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_1p_style, self).__init__()
        self.S = StyleVectorizer(256, 8)
        image_size = 128
        self.triplane_density = Generator(image_size, z_dim)
        rgb_input =  W
        z = torch.randn(1, 256, dtype=torch.float32).cuda()
        self.skips = [4]
        self.z = nn.Parameter(z)
        self.density_net = nn.Linear(self.triplane_density.triplane_feature_all, W)
        self.alpha = nn.Linear(W, 1)
        self.color_net = nn.ModuleList([nn.Linear(W, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W, W) for i in range(D-1)])
        self.color = nn.Linear(W, 64)
        self.noise = image_noise(1, image_size)


    def density(self, x, z, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature
        Outputs:
            sigmas: (N)
        """
        num_layer = self.triplane_density.num_layers
        w_space = latent_to_w(self.S, z, num_layer)
        w_styles = styles_def_to_tensor(w_space)
        triplanes = self.triplane_density(w_styles, self.noise.repeat(z.shape[0],1,1,1))
        plane_coef_point = []
        for triplane in triplanes:
            coord = x
            coord_xy = coord[..., [0,1]]
            coord_yz = coord[..., [1,2]]
            coord_xz = coord[..., [0,2]]
            # de
            # F.grid_sample()
            xy, yz, xz = triplane.split(triplane.shape[1]//3, dim=1)
            plane_coef_point.append(F.grid_sample(xy, coord_xy[:,None], align_corners=True).view(coord_xy.shape[0],-1,coord_xy.shape[1]))
            plane_coef_point.append(F.grid_sample(yz, coord_yz[:,None], align_corners=True).view(coord_yz.shape[0],-1,coord_yz.shape[1]))
            plane_coef_point.append(F.grid_sample(xz, coord_xz[:,None], align_corners=True).view(coord_xz.shape[0],-1,coord_xz.shape[1]))
        plane_coef_point = torch.cat(plane_coef_point, 1)
        plane_coef_point = plane_coef_point.permute(0,2,1)
        h = self.density_net(plane_coef_point)
        m = nn.Sigmoid()
        sigmas = m(self.alpha(h))
        return sigmas, h

    def forward(self, x, z=None, sigma_only=False, render_semantic=False, render_rgb=False, label_feature=None):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        sigmas, h = self.density(x, z, return_feat=True)
        if sigma_only == True:
            return sigmas.unsqueeze(-1)
        if render_rgb == True:
            for i, l in enumerate(self.color_net):
                h = self.color_net[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
            color = self.color(h)
            # color = self.color(h)
        out = sigmas
        out = torch.cat([color, out], -1)
        return out

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4], use_new_activation=False, tensorf=False, semantic_nc = 184, triplane=False, z_dim = 256, L=16):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        # color network
        # self.num_layers = D
        # num_layers = D
        # self.hidden_dim = W
        # hidden_dim = W
        # self.geo_feat_dim = W
        # geo_feat_dim = W
        # sigma_net = []
        # F = 2; log2_T = 19; N_min = 16
        # scale = 1
        # b = 1.38
        # print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        # self.rgb_act = 'Sigmoid'
        # self.xyz_encoder = \
            # tcnn.NetworkWithInputEncoding(
                # n_input_dims=3, n_output_dims=64,
                # encoding_config={
                    # "otype": "Grid",
                       # "type": "Hash",
                    # "n_levels": L,
                    # "n_features_per_level": F,
                    # "log2_hashmap_size": log2_T,
                    # "base_resolution": N_min,
                    # "per_level_scale": b,
                    # "interpolation": "Linear"
                # },
                # network_config={
                    # "otype": "FullyFusedMLP",
                    # "activation": "ReLU",
                    # "output_activation": "None",
                    # "n_neurons": 64,
                    # "n_hidden_layers": 1,
                # }
            # )


        # self.semantic_encoder = \
            # tcnn.NetworkWithInputEncoding(
                # n_input_dims=3, n_output_dims=64,
                # encoding_config={
                    # "otype": "Grid",
                        # "type": "Hash",
                    # "n_levels": L,
                    # "n_features_per_level": F,
                    # "log2_hashmap_size": log2_T,
                    # "base_resolution": N_min,
                    # "per_level_scale": b,
                    # "interpolation": "Linear"
                # },
                # network_config={
                    # "otype": "FullyFusedMLP",
                    # "activation": "ReLU",
                    # "output_activation": "None",
                    # "n_neurons": 64,
                    # "n_hidden_layers": 1,
                # }
        # )
        # self.semantic_net = \
            # tcnn.Network(
                # n_input_dims=64, n_output_dims=semantic_nc,
                # network_config={
                    # "otype": "FullyFusedMLP",
                    # "activation": "ReLU",
                    # "output_activation": "None",
                    # "n_neurons": 64,
                    # "n_hidden_layers": 1,
                # }
            # )
        self.map_net = \
            tcnn.Network(
                n_input_dims=z_dim, n_output_dims= 64,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 4,
                }
        )
        # if triplane:
        # self.triplane = Generator(z_dim)
        self.triplane_density = Generator(z_dim)
        # self.xyz_encoder = \
            # tcnn.NetworkWithInputEncoding(
                # n_input_dims=3, n_output_dims=64,
                # encoding_config={
                    # "otype": "Grid",
                       # "type": "Hash",
                    # "n_levels": L,
                    # "n_features_per_level": F,
                    # "log2_hashmap_size": log2_T,
                    # "base_resolution": N_min,
                    # "per_level_scale": b,
                    # "interpolation": "Linear"
                # },
                # network_config={
                    # "otype": "FullyFusedMLP",
                    # "activation": "ReLU",
                    # "output_activation": "None",
                    # "n_neurons": 64,
                    # "n_hidden_layers": 1,
                # }
             # )
        # else:
            # self.triplane = None
        # rgb_input = 64+63
        # if triplane == True:
        rgb_input =  64  + 127
        z = torch.randn(1, 256, dtype=torch.float32).cuda()
        self.z = nn.Parameter(z)
        self.rgb_act = 'Sigmoid'
        self.rgb_net = \
            tcnn.Network(
                n_input_dims=rgb_input, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
        self.density_net = \
            tcnn.Network(
                n_input_dims=self.triplane_density.triplane_feature_all, n_output_dims=128,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )
        self.encoder_semantic = Embedding(3, 2)
        input_ch_semanitc = 15
        self.skips = [4]
        self.semantic_linears = nn.ModuleList(
            [nn.Linear(input_ch_semanitc, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_semanitc, W) for i in range(D-1)])
        self.semantic_linear = nn.Linear(W, semantic_nc)

    def density(self, x, z, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature
        Outputs:
            sigmas: (N)
        """
        triplanes = self.triplane_density(z)
        plane_coef_point = []
        for triplane in triplanes:
            coord = x
            coord_xy = coord[:, [0,1]]
            coord_yz = coord[:, [1,2]]
            coord_xz = coord[:, [0,2]]
            # de
            # F.grid_sample()
            xy, yz, xz = triplane.split(triplane.shape[1]//3, dim=1)
            plane_coef_point.append(F.grid_sample(xy, coord_xy[None,None], align_corners=True).view(-1, coord_xy.shape[0]))
            plane_coef_point.append(F.grid_sample(yz, coord_yz[None,None], align_corners=True).view(-1, coord_yz.shape[0]))
            plane_coef_point.append(F.grid_sample(xz, coord_xz[None,None], align_corners=True).view(-1, coord_xz.shape[0]))
        plane_coef_point = torch.cat(plane_coef_point, 0).T
        h = self.density_net(plane_coef_point)
        m = nn.Softplus()
        sigmas = m(h[...,:1])
        return sigmas, h[..., 1:]

    def semantic(self, x):
        h = self.encoder_semantic(x*3)
        input_pts = h
        for i, l in enumerate(self.semantic_linears):
            h = self.semantic_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        semantic = self.semantic_linear(h)
        return semantic

    def forward(self, x, z=None, sigma_only=False, render_semantic=False, render_rgb=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        z = self.z
        sigmas, h = self.density(x, z, return_feat=True)
        if sigma_only == True:
            return sigmas.unsqueeze(-1)
        if render_rgb == True:
            z = self.map_net(z)
            color = self.rgb_net(torch.cat((z.repeat(h.shape[0],1), h), 1))
        if render_semantic == True:
            semantic = self.semantic(x)
        out = sigmas
        if render_rgb == True:
            out = torch.cat([color, out], -1)
        if render_semantic == True:
            out = torch.cat([out, semantic], -1)
        return out

class NeRF_torch(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63,
                 skips=[4], use_new_activation=False, semantic_nc = 5, L=2):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF_torch, self).__init__()
        # color network
        self.num_layers = D
        num_layers = D
        self.hidden_dim = W
        hidden_dim = W
        self.geo_feat_dim = W
        geo_feat_dim = W
        sigma_net = []
        self.encoder = Embedding(3, L)
        self.in_dim = 6*L+3
        input_ch = self.in_dim
        self.skips = skips
        self.pts_linears = nn.ModuleList(
           [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.views_linears = nn.ModuleList([nn.Linear(W, W//2)])
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)
        self.feature_linear = nn.Linear(W, W)
        # self.semantic_linears = nn.ModuleList(
            # [nn.Linear(input_ch_semanitc, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch_semanitc, W) for i in range(1)])
        self.semantic_linear = nn.Linear(W, semantic_nc)

    # def semantic(self, x):
        # h = self.encoder_semantic(x*3)
        # input_pts = h
        # for i, l in enumerate(self.semantic_linears):
            # h = self.semantic_linears[i](h)
            # h = F.relu(h)
            # if i in self.skips:
                # h = torch.cat([input_pts, h], -1)
        # semantic = self.semantic_linear(h)
        # return semantic

    def semantic(self, h):
        # h = self.encoder_semantic(x*3)
        # input_pts = h
        # for i, l in enumerate(self.semantic_linears):
            # h = self.semantic_linears[i](h)
            # h = F.relu(h)
            # if i in self.skips:
                # h = torch.cat([input_pts, h], -1)
        semantic = self.semantic_linear(h)
        return semantic

    def density(self, x, return_feat=False):
        h = self.encoder(x*3)
        input_pts = h
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        feature = self.feature_linear(h)
        alpha = self.alpha_linear(h)
        #sigma = F.relu(h[..., 0])
        m = nn.Softplus()
        sigma = m(alpha)
        geo_feat = feature
        # color
        return sigma, geo_feat

    def forward(self, x, sigma_only=False, render_rgb = False, render_semantic = True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        sigmas, h = self.density(x, return_feat=True)
        if sigma_only == True:
            return sigmas.unsqueeze(-1)
        if render_rgb == True:
            for i, l in enumerate(self.views_linears):
                h_ = h
                h_ = self.views_linears[i](h_)
                h_ = F.relu(h_)
            color = self.rgb_linear(h_)
            m = nn.Sigmoid()
            color = m(color)
        if render_semantic == True:
            semantic = self.semantic(h)
        out = sigmas
        if render_rgb == True:
            out = torch.cat([color, out], -1)
        if render_semantic == True:
            out = torch.cat([out, semantic], -1)
        return out

