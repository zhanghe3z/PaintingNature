import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.datasets import dataset_dict
from moviepy.editor import ImageSequenceClip
from models.nets.field import NeRF_1p, NeRF_background
from models.nets.field_stylegan import NeRF_1p_style
from models.nets.discriminator import Discriminator
from models.renderer.renderer import render_rays
from models.diff_aug import DiffAugment
from utils.model_average import ModelAverage, WrappedModel
from models.spade import GauGAN
from utils import *
from losses import loss_dict
from pytorch3d.renderer.cameras import PerspectiveCameras
from metrics import *
import kornia
from pytorch_lightning import LightningModule, Trainer
from einops import rearrange
import tqdm

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)
    # tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens).cpu()
        if mask.sum() == 0:
            continue
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    if N == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None, useMask=True):
        if mask.sum() > 0:
            loss = self.loss(depth_pred[mask[0]], depth_gt[mask])  # * 2 ** (1 - 2)
        else:
            loss = 0
        return loss



def get_checkpoint(checkpoint_path, url=''):
    r"""Get the checkpoint path. If it does not exist yet, download it from
    the url.

    Args:
        checkpoint_path (str): Checkpoint path.
        url (str): URL to download checkpoint.
    Returns:
        (str): Full checkpoint path.
    """
    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = os.getcwd()
    save_dir = os.path.join(os.environ['TORCH_HOME'], '../checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    full_checkpoint_path = os.path.join(save_dir, checkpoint_path)
    if not os.path.exists(full_checkpoint_path):
        os.makedirs(os.path.dirname(full_checkpoint_path), exist_ok=True)
        if is_master():
            print('Downloading {}'.format(url))
            if 'pbss.s8k.io' not in url:
                url = f"https://docs.google.com/uc?export=download&id={url}"
            download_file(url, full_checkpoint_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return full_checkpoint_path

class GauGANLoader(object):
    r"""Manages the SPADE/GauGAN model used to generate pseudo-GTs for training GANcraft.

    Args:
        gaugan_cfg (Config): SPADE configuration.
    """

    def __init__(self, cfg):
        print('[GauGANLoader] Loading GauGAN model.')
        net_G = WrappedModel(ModelAverage(GauGAN(cfg.gen, cfg.data).to('cuda')))
        self.net_G = net_G
        self.net_GG = net_G.module.averaged_model
        self.net_GG.eval()
        self.net_GG.half()
        print('[GauGANLoader] GauGAN loading complete.')

    def update_spade(self):
        self.net_GG = self.net_G.module.averaged_model
        self.net_GG.eval()
        self.net_GG.half()

    def eval(self, label, z=None, style_img=None):
        r"""Produce output given segmentation and other conditioning inputs.
        random style will be used if neither z nor style_img is provided.

        Args:
            label (N x C x H x W tensor): One-hot segmentation mask of shape.
            z: Style vector.
            style_img: Style image.
        """
        inputs = {'label': label[:, :-1].detach().half()}
        random_style = True

        if z is not None:
            random_style = False
            inputs['z'] = z.detach().half()
        elif style_img is not None:
            random_style = False
            inputs['images'] = style_img.detach().half()

        net_GG_output, z = self.net_GG(inputs, random_style=random_style)

        return net_GG_output['fake_images'], z

    def get_pseudo_gt(self, pseudo_gen, voxel_id, z=None, style_img=None, resize_512=True, deterministic=False):
        r"""Evaluating img2img network to obtain pseudo-ground truth images.

        Args:
            pseudo_gen (callable): Function converting mask to image using img2img network.
            voxel_id (N x img_dims[0] x img_dims[1] x max_samples x 1 tensor): IDs of intersected tensors along
            each ray.
            z (N x C tensor): Optional style code passed to pseudo_gen.
            style_img (N x 3 x H x W tensor): Optional style image passed to pseudo_gen.
            resize_512 (bool): If True, evaluate pseudo_gen at 512x512 regardless of input resolution.
            deterministic (bool): If True, disable stochastic label mapping.
        """
        with torch.no_grad():
            fake_masks = voxel_id

            # Generate pseudo GT using GauGAN.
            if resize_512:
                fake_masks_512 = F.interpolate(fake_masks, size=[512, 512], mode='nearest')
            else:
                fake_masks_512 = fake_masks
            pseudo_real_img, z = pseudo_gen(fake_masks_512, z=z, style_img=style_img)

            # NaN Inf Guard. NaN can occure on Volta GPUs.
            nan_mask = torch.isnan(pseudo_real_img)
            inf_mask = torch.isinf(pseudo_real_img)
            pseudo_real_img[nan_mask | inf_mask] = 0.0
            if resize_512:
                pseudo_real_img = F.interpolate(
                    pseudo_real_img, size=[fake_masks.size(2), fake_masks.size(3)], mode='area')
            pseudo_real_img = torch.clamp(pseudo_real_img, -1, 1)

        return pseudo_real_img, z

    def sample_camera(self, data, z):
        r"""Sample camera randomly and precompute everything used by both Gen and Dis.

        Args:
            data (dict):
                images (N x 3 x H x W tensor) : Real images
                label (N x C2 x H x W tensor) : Segmentation map
            pseudo_gen (callable): Function converting mask to image using img2img network.
        Returns:
            ret (dict):
                voxel_id (N x H x W x max_samples x 1 tensor): IDs of intersected tensors along each ray.
                depth2 (N x 2 x H x W x max_samples x 1 tensor): Depths of entrance and exit points for each ray-voxel
                intersection.
                raydirs (N x H x W x 1 x 3 tensor): The direction of each ray.
                cam_ori_t (N x 3 tensor): Camera origins.
                pseudo_real_img (N x 3 x H x W tensor): Pseudo-ground truth image.
                real_masks (N x C3 x H x W tensor): One-hot segmentation map for real images, with translated labels.
                fake_masks (N x C3 x H x W tensor): One-hot segmentation map for sampled camera views.
        """
        device = torch.device('cuda')
        # ================ Assemble a batch ==================
        # Requires: voxel_id, depth2, raydirs, cam_ori_t.
        pseudo_real_img, z = self.get_pseudo_gt(self.eval, data, z = z)

        # =============== Mask translation ================
        return pseudo_real_img, z

class GANLoss():
    def __init__(self, opt):
        self.opt = opt

    def loss(self, input, label, for_real, mask=None, edge_mask=None, cur_ep = -1):
        #--- balancing classes ---
        weight_map = get_class_balancing(self.opt, input, label)
        #--- n+1 loss ---
        target = get_n1_target(self.opt, input, label, for_real)
        loss = F.cross_entropy(input, target, reduction='none')
        # if for_real == False:
        if for_real:
            # if edge_mask is not None:
                # loss = torch.mean(loss * weight_map[:, 0, :, :] * (~edge_mask).float())
            # else:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)
        return loss

def get_class_balancing(opt, input, label):
    # if not opt.no_balancing_inloss:
    class_occurence = torch.sum(label, dim=(0, 2, 3))
    # if opt.contain_dontcare_label:
        # class_occurence[0] = 0
    num_of_classes = (class_occurence > 0).sum()
    coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
    integers = torch.argmax(label, dim=1, keepdim=True)
    # if opt.contain_dontcare_label:
        # coefficients[0] = 0
    weight_map = coefficients[integers]
    # else:
    # weight_map = torch.ones_like(input[:, :, :, :])
    return weight_map


def get_n1_target(opt, input, label, target_is_real):
    targets = get_target_tensor(opt, input, target_is_real)
    num_of_classes = label.shape[1]
    integers = torch.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1
    return integers


def get_target_tensor(opt, input, target_is_real):
    if target_is_real:
        return torch.cuda.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
    else:
        return torch.cuda.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)


def inv_loss(gt, pred, mask):
    if mask.sum()==0:
        return torch.zeros_like((torch.abs(pred - gt)).mean()).to(gt)
    gt = gt.view(gt.shape[0], -1, 1)
    pred = pred.view(gt.shape[0], -1, 1)
    scale, shift = compute_scale_and_shift(pred, gt, torch.ones_like(pred).to(pred))
    pred = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
    loss_d = (torch.abs(pred - gt)[mask]).mean()
    return loss_d

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class Z(nn.Module):
    def __init__(self):
        super(Z, self).__init__()
        z = torch.randn(1, 256, dtype=torch.float32).cuda()
        self.z = nn.Parameter(z).half()
        self.z_orig = nn.Parameter(z).half()

    def forward(self, x):
        self.z = nn.Parameter(x).half()

class cam_embedding(nn.Module):
    def __init__(self, nc):
        super(cam_embedding, self).__init__()
        self.s_embedding = nn.Embedding(nc, 256)

    def forward(self, x):
        return self.s_embedding(x)

class s_embedding(nn.Module):
    def __init__(self, nc):
        super(s_embedding, self).__init__()
        self.s_embedding = nn.Embedding(nc, 32)

    def forward(self, x):
        return self.s_embedding(x)

class GANLoss_fpse(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        r"""GAN loss constructor.

        Args:
            target_real_label (float): Desired output label for the real images.
            target_fake_label (float): Desired output label for the fake images.
        """
        super(GANLoss_fpse, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None

    def forward(self, input_x, t_real, weight=None,
                reduce_dim=True, dis_update=True):
        r"""GAN loss computation.

        Args:
            input_x (tensor or list of tensors): Output values.
            t_real (boolean): Is this output value for real images.
            reduce_dim (boolean): Whether we reduce the dimensions first. This makes a difference when we use
            multi-resolution discriminators.
            weight (float): Weight to scale the loss value.
            dis_update (boolean): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        if isinstance(input_x, list):
            loss = 0
            for pred_i in input_x:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, t_real, weight,
                                        reduce_dim, dis_update)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input_x)
        else:
            return self.loss(input_x, t_real, weight, reduce_dim, dis_update)

    def loss(self, input_x, t_real, weight=None,
             reduce_dim=True, dis_update=True):
        r"""N+1 label GAN loss computation.

        Args:
            input_x (tensor): Output values.
            t_real (boolean): Is this output value for real images.
            reduce_dim (boolean): Whether we reduce the dimensions first. This makes a difference when we use
            multi-resolution discriminators.
            weight (float): Weight to scale the loss value.
            dis_update (boolean): Updating the discriminator or the generator.
        Returns:
            loss (tensor): Loss value.
        """
        assert reduce_dim is True
        pred = input_x['pred'].clone()
        label = input_x['label'].clone()
        batch_size = pred.size(0)

        # ignore label 0
        label[:, 0, ...] = 0
        pred[:, 0, ...] = 0
        pred = F.log_softmax(pred, dim=1)
        assert pred.size(1) == (label.size(1) + 1)
        if dis_update:
            if t_real:
                pred_real = pred[:, :-1, :, :]
                loss = - label * pred_real
                loss = torch.sum(loss, dim=1, keepdim=True)
            else:
                pred_fake = pred[:, -1, None, :, :]  # N plus 1
                loss = - pred_fake
        else:
            assert t_real, "GAN loss must be aiming for real."
            pred_real = pred[:, :-1, :, :]
            loss = - label * pred_real
            loss = torch.sum(loss, dim=1, keepdim=True)

        if weight is not None:
            loss = loss * weight
        if reduce_dim:
            loss = torch.mean(loss)
        else:
            loss = loss.view(batch_size, -1).mean(dim=1)
        return loss



class FeatureMatchingLoss(nn.Module):
    r"""Compute feature matching loss"""
    def __init__(self, criterion='l1'):
        super(FeatureMatchingLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake_features, real_features):
        r"""Return the target vector for the binary cross entropy loss
        computation.
        Args:
           fake_features (list of lists): Discriminator features of fake images.
           real_features (list of lists): Discriminator features of real images.
        Returns:
           (tensor): Loss value.
        """
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j],
                                          real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss

class Renderer_anti(LightningModule):
    def __init__(self, hparams, cfg):
        super(Renderer_anti, self).__init__()
        self.z = Z()
        self.z_2 = Z()
        self.hparams = hparams
        self.loss = loss_dict[hparams.loss_type]()
        self.patch_loss = loss_dict[hparams.patch_loss]()
        self.s1 = SL1Loss()
        self.init_data()
        self.cameras_ndc = PerspectiveCameras(device='cuda',
                                                 focal_length = self.train_dataset.focal_ndc,
                                                 principal_point=self.train_dataset.principal_point_ndc,
                                                 in_ndc=False,
                                                 R=self.train_dataset.R_ndc,
                                                 T=self.train_dataset.T_ndc,
                                                 image_size=self.train_dataset.image_size_ndc)

        self.ce = nn.CrossEntropyLoss(reduction = 'none')
        self.embedding_xyz = self.embedding_ndc
        self.embedding_dir = nn.Identity() #Embedding_tensorf(3, 4)  # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]
        if hparams.use_style:
            self.nerf_coarse = NeRF_1p_style(semantic_nc = self.train_dataset.nc, D=2, noise=self.hparams.noise>0, use_cnn=self.hparams.use_cnn)
        else:
            self.nerf_coarse = NeRF_1p(semantic_nc = self.train_dataset.nc, D=2, noise=self.hparams.noise>0, use_cnn=self.hparams.use_cnn)
        self.models = [self.nerf_coarse]
        self.nerf_b = NeRF_background(D=2, z_dim = 256)
        self.models += [self.nerf_b]
        # self.models += [self.embedding_xyz]
        self.gaugan_model = GauGANLoader(cfg)
        if self.hparams.dis_weight > 0:
            self.D = Discriminator(hparams, self.train_dataset.nc+1)

        self.loss_D = 0
        self.ref_ = None
        if self.hparams.use_fpse <=0:
            self.criterionGAN = GANLoss(hparams)
        else:
            self.criterionGAN = GANLoss_fpse()
            self.featurematching = FeatureMatchingLoss()
        self.l2 = nn.MSELoss()
        
        self.Sobel = Sobel()
        self.image_edge_rid = self.Sobel(torch.ones((1,1,256, 256))) > 0.1
        m = nn.MaxPool2d(9, stride=1, padding=4)
        self.masks_edge_rid = m(m(m(self.image_edge_rid.float().cuda())))
        self.loss_mean = 0
        self.bce_weight = 10
        self.x_max = torch.cuda.FloatTensor([0])
        self.x_min = torch.cuda.FloatTensor([0])
        self.LHQ_fake  = 1
        self.rgb = None

    def embedding_ndc(self,x):
        x = self.cameras_ndc.transform_points_ndc(x)
        x[...,2] = x[...,2]-1
        with torch.no_grad():
            self.x_max = torch.maximum(x.max(), self.x_max)
            self.x_min = torch.minimum(x.min(), self.x_min)
        return x

    def update_z_2(self,):
        z =  nn.Parameter(self.z.z).half()
        self.z_2(z)

    def update_z(self,):
        z =  nn.Parameter(torch.randn(1, 256, dtype=torch.float32).cuda()).half()
        self.z(z)

    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        c2w = batch['c2w']
        uv = batch['coord']
        idx = batch['idx']
        label = batch['label']
        depth = batch['depth'][...,None]
        fake_label = batch['fake_label']
        points = batch['points']
        bg_label = batch['bg_label'].long()
        if 'sky_mask' in batch:
            alphas = batch['sky_mask']
            return rays, c2w, uv, idx, label, fake_label, depth, points, alphas, bg_label
        else:
            return rays, c2w, uv, idx, label, fake_label, depth, points, points, bg_label


    def forward(self, rays, alphas, bbx=None, z=None, perturb=1, render_semantic = True, sky_mask=None, uv = None, depth=None, masks=None, label=None, points=None):
        """Do batched inference on rays using chunk."""
        B = rays.shape[1]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            if depth is not None:
                rendered_ray_chunks = \
                    render_rays(self.models,
                                self.embeddings,
                                rays[:,i:i+self.hparams.chunk],
                                alphas[:,i:i+self.hparams.chunk],
                                bbx,
                                self.hparams.N_samples,
                                self.hparams.use_disp,
                                perturb,
                                self.hparams.noise_std,
                                self.hparams.N_importance,
                                self.hparams.chunk,  # chunk size is effective in val mode
                                self.train_dataset.white_back,
                                render_semantic = render_semantic,
                                sky_mask = sky_mask[:,i:i+self.hparams.chunk],
                                z=z,
                                uv = uv[:,i:i+self.hparams.chunk],
                                depth = depth[:,i:i+self.hparams.chunk],
                                label = label[:,i:i+self.hparams.chunk],
                                points=points[:,i:i+self.hparams.chunk])
                                # masks = masks[i:i+self.hparams.chunk])
            else:
                rendered_ray_chunks = \
                    render_rays(self.models,
                                self.embeddings,
                                rays[:,i:i+self.hparams.chunk],
                                bbx,
                                self.hparams.N_samples,
                                self.hparams.use_disp,
                                perturb,
                                self.hparams.noise_std,
                                self.hparams.N_importance,
                                self.hparams.chunk,  # chunk size is effective in val mode
                                self.train_dataset.white_back,
                                render_semantic = render_semantic,
                                sky_mask = sky_mask,
                                z=z,
                                uv = uv,
                                depth = None,
                                masks = None)


            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 1)
        return results

    def init_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        # kwargs = vars(self.hparams)
        kwargs = self.hparams
        self.train_dataset = dataset(split='train', **kwargs)
        self.H = self.train_dataset.H
        self.W = self.train_dataset.W
        self.val_dataset = dataset(split='val', **kwargs)
        self.test_dataset = dataset(split='debug', **kwargs)

    def configure_optimizers(self):
        li = []
        sc = []
        self.optimizer = get_optimizer(self.hparams, self.models, eps = 1e-15)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        sc += [scheduler]
        li += [self.optimizer]
        if self.hparams.dis_weight > 0:
            self.opt_d = get_optimizer(self.hparams, [self.D], rate=1)
            scheduler_2 = get_scheduler(self.hparams, self.opt_d)
            li +=[self.opt_d]
            sc += [scheduler_2]
        return li, sc

    def train_dataloader(self):
        dataset = DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=False)
        return dataset

    def val_dataloader(self):
        self.val_dataset.read_meta()
        dataset = DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=2,
                          # validate one image (H*W rays) at a time
                          batch_size=1,
                          pin_memory=False)
        return dataset

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.train()
        log = {'lr': get_learning_rate(self.optimizer)}
        rays,  c2w, uv, idx, label_pse, fake_label, depth, points, alphas, bg_label = self.decode_batch(batch)
        if self.hparams.change_style and (batch_idx % 2==0):
            self.update_z_2()
            self.update_z()

        with torch.no_grad():
            torch.cuda.empty_cache()
            b,n,c = rays.shape
            semantic_coarse = rearrange(label_pse.clone(), 'b (h w)->b h w', w=self.W, h=self.H)
            label_gan_idx = semantic_coarse.clone()
            label_gan = torch.cuda.FloatTensor(b, self.train_dataset.nc+1, self.H, self.W).zero_()
            label_gan = label_gan.scatter_(1, label_gan_idx[:,None], 1.0)
            fake_label_gan_idx = torch.ones_like(fake_label).to(label_gan).long() * self.train_dataset.nc
            for label in torch.unique(semantic_coarse):
                mask_ = (semantic_coarse==label)
                semantic_coarse[mask_] = self.train_dataset.label_trans[label.item()]
                mask_fake = (fake_label==self.train_dataset.label_trans[label.item()])
                fake_label_gan_idx[mask_fake] = label
            fake_label_gan = torch.cuda.FloatTensor(b, self.train_dataset.nc+1, self.H, self.W).zero_()
            fake_label_gan = fake_label_gan.scatter_(1, fake_label_gan_idx[:, None], 1.0)

            sky_mask = (semantic_coarse == 105) | (semantic_coarse == 156)
            depth.reshape(-1)[sky_mask.reshape(-1)]=20
            label_map = semantic_coarse
            edge = label_map.clone()
            edge = F.pad(edge.float(), (4,4,4,4), "replicate")
            edge = self.Sobel(edge[:, None].float())
            m = nn.MaxPool2d(9, stride=1)
            edge = m(edge)
            masks_edge = depth.clone().reshape(b,1,self.W, self.H)
            masks_edge = F.pad(masks_edge.float(), (4,4,4,4), "replicate")
            masks_edge = self.Sobel(masks_edge) > 10
            masks_edge = m(masks_edge.float())
            edge_mask = (edge > 0) & (masks_edge > 0)
            h, w = self.H, self.W
            nc = 185
            label = torch.cuda.FloatTensor(b, nc, h, w).zero_()
            semantics = label.clone().scatter_(1, label_map[:, None], 1.0)
            fake_semantics = label.clone().scatter_(1, fake_label[:, None], 1.0)
            if int(self.train_dataset.src_pair_name) != 74:
                label_map_sky = bg_label
                inpaint_sky = ((label_map_sky==105).sum() > (label_map_sky==156).sum())
                if inpaint_sky:
                    label_map_sky[(label_map_sky!=105) & (label_map_sky!=156)] = 105
                else:
                    label_map_sky[(label_map_sky!=105) & (label_map_sky!=156)] = 156
            else:
                label_map_sky  = 156 * torch.ones_like(bg_label).to(bg_label)
                label_map_sky[:, 60:] = bg_label[:, :256-60]
            semantics_sky = label.clone().scatter_(1, label_map_sky[None], 1.0)
            self.eval()
            fake, z = self.gaugan_model.sample_camera(torch.cat((semantics,fake_semantics,semantics_sky),0), self.z.z)
            self.train()
            fake = fake*0.5+0.5
            fake = fake.float()
            sky_bg = fake[-b:]
            fake = fake[:-b]
            torch.cuda.empty_cache()
        rays = rays.squeeze()  # (H*W, 3)
        loss = 0
        DiffAugment_flag = True
        if optimizer_idx == 0:
            z_add = z.repeat(b,1)
            results = self(rays, alphas, render_semantic=False, z=z_add, sky_mask = sky_mask.reshape(b,-1), uv = uv, depth=depth, label=label_pse, points=points)
            if self.hparams.sky_th >0:
                sky_mask = ((points>points.max()-self.hparams.sky_th) | ((points == 0) & (depth>depth.max()-self.hparams.sky_th).squeeze())).float()
            else:
                sky_mask = sky_mask.float().reshape(*points.shape)
            results['rgb_fine'] = results['rgb_fine'] * (1-sky_mask).unsqueeze(-1) + results['sky_color'] * sky_mask.unsqueeze(-1)
            self.rgb = results['rgb_fine'].view(b,self.H,self.W,-1).permute(0,3,1,2).detach().clone()
            if self.hparams.edge_mask == 'True':
                loss_g = self.loss(results['rgb_fine'].view(b,self.H,self.W,-1).permute(0,3,1,2), fake[:b], edge_mask)
            else:
                edge_mask = None
                loss_g = self.loss(results['rgb_fine'].view(b,self.H,self.W,-1).permute(0,3,1,2), fake[:b], edge_mask)
            log['train/loss_vgg'] = loss_g['tot']
            loss += loss_g['tot']
            loss_mean = 0 
            self.loss_mean = loss_mean
            loss = self.hparams.vgg_weight * (loss + loss_mean)
            log['train/loss_mean'] = loss_mean
            if loss == 0:
                import ipdb; ipdb.set_trace()
            if torch.isnan(loss):
                import ipdb; ipdb.set_trace()
        if self.hparams.dis_weight > 0 and (optimizer_idx == 0):
            rgb = results['rgb_fine'].view(b,self.H,self.W,-1).permute(0,3,1,2)
            if self.hparams.use_fpse >0:
                incl_pseudo_real = True
                out = DiffAugment(torch.cat((rgb, label_gan.clone(), fake[:b]), 1), DiffAugment=DiffAugment_flag)
                rgb = out[:, :3]
                label_gan_crop = out[:, 3:-3]
                fake_crop = out[:,-3:]
                data={'fake_masks': label_gan_crop, 'pseudo_real_img': fake_crop}
                net_D_output = self.D(data, rgb, incl_real=False, incl_pseudo_real=incl_pseudo_real)
                output_fake = net_D_output['fake_outputs']
                gan_loss = self.criterionGAN(output_fake, True, dis_update=False)
                fm_loss = self.featurematching(
                net_D_output['fake_features'], net_D_output['pseudo_real_features'])
                loss_G_adv = self.hparams.vgg_weight * fm_loss + gan_loss
                loss_G_adv = loss_G_adv[0]
            else:
                out = DiffAugment(torch.cat((rgb, label_gan.clone(), fake[:b]), 1), DiffAugment=DiffAugment_flag)
                rgb = out[:, :3]
                label_gan_crop = out[:, 3:]
                output_D = self.D(rgb)
                loss_G_adv = self.criterionGAN.loss(output_D, label_gan_crop, for_real=True, cur_ep = self.current_epoch)
            criterion = nn.BCELoss()
            target = torch.where(sky_mask.reshape(b,-1)==True, torch.zeros_like(results['weights']), torch.ones_like(results['weights'])).to(results['weights']).reshape(-1)
            if self.hparams.dis_weight >0.001:
                loss +=  self.hparams.dis_weight * loss_G_adv 
            log['train/loss_G'] = loss_G_adv #+ loss_G_adv2
            log['train/loss'] = loss
        if self.hparams.dis_weight > 0 and (optimizer_idx == 1):
            loss_D = 0
            out = DiffAugment(torch.cat((fake[:b], label_gan.clone()), 1), DiffAugment=DiffAugment_flag, isTrue=True)
            fake_true = out[:, :3]
            label_gan_true = out[:, 3:]
            rgb = self.rgb
            out = DiffAugment(torch.cat((rgb, label_gan.clone()), 1), DiffAugment=DiffAugment_flag, isTrue=True)
            output_D_fake = out[:, :3]
            label_gan_fake = out[:, 3:]
            if self.hparams.use_fpse >0:
                incl_real = False
                incl_pseudo_real = True
                data={'fake_masks': label_gan_true, 'pseudo_real_img': fake_true}
                net_D_output = self.D(data, output_D_fake, incl_real=incl_real, incl_pseudo_real=incl_pseudo_real)
                output_fake = net_D_output['fake_outputs']
                output_pseudo_real = net_D_output['pseudo_real_outputs']
                fake_loss = self.criterionGAN(output_fake, False, dis_update=True)
                true_loss = self.criterionGAN(output_pseudo_real, True, dis_update=True)
                loss_D += fake_loss + true_loss
                loss_D = loss_D[0]
            else:
                output_D_fake = self.D(output_D_fake)
                loss_D_fake = self.criterionGAN.loss(output_D_fake, label_gan_fake, for_real=False, cur_ep = self.current_epoch)
                loss_D += loss_D_fake # + loss_D_fake2
                output_D_real = self.D(fake_true)
                if self.hparams.edge_mask == 'True':
                    loss_D_real = self.criterionGAN.loss(output_D_real, label_gan_true, for_real=True, cur_ep = self.current_epoch)
                else:
                    loss_D_real = self.criterionGAN.loss(output_D_real, label_gan_true, for_real=True, cur_ep = self.current_epoch)
                loss_D += loss_D_real
            loss += loss_D/2
            log['train/loss_dis'] = loss_D
            self.loss_D = loss_D
        log['train/loss'] = loss
        log['x_max'] = self.x_max
        log['x_min'] = self.x_min


        # typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_fine = loss
            log['train/fine'] = psnr_fine

        pb = {
            'loss_dis': self.loss_D,
            'x_max': self.x_max,
            'x_min': self.x_min
        }
        return {'loss': loss,
                'progress_bar': pb,
                'log': log
                    }



    def on_train_epoch_start(self, *args, **kwargs):
        if self.current_epoch >100:
            self.bce_weight = 0
        if ((self.current_epoch+1) % 5 == 0)or(self.hparams.render_spade) or ((self.hparams.demo==True) & ((self.current_epoch+1) % 5 == 0)):
            batch_size = 1
            dataset_test = DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=batch_size,
                          pin_memory=False)
            imgs= []
            for i, batch in tqdm.tqdm(enumerate(dataset_test)):
                sample = batch
                results = self.test_render(batch, i, len(dataset_test))
                dir_name = './results/ndc/{}'.format(self.hparams.exp_name)
                spade_dir_name = './results/ndc/spade_{}'.format(self.hparams.exp_name)
                os.makedirs(dir_name, exist_ok=True)
                os.makedirs(spade_dir_name, exist_ok=True)
                for j in range(len(results['img_fine'])):
                    if 'fname' in batch:
                        fname = os.path.basename(sample['fname']).replace('.JPG', '')
                    else:
                        num = i*batch_size + j
                        fname = f'{num:03d}'
                    img_pred_ = (results['img_fine'][j]*255).detach().cpu().numpy().astype(np.uint8)
                    spade_pred_ = (results['fake'][j]*255).detach().cpu().numpy().astype(np.uint8)
                    # spade_list += [spade_pred_.transpose(1,2,0)]
                    imgs += [img_pred_.transpose(1,2,0)]
                    imageio.imwrite(os.path.join(dir_name, f'{fname}.png'), img_pred_.transpose(1,2,0))
                    imageio.imwrite(os.path.join(spade_dir_name, f'{fname}.png'), spade_pred_.transpose(1,2,0))
                    
            clip = ImageSequenceClip(imgs, fps=30)
            clip.write_videofile(os.path.join(
                dir_name, '{}.mp4'.format(self.current_epoch)), fps=30)
            np.save(os.path.join(
                dir_name, 'z.npy'), self.z.z.detach().cpu().numpy())

            # if self.hparams.demo:
                # self.update_z()


    def validation_step(self, batch, batch_nb):
        W, H = self.hparams.img_wh
        rays,  c2w, uv, idx, label_pse, fake_label, depth, points,alphas, bg_label = self.decode_batch(batch)
        self.train()
        with torch.no_grad():
            b,n,c = rays.shape
            semantic_coarse = rearrange(label_pse.clone(), 'c (h w)->c h w', w=self.W, h=self.H)
            label_gan_idx = semantic_coarse.clone()
            label_gan = torch.cuda.FloatTensor(b, self.train_dataset.nc, self.H, self.W).zero_()
            label_gan = label_gan.scatter_(1, label_gan_idx[None], 1.0)
            fake_label_gan_idx = torch.ones_like(fake_label).to(label_gan).long() * (self.train_dataset.nc-1)
            for label in torch.unique(semantic_coarse):
                mask_ = (semantic_coarse==label)
                semantic_coarse[mask_] = self.train_dataset.label_trans[label.item()]
                mask_fake = (fake_label==self.train_dataset.label_trans[label.item()])
                fake_label_gan_idx[mask_fake] = label
            fake_label_gan = torch.cuda.FloatTensor(b, self.train_dataset.nc, self.H, self.W).zero_()
            fake_label_gan = label_gan.scatter_(1, fake_label_gan_idx[None], 1.0)
            sky_mask = (semantic_coarse == 105) | (semantic_coarse == 156)
            depth.reshape(-1)[sky_mask.reshape(-1)]=20
            label_map = semantic_coarse
            edge = label_map.clone()
            edge = F.pad(edge.float(), (4,4,4,4), "replicate")
            edge = self.Sobel(edge[:, None].float())
            m = nn.MaxPool2d(9, stride=1)
            edge = m(edge)
            masks_edge = depth.clone().reshape(b,1,self.W, self.H)
            masks_edge = F.pad(masks_edge.float(), (4,4,4,4), "replicate")
            masks_edge = self.Sobel(masks_edge) > 10
            masks_edge = m(masks_edge.float())
            edge_mask = (edge > 0) & (masks_edge > 0)
            h, w = self.H, self.W
            nc = 185
            label = torch.cuda.FloatTensor(b, nc, h, w).zero_()
            semantics = label.clone().scatter_(1, label_map[None], 1.0)
            fake_semantics = label.clone().scatter_(1, fake_label[None], 1.0)
            if int(self.train_dataset.src_pair_name) != 74 and (int(self.train_dataset.src_pair_name) != 1):
                label_map_sky = bg_label
                inpaint_sky = ((label_map_sky==105).sum() > (label_map_sky==156).sum())
                if inpaint_sky:
                    label_map_sky[(label_map_sky!=105) & (label_map_sky!=156)] = 105
                else:
                    label_map_sky[(label_map_sky!=105) & (label_map_sky!=156)] = 156
            elif int(self.train_dataset.src_pair_name) == 74:
                label_map_sky  = 156 * torch.ones_like(bg_label).to(bg_label)
                label_map_sky[:, 60:] = bg_label[:, :256-60]
            else:
                label_map_sky = bg_label
            semantics_sky = label.clone().scatter_(1, label_map_sky[None], 1.0)
            self.eval()
            fake, z = self.gaugan_model.sample_camera(torch.cat((semantics,fake_semantics,semantics_sky),0), self.z.z)
            self.train()
            fake = fake*0.5+0.5
            fake = fake.float()
            sky_bg = fake[-1:]
            fake = fake[:-1]
            pse_fake = fake[:1]
            pse_fake = pse_fake.permute(1,0,2,3)
            rgb_mean = torch.zeros_like(pse_fake).to(fake)
            for label in torch.unique(semantic_coarse):
                mask = (semantic_coarse==label)
                rgb_mean[:,mask] = pse_fake[:,mask].mean(dim=-1,keepdim=True)
            torch.cuda.empty_cache()

        rays = rays.squeeze()  # (H*W, 3)

        with torch.no_grad():
            z_add = z.repeat(b,1)
            results = self(rays, alphas, perturb=0, render_semantic=False, z=z_add, sky_mask = sky_mask.reshape(b,-1), uv = uv, depth=depth, label=label_pse, points=points)
        log = {}

        if batch_nb % 5 == 0:
            alphas = visualize_depth(alphas.view(W,H).detach().cpu())
            if self.hparams.sky_th >0:
                sky_mask = ((points>points.max()-self.hparams.sky_th) | ((points == 0) & (depth>depth.max()-self.hparams.sky_th).squeeze())).float()
            else:
                sky_mask = sky_mask.float().reshape(*points.shape)
            results['rgb_fine'] = results['rgb_fine'] * (1-sky_mask).unsqueeze(-1) + sky_bg.permute(0,2,3,1).reshape(*results['rgb_fine'].shape)* sky_mask.unsqueeze(-1)
            img_fine = results[f'rgb_fine'].view(H, W, 3).cpu()
            img_fine = img_fine.permute(2, 0, 1)
            label_map = tens_to_lab(label_map[0], 183)
            fake_label = tens_to_lab(fake_label[0], 183)
            out = DiffAugment(torch.cat((fake[:1], torch.from_numpy(label_map).permute(2, 0, 1).reshape(*fake[:1].shape).to(fake)),1), DiffAugment=True, isTrue=True)
            fake_crop = out[:,:3]
            label_crop = out[:,3:]
            fake_crop = F.interpolate(fake_crop, scale_factor=2).detach().cpu()[0]
            label_crop = F.interpolate(label_crop, scale_factor=2).detach().cpu()[0]
            label_map = torch.from_numpy(label_map).to(img_fine).permute(2, 0, 1)
            fake_label = torch.from_numpy(fake_label).to(img_fine).permute(2, 0, 1)
            depth = visualize_depth(
                depth.view(H,W))  # (3, H, W)
            fake_LHQ = fake.detach().cpu()[1:]
            fake = fake.detach().cpu()[:1]
            edge_mask = edge_mask.float().repeat(1,3,1,1)[0].cpu()
            masks_edge = masks_edge.float().repeat(1,3,1,1)[0].cpu()
            add = edge_mask * 0.5 + img_fine * 0.5
            add_ = edge_mask * 0.5 + fake * 0.5
            add_img_fine = img_fine*0.5+label_map*0.5
            add_img_fake = fake[0]*0.5+label_map*0.5
            if self.hparams.use_fpse > 0:
                Discriminator_label = add_img_fake
            else:
                Discriminator_label = self.D(results[f'rgb_fine'].view(-1,self.H,self.W,3).permute(0,3,1,2))
                Discriminator_label = Discriminator_label[0].argmax(0).detach().cpu()
                Discriminator_label = torch.from_numpy(tens_to_lab(Discriminator_label, 183)).to(label_map).permute(2, 0, 1)

            stack = torch.stack([img_fine, fake_crop, label_crop, Discriminator_label, add_img_fake, fake[0], fake_LHQ[0], rgb_mean[:,0].cpu(), label_map, fake_label, depth, edge_mask, add, add_[0], masks_edge, alphas])
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)

        log['val_psnr'] = rays.mean()
        self.val_dataset.read_meta()
        self.train()
        return log


    def test_render(self, batch, i = 0, num=255):
        H, W = self.H, self.W
        rays,  c2w, uv, idx, label_pse, fake_label, depth, points, sky_mask_render, bg_label = self.decode_batch(batch)
        rays = rays.cuda()
        fake_label = fake_label.cuda()
        c2w = c2w.cuda()
        uv = uv.cuda()
        idx = idx
        label_pse = label_pse.cuda()
        depth = depth.cuda()
        points =  points.cuda()
        bg_label = bg_label.cuda()
        self.train()
        b,n,c = rays.shape
        rays = rays.squeeze()  # (H*W, 3)
        with torch.no_grad():
            torch.cuda.empty_cache()
            semantic_coarse = rearrange(label_pse.clone(), 'c (h w)->c h w', w=self.W, h=self.H)
            semantic_coarse = semantic_coarse.long()
            label_gan_idx = semantic_coarse.clone()
            label_gan = torch.cuda.FloatTensor(b, self.train_dataset.nc, self.H, self.W).zero_()
            label_gan = label_gan.scatter_(1, label_gan_idx[None], 1.0)
            fake_label_gan_idx = torch.ones_like(fake_label).to(label_gan).long() * (self.train_dataset.nc-1)
            for label in torch.unique(semantic_coarse):
                mask_ = (semantic_coarse==label)
                semantic_coarse[mask_] = self.train_dataset.label_trans[label.item()]
                mask_fake = (fake_label==self.train_dataset.label_trans[label.item()])
                fake_label_gan_idx[mask_fake] = label
            fake_label_gan = torch.cuda.FloatTensor(b, self.train_dataset.nc, self.H, self.W).zero_()
            fake_label_gan = label_gan.scatter_(1, fake_label_gan_idx[None], 1.0)
            sky_mask = (semantic_coarse == 105) | (semantic_coarse == 156)
            depth.reshape(-1)[sky_mask.reshape(-1)]=20
            label_map = semantic_coarse
            edge = label_map.clone()
            edge = F.pad(edge.float(), (4,4,4,4), "replicate")
            edge = self.Sobel(edge[:, None].float())
            m = nn.MaxPool2d(9, stride=1)
            edge = m(edge)
            masks_edge = depth.clone().reshape(b,1,self.W, self.H)
            masks_edge = F.pad(masks_edge.float(), (4,4,4,4), "replicate")
            masks_edge = self.Sobel(masks_edge) > 10
            masks_edge = m(masks_edge.float())
            edge_mask = (edge > 0) & (masks_edge > 0)
            h, w = self.H, self.W
            nc = 185
            label = torch.cuda.FloatTensor(b, nc, h, w).zero_()
            semantics = label.clone().scatter_(1, label_map[None], 1.0)
            fake_semantics = label.clone().scatter_(1, fake_label[None], 1.0)
            if (int(self.train_dataset.src_pair_name) != 74) and (int(self.train_dataset.src_pair_name) != 38):
                label_map_sky = bg_label
                inpaint_sky = ((label_map_sky==105).sum() > (label_map_sky==156).sum())
                if inpaint_sky:
                    label_map_sky[(label_map_sky!=105) & (label_map_sky!=156)] = 105
                else:
                    label_map_sky[(label_map_sky!=105) & (label_map_sky!=156)] = 156
            else:
                label_map_sky  = 105 * torch.ones_like(bg_label).to(bg_label)
                label_map_sky[:, 5*60:] = bg_label[:, :5*(256-60)]
                label_map_sky[label_map_sky==156] = 105
            semantics_sky = label.clone().scatter_(1, label_map_sky[None], 1.0)
            self.eval()
            if self.hparams.change_style:
                if i < num/2:
                    z = (self.z_2.z - self.z.z) * 2 * i/num + self.z.z
                else:
                    z = (self.z.z - self.z_2.z) * 2 * (i-num/2)/num + self.z_2.z
            else:
                z = self.z.z

            fake, z = self.gaugan_model.sample_camera(torch.cat((semantics,fake_semantics,semantics_sky),0), z)
            self.train()
            fake = fake*0.5+0.5
            fake = fake.float()
            torch.cuda.empty_cache()
        rays = rays.squeeze()  # (H*W, 3)
        z_add = z.repeat(b,1)
        results = self(rays, sky_mask, perturb=0, render_semantic=False, z=z_add, sky_mask = sky_mask.reshape(b,-1), uv = uv, depth=depth, label=label_pse, points=points)
        sky_mask_render = sky_mask_render.to(fake)
        with torch.no_grad():
            if torch.all(points.reshape(-1) == sky_mask_render.reshape(-1)):
                if self.hparams.sky_th>0:
                    sky_mask = ((points>points.max()-self.hparams.sky_th) | ((points == 0) & (depth>points.max()-self.hparams.sky_th).squeeze())).float()
                else:
                    sky_mask = sky_mask.float()
                results['rgb_fine'] = results['rgb_fine'].reshape(-1,3) * (1-sky_mask).reshape(-1,1) + fake[-b:].reshape(3, -1).T * sky_mask.reshape(-1,1)
                img_fine = results[f'rgb_fine']
                img_fine = img_fine.view(b, H, W, 3).cpu().permute(0,3,1,2)
            else:
                img_fine = results[f'rgb_fine']
                sky_mask_render = sky_mask_render.float().view(b, 1, H, W)
                sky_mask_render = sky_mask_render.reshape(b,-1,1).to(fake)
                img_fine = img_fine * (1-sky_mask_render) + fake[-b:].reshape(b,3,-1).permute(0,2,1) * sky_mask_render
                img_fine = img_fine.view(b, H, W, 3).cpu().permute(0,3,1,2)

        img_fine = kornia.filters.box_blur(img_fine, (5,5), border_type='replicate')
        img_fine = F.interpolate(img_fine, scale_factor=1/5)
        self.train()
        ret = {"img_fine": img_fine, "fake":fake}
        return ret

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        return {'progress_bar': {'val_psnr': mean_psnr},
                'log': {'val/psnr': mean_psnr}
                }
