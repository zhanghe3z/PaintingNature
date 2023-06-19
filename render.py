import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from models.rendering import render_rays
from utils import load_ckpt
from models.datasets import dataset_dict
from models.datasets.depth_utils import *
from configs.config import Config
import cv2
from PIL import Image
from moviepy.editor import ImageSequenceClip
from models.render_anti import Renderer_anti as Renderer_anti
torch.backends.cudnn.benchmark = True

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
    save_dir = os.path.join(os.environ['TORCH_HOME'], './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    full_checkpoint_path = os.path.join(save_dir, checkpoint_path)
    if not os.path.exists(full_checkpoint_path):
        raise AssertionError("Checkpoint path is error!")
        # os.makedirs(os.path.dirname(full_checkpoint_path), exist_ok=True)
        # if is_master():
            # print('Downloading {}'.format(url))
            # if 'pbss.s8k.io' not in url:
                # url = f"https://docs.google.com/uc?export=download&id={url}"
            # download_file(url, full_checkpoint_path)
    # if dist.is_available() and dist.is_initialized():
        # dist.barrier()
    return full_checkpoint_path

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

def visualize_depth(x, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    # x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    # x_ = T.ToTensor()(x_) # (3, H, W)
    return x_




def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--change_style', default=False, action='store_true')
    parser.add_argument('--root_dir', type=str,
                        required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        help='which dataset to validate')

    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--use_cnn', default=False,action="store_true")
    parser.add_argument('--new_pose', default=False,action="store_true")
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')
    parser.add_argument('--angle', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--test_name', type=str, default='-1')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=1024 * 32 * 4,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--test', default=True)
    parser.add_argument('--use_semantic_embedding', default=False, action="store_true")
    parser.add_argument('--demo', default=False,
                        action="store_true")  # use depth
    parser.add_argument('--timestamp', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--depth_type', type=str,
                        default='nerf')  # depth supervision
    parser.add_argument('--save_depth', default=False, action="store_true")
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes', 'npy', 'png'],
                        help='which format to save')
    parser.add_argument('--model', type=str, default="nerf")
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'ft', 'clip', 'l2_ssim', 'l2_vgg'], help='loss to use')
    parser.add_argument('--patch_loss', type=str, default='mse',
                        choices=['mse', 'ft', 'clip', 'l2_ssim', 'l2_vgg'], help='loss to use')
    parser.add_argument('--dis_weight', type=float, default=0.0)
    parser.add_argument('--sky_th', type=float, default=-1.)
    parser.add_argument('--use_style', default=False, action="store_true")
    parser.add_argument('--noise', type=float,default=1)
    parser.add_argument('--use_fpse', type=float, default=0.)
    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    # chunk = 1024*32 * 8
    chunk = 1024*32 * 16
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        # render_rays_hog(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=False)
        # test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


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

if __name__ == "__main__":
    args = get_opts()
    if args.timestamp == "":
        args.timestamp = args.ckpt_path.split('/')[1]
        print("[timestamp auto set]", args.timestamp)
    w, h = args.img_wh
    far=False
    if args.test_name=='-1':
        args.test_name=''
        far=True
    dic = torch.load(args.ckpt_path)
    cfg = Config('configs/lhq.yaml')
    system = Renderer_anti(args, cfg)
    load_ckpt(system.models[0], args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(system.models[1], args.ckpt_path, model_name='nerf_b')
    load_ckpt(system.z, args.ckpt_path, model_name='z')
    dataset = dataset_dict[args.dataset_name](split='debug', **vars(args))
    args.batch_size = 1
    dataset = DataLoader(dataset,
                          shuffle=False,
                          num_workers=0,
                          batch_size=args.batch_size,
                          pin_memory=False)
    imgs = []
    psnrs = []
    default_checkpoint_path = 'lhq' + '-' + cfg.pretrained_weight + '.pt'
    checkpoint = get_checkpoint(default_checkpoint_path, cfg.pretrained_weight)
    ckpt = torch.load(checkpoint)
    system.gaugan_model.net_G.load_state_dict(ckpt['net_G'])
    system.gaugan_model.update_spade()
    dir_name = f'results/{args.dataset_name}/{args.scene_name}/{args.timestamp}'
    semantic_dir_name = f'results/{args.dataset_name}/{args.scene_name}/{args.timestamp}/semantic'
    depth_dir_name  = f'results/{args.dataset_name}/{args.scene_name}/{args.timestamp}/depth'
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(semantic_dir_name, exist_ok=True)
    os.makedirs(depth_dir_name, exist_ok=True)
    z = torch.randn(1, 256, dtype=torch.float32).cuda()
    z = z.half()
    system = system.cuda().eval()
    for i, batch in tqdm(enumerate(dataset)):
        sample = batch
        results = system.test_render(batch)
        for j in range(len(results['img_fine'])):
            if 'fname' in batch:
                fname = os.path.basename(sample['fname']).replace('.JPG', '')
            else:
                num = i*args.batch_size + j
                fname = f'{num:03d}'
            img_pred_ = (results['img_fine'][j]*255).detach().cpu().numpy().astype(np.uint8)
            imgs += [img_pred_.transpose(1,2,0)]
            imageio.imwrite(os.path.join(dir_name, f'{fname}.png'), img_pred_.transpose(1,2,0))
            if 'depth_coarse' in results:
                imageio.imwrite(os.path.join(depth_dir_name, f'{fname}.png'), (results['depth_coarse']*255).detach().cpu().numpy().astype(np.uint8).transpose(1,2,0))

    clip = ImageSequenceClip(imgs, fps=30)
    clip.write_videofile(os.path.join(
        dir_name, f'{args.scene_name}.mp4'), fps=30)

