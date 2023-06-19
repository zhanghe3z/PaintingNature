import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

import torch

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    
    x = depth.cpu().numpy()
    mask = (x <=0.000001)
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    x_[:, torch.from_numpy(mask)] = 1

    return x_
