import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random
seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.GroupNorm(4, *args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


# class NoiseInjection(nn.Module):
    # def __init__(self, ch, height, width):
        # super().__init__()
        # self.noise = nn.Parameter(0.001 * torch.randn(1, ch, height, width, device='cuda'), requires_grad=True)

    # def forward(self, feat, noise=None):
        # if noise is None:
        # batch, _, height, width = feat.shape
        # return torch.cat((feat, self.noise.repeat(batch,1,1,1)),1)
class NoiseInjection(nn.Module):
    def __init__(self, ch, height, width):
        super().__init__()
        self.noise = nn.Parameter(0.001 * torch.randn(1, ch, height, width, device='cuda'), requires_grad=True)

    def forward(self, feat, noise=None):
        batch, _, height, width = feat.shape
        return feat + self.noise.repeat(batch,1,1,1)

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)

class SmoothBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(conv2d(ch_in, ch_in, 3, 1, 4, dilation=4, bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    batchNorm2d(ch_in),
                                    conv2d(ch_in, ch_in, 3, 1, 4, dilation=4, bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    batchNorm2d(ch_in),
                                  conv2d(ch_in, ch_out, 3, 1, 4, dilation=4, bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.main(x)

class ConvOut(nn.Module):
    def __init__(self, in_channels, style_dim, hidden_channels=256):
        super().__init__()

        self.fc_z_cond = nn.Linear(style_dim, 2 * 2 * hidden_channels)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv2a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)

        self.conv3a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)

        self.conv4a = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv4b = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)

        self.conv4 = nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def modulate(self, x, w, b):
        w = w[..., None, None]
        b = b[..., None, None]
        return x * (w+1) + b

    def forward(self, x, z):
        r"""Forward network.

        Args:
            x (N x in_channels x H x W tensor): Intermediate feature map
            z (N x style_dim tensor): Style codes.
        """
        z = self.fc_z_cond(z)
        adapt = torch.chunk(z, 2 * 2, dim=-1)

        y = self.act(self.conv1(x))

        y = y + self.conv2b(self.act(self.conv2a(y)))
        y = self.act(self.modulate(y, adapt[0], adapt[1]))

        y = y + self.conv3b(self.act(self.conv3a(y)))
        y = self.act(self.modulate(y, adapt[2], adapt[3]))

        y = y + self.conv4b(self.act(self.conv4a(y)))
        y = self.act(y)

        y = self.conv4(y)

        return y



class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


def UpBlock_bi(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block

def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        # NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        # NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block

def UpBlockComp_bi(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        #convTranspose2d(in_planes, out_planes*2, 4, 2, 1, bias=False),
        # NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        # NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block



class Generator(nn.Module):
    def __init__(self, z_dim=256, ngf=64, nz=256, nc=3, im_size=1024, out_dim=256, noise=False, use_cnn = False):
        super(Generator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size
        triplane_per_feature = 12
        triplane_feature = triplane_per_feature * 3
        nc = triplane_feature
        self.triplane_feature = triplane_feature
        if use_cnn == False:
            self.triplane_feature_all = triplane_feature * 5# + triplane_feature
        else:
            self.triplane_feature_all = triplane_feature * 4  + triplane_feature//4
        self.use_cnn = use_cnn
        self.init = InitLayer(z_dim, channel=nfc[4])

        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock_bi(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.smooth_32 = SmoothBlock(nfc[32],nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.smooth_64 = SmoothBlock(nfc[64],nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.smooth_128 = SmoothBlock(nfc[128],nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])
        self.smooth_256 = SmoothBlock(nfc[256],nfc[256])

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])
        # self.se_512 = SEBlock(nfc[32], nfc[512])
        # self.se_1024 = SEBlock(nfc[64], nfc[1024])
        if noise == True:
            self.to_32 = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
            self.to_64 = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
            self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
            self.to_256 = conv2d(nfc[256], nc, 1, 1, 1, bias=False)
            self.to_big = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
        else:
            self.to_32 = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
            self.to_64 = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
            self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
            self.to_256 = conv2d(nfc[256], nc, 1, 1, 1, bias=False)
            self.to_big = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
        # self.to_256_v1 = conv2d(nfc[256], nc, 1, 1, 1, bias=False)
        # self.to_128_v1 = conv2d(nfc[128], nc, 1, 1, 1, bias=False)
        # self.to_big_v2 = conv2d(nfc[1024], nc//2, 3, 1, 1, bias=False)
        # self.to_big_1024 = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
        self.noise = noise
        if noise==True:
            self.noise_32 = NoiseInjection(nfc[32], 32, 32)
            self.noise_64 = NoiseInjection(nfc[64], 64, 64)
            self.noise_128 = NoiseInjection(nfc[128], 128, 128)
            self.noise_256 = NoiseInjection(nfc[256], 256, 256)
            # self.noise_512 = NoiseInjection(nc, 512, 512)

        # if im_size > 256:
        self.feat_512 = UpBlockComp_bi(nfc[256], nfc[512])
        # self.smooth_512 = SmoothBlock(nfc[512],nfc[512])
        # if use_cnn ==False:
            # self.feat_1024 = UpBlockComp(nfc[512], nfc[1024])
        # self.se_512 = SEBlock(nfc[32], nfc[512])
        # if im_size > 512:
        # self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        # self.down_from_big_128 = nn.Sequential(
                                # conv2d(nfc[128], nfc[64], 4, 2, 1, bias=False),
                                # nn.LeakyReLU(0.2, inplace=True))
        # self.down_from_big_256 = nn.Sequential(
                                    # conv2d(nfc[256], nfc[128], 4, 2, 1, bias=False),
                                    # nn.LeakyReLU(0.2, inplace=True))

        # self.down_4_256  = DownBlockComp(nfc[256], nfc[128])
        # self.down_4_128 = DownBlockComp(nfc[128], nfc[64])
        hidden_channels = 256
        self.out_in_channels = out_dim
        # self.conv1 = nn.Sequential(nn.Conv2d(64, hidden_channels, 1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)
        # self.conv3a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        # self.conv3b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)
        # self.conv4a = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        # self.conv4b = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        # self.conv4 = nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0))
        # self.conv_out = nn.Sequential(nn.Conv2d(self.out_in_channels, hidden_channels, 1, stride=1, padding=0),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0))
        self.conv_out = ConvOut(self.out_in_channels, 256, hidden_channels=hidden_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.feat_512 = UpBlockComp_half(nfc[256], nfc[512])
        # self.feat_1024 = UpBlockComp_half(nfc[512], nfc[1024])

        # self.feat_2048 = UpBlock(nfc[2048], nfc[2048])
    def forward(self, input):
        out = []
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)
        if self.noise==True:
            feat_32 = self.smooth_32(self.noise_32(feat_32))
            out.append(self.to_32(feat_32))
        else:
            out.append(self.to_32(feat_32))
        feat_64  = self.se_64(feat_4, self.feat_64(feat_32))
        if self.noise==True:
            feat_64 = self.smooth_64(self.noise_64(feat_64))
            out.append(self.to_64(feat_64))
        else:
            out.append(self.to_64(feat_64))

        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )
        if self.noise ==True:
            feat_128 = self.smooth_128(self.noise_128(feat_128))
            out.append(self.to_128(feat_128))
        else:
            out.append(self.to_128(feat_128))

        if self.noise ==True:
            feat_256 = self.se_256(feat_16,self.feat_256(feat_128))
            feat_256 = self.smooth_256(feat_256)
            out.append(self.to_256(self.noise_256(feat_256)))
        else:
            feat_256 = self.se_256(feat_16, self.feat_256(feat_128))
            out.append(self.to_256(feat_256))
        # feat_128_v1 = self.down_from_big_256(feat_256)
        # feat_256_v1 = self.feat_256_v1(feat_128_v1)
        # feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        # out.append(self.to_big(self.to_256_v1(feat_256_v1)))
        feat_512 = self.feat_512(feat_256)
        # feat_512 = self.smooth_512(feat_512)
        # feat_512 = self.se_512(feat_32, feat_512)
        # if self.noise ==True:
        # out.append(self.noise_512(self.to_big(feat_512)))
        # else:
        out.append(self.to_big(feat_512))
        # if self.use_cnn == False:
            # feat_1024 = self.feat_1024(feat_512)
            # out.append(self.to_big_v2(feat_1024))

        # feat_1024 = self.feat_1024(feat_512)
        # feat_1024 = self.se_512(feat_64, feat_1024)
        # out.append(self.to_big_1024(feat_1024))
        return out


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                                    batchNorm2d(nfc[512]),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )

        self.down_4  = DownBlockComp(nfc[512], nfc[256])
        self.down_8  = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
                            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])
        
        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, imgs, label, part=None):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)
        
        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)
        
        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        #rf_0 = torch.cat([self.rf_big_1(feat_last).view(-1),self.rf_big_2(feat_last).view(-1)])
        #rff_big = torch.sigmoid(self.rf_factor_big)
        rf_0 = self.rf_big(feat_last).view(-1)

        feat_small = self.down_from_small(imgs[1])
        #rf_1 = torch.cat([self.rf_small_1(feat_small).view(-1),self.rf_small_2(feat_small).view(-1)])
        rf_1 = self.rf_small(feat_small).view(-1)

        if label=='real':    
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            assert part is not None
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,8:])

            return torch.cat([rf_0, rf_1]) , [rec_img_big, rec_img_small, rec_img_part]

        return torch.cat([rf_0, rf_1])


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]

class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_small = nn.Sequential( 
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False), 
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )
        self.rf_small = nn.Sequential(
                            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)
        
    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)
        if label=='real':    
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf



class Generator_high(nn.Module):
    def __init__(self, z_dim=256, ngf=64, nz=256, nc=3, im_size=1024, out_dim=256, noise=False, use_cnn = False):
        super(Generator_high, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size
        triplane_per_feature = 12
        triplane_feature = triplane_per_feature * 3
        nc = triplane_feature
        self.triplane_feature = triplane_feature
        if use_cnn == False:
            self.triplane_feature_all = triplane_feature * 6# + triplane_feature
        else:
            self.triplane_feature_all = triplane_feature * 4  + triplane_feature//4
        self.use_cnn = use_cnn
        self.init = InitLayer(z_dim, channel=nfc[4])

        self.feat_8   = UpBlockComp(nfc[4], nfc[8])
        self.feat_16  = UpBlock_bi(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32])
        self.smooth_32 = SmoothBlock(nfc[32],nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.smooth_64 = SmoothBlock(nfc[64],nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])
        self.smooth_128 = SmoothBlock(nfc[128],nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])
        self.smooth_256 = SmoothBlock(nfc[256],nfc[256])

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])
        # self.se_512 = SEBlock(nfc[32], nfc[512])
        # self.se_1024 = SEBlock(nfc[64], nfc[1024])
        if noise == True:
            self.to_32 = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
            self.to_64 = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
            self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
            self.to_256 = conv2d(nfc[256], nc, 1, 1, 1, bias=False)
            self.to_512 = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
            self.to_big = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
        else:
            self.to_32 = conv2d(nfc[32], nc, 1, 1, 0, bias=False)
            self.to_64 = conv2d(nfc[64], nc, 1, 1, 0, bias=False)
            self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
            self.to_256 = conv2d(nfc[256], nc, 1, 1, 1, bias=False)
            self.to_512 = conv2d(nfc[512], nc, 3, 1, 1, bias=False)
            self.to_big = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
        # self.to_256_v1 = conv2d(nfc[256], nc, 1, 1, 1, bias=False)
        # self.to_128_v1 = conv2d(nfc[128], nc, 1, 1, 1, bias=False)
        # self.to_big_v2 = conv2d(nfc[1024], nc//2, 3, 1, 1, bias=False)
        # self.to_big_1024 = conv2d(nfc[1024], nc, 3, 1, 1, bias=False)
        self.noise = noise
        if noise==True:
            self.noise_32 = NoiseInjection(nfc[32], 32, 32)
            self.noise_64 = NoiseInjection(nfc[64], 64, 64)
            self.noise_128 = NoiseInjection(nfc[128], 128, 128)
            self.noise_256 = NoiseInjection(nfc[256], 256, 256)
            self.noise_512 = NoiseInjection(nfc[512], 512, 512)

        # if im_size > 256:
        self.feat_512 = UpBlockComp(nfc[256], nfc[512])
        # self.smooth_512 = SmoothBlock(nfc[512],nfc[512])
        # if use_cnn ==False:
            # self.feat_1024 = UpBlockComp(nfc[512], nfc[1024])
        self.se_512 = SEBlock(nfc[32], nfc[512])
        # if im_size > 512:
        self.feat_1024 = UpBlock_bi(nfc[512], nfc[1024])

        # self.down_from_big_128 = nn.Sequential(
                                # conv2d(nfc[128], nfc[64], 4, 2, 1, bias=False),
                                # nn.LeakyReLU(0.2, inplace=True))
        # self.down_from_big_256 = nn.Sequential(
                                    # conv2d(nfc[256], nfc[128], 4, 2, 1, bias=False),
                                    # nn.LeakyReLU(0.2, inplace=True))

        # self.down_4_256  = DownBlockComp(nfc[256], nfc[128])
        # self.down_4_128 = DownBlockComp(nfc[128], nfc[64])
        hidden_channels = 256
        self.out_in_channels = out_dim
        # self.conv1 = nn.Sequential(nn.Conv2d(64, hidden_channels, 1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)
        # self.conv3a = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        # self.conv3b = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False)
        # self.conv4a = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        # self.conv4b = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        # self.conv4 = nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0))
        # self.conv_out = nn.Sequential(nn.Conv2d(self.out_in_channels, hidden_channels, 1, stride=1, padding=0),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
        # nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1, bias=False),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0),
        # nn.LeakyReLU(0.2, inplace=True),
        # nn.Conv2d(hidden_channels, 3, 1, stride=1, padding=0))
        self.conv_out = ConvOut(self.out_in_channels, 256, hidden_channels=hidden_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.feat_512 = UpBlockComp_half(nfc[256], nfc[512])
        # self.feat_1024 = UpBlockComp_half(nfc[512], nfc[1024])

        # self.feat_2048 = UpBlock(nfc[2048], nfc[2048])
    def forward(self, input):
        out = []
        feat_4   = self.init(input)
        feat_8   = self.feat_8(feat_4)
        feat_16  = self.feat_16(feat_8)
        feat_32  = self.feat_32(feat_16)
        if self.noise==True:
            feat_32 = self.smooth_32(self.noise_32(feat_32))
            out.append(self.to_32(feat_32))
        else:
            out.append(self.to_32(feat_32))
        feat_64  = self.se_64(feat_4, self.feat_64(feat_32))
        if self.noise==True:
            feat_64 = self.smooth_64(self.noise_64(feat_64))
            out.append(self.to_64(feat_64))
        else:
            out.append(self.to_64(feat_64))

        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )
        if self.noise ==True:
            feat_128 = self.smooth_128(self.noise_128(feat_128))
            out.append(self.to_128(feat_128))
        else:
            out.append(self.to_128(feat_128))

        if self.noise ==True:
            feat_256 = self.se_256(feat_16,self.feat_256(feat_128))
            feat_256 = self.smooth_256(feat_256)
            out.append(self.to_256(self.noise_256(feat_256)))
        else:
            feat_256 = self.se_256(feat_16, self.feat_256(feat_128))
            out.append(self.to_256(feat_256))
        # feat_128_v1 = self.down_from_big_256(feat_256)
        # feat_256_v1 = self.feat_256_v1(feat_128_v1)
        # feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        # out.append(self.to_big(self.to_256_v1(feat_256_v1)))
        feat_512 = self.feat_512(feat_256)
        # feat_512 = self.smooth_512(feat_512)
        feat_512 = self.se_512(feat_32, feat_512)
        # if self.noise ==True:
        out.append(self.to_512(self.noise_512(feat_512)))
        # else:
        # out.append(self.to_big(feat_512))
        # if self.use_cnn == False:
            # feat_1024 = self.feat_1024(feat_512)
            # out.append(self.to_big_v2(feat_1024))

        feat_1024 = self.feat_1024(feat_512)
        # feat_1024 = self.se_512(feat_64, feat_1024)
        out.append(self.to_big(feat_1024))
        return out

