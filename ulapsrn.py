import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _Shared_Source_Residual_Block(nn.Module):
    def __init__(self, D=5, R=2):
        super(_Shared_Source_Residual_Block, self).__init__()
        self.D = D
        self.R = R
        conv_block = []
        for i in range(0, self.D):
            conv_block.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
            if i < self.D-1:
                conv_block.append(nn.LeakyReLU(0.2, inplace=True))
        self.cov_block = nn.Sequential(*conv_block)

    def forward(self, x):
        output = x
        for i in range(0, self.R):
            output = x + self.cov_block(output)
        return output

class _Distinct_Source_Residual_Block(nn.Module):
    def __init__(self, D=10, R=1):
        super(_Distinct_Source_Residual_Block, self).__init__()
        self.D = D
        self.R = R
        conv_block = []
        for i in range(0, self.D):
            conv_block.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False))
            if i < self.D-1:
                conv_block.append(nn.LeakyReLU(0.2, inplace=True))
        self.cov_block = nn.Sequential(*conv_block)

    def forward(self, x):
        output = x
        for i in range(0, self.R):
            output = output + self.cov_block(output)
        return output

class _Downsample_Block(nn.Module):
    def __init__(self):
        super(_Downsample_Block, self).__init__()
        self.down_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False), # down-sampling
            nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d(2)
        )

    def forward(self, x):
        output = self.down_block(x)
        return output

class _Upsample_Block(nn.Module):
    def __init__(self):
        super(_Upsample_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output

class Net_Quarter_Deep_Mask(nn.Module):
    def __init__(self):
        super(Net_Quarter_Deep_Mask, self).__init__()

        recursive_block = _Distinct_Source_Residual_Block

        self.conv_input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.en_feature_original = self.make_layer(recursive_block)
        self.en_downsample_half = self.make_layer(_Downsample_Block)
        self.en_feature_half = self.make_layer(recursive_block)
        self.en_downsample_quarter = self.make_layer(_Downsample_Block)
        self.en_feature_quarter = self.make_layer(recursive_block)
        self.conv_R_quarter = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        input_mask = (x[0] == 0).float()
        data_quarter = x[1]
        input = torch.cat((x[0], input_mask), dim=1)
        en_feature_original  = self.en_feature_original(self.relu(self.conv_input(input)))
        en_downsample_half = self.en_downsample_half(en_feature_original)
        en_feature_half = self.en_feature_half(en_downsample_half)
        en_downsample_quarter = self.en_downsample_quarter(en_feature_half)
        en_feature_quarter = self.en_feature_quarter(en_downsample_quarter)
        refined_quarter = data_quarter + self.conv_R_quarter(en_feature_quarter)
        return refined_quarter, en_feature_quarter, en_feature_half, en_feature_original

class Net_Quarter_Half(nn.Module):
    def __init__(self, pretrained=None):
        super(Net_Quarter_Half, self).__init__()

        recursive_block = _Distinct_Source_Residual_Block
        # self.QuarterNet = Net_Quarter()
        self.QuarterNet = Net_Quarter_Deep_Mask()
        self.de_upsample_half = self.make_layer(_Upsample_Block)
        self.de_feature_half = self.make_layer(recursive_block, 20, 1) # this is long decoder for large receptive field
        self.conv_R_half = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample_img_half = nn.Upsample(scale_factor=2, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, *args):
        layers = []
        layers.append(block(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        input, data_quarter = x
        refined_quarter, en_feature_quarter, en_feature_half, en_feature_original = self.QuarterNet((input, data_quarter))
        de_upsample_half = self.de_upsample_half(en_feature_quarter) + en_feature_half
        de_feature_half = self.de_feature_half(de_upsample_half)
        refined_half = self.upsample_img_half(refined_quarter) + self.conv_R_half(de_feature_half)
        return refined_half, refined_quarter, de_feature_half, en_feature_original


class Net_Quarter_Half_Mapping(nn.Module):
    def __init__(self, pretrained=None):
        super(Net_Quarter_Half_Mapping, self).__init__()

        recursive_block = _Distinct_Source_Residual_Block
        # self.QuarterNet = Net_Quarter()
        self.QuarterNet = Net_Quarter_Deep_Mask()
        self.de_upsample_half = self.make_layer(_Upsample_Block)
        self.de_feature_half = self.make_layer(recursive_block, 10, 1) # this is long decoder for large receptive field
        self.mapping_feature_half = self.make_layer(recursive_block, 10, 1)
        self.conv_R_half = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample_img_half = nn.Upsample(scale_factor=2, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, *args):
        layers = []
        layers.append(block(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        input, data_quarter = x
        refined_quarter, en_feature_quarter, en_feature_half, en_feature_original = self.QuarterNet((input, data_quarter))
        de_upsample_half = self.de_upsample_half(en_feature_quarter) + self.mapping_feature_half(en_feature_half)
        de_feature_half = self.de_feature_half(de_upsample_half)
        refined_half = self.upsample_img_half(refined_quarter) + self.conv_R_half(de_feature_half)
        return refined_half, refined_quarter, de_feature_half, en_feature_original

class Net_Quarter_Half_Original(nn.Module):
    def __init__(self):
        super(Net_Quarter_Half_Original, self).__init__()

        recursive_block = _Distinct_Source_Residual_Block
        self.HalfNet = Net_Quarter_Half()
        self.de_upsample_original = self.make_layer(_Upsample_Block)
        self.de_feature_original = self.make_layer(recursive_block, 40, 1)
        self.conv_R_original = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample_img_original = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        # self.upsample_img_original = nn.Upsample(scale_factor=2, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, *args):
        layers = []
        layers.append(block(*args))
        return nn.Sequential(*layers)

    def forward(self, x):
        input, data_quarter = x
        refined_half, refined_quarter, de_feature_half, en_feature_original = self.HalfNet((input, data_quarter))
        de_upsample_original = self.de_upsample_original(de_feature_half) + en_feature_original
        de_feature_original = self.de_feature_original(de_upsample_original)
        refined_original = self.upsample_img_original(refined_half) + self.conv_R_original(de_feature_original)
        return refined_original, refined_half, refined_quarter

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        parameters_share = False
        recursive_block = _Distinct_Source_Residual_Block

        # F1 is just downsampled, F2 passed feature embedding
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.en_feature_original = self.make_layer(recursive_block)
        self.en_downsample_half = self.make_layer(_Downsample_Block)
        self.en_feature_half = self.make_layer(recursive_block)
        if parameters_share == False:
            self.en_downsample_quarter = self.make_layer(_Downsample_Block)
            self.en_feature_quarter = self.make_layer(recursive_block)
        else:
            self.en_downsample_quarter = self.en_downsample_half
            self.en_feature_quarter = self.en_feature_half

        self.de_feature_quarter = self.make_layer(recursive_block) # takes the features right before the image
        self.de_upsample_half = self.make_layer(_Upsample_Block)
        if parameters_share == False:
            self.de_feature_half = self.make_layer(recursive_block)
            self.de_upsample_original = self.make_layer(_Upsample_Block)
            self.de_feature_original = self.make_layer(recursive_block)
        else:
            self.de_feature_half = self.de_feature_quarter
            self.de_upsample_original = self.de_upsample_half
            self.de_feature_original = self.de_feature_quarter

        self.de_upsample_img_half = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
        if parameters_share == False:
            self.de_upsample_img_original = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
            # self.de_upsample_img_original = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.de_upsample_img_original = self.de_upsample_img_half

        self.conv_R_quarter = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        if parameters_share == False:
            self.conv_R_half = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_R_original = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv_R_half = self.conv_R_quarter
            self.conv_R_original = self.conv_R_quarter

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        input, data_quarter = x

        en_feature_original  = self.relu(self.conv_input(input))
        en_downsample_half = self.en_downsample_half(en_feature_original)
        en_feature_half = self.en_feature_half(en_downsample_half)
        en_downsample_quarter = self.en_downsample_quarter(en_feature_half)

        en_feature_quarter = self.en_feature_quarter(en_downsample_quarter)
        refined_quarter = data_quarter + self.conv_R_quarter(en_feature_quarter)

        de_feature_quarter = self.de_feature_quarter(en_feature_quarter)
        de_upsample_half = self.de_upsample_half(de_feature_quarter)# + en_feature_half
        de_feature_half = self.de_feature_half(de_upsample_half)
        refined_half = self.de_upsample_img_half(refined_quarter) + self.conv_R_half(de_feature_half)

        de_upsample_original = self.de_upsample_original(de_feature_half)# + en_feature_original
        de_feature_original = self.de_feature_original(de_upsample_original)
        refined = self.de_upsample_img_original(refined_half) + self.conv_R_original(de_feature_original)

        return refined_quarter, refined_half, refined

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error)
        return loss

class L1_Gradient_loss(nn.Module):
    def __init__(self):
        super(L1_Gradient_loss, self).__init__()
        self.eps = 1e-6
        self.crit = L1_Charbonnier_loss()

    def forward(self, X, Y):
        xgin = X[:,:,1:,:] - X[:,:,0:-1,:]
        ygin = X[:,:,:,1:] - X[:,:,:,0:-1]
        xgtarget = Y[:,:,1:,:] - Y[:,:,0:-1,:]
        ygtarget = Y[:,:,:,1:] - Y[:,:,:,0:-1]

        xl = self.crit(xgin, xgtarget)
        yl = self.crit(ygin, ygtarget)
        return (xl + yl) * 0.5

class Patch_Discontinuity_loss(nn.Module):
    def __init__(self, kernel_size=5):
        super(Patch_Discontinuity_loss, self).__init__()
        self.eps = 1e-6
        self.crit = nn.MSELoss()
        psize = kernel_size // 2
        self.pool_xgin = nn.MaxPool2d(kernel_size, 1, padding=psize)
        self.pool_ygin = nn.MaxPool2d(kernel_size, 1, padding=psize)
        self.pool_xgtarget = nn.MaxPool2d(kernel_size, 1, padding=psize)
        self.pool_ygtarget = nn.MaxPool2d(kernel_size, 1, padding=psize)

    def forward(self, X, Y):
        b, c, h, w = X.size()
        xgtarget = torch.abs(Y[:,:,1:,:] - Y[:,:,0:-1,:])
        ygtarget = torch.abs(Y[:,:,:,1:] - Y[:,:,:,0:-1])
        xmask = (Y[:,:,1:,:] > 0).float() * (Y[:,:,0:-1,:] > 0).float() * (xgtarget > 0.1).float()
        ymask =  (Y[:,:,:,1:] > 0).float() * (Y[:,:,:,0:-1] > 0).float() * (ygtarget > 0.1).float()
        ygin = torch.abs(X[:,:,:,1:] - X[:,:,:,0:-1]) * ymask
        xgin = torch.abs(X.narrow(2, 1, h-1) - X.narrow(2, 0, h-1)) * xmask
        xgtarget2 = xgtarget * xmask
        ygtarget2 = ygtarget * ymask

        xl = self.crit(self.pool_xgin(xgin), self.pool_xgtarget(xgtarget2))
        yl = self.crit(self.pool_ygin(ygin), self.pool_ygtarget(ygtarget2))
        return (xl + yl) * 0.5


# Tukey loss in Robust Optimization for Deep Regression (ICCV)
class TukeyLoss(nn.Module):
    def __init__(self):
        super(TukeyLoss, self).__init__()
        self.epoch = 0

    def setIter(self, epoch):
        self.epoch = epoch

    def mad(self, x):
        med = torch.median(x)
        return torch.median(torch.abs(x - med))

    def forward(self, X, Y):
        res = Y-X
        MAD = 1.4826 * self.mad(res)

        if self.epoch < 20:
            MAD = MAD * 7

        resMAD = res / MAD
        c = 4.6851
        yt = (c*c/6) * (1 - (1-(resMAD/c)**2)**3)
        yt = torch.clamp(yt, 0, c*c/6)
        return torch.mean(yt)


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class L1_Masked_Charbonnier_loss(nn.Module):
    """L1 Masked Charbonnierloss. (ignore large gap between input and GT)"""
    def __init__(self, kernel_size):
        super(L1_Masked_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
        psize = kernel_size // 2
        self.pool = nn.MaxPool2d(kernel_size, 1, padding=psize)

    def forward(self, X, Y, G):
        mask = 1 - ((self.pool(torch.abs(Y-G)) > 0.2) * (Y == 0)).float()
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps ) * mask
        loss = torch.mean(error)
        return loss


class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        self.features = nn.Sequential(
            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.features(input)
        # out = self.sigmoid(out)
        return out


class SRGAN_Discriminator(nn.Module):
    def __init__(self):
        super(SRGAN_Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.net(x)
        # batch_size = x.size(0)
        # return F.sigmoid(self.net(x).view(batch_size))

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class CNPNet(nn.Module):
    def __init__(self):
        super(CNPNet, self).__init__()
        self.feature_extraction1 = self.FeatureExtraction(2, 56)
        self.feature_extraction2 = self.FeatureExtraction(56, 56)
        self.feature_extraction3 = self.FeatureExtraction(56, 56)
        self.mapping1 = self.Mapping()
        self.mapping2 = self.Mapping()
        self.mapping3 = self.Mapping()
        self.adjustment1 = self.Adjustment()
        self.adjustment2 = self.Adjustment()
        self.adjustment3 = self.Adjustment()
        self.downsample = nn.MaxPool2d(2)
        self.upsample3 = nn.ConvTranspose2d(in_channels=56, out_channels=56, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample2 = nn.ConvTranspose2d(in_channels=56, out_channels=56, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_img = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample_avg = nn.AvgPool2d(2)

        for m in self.modules():
            print(m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def Mapping(self):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=12, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=12, out_channels=56, kernel_size=1, stride=1, padding=0, bias=False),
        )
        return layers

    def FeatureExtraction(self, in_channels, out_channels):
        layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return layers

    def Adjustment(self):
        layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=56, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
        )
        return layers

    def forward(self, x):
        input, data_half, data_quarter = x
        input_mask = (input == 0).float()
        fs1 = self.feature_extraction1(torch.cat((input, input_mask), dim=1))
        fs2 = self.feature_extraction2(self.downsample(fs1))
        fs3 = self.feature_extraction3(self.downsample(fs2))
        mp1 = self.mapping1(fs1)
        mp2 = self.mapping2(fs2)
        mp3 = self.mapping3(fs3)
        input_half = self.downsample_avg(input)
        input_quarter = self.downsample_avg(input_half)
        pred_quarter = self.adjustment3(mp3) + input_quarter
        half_feature = mp2 + self.upsample3(mp3)
        pred_half = self.adjustment2(half_feature) + input_half
        # pred_half = self.adjustment2(half_feature) + self.upsample_img(pred_quarter)
        original_feature = mp1 + self.upsample2(half_feature)
        pred_original = self.adjustment1(original_feature) + input
        # pred_original = self.adjustment1(original_feature) + self.upsample_img(pred_half)

        return pred_original, pred_half, pred_quarter

def weight_init(m):
    print(m)
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, nn.ConvTranspose2d):
        c1, c2, h, w = m.weight.data.size()
        weight = get_upsample_filter(h)
        m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
        if m.bias is not None:
            m.bias.data.zero_()
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

class CNPNet_Pyramid(nn.Module):
    def __init__(self, N=3):
        super(CNPNet_Pyramid, self).__init__()
        self.N = N
        self.feature_extraction = nn.ModuleList([self.FeatureExtraction(2, 56)])
        self.mapping = nn.ModuleList([self.Mapping()])
        self.adjustment = nn.ModuleList([self.Adjustment()])
        self.upsample = nn.ModuleList([nn.ConvTranspose2d(in_channels=56, out_channels=56, kernel_size=4, stride=2, padding=1, bias=False)])
        for i in range(0, N-1):
            self.feature_extraction.append(self.FeatureExtraction(56, 56))
            self.mapping.append(self.Mapping())
            self.adjustment.append(self.Adjustment())
            self.upsample.append(nn.ConvTranspose2d(in_channels=56, out_channels=56, kernel_size=4, stride=2, padding=1, bias=False))

        self.downsample = nn.MaxPool2d(2)
        self.downsample_avg = nn.AvgPool2d(2)
        self.upsample_img = nn.Upsample(scale_factor=2, mode='nearest')

    def Mapping(self):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=12, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=12, out_channels=56, kernel_size=1, stride=1, padding=0, bias=False),
        )
        return layers

    def FeatureExtraction(self, in_channels, out_channels):
        layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return layers

    def Adjustment(self):
        layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=56, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
        )
        return layers

    def forward(self, x):
        N = self.N
        # input, data_half, data_quarter = x
        input = x
        input_mask = (input == 0).float()
        input_masked = torch.cat((input, input_mask), dim=1)
        fs = [self.feature_extraction[0](input_masked)]
        ms = [self.mapping[0](fs[0])]
        for i in range(1, N):
            fs.append(self.feature_extraction[i](self.downsample(fs[i-1])))
            ms.append(self.mapping[i](fs[i]))
        depth_pyramid = [input]
        for i in range(1, N):
            depth_pyramid.append(self.downsample_avg(depth_pyramid[i-1]))

        r = ms[N-1]
        pred = [self.adjustment[N-1](r) + depth_pyramid[N-1]]
        j = 0
        for i in range(N-2, -1, -1):
            r = ms[i] + self.upsample[i+1](ms[i+1])
            pred.append(self.adjustment[i](r) + depth_pyramid[i])
            # pred.append(self.adjustment[i](r) + self.upsample_img(pred[j]))
            j = j + 1
        pred.reverse()
        return pred

class ResCNPNet(nn.Module):
    def __init__(self, N=3):
        super(ResCNPNet, self).__init__()
        self.N = N
        self.feature_extraction = nn.ModuleList([self.FeatureExtraction(2, 56)])
        self.mapping = nn.ModuleList([self.Mapping()])
        self.adjustment = nn.ModuleList([self.Adjustment()])
        self.upsample = nn.ModuleList([nn.ConvTranspose2d(in_channels=56, out_channels=56, kernel_size=4, stride=2, padding=1, bias=False)])
        for i in range(0, N-1):
            self.feature_extraction.append(self.FeatureExtraction(56, 56))
            self.mapping.append(self.Mapping())
            self.adjustment.append(self.Adjustment())
            self.upsample.append(nn.ConvTranspose2d(in_channels=56, out_channels=56, kernel_size=4, stride=2, padding=1, bias=False))

        self.downsample = nn.MaxPool2d(2)
        self.downsample_avg = nn.AvgPool2d(2)
        self.upsample_img = nn.Upsample(scale_factor=2, mode='bilinear')

    def Mapping(self):
        layers = nn.Sequential(
            ResidualBlock(56),
            ResidualBlock(56),
        )
        return layers

    def FeatureExtraction(self, in_channels, out_channels):
        layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )
        return layers

    def Adjustment(self):
        layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=56, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False),
        )
        return layers

    def forward(self, x):
        N = self.N
        input, data_half, data_quarter = x
        input_mask = (input == 0).float()
        input_masked = torch.cat((input, input_mask), dim=1)
        fs = [self.feature_extraction[0](input_masked)]
        ms = [self.mapping[0](fs[0])]
        for i in range(1, N):
            fs.append(self.feature_extraction[i](self.downsample(fs[i-1])))
            ms.append(self.mapping[i](fs[i]))
        depth_pyramid = [input]
        for i in range(1, N):
            depth_pyramid.append(self.downsample_avg(depth_pyramid[i-1]))

        r = ms[N-1]
        pred = [self.adjustment[N-1](r) + depth_pyramid[N-1]]
        j = 0
        for i in range(N-2, -1, -1):
            r = ms[i] + self.upsample[i+1](ms[i+1])
            pred.append(self.adjustment[i](r) + depth_pyramid[i])
            # pred.append(self.adjustment[i](r) + self.upsample_img(pred[j]))
            j = j + 1
        pred.reverse()
        return pred

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.prelu(x)
        # x = self.conv(x)
        # x = self.pixel_shuffle(x)
        # x = self.prelu(x)
        return x

class SRGAN_Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(SRGAN_Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        # return (F.tanh(block8) + 1) / 2
        return block8

class SRGAN_Discriminator(nn.Module):
    def __init__(self):
        super(SRGAN_Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        # return self.net(x)
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))



class MyNewFlowerNetwork(nn.Module):
    def __init__(self, num_of_classes):
        self.dpn = model_factory.create_model('dpn131', num_classes=1000, pretraiend=False)
        self.dpn.load_state_dict(torch.load('./pretrained/dpn131.pth'))
        self.dpn.classifier = nn.Conv2(2688, 100, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        return self.dpn.forward(x)
