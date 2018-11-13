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
