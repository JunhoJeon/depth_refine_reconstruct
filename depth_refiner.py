import sys
import io
import torch
from torch.autograd import Variable
import numpy as np
import math
import cv2
import scipy
import scipy.misc
from PIL import Image
from ulapsrn import Net_Quarter_Half_Original


def generate_grid(h, w, fov):
    x = (torch.arange(1, w + 1) - (w + 1) / 2) / (w / 2) * math.tan(fov / 2 / 180 * math.pi)
    y = -(torch.arange(1, h + 1) - (h + 1) / 2) / (h / 2) * math.tan(fov / 2 / 180 * math.pi) * (h / w)
    grid = torch.stack([x.repeat(h, 1), y.repeat(w, 1).t(), torch.ones(h, w, dtype=torch.int64)], 0)
    return grid.type(torch.FloatTensor)


def get_normal(x):
    [b, c, h, w] = x.size()
    grid = generate_grid(482, 642, 60)
    ph = (482 - h) // 2
    pw = (642 - w) // 2
    grid = grid.narrow(1, ph + 1, h).narrow(2, pw + 1, w)
    padding = torch.nn.ReflectionPad2d((1, 1, 1, 1))
    v = x.repeat(1, 3, 1, 1)
    pv = padding(v * grid)
    gx = pv.narrow(3, 0, w).narrow(2, 0, h) / 2 - pv.narrow(3, 2, w).narrow(2, 0, h) / 2
    gy = pv.narrow(2, 2, h).narrow(3, 0, w) / 2 - pv.narrow(2, 0, h).narrow(3, 0, w) / 2
    crs = gx.cross(gy, 1)
    norm = crs.norm(2, 1, keepdim=True).repeat(1, 3, 1, 1)
    n = -crs / (norm.clamp(min=1e-8))
    return n


class DepthPredictor:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        print("===> Building model")
        self.model = Net_Quarter_Half_Original()
        print("===> Setting model")
        self.model = self.model.cuda(self.gpu_id)
        # weights = torch.load('model_original_deep_mask_normal_ssim_tv_upconv/model_epoch_99.pth')
        weights = torch.load('./model_epoch_99.pth')
        self.model.load_state_dict(weights['model'].state_dict())
        self.model.eval()
        print('=> warming up')
        data_bytes = open('raw_depth0195.png', 'rb').read()
        pil_img = Image.open(io.BytesIO(data_bytes))
        self.__call__(pil_img)
        print('=> loading done')

    def __call__(self, pil_input):
        with torch.no_grad():
            print("===> Loading Input Depth")
            # pil_input = pil_input.crop((40, 60, 600, 460))
            img_numpy = np.array(pil_input).astype(np.uint16)
            h, w = img_numpy.shape
            img_numpy_quarter = np.array(pil_input.resize((w // 4, h // 4), Image.NEAREST)).astype(np.uint16)
            input = Variable(torch.from_numpy(img_numpy.astype(np.int32)).unsqueeze(0).unsqueeze(0)).float()/1000.0
            input_quarter = Variable(torch.from_numpy(img_numpy_quarter.astype(np.int32)).unsqueeze(0).unsqueeze(0)).float() / 1000.0
            input = input.cuda(self.gpu_id)
            input_quarter = input_quarter.cuda(self.gpu_id)
            print("===> Testing")
            pred_original, pred_half, pred_quarter = self.model((input, input_quarter))
            res = pred_original

            depth = res.data.squeeze().cpu().numpy()
            res_med = torch.from_numpy(cv2.medianBlur(depth, 3)).unsqueeze(0).unsqueeze(0)
            res_med_img = (res_med[0][0]*1000).numpy().astype(np.uint16)

            # auxiliary process for surface normal estimation
            # normal_input = (get_normal(input.cpu()) + 1) / 2.0
            # normal_output_med = (get_normal(res_med.cpu()) + 1) / 2.0

            # normal_input = scipy.misc.toimage(normal_input.squeeze().data.cpu().numpy().transpose((1, 2, 0)))
            # normal_output_med = scipy.misc.toimage(normal_output_med.squeeze().data.cpu().numpy().transpose((1, 2, 0)))
            return res_med_img

    def get_input(self, pil_input):
        with torch.no_grad():
            print("===> Loading Input Depth")
            # pil_input = pil_input.crop((40, 60, 600, 460))
            img_numpy = np.array(pil_input).astype(np.uint16)
            h, w = img_numpy.shape
            input = Variable(torch.from_numpy(img_numpy.astype(np.int32)).unsqueeze(0).unsqueeze(0)).float() / 1000.0
            normal_input = (get_normal(input) + 1) / 2.0
            normal_input = scipy.misc.toimage(normal_input.squeeze().data.cpu().numpy().transpose((1, 2, 0)))
            return normal_input
