import torch.nn.functional as F
import torch
import numpy as np
import cv2 as cv
import os
device = torch.device('cuda:0')

def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [n, h, w] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=True):
    """
        Crop before upsample.
        proto_out: [mask_dim, mask_h, mask_w]
        out_masks: [n, mask_dim], n is number of masks after nms
        bboxes: [n, 4], n is number of masks after nms
        shape:input_image_size, (h, w)
        return: h, w, n
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

def format_img(np_img):
    # YOLOV5 输入的预处理
    _H, _W, _ = np_img.shape
    im = np.zeros((640, 640, 3), dtype=np.uint8)
    im[...] = 114
    factor_w = _W / 640
    factor_h = _H / 640
    factor = max(factor_w, factor_h)

    img = cv.resize(np_img, (int(_W / factor), int(_H / factor)))
    _H, _W, _ = img.shape
    dif_w = int((640 - _W) / 2)
    dif_h = int((640 - _H) / 2)
    im[dif_h: dif_h + _H, dif_w: dif_w + _W] = img

    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).half().to(device)  # float() half() 要与模型匹配
    im /= 255  # 0 - 255 to 0.0 - 1.0
    # 返回一些预处理后的图像, 和预处理时附加的信息
    return im, dif_w, dif_h, factor

def from_path_get_name(path):
    name, img_type = os.path.basename(path).split('.')
    return name, img_type
