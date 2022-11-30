import random
import time

import tensorrt as trt
import torch
from collections import OrderedDict, namedtuple
import cv2 as cv
import numpy as np
from tools import process_mask, format_img, from_path_get_name


# 模型部分
confidence = 0.5
nms_threshold = 0.4
input_shape = (1, 3, 640, 640)
N, C, H, W = input_shape
logger = trt.Logger(trt.Logger.INFO)
device = torch.device('cuda:0')
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
bindings = OrderedDict()
bindings_addrs = OrderedDict()
context = None

# 加载模型
def load_model(model_path):
    global bindings_addrs, context
    with open(model_path, 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = model.get_binding_shape(index)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        bindings_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = model.create_execution_context()


def nms(output):
    indices = cv.dnn.NMSBoxes(output[:, :4], output[:, 4], confidence, nms_threshold)
    return indices

def apply_masks(img_src, masks, format_info):
    dif_w, dif_h = format_info[1: 3]
    board = torch.zeros((H, W, C), dtype=torch.uint8, device=device)
    for i, mask in enumerate(masks):
        b, g, r = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        color = torch.tensor((b, g, r), dtype=torch.uint8, device=device)
        board[mask.bool()] = color
        b_ = board[dif_h: H - dif_h, dif_w: W - dif_w, :]  # 截取正确的部分
        mask_ = b_.cpu().numpy()
        mask_ = cv.resize(mask_, list(reversed(img_src.shape[:2])))
        img_src = cv.add(img_src, mask_)
        board[...] = 0
    return img_src


def apply_bboxes(img_src, bboxes, class_conf):
    for box in bboxes:
        box = box.astype(np.int32)
        cv.rectangle(img_src, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 3)
def process(format_info):
    if context is not None:
        img_data, dif_w, dif_h, factor = format_info
        bindings_addrs['images'] = img_data.data_ptr()
        context.execute_v2(list(bindings_addrs.values()))
        out_prob, proto_mask = bindings['output0'].data.squeeze(), bindings['output1'].data.squeeze()
        out_prob = out_prob[out_prob[:, 4] > confidence]
        out_prob[:, 0] -= out_prob[:, 2] / 2
        out_prob[:, 1] -= out_prob[:, 3] / 2
        # 续87行 xywh 形成xyx

        # 拷贝一份计算nms
        out_prob_clone = out_prob.clone()  # format时的偏差
        out_prob_clone[:, 0] -= dif_w
        out_prob_clone[:, 1] -= dif_h  # 在GPU上先做减法
        value, idx = torch.max(out_prob_clone[:, 5: -32] * out_prob_clone[:, 4: 5], dim=1)  # 相乘得到分类概率 32是解mask用到的向量
        out_prob_clone[:, 4] = value  # 写入confidence
        out_prob_clone[:, 5] = idx
        all_box = out_prob_clone[:, :6].cpu().numpy()
        indices = nms(all_box)
        boxes = all_box[indices, :4] * factor
        class_conf = all_box[indices, 4:]


        out_prob = out_prob[indices].float()
        out_prob[:, 2] += out_prob[:, 0]
        out_prob[:, 3] += out_prob[:, 1]

        # 计算mask
        masks = process_mask(proto_mask, out_prob[:, -32:], out_prob[:, :4], img_data.shape[-2:])
        return boxes, masks, class_conf
    return None, None, None

def video(camera_usb=0):
    cap = cv.VideoCapture(camera_usb)
    while 1:
        flag, img_src = cap.read()
        if not flag:
            break
        t = time.time()
        format_info = format_img(img_src)
        bboxes, masks, class_conf = process(format_info)
        if bboxes is not None:
            apply_bboxes(img_src, bboxes, class_conf)
            apply_masks(img_src, masks, format_info)
        print("FPS:", 1 / (time.time() - t))
        cv.imshow("1", img_src)
        cv.waitKey(1)

def picture(img_path):
    img_src = cv.imread(img_path)
    format_info = format_img(img_src)
    bboxes, masks, class_conf = process(format_info)
    apply_bboxes(img_src, bboxes, class_conf)
    img_src = apply_masks(img_src, masks, format_info)
    name, img_type = from_path_get_name(img_path)
    cv.imwrite(name + '-seg.' + img_type, img_src)

def main():
    model_path = "yolov5m-seg.engine"
    img_path = "zidane.jpg"
    camera_usb = 0


    load_model(model_path)
    picture(img_path)  # 处理一张图像
    # video(camera_usb) # 处理摄像头捕捉的图像


if __name__ == '__main__':
    main()
