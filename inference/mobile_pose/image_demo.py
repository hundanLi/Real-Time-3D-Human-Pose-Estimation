#!/usr/bin/env python
# coding: utf-8

# # 基于mobilenet的图像3D姿态估计
import mxnet as mx
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import heatmap_to_coord, upscale_bbox_fn, crop_resize_normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import time
import torch
import sys

from common.camera import camera_to_world
from common.model import TemporalModel
from common.generators import UnchunkedGenerator

sys.path.append('../../')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_screen_coordinates(X, w, h):
    """
    坐标标准化
    Args:
        X:
        w:
        h:

    Returns:

    """
    assert X.shape[-1] == 2

    # 将 x 坐标 从[0, w] 映射到 [-1, 1], 同时保留宽高比
    # noinspection PyTypeChecker
    return X / w * 2 - [1, h / w]


def get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, causal=False, channels=1024):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print('Loading checkpoint', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))

    pose3d_predictor = TemporalModel(17, 2, 17, filter_widths=filter_widths, causal=causal, channels=channels)
    receptive_field = pose3d_predictor.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pose3d_predictor.load_state_dict(checkpoint['model_pos'])

    return pose3d_predictor.to(device).eval()


# 画图函数
# noinspection PyUnresolvedReferences
def render_image(coords_3d, skeleton, azim, input_video_frame, save=True):
    # 人数
    num_persons = len(coords_3d)

    fig = plt.figure(figsize=(6 * (1 + num_persons), 6), dpi=100)
    # 输入图像
    ax_in = fig.add_subplot(1, 1 + num_persons, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')
    ax_in.imshow(input_video_frame, aspect='equal')

    ax_3d_list = []
    # plot 3D axes
    for i in range(num_persons):
        ax_3d = fig.add_subplot(1, 1 + num_persons, i + 2, projection='3d')
        ax_3d.view_init(elev=15., azim=azim)
        # ax_3d.view_init(elev=15, azim=70)
        # set 长度范围
        radius = 2
        ax_3d.set_xlim3d([-radius / 2, radius / 2])
        ax_3d.set_zlim3d([0, radius])
        ax_3d.set_ylim3d([-radius / 2, radius / 2])
        ax_3d.set_aspect('equal')
        ax_3d.set_title("Reconstruction-{}".format(i + 1))
        # 坐标轴刻度
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])
        ax_3d.dist = 7.5
        ax_3d_list.append(ax_3d)

    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue
        col = 'red' if j in skeleton.joints_right() else 'black'
        # 画图3D

        for pi, pos in enumerate(coords_3d):
            ax_3d_list[pi].plot([pos[j, 0], pos[j_parent, 0]],
                                [pos[j, 1], pos[j_parent, 1]],
                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)

    if save:
        plt.savefig('result.svg')


class Skeleton:
    @staticmethod
    def parents():
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    @staticmethod
    def joints_right():
        return [1, 2, 3, 9, 10]


# 预测3d坐标
def predict_3d_pos(test_generator, predictor):
    with torch.no_grad():
        predictor.eval()
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = predictor(inputs_2d)
            return predicted_3d_pos.squeeze(0).cpu().numpy()


def detector_to_mobile_pose(img, class_ids, scores, bounding_boxs,
                            output_shape=(256, 192), scale=1.25, ctx=mx.cpu(),
                            thr=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # print("class_ids.shape: ", class_ids.shape)
    # print("class_ids[0, 0, 0]", class_ids[0, 0, 0])
    L = class_ids.shape[1]
    start = time.time()
    upscale_bbox = []
    for i in range(L):
        in_start = time.time()
        if class_ids[0][i].asscalar() != 0:
            break
        # print("decision-1 time: {:.3f}".format(time.time() - in_start))
        in_start = time.time()
        if scores[0][i].asscalar() < thr:
            break
        # print("decision-2 time: {:.3f}".format(time.time() - in_start))
        bbox = bounding_boxs[0][i]
        new_bbox = upscale_bbox_fn(bbox.asnumpy().tolist(), img, scale=scale)
        upscale_bbox.append(new_bbox)
    # print("filter time: {:.3f}".format(time.time() - start))
    start = time.time()
    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(img, upscale_bbox, output_shape, mean=mean, std=std)
        pose_input = pose_input.as_in_context(ctx)
    else:
        pose_input = None
    # print("crop & transform time: {:.3f}".format(time.time() - start))
    return pose_input, upscale_bbox


# 加载3d模型
ckpt_dir = '../../checkpoint/detectron_pt_coco'
ckpt_name = 'arc_1_ch_512_epoch_40.bin'
filter_widths = [1, 1, 1]
pose3d_predictor = get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, channels=512)

# 加载2d模型
detector_name = ['yolo3_mobilenet1.0_coco', 'yolo3_darknet53_coco']
posenet_name = ['mobile_pose_mobilenetv3_small', 'mobile_pose_resnet18_v1b', 'mobile_pose_resnet50_v1b']
detector = model_zoo.get_model(detector_name[0], pretrained=True)
pose_net = model_zoo.get_model(posenet_name[1], pretrained=True)
detector.hybridize(static_alloc=True, static_shape=True)
pose_net.hybridize(static_alloc=True, static_shape=True)


# reset the detector to only detect human
# noinspection PyUnresolvedReferences
detector.reset_class(['person'], reuse_weights=['person'])

im_fname = '../images/liuyifei2.jpg'

# 1.预处理输入图像和检测人体

x, img = data.transforms.presets.yolo.load_test(im_fname, short=360)
# detector.summary(x)
# print("x.shape:", x.shape)

start = time.time()

# detect persons and bbox,
class_IDs, scores, bounding_boxes = detector(x)  # shape: [sample_idx, class_idx, instance]
# print("bounding_boxes.shape", bounding_boxes.shape, "bounding_boxes[0, 0]:", bounding_boxes[0, 0])

# 2.预处理检测器的输出张量作为mobile_pose的输入
pose_input, upscale_bbox = detector_to_mobile_pose(img, class_IDs, scores, bounding_boxes)
print("detector cost time: {:.3f} seconds".format(time.time() - start))
prepare_end = time.time()

# 4.2d关节点预测
# pose_net.summary(pose_input)
predicted_heatmap = pose_net(pose_input)
pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
# print("type(pre_coords): {}, shape(pre_coords): {}".format(type(pred_coords), pred_coords.shape))
# print("pred_coords: {}".format(pred_coords))

print("2d pose predictor cost time: {:.3f} seconds".format(time.time() - prepare_end))

# 5.显示2d姿态
ax = utils.viz.plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxes, scores, box_thresh=0.5,
                              keypoint_thresh=0.2)
# print(pred_coords)
# 6.坐标标准化
kps = normalize_screen_coordinates(pred_coords.asnumpy(), w=img.shape[1], h=img.shape[0])
# print('kps.type: {}, kps.shape: {}'.format(type(kps), kps.shape))

# 7.2d keypoints生成器
receptive_field = pose3d_predictor.receptive_field()
pad = (receptive_field - 1) // 2  # Padding on each side
causal_shift = 0

# 创建生成器作为3d预测器的输入
generator = UnchunkedGenerator(None, None, [kps], pad=pad, causal_shift=causal_shift, augment=False)

# 8.3d姿势估计和显示
start_time = time.time()
prediction = predict_3d_pos(generator, pose3d_predictor)
print("3d pose predictor cost time: {:.3f} seconds".format(time.time() - start_time))
# print("prediction.shape: ", prediction.shape)

rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
prediction = camera_to_world(prediction, R=rot, t=0)

# We don't have the trajectory, but at least we can rebase the height
prediction[:, :, 2] -= np.min(prediction[:, :, 2])
# prediction = np.stack([prediction[0]])
# 9.渲染图像
render_image(coords_3d=prediction, skeleton=Skeleton,
             azim=np.array(70., dtype=np.float32),
             input_video_frame=img)

elapsed = time.time() - start
print("Total elapsed time of 3d pose prediction: {:.3f} seconds".format(elapsed))

plt.show()

if __name__ == '__main__':
    pass
