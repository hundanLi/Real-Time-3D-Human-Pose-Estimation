#!/usr/bin/env python
# coding: utf-8

# # 基于alphapose的图像3D姿态估计
import os
import sys
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
import torch

sys.path.append('../../')

from inference.commons import get_pose3d_predictor, normalize_screen_coordinates, predict_3d_pos, Skeleton
from common.camera import camera_to_world
from common.generators import UnchunkedGenerator


# 画图函数
# noinspection PyUnresolvedReferences
def render_image(coords_3d, skeleton, azim, input_video_frame, save=True, save_path="result.jpg"):
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
        plt.savefig(save_path)


ckpt_dir = '../../checkpoint/detectron_pt_coco'
ckpt_name = 'arc_1_ch_1024_epoch_40.bin'
filter_widths = [1, 1, 1]
pose3d_predictor = get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths)

detector_name = ['yolo3_mobilenet1.0_coco', 'yolo3_darknet53_coco']
detector = model_zoo.get_model(detector_name[0], pretrained=True)
pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

# reset the detector to only detect human
# noinspection PyUnresolvedReferences
detector.reset_class(['person'], reuse_weights=['person'])


def predict(img_path):
    # 1.预处理输入图像和检测人体
    x, img = data.transforms.presets.yolo.load_test(img_path, short=256)

    start = time.time()

    # detect persons and bbox
    class_ids, scores, bounding_boxes = detector(x)
    # 2.预处理检测器的输出张量作为alpha_pose的输入
    pose_input, upscale_bbox = detector_to_alpha_pose(img, class_ids, scores, bounding_boxes)
    global detector_time
    detector_time += (time.time() - start)
    print("detector cost time: {:.3f} seconds".format(time.time() - start))


    # 3.预测关节点
    start_time = time.time()
    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
    global predictor_2d_time
    predictor_2d_time += (time.time() - start_time)
    print("2d pose predictor cost time: {:.3f} seconds".format(time.time() - start_time))

    # 4.显示2d姿态
    # ax = utils.viz.plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxes, scores, box_thresh=0.5,
    #                               keypoint_thresh=0.2)

    # 5.坐标标准化
    start_time = time.time()
    kps = normalize_screen_coordinates(pred_coords.asnumpy(), w=img.shape[1], h=img.shape[0])

    receptive_field = pose3d_predictor.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    # 6.创建生成器作为3d预测器的输入
    generator = UnchunkedGenerator(None, None, [kps], pad=pad, causal_shift=causal_shift, augment=False)

    # 7.3d姿势估计和显示
    prediction = predict_3d_pos(generator, pose3d_predictor)
    global predictor_3d_time, full_time
    predictor_3d_time += time.time() - start_time
    full_time += time.time() - start
    print("3d pose predictor cost time: {:.3f} seconds".format(time.time() - start_time))

    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    elapsed = time.time() - start
    print("Total elapsed time of predicting image {}: {:.3f} seconds".format(img_path, elapsed))
    return prediction, img


def predict_imgs(img_dir, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for parent, _, files in os.walk(img_dir):
        frames = 0
        for file in files:
            img_path = os.path.join(parent, file)
            save_path = os.path.join(result_dir, file)
            # 读取图像和预测
            prediction, img = predict(img_path)
            if prediction is None:
                print("No humans found in image file:", img_path)
                continue
            # 渲染图像
            render_image(coords_3d=prediction, skeleton=Skeleton,
                         azim=np.array(70., dtype=np.float32),
                         input_video_frame=img, save=True, save_path=save_path)
            frames += 1
            plt.close()
        print(detector_time, predictor_2d_time, predictor_3d_time, full_time)
        print("detector fps:{:.3f}, 2d fps: {:.3f}, 3d fps: {:.3f}, fps: {:.3f}".format(frames / detector_time,
                                                                                        frames / predictor_2d_time,
                                                                                        frames / predictor_3d_time,
                                                                                        frames / full_time))


def predict_img(img_path: str, show=True, save=True):
    file_name = img_path.rsplit(".", 1)[0]
    save_path = file_name.rsplit('/', 1)[-1] + "_out.jpg"
    # print(save_path)
    # 读取图像和预测
    prediction, img = predict(img_path)
    if prediction is None:
        print("No humans found in image file:", img_path)
        return
        # 渲染图像
    render_image(coords_3d=prediction, skeleton=Skeleton,
                 azim=np.array(70., dtype=np.float32),
                 input_video_frame=img, save=save, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close()


def bench_test(img_path):
    frames = 100
    start = time.time()
    for i in range(frames):
        predict_img(img_path, show=False, save=False)
    fps = frames / (time.time() - start)
    print("Fps: {:.3f}".format(fps))


if __name__ == '__main__':
    detector_time = 0
    predictor_2d_time = 0
    predictor_3d_time = 0
    full_time = 0
    image_file = "../images/mpi_inf_3dhp_354.png"
    # predict_img(image_file)

    # bench_test(image_file)

    image_dir = '../images'
    result_dir = 'output'
    predict_imgs(image_dir, result_dir)
