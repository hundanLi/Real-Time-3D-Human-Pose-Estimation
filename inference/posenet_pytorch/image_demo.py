import os

import cv2
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.path.append("../../")
import posenet

from utils import get_pose3d_predictor, normalize_screen_coordinates, predict_3d_pos, Skeleton
from common.camera import camera_to_world
from common.generators import UnchunkedGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载3d模型
ckpt_dir = '../../checkpoint/detectron_pt_coco'
ckpt_name = 'arc_1_ch_1024_epoch_40.bin'
filter_widths = [1, 1, 1]
pose3d_predictor = get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, channels=1024)
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

# 加载2D模型
model = posenet.load_model(args.model)
model = model.to(device)
output_stride = model.output_stride


# 预测2d关键点坐标
def predict_2d_joints(filename, min_pose_score=0.5, min_part_score=0.1):
    # 读取图像文件
    input_image, skeleton_2d_image, output_scale = posenet.read_imgfile(
        filename, scale_factor=args.scale_factor, output_stride=output_stride)
    # 检测热图offset
    start_time = time.time()
    with torch.no_grad():
        # noinspection PyArgumentList
        input_image = torch.Tensor(input_image).to(device)
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

    # 检测坐标点
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=10,
        min_pose_score=min_pose_score)
    keypoint_coords *= output_scale
    global predictor_2d_time
    predictor_2d_time += time.time() - start_time
    # 显示结果
    skeleton_2d_image = posenet.draw_skel_and_kp(
        skeleton_2d_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=min_pose_score, min_part_score=min_part_score)
    skeleton_2d_image = cv2.cvtColor(skeleton_2d_image, cv2.COLOR_BGR2RGB)
    # 交换xy在数组中的位置
    coords_2d = np.zeros_like(keypoint_coords)
    coords_2d[:, :, 0], coords_2d[:, :, 1] = keypoint_coords[:, :, 1], keypoint_coords[:, :, 0]
    return skeleton_2d_image, coords_2d


def predict_3d_joints(predictor, coords_2d, w, h):
    # 坐标标准化
    start_time = time.time()
    kps = normalize_screen_coordinates(coords_2d, w, h)
    # print('kps.type: {}, kps.shape: {}'.format(type(kps), kps.shape))

    # 2d keypoints生成器
    receptive_field = predictor.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    # 创建生成器作为3d预测器的输入
    generator = UnchunkedGenerator(None, None, [kps], pad=pad, causal_shift=causal_shift, augment=False)
    prediction = predict_3d_pos(generator, predictor)
    global predictor_3d_time
    predictor_3d_time += time.time() - start_time
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    return prediction


# noinspection PyUnresolvedReferences
def render_image(coords_3d, skeleton, azim, input_video_frame, save=False, save_path="result.jpg"):
    # 人数
    num_persons = len(coords_3d)
    if num_persons == 0:
        num_persons = 1
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

    if len(coords_3d) > 0:
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


def predict_img(img_path, show=True, save=True):
    start = time.time()
    save_path = img_path.rsplit(".", 1)[0].rsplit("/", 1)[-1] + "_out.jpg"
    # 检测2d关键点和渲染出2d姿势图
    draw_image, coords_2d = predict_2d_joints(img_path, min_pose_score=0.5, min_part_score=0.1)
    # print("2d predictor cost time: {:.3f} seconds.".format(time.time() - start))
    # 检测3d坐标
    prediction = predict_3d_joints(pose3d_predictor, coords_2d, draw_image.shape[1], draw_image.shape[0])
    print("Total cost time of predicting image {}: {:.3f} seconds.".format(img_path, time.time() - start))
    # 渲染2d/3d姿势图
    render_image(coords_3d=prediction, skeleton=Skeleton, azim=70., input_video_frame=draw_image, save=save,
                 save_path=save_path)
    if show:
        plt.show()
    else:
        plt.close()


def predict_imgs(image_dir, result_dir):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for parent, _, files in os.walk(image_dir):
        start = time.time()
        frames = 0
        for file in files:
            img_path = os.path.join(parent, file)
            save_path = os.path.join(result_dir, file)
            # 读取图像和预测
            draw_image, coords_2d = predict_2d_joints(img_path, min_pose_score=0.5, min_part_score=0.1)
            if len(coords_2d) == 0:
                print("No humans found in image file:", img_path)
                continue
            # 检测3d坐标
            prediction = predict_3d_joints(pose3d_predictor, coords_2d, draw_image.shape[1], draw_image.shape[0])
            frames += 1
            # 渲染图像
            render_image(coords_3d=prediction, skeleton=Skeleton, azim=70., input_video_frame=draw_image, save=True,
                         save_path=save_path)
            plt.close()
        print("2d predictor fps: {:.3f}, 3d predictor fps: {:.3f}, fps: {:.3f}".format(
            frames / predictor_2d_time,
            frames / predictor_3d_time,
            frames / (predictor_3d_time + predictor_2d_time)
        )
        )


if __name__ == "__main__":
    predictor_2d_time = 0
    predictor_3d_time = 0
    img_file = '../images/liuyifei1.jpg'
    # predict_img(img_file)

    img_dir = '../images'
    result_dir = 'output'
    predict_imgs(img_dir, result_dir)
