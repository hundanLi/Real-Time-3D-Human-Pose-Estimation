import os
import sys
import time
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from mxnet import nd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import mxnet

sys.path.append("../../")

# 1. 加载目标检测器和2d关键点检测器
detector_name = ['yolo3_mobilenet1.0_coco', 'yolo3_darknet53_coco']
posenet_name = ['mobile_pose_mobilenetv3_small', 'mobile_pose_resnet18_v1b', 'mobile_pose_resnet50_v1b']
detector = model_zoo.get_model(detector_name[1], pretrained=True)
pose_net = model_zoo.get_model(posenet_name[2], pretrained=True)

# reset the detector to only detect human
# noinspection PyUnresolvedReferences
detector.reset_class(['person'], reuse_weights=['person'])


def detect_2d_joints(frame, short=360):
    """
    Args:
        short: 较短边resize大小
        frame: 任意尺寸的RGB图像

    Returns: 处理过的图像(ndarray)，关节点坐标(NDArray)以及置信度等显示2d姿势相关的要素
    """
    # 缩放图像和生成目标检测器输入张量
    frame = nd.array(frame)
    x, img = data.transforms.presets.yolo.transform_test(frame, short=short)
    print(x.shape, img.shape)
    # 检测人体
    class_ids, scores, bounding_boxes = detector(x)
    # 生成posenet的输入张量
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_ids, scores, bounding_boxes)
    # 预测关节点
    predict_heatmap = pose_net(pose_input)
    predict_coords, confidence = heatmap_to_coord(predict_heatmap, upscale_bbox)

    # 显示2d姿态
    # ax = utils.viz.plot_keypoints(img, predict_coords, confidence, class_ids, bounding_boxes, scores)

    return {
        'img': img,
        'coords': predict_coords,
        'confidence': confidence,
        'class_ids': class_ids,
        'bboxes': bounding_boxes,
        'scores': scores
    }


if __name__ == '__main__':
    im_filename = 'images/liuyifei2.jpg'
    frame = cv2.cvtColor(cv2.imread(im_filename), cv2.COLOR_BGR2RGB)
    joints_dict = detect_2d_joints(frame)
    utils.viz.plot_keypoints(**joints_dict)
    plt.show()
