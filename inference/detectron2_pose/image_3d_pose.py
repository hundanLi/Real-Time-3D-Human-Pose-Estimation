#!/usr/bin/env python
# coding: utf-8

# # 图像3D人体姿态估计

# In[1]:


import cv2
import matplotlib.pyplot as plt
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import os
import time
import torch
import sys

sys.path.append('../../')


# ## 2. 2d关节点检测

# ### 2.1 detectron2

# #### 2.1.1 加载2d检测器

# In[4]:


def init_kps_predictor(config_path, weights_path, cuda=True):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = weights_path
    if not cuda:
        cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    return predictor


model_config_path = './keypoint_rcnn_R_101_FPN_3x.yaml'
model_weights_path = './model_R101.pkl'
kps_predictor = init_kps_predictor(model_config_path, model_weights_path, cuda=False)


# #### 2.1.2 检测2d关节点坐标

# In[5]:


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def predict_kps(kps_predictor, img):
    """
        kps_predictor: The detectron's 2d keypoints predictor
        img_generator:  Images source
    """
    # Predict kps:

    pose_output = kps_predictor(img)

    if len(pose_output["instances"].pred_boxes.tensor) > 0:
        cls_boxes = pose_output["instances"].pred_boxes.tensor[0].cpu().numpy()
        cls_keyps = pose_output["instances"].pred_keypoints[0].cpu().numpy()
    else:
        cls_boxes = np.full((4,), np.nan, dtype=np.float32)
        cls_keyps = np.full((17, 3), np.nan, dtype=np.float32)  # nan for images that do not contain human

    return cls_keyps


# ## 3. 3d姿势检测

# ### 3.1 加载3d预测模型

# In[31]:


# prev_dir = os.getcwd()
# os.chdir('/home/li/python/pose-estimation/3d/VideoPose3D')
# print("Change work dir from {} to {}".format(prev_dir, os.getcwd()))
from common.model import *


def get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, causal=False):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print('Loading checkpoint', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))

    pose3d_predictor = TemporalModel(17, 2, 17, filter_widths=filter_widths, causal=causal)
    receptive_field = pose3d_predictor.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pose3d_predictor.load_state_dict(checkpoint['model_pos'])

    if torch.cuda.is_available():
        pose3d_predictor = pose3d_predictor.cuda()

    return pose3d_predictor


ckpt_dir = '../../checkpoint/detectron_pt_coco'
ckpt_name = 'arc_1_ch_1024_epoch_40.bin'
filter_widths = [1, 1, 1]
pose3d_predictor = get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths)
_ = pose3d_predictor.eval()

# ### 3.2 预处理2d keypoints

# In[32]:


from common.camera import *
from common.generators import UnchunkedGenerator

# keypoints_symmetry= [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
# kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
# joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
# kps = torch.from_numpy(kps).unsqueeze(0)
# print(kps.size())
receptive_field = pose3d_predictor.receptive_field()
pad = (receptive_field - 1) // 2  # Padding on each side
causal_shift = 0

# ### 3.3 画图辅助函数

# In[33]:


from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D


# 画图
# noinspection PyUnresolvedReferences
def render_image(pos_3d, skeleton, azim, input_video_frame, fig):
    ax = fig.add_subplot(121)
    ax.set_aspect('equal')
    ax.imshow(input_video_frame)
    ax.axis("off")

    # 3D
    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(elev=15., azim=azim)
    # set 长度范围
    radius = 2
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, 2])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_aspect('equal')
    # 坐标轴刻度
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5

    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()

    pos = pos_3d['Reconstruction'][0]
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        col = 'red' if j in skeleton.joints_right() else 'black'
        # 画图3D
        ax.plot([pos[j, 0], pos[j_parent, 0]],
                [pos[j, 1], pos[j_parent, 1]],
                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col)


# ### 3.4 图像3d姿势估计

# In[34]:


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
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


def predict_images(image_dir: str = '../images'):
    import os
    filenames = os.listdir(image_dir)
    image_files = [os.path.join(image_dir, fn) for fn in filenames]
    for i, img_file in enumerate(image_files):
        figure = plt.figure(figsize=(12, 6), dpi=100)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        hw = (512, int(h / w * 512))
        img = cv2.resize(img, hw)

        # 二维姿态生成器
        start = time.time()
        kps = predict_kps(kps_predictor, img)
        # print(kps)
        print("Spending {:.2f} seconds to predict 2d pose.".format(time.time() - start))
        # 标准化，去掉概率列，只保留坐标值
        kps = normalize_screen_coordinates(kps[..., :2], w=img.shape[1], h=img.shape[0])

        kps = torch.from_numpy(kps).unsqueeze(0).numpy()
        # 创建生成器作为3d预测器的输入
        generator = UnchunkedGenerator(None, None, [kps], pad=pad, causal_shift=causal_shift, augment=False)

        # 三维姿态估计
        start = time.time()

        prediction = predict_3d_pos(generator, pose3d_predictor)
        rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
        prediction = camera_to_world(prediction, R=rot, t=0)

        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        pos_3d = {'Reconstruction': prediction}

        # 渲染图像
        render_image(pos_3d=pos_3d, skeleton=Skeleton(),
                     azim=np.array(70., dtype=np.float32),
                     input_video_frame=img, fig=figure)

        elapsed = time.time() - start
        print("Spending {:.2f} seconds to predict image: {}".format(elapsed, img_file))

        figure.tight_layout()

        plt.savefig("images/" + str(i + 1) + '.png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    predict_images()
