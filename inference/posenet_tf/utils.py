import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
import sys
sys.path.append("../../")

from common.model import TemporalModel

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
def render_image(coords_3d, skeleton, azim, input_video_frame, save=False, show=False):
    # 人数
    num_persons = len(coords_3d)
    if num_persons == 0:
        num_persons = 1
    fig = plt.figure(figsize=(6 * (1 + num_persons), 6), dpi=100)
    canvas = FigureCanvasAgg(fig)

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
        plt.savefig('result.svg')
    if show:
        plt.show()
    else:
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close()
        return image


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
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)

            # Positional model
            predicted_3d_pos = predictor(inputs_2d)
            return predicted_3d_pos.squeeze(0).cpu().numpy()
