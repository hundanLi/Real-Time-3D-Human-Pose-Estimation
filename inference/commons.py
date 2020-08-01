import os
import sys
import torch
import numpy as np

sys.path.append('../../')
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
