import torch
from thop import profile

# 加载3D模型
from inference.commons import get_pose3d_predictor

ckpt_dir = 'checkpoint/detectron_pt_coco'
ckpt_name = 'arc_27_epoch_40.bin'
filter_widths = [3, 3, 3]
pose3d_predictor = get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, channels=1024)

model_params = 0
for parameter in pose3d_predictor.parameters():
    model_params += parameter.numel()
print('Trainable parameter count:', model_params)

if __name__ == '__main__':
    pass

