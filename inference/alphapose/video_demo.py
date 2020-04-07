from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from mxnet import nd
# ++++++++++++++++++++注意+++++++++++++++++++++++++
# ！！！先导入mxnet和gluon，再导入pytorch中的torch！！！
# ++++++++++++++++++++注意+++++++++++++++++++++++++
import os
import sys
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from common.camera import camera_to_world
from common.generators import UnchunkedGenerator
from common.model import TemporalModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append("../../")

# 1. 加载目标检测器和2d关键点检测器
detector_name = ['yolo3_mobilenet1.0_coco', 'yolo3_darknet53_coco']
detector = model_zoo.get_model(detector_name[1], pretrained=True)
pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

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
    # print(x.shape, img.shape)
    # 检测人体
    class_ids, scores, bounding_boxes = detector(x)
    # 生成posenet的输入张量
    pose_input, upscale_bbox = detector_to_alpha_pose(img, class_ids, scores, bounding_boxes)
    # 预测关节点
    predict_heatmap = pose_net(pose_input)
    predict_coords, confidence = heatmap_to_coord_alpha_pose(predict_heatmap, upscale_bbox)

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


def normalize_screen_coordinates(X, w, h):
    """ 对坐标进行标准化以适应3d预测器的输入
    Args:
        X: ndarray类型的二维坐标
        w: 图像宽度
        h: 图像高度
    Returns: 标准化坐标数组

    """
    assert X.shape[-1] == 2
    # 将 x 坐标 从[0, w] 映射到 [-1, 1], 同时保留宽高比
    # noinspection PyTypeChecker
    return X / w * 2 - [1, h / w]


def get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, causal=False, channels=1024):
    """
    加载3d关节点坐标预测器
    Args:
        ckpt_dir:
        ckpt_name:
        filter_widths:
        causal:

    Returns: pose3d_predictor

    """
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print('Loading checkpoint', ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))

    pose3d_predictor = TemporalModel(17, 2, 17, filter_widths=filter_widths, causal=causal, channels=channels)
    receptive_field = pose3d_predictor.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pose3d_predictor.load_state_dict(checkpoint['model_pos'])

    return pose3d_predictor.to(device).eval()


def joints_2d_generator(joints_coords):
    """
    2d关节点坐标生成器
    Args:
        joints_coords: 坐标

    Returns: 生成器

    """
    pad = 0
    causal_shift = 0
    kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
    kps_right = [2, 4, 6, 8, 10, 12, 14, 16]
    generator = UnchunkedGenerator(None, None, [joints_coords], pad=pad, causal_shift=causal_shift, augment=True,
                                   kps_left=kps_left, kps_right=kps_right)
    return generator


# noinspection PyUnresolvedReferences
def render_image(coords_3d, skeleton, img, coords, confidence, class_ids, bboxes, scores):
    fig = plt.figure(figsize=(12, 6), dpi=150)
    canvas = FigureCanvas(fig)
    # plot input frame
    ax_in = fig.add_subplot(1, 2, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')
    utils.viz.plot_keypoints(img, coords, confidence, class_ids, bboxes, scores, ax=ax_in)

    # 3D
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d.view_init(elev=15., azim=np.array(70., dtype=np.float32))
    # set 长度范围
    radius = 2.0
    ax_3d.set_xlim3d([-radius / 2, radius / 2])
    ax_3d.set_zlim3d([0, radius * 0.8])
    ax_3d.set_ylim3d([-radius / 2, radius / 2])
    ax_3d.set_aspect('equal')
    # 坐标轴刻度
    ax_3d.set_xticklabels([])
    ax_3d.set_yticklabels([])
    ax_3d.set_zticklabels([])
    ax_3d.dist = 7.5
    ax_3d.set_title('3D Pose Reconstruction')

    # array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15])
    parents = skeleton.parents()

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        col = 'red' if j in skeleton.joints_right() else 'black'
        # 画图3D
        ax_3d.plot([coords_3d[j, 0], coords_3d[j_parent, 0]],
                   [coords_3d[j, 1], coords_3d[j_parent, 1]],
                   [coords_3d[j, 2], coords_3d[j_parent, 2]], zdir='z', c=col)

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return image


# noinspection PyMethodMayBeStatic
class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def predict_3d_pos(joints_generator, predictor):
    """
    # 预测3d坐标
    Args:
        joints_generator: 2d关键点坐标生成器
        predictor: 3d预测器

    Returns: 3d坐标

    """
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    pos_3d = []
    with torch.no_grad():
        predictor.eval()
        for _, batch, batch_2d in joints_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
            # Positional model
            predicted_3d_pos = predictor(inputs_2d)

            # Test-time augmentation (if enabled)
            if joints_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            return predicted_3d_pos.squeeze(0).cpu().numpy()


def video_pose(filepath, ckpt_dir, ckpt_name, filter_widths, show=False, channels=1024, save_file='output.mp4'):
    # 加载3d姿势估计器
    pose3d_predictor = get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, channels=channels)

    receive_field = 1
    for i in filter_widths:
        receive_field *= i
    #     print(receive_field)
    half = receive_field // 2
    # 读取视频
    cap = cv2.VideoCapture(filepath)
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # 帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    pause = int(1000 / fps)

    if show:
        # 宽高
        cv2.namedWindow('Video', 0)
        cv2.resizeWindow('Video', 960, 540)

    # 帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 保存视频文件
    wh = (1280, 720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_avi = cv2.VideoWriter('output.mp4', fourcc, fps, wh)

    coords_2d_list = []
    dicts = []
    i = 0
    # 因为设置了数据生成器的pad=0，因此需要获取前receive_field//2帧做准备
    print("Preparing...")
    while i < half:
        ret_val, frame = cap.read()
        if ret_val != 1:
            print("Video is too short!")
            output_avi.release()
            cap.release()
            cv2.destroyAllWindows()
            return
        # noinspection PyBroadException
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            continue

        # 生成2d关键点
        joints_dict = detect_2d_joints(frame)
        dicts.append(joints_dict)
        img, predict_coords = joints_dict['img'], joints_dict['coords']
        normalized_coords = normalize_screen_coordinates(predict_coords.asnumpy()[0], w=img.shape[1], h=img.shape[0])
        coords_2d_list.append(normalized_coords)
        i += 1

    print("Starting to predict 3d pose...")
    fps_time = time.time()
    while True:
        #  获取帧
        if i > receive_field and len(dicts) < 1:
            break
        i += 1
        ret_val, frame = cap.read()
        if ret_val == 1:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                continue
            # 生成2d关键点
            joints_dict = detect_2d_joints(frame)
            dicts.append(joints_dict)
            img, predict_coords = joints_dict['img'], joints_dict['coords']
            normalized_coords = normalize_screen_coordinates(predict_coords.asnumpy()[0], w=img.shape[1],
                                                             h=img.shape[0])
            coords_2d_list.append(normalized_coords)

        joints_dict = dicts[0]
        if i > half + 1:
            # 去除最左端的无用帧
            coords_2d_list = coords_2d_list[1:]
            dicts = dicts[1:]

        if len(coords_2d_list) < receive_field:
            if i < receive_field:
                # 视频开头小于receive_field帧时，在左边进行pad操作
                #                 print("kps_list length is {}, padding {} frames to left end.".format(len(kps_list), half))
                while len(coords_2d_list) < receive_field:
                    coords_2d_list.insert(0, coords_2d_list[0])
            else:
                # 视频末尾不足receive_field帧时，在右边进行pad操作
                #                 print("kps_list length is {}, padding 1 frames to right end.".format(len(kps_list)))
                coords_2d_list.append(coords_2d_list[-1])

        # 构造2d关键点生成器
        kps_2d = np.stack(coords_2d_list)
        generator = joints_2d_generator(kps_2d)
        #         print(generator.num_frames())

        # 3d关键点预测
        predictions = predict_3d_pos(generator, pose3d_predictor)
        #         print('predictions.shape: ', predictions.shape)

        rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
        predictions = camera_to_world(predictions, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        predictions[:, :, 2] -= np.min(predictions[:, :, 2])

        coords_3d = predictions[0]

        #         print('predicted {} frame, elapsed time: {:.3f} seconds.'.format(predictions.shape[0], time.time() - fps_time))
        fps = 1.0 / (time.time() - fps_time)

        # 渲染图像
        result_image = render_image(coords_3d=coords_3d, skeleton=Skeleton(), **joints_dict)
        cv2.putText(result_image, "FPS: %f" % fps, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        if show:
            # 实时显示
            cv2.imshow('Video', result_image)
            if cv2.waitKey(pause) & 0xff == ord('q'):
                break

        # resize and write
        to_write = cv2.resize(result_image, wh)
        output_avi.write(to_write)
        fps_time = time.time()
    output_avi.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = '../videos/kunkun_cut.mp4'
    filename = video_path.rsplit('/', 1)[-1]
    file_ext = filename.rsplit('.', 1)[-1]
    output_path = filename.rsplit('.', 1)[0] + "_output." + file_ext
    video_pose(video_path, ckpt_dir='../../checkpoint/detectron_pt_coco',
               ckpt_name='arc_27_ch_512_epoch_30.bin', filter_widths=[3, 3, 3], show=True, channels=512,
               save_file=output_path)
    print("Finish prediction...")
