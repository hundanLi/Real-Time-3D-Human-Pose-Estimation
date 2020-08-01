import torch
import cv2
import time
import argparse
import numpy as np
import sys
import posenet
from utils import get_pose3d_predictor, normalize_screen_coordinates, predict_3d_pos, render_image, Skeleton

sys.path.append("../../")
from common.camera import camera_to_world
from common.generators import UnchunkedGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载2D模型
model = posenet.load_model(args.model)
model = model.to(device)
output_stride = model.output_stride
# 加载3d模型
ckpt_dir = '../../checkpoint/detectron_pt_coco'
ckpt_name = 'arc_27_ch_512_epoch_80.bin'
filter_widths = [3, 3, 3]
pose3d_predictor = get_pose3d_predictor(ckpt_dir, ckpt_name, filter_widths, channels=512)
rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)


# 预测2d关键点坐标
def predict_2d_joints(frame, min_pose_score=0.7):
    # 读取图像文件
    input_image, draw_image, output_scale = posenet.process_input(frame, scale_factor=args.scale_factor,
                                                                  output_stride=output_stride)
    # 检测热图offset
    # noinspection PyArgumentList
    input_image = torch.Tensor(input_image).to(device)
    with torch.no_grad():
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

    # 检测坐标点
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=5,
        min_pose_score=min_pose_score)
    keypoint_coords *= output_scale
    # 显示结果
    skeleton_2d_image = posenet.draw_skel_and_kp(
        draw_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=min_pose_score, min_part_score=0.2)
    skeleton_2d_image = cv2.cvtColor(skeleton_2d_image, cv2.COLOR_BGR2RGB)
    # 交换xy在数组中的位置
    coords_2d = np.zeros_like(keypoint_coords)
    coords_2d[:, :, 0], coords_2d[:, :, 1] = keypoint_coords[:, :, 1], keypoint_coords[:, :, 0]
    return skeleton_2d_image, coords_2d


def predict_3d_joints(predictor, coords_2d, w, h):
    # 坐标标准化
    kps = normalize_screen_coordinates(coords_2d, w, h)
    # print('kps.type: {}, kps.shape: {}'.format(type(kps), kps.shape))

    # 2d keypoints生成器
    receptive_field = predictor.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    # 创建生成器作为3d预测器的输入
    generator = UnchunkedGenerator(None, None, [kps], pad=pad, causal_shift=causal_shift, augment=False)
    prediction = predict_3d_pos(generator, predictor)
    prediction = camera_to_world(prediction, R=rot, t=0)
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    return prediction


def video_pose(filepath, show=True, save=False, save_file='output.mp4'):
    receive_field = 1
    for i in filter_widths:
        receive_field *= i
    half = receive_field // 2
    cap = cv2.VideoCapture(filepath)
    w, h = int(2 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(2 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Resize width and height: ", w, h)
    fps = cap.get(cv2.CAP_PROP_FPS)
    win_name = "video"
    if show:
        cv2.namedWindow(win_name, 0)
        cv2.resizeWindow(win_name, w, h)

    output_video = None

    coords_2d_list = []
    skeleton_2d_images = []
    i = 0
    print("Preparing...")
    while i < half:
        ret, frame = cap.read()
        if not ret:
            print("Video is too short!")
            cap.release()
            cv2.destroyAllWindows()
            return
        # 预测2d关键点和渲染了2d姿势的原图像
        skeleton_2d_image, coords_2d = predict_2d_joints(frame, min_pose_score=0.5)
        if len(coords_2d) > 0:
            coords_2d_list.append(coords_2d[0])
        else:
            continue
        skeleton_2d_images.append(skeleton_2d_image)
        i += 1
    if save:
        # 输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(save_file, fourcc, fps, (w, h))

    print("Start to predict 3d pose...")
    fps_time = time.time()
    while True:
        i += 1
        if i > receive_field and len(skeleton_2d_images) < 1:
            break
        ret, frame = cap.read()
        if ret:
            skeleton_2d_image, coords_2d = predict_2d_joints(frame, min_pose_score=0.5)
            if len(coords_2d) > 0:
                coords_2d_list.append(coords_2d[0])
            else:
                continue
            skeleton_2d_images.append(skeleton_2d_image)

        draw_frame = skeleton_2d_images[0]
        if i > half + 1:
            # 去除最左端的无用帧
            coords_2d_list = coords_2d_list[1:]
            skeleton_2d_images = skeleton_2d_images[1:]

        if len(coords_2d_list) < receive_field:
            if i < receive_field:
                # 视频开头小于receive_field帧时，在左边进行pad操作
                while len(coords_2d_list) < receive_field:
                    coords_2d_list.insert(0, coords_2d_list[0])
            else:
                # 视频末尾不足receive_field帧时，在右边进行pad操作
                if len(coords_2d_list) > 0:
                    coords_2d_list.append(coords_2d_list[-1])
                else:
                    break
        # 预测3d坐标
        predictions = predict_3d_joints(pose3d_predictor, np.stack(coords_2d_list), draw_frame.shape[1],
                                        draw_frame.shape[0])
        coords_3d = predictions[:1]
        fps = 1 / (time.time() - fps_time)
        result_image = render_image(coords_3d=coords_3d, skeleton=Skeleton, azim=70., input_video_frame=draw_frame,
                                    save=False)
        cv2.putText(result_image, "FPS: %.3f" % fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        if show:
            cv2.imshow(win_name, result_image)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        if save:
            result_image = cv2.resize(result_image, (w, h))
            output_video.write(result_image)

        fps_time = time.time()
    if save:
        output_video.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Finish prediction")


if __name__ == '__main__':
    video_path = '../videos/basketball-3.mp4'
    filename = video_path.rsplit('/', 1)[-1]
    file_ext = filename.rsplit('.', 1)[-1]
    output_path = filename.rsplit('.', 1)[0] + "_output." + file_ext
    video_pose(video_path, show=True, save=True, save_file=output_path)
