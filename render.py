import os

from common.arguments import parse_args
from common.camera import normalize_screen_coordinates, camera_to_world, image_coordinates
from common.generators import UnchunkedGenerator
from common.loss import *
from common.mocap_dataset import MocapDataset
from common.model import TemporalModel
from common.utils import deterministic_random

args = parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset() -> MocapDataset:
    """
    加载数据集
    Returns: dataset
    """
    print('Loading custom dataset...')
    if args.dataset.startswith('custom'):
        # 自定义数据集是2d关键点集，用于预测3d关键点
        from common.custom_dataset import CustomDataset
        dataset_ = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    else:
        raise KeyError('Invalid dataset')

    return dataset_


# noinspection PyShadowingNames
def load_2d_keypoints():
    print('Loading 2D detections...')
    keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    print(keypoints_metadata['layout_name'])
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()
    return keypoints, keypoints_metadata, keypoints_symmetry, kps_left, kps_right, joints_left, joints_right


def preprocess():
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[
                subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
                action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                if args.keypoint_probs:
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                else:
                    kps = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            # print("len of poses_2d[0]: {}".format(len(poses_2d[0])))
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


# noinspection PyShadowingNames
def create_model():
    # 加载模型
    filter_widths = [int(x) for x in args.architecture.split(',')]
    model_eval = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                               dataset.skeleton().num_joints(),
                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                               channels=args.channels,
                               dense=args.dense)

    receptive_field = model_eval.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    if args.causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model_eval.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    model_eval.to(device)
    return model_eval, causal_shift, pad


def evaluate(generator):
    with torch.no_grad():
        model_eval.eval()
        for _, _, batch_2d in generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
            print('input_2d.shape:', inputs_2d.shape)
            # Positional model
            predicted_3d_pos = model_eval(inputs_2d)

            # Test-time augmentation (if enabled)
            if generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            return predicted_3d_pos.squeeze(0).cpu().numpy()


if __name__ == '__main__':
    dataset = load_dataset()
    keypoints, keypoints_metadata, keypoints_symmetry, kps_left, kps_right, joints_left, joints_right = load_2d_keypoints()
    preprocess()

    # 加载渲染测试集
    subjects_test = [args.viz_subject]
    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)
    # print(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints())

    # 创建模型和加载模型参数
    model_eval, causal_shift, pad = create_model()
    if not args.evaluate or not args.resume:
        print('Invalid arguments: ', args.evaluate, not args.resume)
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_eval.load_state_dict(checkpoint['model_pos'])

    if not args.render:
        print("Invalid argument:", args.render)

    # 渲染3D姿势
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()

    print('INFO: this action is unlabeled. Ground truth will not be rendered.')
    print('kps_left:', kps_left, 'kps_right:', kps_right)
    print('joints_left:', joints_left, 'joints_right:', joints_right)
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    print('INFO: Testing on {} frames'.format(gen.num_frames()))

    prediction = evaluate(gen)
    print('prediction.shape: ', prediction.shape)

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]

        # If the ground truth is not available, take the camera extrinsic params from a random subject.
        # They are almost the same, and anyway, we only need this for visualization purposes.
        rot = None
        for subject in dataset.cameras():
            if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                break
        prediction = camera_to_world(prediction, R=rot, t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
        # print('w, h:', cam['res_w'], cam['res_h'])
        from common.visualization import render_animation

        print("rot:", rot)
        print("cam['azimuth']:", cam['azimuth'])
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip, show=args.viz_show)
