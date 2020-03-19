import errno
import os
import sys
from time import time

import torch.optim as optim

from common.arguments import parse_args
from common.camera import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.loss import *
from common.mocap_dataset import MocapDataset
from common.model import *
from common.utils import deterministic_random

args = parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mk_checkpoint_dir() -> None:
    """
    创建checkpoint目录
    Returns: None
    """
    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)


def load_dataset() -> MocapDataset:
    """
    加载数据集
    Returns: dataset
    """
    print('Loading dataset...')
    dataset_path = 'data/data_3d_' + args.dataset + '.npz'
    if args.dataset == 'h36m':
        # Human3.6M的3d关键点数据集
        from common.h36m_dataset import Human36mDataset
        dataset_ = Human36mDataset(dataset_path)
    elif args.dataset.startswith('humaneva'):
        # Human-eva的3d关键点数据集
        from common.humaneva_dataset import HumanEvaDataset
        dataset_ = HumanEvaDataset(dataset_path)
    elif args.dataset.startswith('custom'):
        # 自定义数据集是2d关键点集，用于预测3d关键点
        from common.custom_dataset import CustomDataset
        dataset_ = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
    else:
        raise KeyError('Invalid dataset')

    return dataset_


def prepare_data():
    print('Preparing data...')
    print(dataset.subjects())
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d


# noinspection PyShadowingNames
def load_2d_keypoints():
    print('Loading 2D detections...')
    keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
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


# noinspection PyShadowingNames
def fetch_data(subjects, action_filter=None, subset=1, parse_3d_poses=True):
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
    if not args.disable_optimizations and not args.dense and args.stride == 1:
        # Use optimized model for single-frame predictions
        print('Instantiate TemporalModelOptimized1f...')
        model_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                               dataset.skeleton().num_joints(),
                                               filter_widths=filter_widths, causal=args.causal,
                                               dropout=args.dropout,
                                               channels=args.channels)
    else:
        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization)
        # fall back to normal model
        model_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                    dataset.skeleton().num_joints(),
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                    channels=args.channels,
                                    dense=args.dense)
    model_eval = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                               dataset.skeleton().num_joints(),
                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                               channels=args.channels,
                               dense=args.dense)
    # add to tensorboard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('runs/')
    x = torch.randn(size=(2, 392, 17, 2), dtype=torch.float32)
    writer.add_graph(model_eval, x)

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

    model_train.to(device)
    model_eval.to(device)
    return model_train, model_eval, causal_shift, pad


if __name__ == '__main__':
    mk_checkpoint_dir()
    dataset = load_dataset()
    prepare_data()
    keypoints, keypoints_metadata, keypoints_symmetry, kps_left, kps_right, joints_left, joints_right = load_2d_keypoints()
    preprocess()

    # 用于训练和测试的对象
    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)
    # 加载测试数据集
    cameras_valid, poses_valid, poses_valid_2d = fetch_data(subjects_test, action_filter)
    # 加载训练数据集
    cameras_train, poses_train, poses_train_2d = fetch_data(subjects_train, action_filter, subset=args.subset)

    # 创建模型
    model_train, model_eval, causal_shift, pad = create_model()
    # 超参数和优化器
    lr = args.learning_rate
    optimizer = optim.Adam(model_train.parameters(), lr=lr, amsgrad=True)
    lr_decay = args.lr_decay
    initial_momentum = 0.1
    final_momentum = 0.001

    # 训练数据生成器,小批量
    train_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d,
                                       args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                       joints_right=joints_right)
    # 验证数据生成器
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    # 测试数据生成器
    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                        joints_right=joints_right)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    epoch = 0
    # 加载预训练模型参数
    if args.resume:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_train.load_state_dict(checkpoint['model_pos'])
        model_eval.load_state_dict(checkpoint['model_pos'])

        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        lr = checkpoint['lr']

    # 开始训练
    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    print("Training on ", device)
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        # 关节点总个数
        N = 0
        model_train.train()

        # 监督学习
        step = 1
        batch_time = time()
        for _, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).to(device)
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)

            # 第一列
            inputs_3d[:, :, 0] = 0
            # 梯度清零
            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_train(inputs_2d)
            # 计算误差
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            # 累加每个关节点的误差
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            # 反向传播
            loss_total = loss_3d_pos
            loss_total.backward()
            # 参数优化
            optimizer.step()

            if step % 100 == 0:
                print("epoch {}/{} step {} loss:{:.6f} time elapsed: {:.2f} seconds"
                      .format(epoch + 1, args.epochs, step, loss_total.item(), time() - batch_time))
                with open('train-log.log', 'a+') as f:
                    print("epoch {}/{} step {} loss:{:.6f} time elapsed: {:.2f} seconds"
                          .format(epoch + 1, args.epochs, step, loss_total.item(), time() - batch_time),
                          file=f)
                batch_time = time()
            step += 1
        losses_3d_train.append(epoch_loss_3d_train / N)

        # 每轮次后的验证
        if not args.no_eval:
            with torch.no_grad():
                model_eval.load_state_dict(model_train.state_dict())
                model_eval.eval()

                epoch_loss_3d_valid = 0
                epoch_loss_2d_valid = 0
                N = 0

                # 在测试集上验证
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32')).to(device)
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
                    inputs_3d[:, :, 0] = 0
                    # Predict 3D poses
                    predicted_3d_pos = model_eval(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # 在训练集上验证
                epoch_loss_3d_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32')).to(device)
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_eval(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0
        # 统计时间,单位为分钟
        elapsed = (time() - start_time) / 60
        # 日志打印
        if args.no_eval:
            print('Epoch [%d], elapsed time: %.2f min, lr: %f, 3d_train_loss: %f mm' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:
            print('Epoch [%d], elapsed time: %.2f min, lr: %f, 3d_train_loss: %f mm, 3d_eval_loss: %f mm, '
                  '3d_valid_loss: %f mm' % (
                      epoch + 1,
                      elapsed,
                      lr,
                      losses_3d_train[-1] * 1000,
                      losses_3d_train_eval[-1] * 1000,
                      losses_3d_valid[-1] * 1000))

        # 学习率指数衰减
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # 批标准化动量衰减
        momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
        model_train.set_bn_momentum(momentum)
        # 保存检查点
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_train.state_dict(),
            }, chk_path)

        # 绘制并保存训练损失曲线
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['training loss', 'eval loss on training set', 'val loss on testing set'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
            plt.close('all')
