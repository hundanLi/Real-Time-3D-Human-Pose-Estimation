#  Real-time 3D human pose estimation.

This repository is for **real-time 3D human pose estimation**.

## Dependencies

- Ubuntu 18.04
- Python 3.6
- Pytorch 1.4.0
- GluonCV 0.6
- Numpy 1.18，opencv-python 4.2.0，matplotlib 2.2.5
- Optional: TensorFlow 1.14, detectron2

## Dataset setup

`${PROJECT_ROOT}` represents the root directory of the repository.

- 3D ground-truth poses

  ```bash
  $ cd ${PROJECT_ROOT}/data
  $ wget wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
  $ python prepare_data_h36m.py --from-archive h36m.zip
  ```

- 2D poses for training

  ```bash
  $ cd ${PROJECT_ROOT}/data
  $ wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_pt_coco.npz
  ```

## Training from scratch

For example, use the following command to train a model with a receive-field of 27-frames for 40 epochs.

```bash
$ cd ${PROJECT_ROOT}
$ python train.py -e 40 -k detectron_pt_coco -arc 3,3,3 -c checkpoint/detectron_pt_coco \ 
--checkpoint-frequency 10 --export-training-curves --no_kp_probs
```

For more about arguments, see the source code [arguments.py](common/arguments.py).

## Evaluation

To evaluation the trained model, using the following command.

```bash
$ python val.py -k detectron_pt_coco -arc 3,3,3 -c checkpoint/detectron_pt_coco \ 
 --no_kp_probs  --evaluate epoch_40.bin
```

## Inference

To inference an image or video end-to-end, running the script in the  `inference` directory. For example, using the `simple_baseline_pose` as 2D human pose estimator, you can run the `image_demo.py` under `simple_pose` directory to inference an image or run the `video_demo.py` to inference a video. You can change the code to specify an image or video file for inference before running.

To use posenet or mask-rcnn as a 2D human pose estimator, you need to learn more about [posenet-python](https://github.com/rwightman/posenet-python) and [detectron2](https://github.com/facebookresearch/detectron2).