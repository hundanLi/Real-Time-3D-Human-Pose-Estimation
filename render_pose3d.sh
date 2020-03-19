#!/bin/bash
source /home/li/python/virtualenv/pytorch-env/bin/activate

python render.py -d custom -k myvideos -arc 1,1,1 -c checkpoint/detectron_pt_coco/ --evaluate arc_1_epoch_40.bin \
--render --viz-subject detectron2 --viz-action custom --viz-camera 0 --viz-video ./data/input_video.mp4 \
--viz-output arc_1_epoch_40.mp4 --viz-size 6 --viz-no-ground-truth #--viz-show

deactivate
