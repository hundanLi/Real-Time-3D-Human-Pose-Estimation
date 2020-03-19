#!/bin/bash
echo "Start training at $(date)"
source /home/li/python/virtualenv/pytorch-env/bin/activate
python train.py -e 1 -k detectron_pt_coco -arc 3,3,3,3,3 -b 16 -c checkpoint/detectron_pt_coco --checkpoint-frequency 1 --export-training-curves --no_kp_probs
deactivate
echo "Finish training at $(date)"
