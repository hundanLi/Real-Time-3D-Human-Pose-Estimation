#!/bin/bash
source ~/python/virtualenv/pytorch-env/bin/activate
python prepare_data_h36m.py --from-archive h36m.zip
deactivate
