#!/bin/bash

datapath=data
savepath=output
checkpoint=output/model_last.pth

#set PYTHONPATH
export PYTHONPATH="/content/PyTorch-RFNet:${PYTHONPATH}"

#train
python3 rfnet/train.py --batch_size=1 --iter_per_epoch 300 --datapath $datapath --savepath $savepath --num_epochs 300 --lr 2e-4 --region_fusion_start_epoch 20 

#test
python3 rfnet/test.py --batch_size=1 --datapath $datapath --savepath $savepath --checkpoint $checkpoint

