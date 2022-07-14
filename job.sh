#!/bin/bash

batch_size=1
patch_size=80
iterations=300
epochs=300

datapath=data
savepath=output
checkpoint=output/model_last.pth

#set PYTHONPATH
export PYTHONPATH="/content/PyTorch-RFNet:${PYTHONPATH}"

#train
python3 rfnet/train.py --batch_size $batch_size --patch_size $patch_size --iter_per_epoch $iterations --datapath $datapath --savepath $savepath --num_epochs $epochs --lr 2e-4 --region_fusion_start_epoch 20 

#test
python3 rfnet/test.py --batch_size $batch_size --patch_size $patch_size --datapath $datapath --savepath $savepath --checkpoint $checkpoint

