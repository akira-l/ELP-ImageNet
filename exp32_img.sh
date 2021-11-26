#!/bin/sh
#****************************************************************#
# ScriptName: exp_img.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2021-09-18 11:10
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-11-10 17:17
# Function: 
#***************************************************************#


#nohup python train_img.py > rerun_imagenet_bs512.log 2>&1 & 

#nohup python -m torch.distributed.launch --master_port 29502 ddp_train.py --tb 512 --lr 0.0002 --save './net_model/_91920_imagenet/weights__base__18_2502_0.7484_0.8966.pth' > rerun_imagenet_ddp_512_l3.log 2>&1 & 

#nohup python -m torch.distributed.launch --master_port 29502 ddp_train.py --tb 512 --lr 0.0002 --save './net_model/_91920_imagenet/weights__base__18_2502_0.7484_0.8966.pth' > rerun_imagenet_ddp_512_l3.log 2>&1 & 

#read -p "input job name : " job_discribe
time_stamp=`date "+%m-%d-%H-%M"`
#job_name="${job_discribe}-${time_stamp}"

#echo "Job name: $job_name"


log_name="resnet50_rerun-bs1024_00008_sum-ver_gamma1-${time_stamp}.log"

nohup python train_img.py --tb 1024 --tnw 16 --lr 0.0008 --alpha 0.5 > ${log_name} 2>&1 & 

