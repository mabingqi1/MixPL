#!/bin/bash

# show env，查看申请的卡是否正常
nvidia-smi

# install, 安装一些依赖，docker如果存在可以不用安装
# apt-get update
# apt-get install -y libgl1-mesa-glx
# apt-get install -y libglib2.0-dev

# conda env, 激活自己的conda，注意conda安装路径
source /yinghepool/miniconda3/etc/profile.d/conda.sh
# export CONDA_ENVS_PATH=/yinghepool/mabingqi/envs
conda activate /yinghepool/mabingqi/envs/mmdet
# run
cd /yinghepool/mabingqi/MixPL

# check args
CONFIG=/yinghepool/mabingqi/MixPL/projects/MixPL/configs/yh_qixiongjiye@100-mixpl_dino.py
WORK_DIR=./work_dirs/$(basename ${CONFIG%.*})
GPUS=8
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29512}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} \
    --work-dir $WORK_DIR \