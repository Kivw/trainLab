#!/bin/bash

# Usage: ./run_train.sh 0,1,2,3 path/to/config.yaml

# 第一个参数：CUDA_VISIBLE_DEVICES 列表，例如 "0,1,2,3"
export CUDA_VISIBLE_DEVICES=$1

# 第二个参数：YAML 配置文件路径
CFG_FILE=$2

# 可选：额外命令行参数
shift 2
OPTS="$@"

# 启动训练
python -u train.py 

