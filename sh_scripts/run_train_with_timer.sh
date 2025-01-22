#!/bin/bash

# 环境变量配置
MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 3 --num_heads 4 --num_head_channels 64 --num_heads_upsample -1 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 4"

# 运行训练脚本，并将其放入后台
python scripts/image_train.py --data_dir /data2/my_ris_v3_final $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS &
TRAIN_PID=$!  # 获取训练脚本的进程 ID

echo "Training started with PID $TRAIN_PID. Waiting for 5 minutes..."

# 等待 5 分钟
sleep 300

# 设置 DIFFUSION_TRAINING_TEST 环境变量并通知训练脚本退出
echo "Setting DIFFUSION_TRAINING_TEST to stop training."
export DIFFUSION_TRAINING_TEST=1
sleep 300
kill -SIGUSR1 $TRAIN_PID  # 发送信号以触发条件退出（如有必要）