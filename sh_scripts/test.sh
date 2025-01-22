# 环境变量配置
MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 3 --num_heads 4 --num_head_channels 64 --num_heads_upsample -1 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr_anneal_steps 120 --lr 2e-5 --batch_size 4"

# 运行训练脚本，并将其放入后台
python /root/guided-diffusion/scripts/image_train.py --data_dir /data2/my_ris_v3_final $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
