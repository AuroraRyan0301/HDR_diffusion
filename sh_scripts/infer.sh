# 环境变量配置
MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 3 --num_heads 4 --num_head_channels 64 --num_heads_upsample -1 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 4"

python scripts/image_sample.py --model_path /data2/outputs/openai-2025-01-23-09-57-32-213938/model076000.pt $MODEL_FLAGS $DIFFUSION_FLAGS