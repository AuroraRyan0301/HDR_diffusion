MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 3 --num_heads 4 --num_head_channels 64 --num_heads_upsample -1 --learn_sigma True --resblock_updown True --dropout 0.1 --use_fp16 True --attention_resolutions 32,16,8 --use_new_attention_order False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm True"

python scripts/image_sample.py --model_path /data2/indoor.pt $MODEL_FLAGS $DIFFUSION_FLAGS