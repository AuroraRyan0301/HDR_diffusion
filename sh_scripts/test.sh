MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 3 --num_heads 4 --num_head_channels 64 --num_heads_upsample -1 --learn_sigma True --resblock_updown True --dropout 0.1 --use_fp16 True --attention_resolutions 32,16,8 --use_new_attention_order False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 5e-5 --batch_size 4 --use_checkpoint True"

python scripts/image_train.py --data_dir /data2/my_ris_v3_final $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS