echo "=== 빠른 테스트를 위한 저해상도 학습 ==="

python train_nerf.py \
    --datadir ./data/nerf_synthetic/lego \
    --dataset_type synthetic \
    --white_bkgd \
    --downsample 8.0 \
    --N_coarse 32 \
    --N_fine 64 \
    --n_iters 50000 \
    --batch_size 512 \
    --lr 5e-4 \
    --lr_decay 250 \
    --val_freq 1000 \
    --save_freq 5000 \
    --video_freq 10000 \
    --logdir ./logs_quick
