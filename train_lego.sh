echo "=== Lego 데이터셋 학습 스크립트 ==="

# 기본 설정으로 Lego 학습
python train_nerf.py \
    --datadir ./data/nerf_synthetic/lego \
    --dataset_type synthetic \
    --white_bkgd \
    --N_coarse 64 \
    --N_fine 128 \
    --n_iters 200000 \
    --batch_size 1024 \
    --lr 5e-4 \
    --lr_decay 250 \
    --val_freq 2500 \
    --save_freq 10000 \
    --video_freq 50000 \
    --logdir ./logs

