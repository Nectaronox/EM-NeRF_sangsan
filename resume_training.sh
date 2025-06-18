echo "=== 체크포인트에서 학습 재개 ==="

# 가장 최근 체크포인트 찾기
LATEST_CKPT=$(ls -t ./logs/*/checkpoint_*.pt | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Resuming from $LATEST_CKPT"

python train_nerf.py \
    --datadir ./data/nerf_synthetic/lego \
    --dataset_type synthetic \
    --white_bkgd \
    --resume $LATEST_CKPT \
    --n_iters 300000 \
    --logdir ./logs
