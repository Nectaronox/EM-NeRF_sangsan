echo "=== 모든 Synthetic 객체 학습 ==="

OBJECTS=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")

for obj in "${OBJECTS[@]}"; do
    echo "Training $obj..."
    python train_nerf.py \
        --datadir ./data/nerf_synthetic/$obj \
        --dataset_type synthetic \
        --white_bkgd \
        --n_iters 200000 \
        --logdir ./logs_all/$obj
done

