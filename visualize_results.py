import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_training_curves(log_dir):
    """학습 곡선 시각화"""
    # 로그 파일에서 데이터 읽기
    # (실제로는 텐서보드나 wandb 사용 추천)
    pass

def create_comparison_grid(log_dir):
    """렌더링 결과 비교 그리드 생성"""
    test_images = sorted(glob.glob(os.path.join(log_dir, 'test_*.png')))
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    n_images = min(len(test_images), 9)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(n_images):
        img = plt.imread(test_images[i])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'View {i}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'test_grid.png'))
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    args = parser.parse_args()
    
    create_comparison_grid(args.log_dir)