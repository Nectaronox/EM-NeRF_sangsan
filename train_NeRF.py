import argparse
import os
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# 위에서 구현한 NeRF 모듈들을 import (실제로는 별도 파일로 저장)
# from improved_nerf import NeRF_Complete, NeRFDataset, train_nerf, render_image, get_rays

def create_args():
    """학습 인자 설정"""
    parser = argparse.ArgumentParser(description='NeRF Training')
    
    # 데이터 관련
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to dataset (e.g., nerf_synthetic/lego)')
    parser.add_argument('--dataset_type', type=str, default='synthetic',
                        choices=['synthetic', 'llff'], help='Dataset type')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='Downsample factor for images')
    parser.add_argument('--testskip', type=int, default=8,
                        help='Skip every N images in test set')
    
    # 모델 관련
    parser.add_argument('--pos_L', type=int, default=10,
                        help='Positional encoding levels for positions')
    parser.add_argument('--dir_L', type=int, default=4,
                        help='Positional encoding levels for directions')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for MLPs')
    parser.add_argument('--use_viewdirs', action='store_true', default=True,
                        help='Use view directions')
    parser.add_argument('--use_ndc', action='store_true', default=False,
                        help='Use NDC coordinates (for forward-facing scenes)')
    
    # 렌더링 관련
    parser.add_argument('--N_coarse', type=int, default=64,
                        help='Number of coarse samples')
    parser.add_argument('--N_fine', type=int, default=128,
                        help='Number of fine samples')
    parser.add_argument('--chunk_size', type=int, default=1024*4,
                        help='Chunk size for rendering')
    parser.add_argument('--white_bkgd', action='store_true', default=True,
                        help='Use white background for synthetic data')
    
    # 학습 관련
    parser.add_argument('--n_iters', type=int, default=200000,
                        help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size (number of rays)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=int, default=250,
                        help='Learning rate decay in thousands of steps')
    
    # 로깅 관련
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--val_freq', type=int, default=2500,
                        help='Validation frequency')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Checkpoint save frequency')
    parser.add_argument('--video_freq', type=int, default=50000,
                        help='Video generation frequency')
    
    # 기타
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def setup_experiment(args):
    """실험 환경 설정"""
    # 로그 디렉토리 생성
    exp_name = f"{os.path.basename(args.datadir)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.logdir = os.path.join(args.logdir, exp_name)
    os.makedirs(args.logdir, exist_ok=True)
    
    # 설정 저장
    with open(os.path.join(args.logdir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')
    
    # 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # CUDA 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return args

def create_spiral_poses(poses, n_frames=120, n_rots=2, zrate=0.5):
    """나선형 카메라 경로 생성 (비디오 렌더링용)"""
    c2w = poses[0]
    up = c2w[:3, 1]
    focus = np.array([0., 0., 0.])
    
    # 반지름 계산
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, axis=0)
    
    spiral_poses = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames):
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = c - focus
        z = z / np.linalg.norm(z)
        
        # 카메라 행렬 구성
        y = up
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        
        pose = np.stack([x, y, z, c], axis=-1)
        spiral_poses.append(pose)
    
    return np.stack(spiral_poses, axis=0)

@torch.no_grad()
def render_video(model, dataset, args, n_frames=120):
    """비디오 렌더링"""
    model.eval()
    
    # 나선형 카메라 경로 생성
    poses = dataset.poses
    spiral_poses = create_spiral_poses(poses, n_frames=n_frames)
    
    H, W = dataset.H, dataset.W
    focal = dataset.focal
    
    frames = []
    for i, pose in enumerate(tqdm(spiral_poses, desc="Rendering video")):
        pose = torch.FloatTensor(pose).to(args.device)
        rgb = render_image(model, pose, H, W, focal, dataset.near, dataset.far, args)
        frames.append((rgb * 255).astype(np.uint8))
    
    # 비디오 저장 (imageio 필요)
    import imageio
    video_path = os.path.join(args.logdir, f'spiral_{model.training_iter:06d}.mp4')
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    print(f"Video saved to {video_path}")

def plot_metrics(metrics, save_path):
    """학습 메트릭 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 플롯
    ax1.plot(metrics['iters'], metrics['losses'])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # PSNR 플롯
    ax2.plot(metrics['iters'], metrics['psnrs'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR')
    ax2.set_title('Training PSNR')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """메인 실행 함수"""
    # 인자 파싱 및 설정
    args = create_args()
    args = setup_experiment(args)
    
    print(f"Experiment: {os.path.basename(args.logdir)}")
    print(f"Device: {args.device}")
    
    # 데이터셋 로드
    print("Loading dataset...")
    train_dataset = NeRFDataset(args.datadir, split='train', downsample=args.downsample)
    val_dataset = NeRFDataset(args.datadir, split='val', downsample=args.downsample)
    test_dataset = NeRFDataset(args.datadir, split='test', downsample=args.downsample, 
                               testskip=args.testskip)
    
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Image shape: {train_dataset.H} x {train_dataset.W}")
    
    # 모델 생성
    print("Creating model...")
    model = NeRF_Complete(
        pos_L=args.pos_L,
        dir_L=args.dir_L,
        hidden_dim=args.hidden_dim,
        use_viewdirs=args.use_viewdirs,
        use_ndc=args.use_ndc
    ).to(args.device)
    
    # 체크포인트 로드 (있을 경우)
    start_iter = 0
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_iter = checkpoint['iter']
        print(f"Resumed from iteration {start_iter}")
    
    # 모델에 iteration 추적 추가
    model.training_iter = start_iter
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate 스케줄러
    def lr_lambda(iter):
        decay_rate = 0.1
        decay_steps = args.lr_decay * 1000
        return decay_rate ** (iter / decay_steps)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 메트릭 추적
    metrics = {
        'iters': [],
        'losses': [],
        'psnrs': [],
        'val_psnrs': []
    }
    
    # 학습 시작
    print("Starting training...")
    model.train()
    
    pbar = tqdm(range(start_iter, args.n_iters), desc="Training")
    for i in pbar:
        model.training_iter = i
        
        # 랜덤 이미지 선택
        idx = np.random.randint(0, len(train_dataset))
        data = train_dataset[idx]
        
        # 광선 샘플링
        rays_o, rays_d, target_rgb = sample_rays_batch(
            data['image'], data['pose'],
            train_dataset.focal, train_dataset.H, train_dataset.W,
            N_rays=args.batch_size
        )
        
        # GPU로 이동
        rays_o = rays_o.to(args.device)
        rays_d = rays_d.to(args.device)
        target_rgb = target_rgb.to(args.device)
        
        # Forward pass
        outputs = model.render_rays(
            rays_o, rays_d,
            train_dataset.near, train_dataset.far,
            N_coarse=args.N_coarse, N_fine=args.N_fine,
            perturb=True, white_bkgd=args.white_bkgd,
            H=train_dataset.H, W=train_dataset.W, focal=train_dataset.focal
        )
        
        # Loss 계산
        loss = compute_loss(outputs, target_rgb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 메트릭 기록
        if i % 100 == 0:
            psnr = compute_psnr(F.mse_loss(outputs['rgb_fine'], target_rgb))
            metrics['iters'].append(i)
            metrics['losses'].append(loss.item())
            metrics['psnrs'].append(psnr.item())
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{psnr.item():.2f}",
                'LR': f"{scheduler.get_last_lr()[0]:.6f}"
            })
        
        # Validation
        if i % args.val_freq == 0 and i > 0:
            model.eval()
            val_psnr = validate_full(model, val_dataset, args)
            metrics['val_psnrs'].append(val_psnr)
            model.train()
            
            # 메트릭 플롯 저장
            plot_metrics(metrics, os.path.join(args.logdir, 'metrics.png'))
        
        # 체크포인트 저장
        if i % args.save_freq == 0 and i > 0:
            checkpoint_path = os.path.join(args.logdir, f'checkpoint_{i:06d}.pt')
            torch.save({
                'iter': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args,
            }, checkpoint_path)
            print(f"\nCheckpoint saved to {checkpoint_path}")
        
        # 비디오 생성
        if i % args.video_freq == 0 and i > 0:
            render_video(model, test_dataset, args)
    
    # 최종 테스트
    print("\nFinal evaluation on test set...")
    model.eval()
    test_psnrs = []
    
    for idx in tqdm(range(len(test_dataset)), desc="Testing"):
        data = test_dataset[idx]
        pose = data['pose'].to(args.device)
        H, W = test_dataset.H, test_dataset.W
        focal = test_dataset.focal
        
        rgb = render_image(model, pose, H, W, focal, test_dataset.near, test_dataset.far, args)
        target = data['image'].numpy()
        
        mse = np.mean((rgb - target) ** 2)
        psnr = -10. * np.log10(mse)
        test_psnrs.append(psnr)
        
        # 테스트 이미지 저장
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(rgb)
        ax1.set_title(f'Rendered (PSNR: {psnr:.2f})')
        ax1.axis('off')
        ax2.imshow(target)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.logdir, f'test_{idx:03d}.png'))
        plt.close()
    
    avg_psnr = np.mean(test_psnrs)
    print(f"\nTest set average PSNR: {avg_psnr:.2f}")
    
    # 최종 결과 저장
    with open(os.path.join(args.logdir, 'results.txt'), 'w') as f:
        f.write(f"Test set average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Individual test PSNRs: {test_psnrs}\n")

@torch.no_grad()
def validate_full(model, dataset, args):
    """전체 검증 세트에 대한 평가"""
    model.eval()
    psnrs = []
    
    for idx in range(len(dataset)):
        data = dataset[idx]
        pose = data['pose'].to(args.device)
        H, W = dataset.H, dataset.W
        focal = dataset.focal
        
        rgb = render_image(model, pose, H, W, focal, dataset.near, dataset.far, args)
        target = data['image'].numpy()
        
        mse = np.mean((rgb - target) ** 2)
        psnr = -10. * np.log10(mse)
        psnrs.append(psnr)
    
    avg_psnr = np.mean(psnrs)
    print(f"\nValidation average PSNR: {avg_psnr:.2f}")
    
    return avg_psnr

if __name__ == "__main__":
    main()


