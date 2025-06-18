
import torch
import numpy as np
import matplotlib.pyplot as plt
from improved_nerf import NeRF_Complete, NeRFDataset, render_image, create_spiral_poses
import argparse
import os
from tqdm import tqdm
import imageio

def render_test_set(model, dataset, args):
    """테스트 세트 렌더링"""
    model.eval()
    
    for idx in tqdm(range(len(dataset)), desc="Rendering test set"):
        data = dataset[idx]
        pose = data['pose'].to(args.device)
        H, W = dataset.H, dataset.W
        focal = dataset.focal
        
        rgb = render_image(model, pose, H, W, focal, dataset.near, dataset.far, args)
        
        # 저장
        save_path = os.path.join(args.output_dir, f'test_{idx:03d}.png')
        plt.imsave(save_path, rgb)

def render_novel_views(model, dataset, args, n_frames=120):
    """새로운 시점에서 렌더링"""
    model.eval()
    
    # 나선형 경로 생성
    poses = dataset.poses
    spiral_poses = create_spiral_poses(poses, n_frames=n_frames)
    
    H, W = dataset.H, dataset.W
    focal = dataset.focal
    
    frames = []
    for i, pose in enumerate(tqdm(spiral_poses, desc="Rendering novel views")):
        pose = torch.FloatTensor(pose).to(args.device)
        rgb = render_image(model, pose, H, W, focal, dataset.near, dataset.far, args)
        
        # 이미지 저장
        save_path = os.path.join(args.output_dir, f'novel_{i:03d}.png')
        plt.imsave(save_path, rgb)
        
        frames.append((rgb * 255).astype(np.uint8))
    
    # 비디오 저장
    video_path = os.path.join(args.output_dir, 'novel_views.mp4')
    imageio.mimwrite(video_path, frames, fps=30, quality=8)
    print(f"Video saved to {video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./renders',
                        help='Output directory')
    parser.add_argument('--render_test', action='store_true',
                        help='Render test set')
    parser.add_argument('--render_novel', action='store_true',
                        help='Render novel views')
    parser.add_argument('--n_frames', type=int, default=120,
                        help='Number of frames for novel view video')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 체크포인트 로드
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # 모델 생성
    model = NeRF_Complete(
        pos_L=10, dir_L=4, hidden_dim=256, 
        use_viewdirs=True, use_ndc=False
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 데이터셋 로드
    dataset = NeRFDataset(args.datadir, split='test', testskip=1)
    
    # 렌더링 인자 설정
    args.N_coarse = 64
    args.N_fine = 128
    args.chunk_size = 1024 * 4
    args.white_bkgd = True
    
    # 렌더링 수행
    if args.render_test:
        render_test_set(model, dataset, args)
    
    if args.render_novel:
        render_novel_views(model, dataset, args, n_frames=args.n_frames)

if __name__ == "__main__":
    main()
